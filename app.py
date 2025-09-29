# app.py
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, BigInteger, DateTime, func
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import time
import requests
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from datetime import datetime
from cachetools import cached, TTLCache
from contextlib import contextmanager
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import io
import base64

# Инициализация
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1_000_000, backupCount=3),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Конфигурация SQLAlchemy
Base = declarative_base()
engine = None
SessionLocal = None
db_connection_active = True


# Модели данных
class CryptoRate(Base):
    __tablename__ = 'crypto_rates'

    id = Column(Integer, primary_key=True)
    crypto = Column(String(50), nullable=False)
    currency = Column(String(10), nullable=False)
    rate = Column(Float(precision=8), nullable=False)
    source = Column(String(100), nullable=False)
    timestamp = Column(BigInteger, nullable=False)


class AppLog(Base):
    __tablename__ = 'app_logs'

    id = Column(Integer, primary_key=True)
    service = Column(String(50), default='crypto_api')
    component = Column(String(50), default='backend')
    message = Column(Text, nullable=False)
    level = Column(String(20), nullable=False)
    traceback = Column(Text)
    user_id = Column(String(36))
    timestamp = Column(BigInteger, nullable=False)


def log_message(
        message: str,
        level: str = 'info',
        service: str = 'crypto_api',
        component: str = 'backend',
        traceback: str = None,
        user_id: str = None
):
    """Запись лога в БД и файл"""
    logging.log(
        getattr(logging, level.upper()),
        f"[{service}.{component}] User={user_id} | {message}",
        exc_info=traceback is not None
    )

    if db_connection_active:
        try:
            with get_db() as db:
                log_entry = AppLog(
                    message=message,
                    level=level,
                    service=service,
                    component=component,
                    traceback=traceback,
                    user_id=user_id,
                    timestamp=int(time.time())
                )
                db.add(log_entry)
                db.commit()
        except SQLAlchemyError as e:
            logging.error(f"Ошибка записи лога в БД: {e}")


def init_db_connection():
    """Инициализация подключения к базе данных"""
    global engine, SessionLocal, db_connection_active

    try:
        DATABASE_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
        engine = create_engine(
            DATABASE_URI,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine
            )
        )
        db_connection_active = True
        log_message("Инициализировано подключение к базе данных", 'info')
    except SQLAlchemyError as e:
        db_connection_active = False
        log_message(f"Ошибка подключения к базе данных: {e}", 'error')
        raise


@contextmanager
def get_db():
    """Контекстный менеджер для работы с сессией"""
    global db_connection_active

    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        log_message("Попытка доступа к отключенной БД", 'warning')
        raise RuntimeError("Соединение с базой данных отключено")

    if SessionLocal is None:
        init_db_connection()

    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        db.rollback()
        log_message(f"Ошибка БД: {e}", 'error')
        raise
    finally:
        db.close()


def init_db():
    """Инициализация структуры БД"""
    if not db_connection_active:
        log_message("Попытка инициализации БД при отключенном соединении", 'error')
        return

    try:
        Base.metadata.create_all(bind=engine)
        log_message("База данных инициализирована", 'info')
    except SQLAlchemyError as e:
        log_message(f"Ошибка инициализации БД: {e}", 'error')


# Константы
TOP_CRYPTO = ['bitcoin', 'ethereum', 'binancecoin']
CURRENCIES = ['usd', 'eur', 'gbp', 'jpy', 'cny', 'rub']
PERIODS = [
    {'value': '1', 'label': '1 день'},
    {'value': '7', 'label': '7 дней'},
    {'value': '30', 'label': '30 дней'},
    {'value': '90', 'label': '90 дней'},
    {'value': '180', 'label': '180 дней'},
    {'value': '365', 'label': '1 год'}
]
crypto_list_cache = TTLCache(maxsize=1, ttl=3600)


@cached(crypto_list_cache)
def load_crypto_list():
    """Загружает список криптовалют"""
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/list",
            timeout=5
        )
        response.raise_for_status()

        all_crypto = [c['id'] for c in response.json()]
        sorted_crypto = TOP_CRYPTO.copy()

        for crypto in all_crypto:
            if crypto not in TOP_CRYPTO:
                sorted_crypto.append(crypto)

        return sorted_crypto

    except requests.RequestException as e:
        logging.warning(f"Ошибка загрузки списка: {e}. Используем базовый набор.")
        return TOP_CRYPTO


def get_crypto_rate(crypto: str, currency: str) -> dict:
    """Получает курс криптовалюты к валюте"""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies={currency}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if currency not in data.get(crypto, {}):
            return {
                'success': False,
                'error': f"Курс для {crypto}/{currency} не найден",
                'source': 'coingecko',
                'timestamp': int(time.time())
            }

        return {
            'success': True,
            'rate': data[crypto][currency],
            'source': 'coingecko',
            'timestamp': int(time.time())
        }

    except requests.RequestException as e:
        return {
            'success': False,
            'error': f"Ошибка API: {str(e)}",
            'source': 'coingecko',
            'timestamp': int(time.time())
        }


class CoinGeckoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
    
    def get_ohlc(self, coin_id, vs_currency, days):
        """
        Получение OHLC данных (Open, High, Low, Close)
        
        Args:
            coin_id: идентификатор монеты (e.g., 'bitcoin')
            vs_currency: валюта (e.g., 'usd')
            days: период (1, 7, 14, 30, 90, 180, 365)
        """
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Конвертируем в DataFrame
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                logging.error(f"Ошибка API CoinGecko: {response.status_code}")
                return None
        except requests.RequestException as e:
            logging.error(f"Ошибка запроса к CoinGecko: {e}")
            return None

    def generate_plot(self, data, crypto, currency, period):
        """Генерирует график и возвращает его в base64"""
        if data is None or data.empty:
            return None
            
        try:
            # Создаем график
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['close'], linewidth=2)
            plt.title(f'{crypto.upper()}/{currency.upper()} Price ({period} дней)', fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(f'Price ({currency.upper()})', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Конвертируем график в base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plt.close()
            
            plot_url = base64.b64encode(img.getvalue()).decode()
            return f"data:image/png;base64,{plot_url}"
            
        except Exception as e:
            logging.error(f"Ошибка генерации графика: {e}")
            return None


@app.route('/disconnect_db', methods=['POST'])
def disconnect_db():
    """Отключение подключения к базе данных"""
    global db_connection_active, engine

    db_connection_active = False
    if engine:
        engine.dispose()

    flash("Соединение с базой данных отключено", 'warning')
    log_message("Ручное отключение базы данных", 'info')
    return redirect(url_for('index'))


@app.route('/connect_db', methods=['POST'])
def connect_db():
    """Включение подключения к базе данных"""
    global db_connection_active

    try:
        init_db_connection()
        db_connection_active = True
        flash("Соединение с базой данных восстановлено", 'success')
        log_message("Ручное подключение базы данных", 'info')
    except SQLAlchemyError as e:
        flash(f"Ошибка подключения: {str(e)}", 'error')
        log_message(f"Ошибка подключения к БД: {e}", 'error')

    return redirect(url_for('index'))


@app.route('/', methods=['GET', 'POST'])
def index():
    rate_data = None
    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')

        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            log_message("Не выбрана криптовалюта или валюта", 'warning')
            return redirect('/')

        rate_data = get_crypto_rate(crypto, currency)

        if not rate_data['success']:
            flash(rate_data['error'], 'error')
            log_message(rate_data['error'], 'error')
            return redirect('/')

        try:
            with get_db() as db:
                rate_entry = CryptoRate(
                    crypto=crypto,
                    currency=currency,
                    rate=round(rate_data['rate'], 8),
                    source=rate_data['source'],
                    timestamp=rate_data['timestamp']
                )
                db.add(rate_entry)
                db.commit()

            msg = f"Курс {crypto.upper()}/{currency.upper()}: {rate_data['rate']:.4f}"
            flash(msg, 'success')
            log_message(msg, 'info')

        except SQLAlchemyError as e:
            error_msg = f"Ошибка БД: {str(e)}"
            flash(error_msg, 'error')
            log_message(error_msg, 'error', traceback=str(e))

    cryptos = load_crypto_list()
    return render_template('index.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=PERIODS,
                           rate=rate_data.get('rate') if request.method == 'POST' else None,
                           db_connected=db_connection_active)


@app.route('/chart', methods=['GET', 'POST'])
def chart():
    """Страница с графиком курса криптовалюты"""
    plot_url = None
    error = None
    
    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')
        period = request.form.get('period', '7')
        
        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            return redirect(url_for('chart'))
        
        try:
            api = CoinGeckoAPI()
            data = api.get_ohlc(crypto, currency, period)
            
            if data is not None:
                plot_url = api.generate_plot(data, crypto, currency, period)
                if plot_url:
                    log_message(f"Сгенерирован график для {crypto}/{currency} за {period} дней", 'info')
                else:
                    error = "Ошибка генерации графика"
            else:
                error = "Не удалось получить данные для построения графика"
                
        except Exception as e:
            error = f"Ошибка: {str(e)}"
            log_message(f"Ошибка построения графика: {e}", 'error')
    
    cryptos = load_crypto_list()
    return render_template('chart.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=PERIODS,
                           plot_url=plot_url,
                           error=error)


@app.route('/crypto_table')
def show_crypto_table():
    """Отображает таблицу курсов криптовалют"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    try:
        with get_db() as db:
            # Запрашиваем только существующие столбцы
            rates = db.query(
                CryptoRate.crypto,
                CryptoRate.currency,
                CryptoRate.rate,
                CryptoRate.source,
                CryptoRate.timestamp
            ).order_by(CryptoRate.timestamp.desc()).limit(100).all()

        # Форматируем данные для шаблона
        rates_data = [{
            'crypto': r.crypto,
            'currency': r.currency,
            'rate': r.rate,
            'date_time': datetime.fromtimestamp(r.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'source': r.source
        } for r in rates]

        return render_template('data_table.html',
                               title='Курсы криптовалют',
                               data=rates_data,
                               columns=['crypto', 'currency', 'rate', 'date_time', 'source'])
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        return redirect(url_for('index'))


@app.route('/log_table')
def show_log_table():
    """Отображает таблицу логов"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    try:
        with get_db() as db:
            logs = db.query(
                AppLog.id,
                AppLog.service,
                AppLog.component,
                AppLog.message,
                AppLog.level,
                AppLog.traceback,
                AppLog.user_id,
                AppLog.timestamp
            ).order_by(AppLog.timestamp.desc()).limit(100).all()

        logs_data = [{
            'date_time': datetime.fromtimestamp(l.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'level': l.level,
            'message': l.message,
            'service': l.service,
            'component': l.component,
            'user_id': l.user_id,
            'traceback': l.traceback
        } for l in logs]

        return render_template('data_table.html',
                               title='Логи приложения',
                               data=logs_data,
                               columns=['date_time', 'level', 'message', 'service', 'component', 'user_id',
                                        'traceback'])
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        return redirect(url_for('index'))


if __name__ == '__main__':
    init_db_connection()
    init_db()
    log_message("Приложение запущено", 'info')
    app.run(debug=True)
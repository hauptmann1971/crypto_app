# app.py
from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, BigInteger, DateTime, func, Boolean
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import time
import requests
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from datetime import datetime, timedelta
from cachetools import cached, TTLCache
from contextlib import contextmanager
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import io
import base64
import telegram_auth as t_a
import phonenumbers
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional
from crypto_chart import crypto_chart_api
import threading

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
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Конфигурация SQLAlchemy
Base = declarative_base()
engine = None
SessionLocal = None
db_connection_active = True

# Конфигурация Telegram
BOT_USERNAME = os.getenv('BOT_USERNAME', '@romanov_crypto_currency_bot')
BOT_TOKEN = os.getenv('BOT_TOKEN', '8264247176:AAFByVrbcY8K-aanicYu2QK-tYRaFNq0lxY')

# Словарь популярных криптовалют (50 штук)
POPULAR_CRYPTOS = [
    'bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano', 'solana',
    'polkadot', 'dogecoin', 'matic-network', 'stellar', 'litecoin', 'chainlink',
    'bitcoin-cash', 'ethereum-classic', 'monero', 'eos', 'tezos', 'aave',
    'cosmos', 'uniswap', 'tron', 'neo', 'vechain', 'theta-token', 'filecoin',
    'algorand', 'maker', 'compound-governance-token', 'dash', 'zcash',
    'decred', 'waves', 'ontology', 'icon', 'zilliqa', 'bittorrent',
    'pancakeswap-token', 'sushi', 'curve-dao-token', 'yearn-finance',
    'balancer', 'uma', 'renbtc', 'helium', 'chiliz', 'enjincoin',
    'axie-infinity', 'the-sandbox', 'decentraland', 'gala'
]

# Глобальные переменные для асинхронной загрузки
FULL_CRYPTO_LIST = POPULAR_CRYPTOS.copy()
CRYPTO_LOADING = False
CRYPTO_LOADED = False

@dataclass
class TelegramUser:
    id: int
    first_name: str
    auth_date: int
    hash: str
    username: Optional[str] = None
    photo_url: Optional[str] = None
    last_name: Optional[str] = None

def verify_telegram_authentication(data: dict, bot_token: str) -> bool:
    """
    Проверяет данные авторизации Telegram
    """
    try:
        # Проверяем обязательные поля
        required_fields = ['id', 'first_name', 'auth_date', 'hash']
        for field in required_fields:
            if field not in data:
                return False

        # Проверяем, что данные не устарели (не старше 24 часов)
        auth_date = datetime.fromtimestamp(int(data['auth_date']))
        if datetime.now() - auth_date > timedelta(hours=24):
            return False

        # Формируем строку для проверки
        data_check_string = '\n'.join(
            f'{key}={value}'
            for key, value in sorted(data.items())
            if key != 'hash'
        )

        # Вычисляем секретный ключ
        secret_key = hashlib.sha256(bot_token.encode()).digest()

        # Вычисляем хэш
        computed_hash = hmac.new(
            secret_key,
            data_check_string.encode(),
            hashlib.sha256
        ).hexdigest()

        # Сравниваем хэши безопасным способом
        return hmac.compare_digest(computed_hash, data['hash'])

    except Exception as e:
        print(f"Error verifying Telegram auth: {e}")
        return False

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


# Добавь эту модель после других моделей
class TelegramUser(Base):
    __tablename__ = 'telegram_users'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100))
    username = Column(String(100))
    photo_url = Column(Text)
    auth_date = Column(BigInteger, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    last_login = Column(BigInteger, nullable=False)
    is_active = Column(Boolean, default=True)


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

    db = SessionLocal
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


def load_full_crypto_list_async():
    """Асинхронно загружает полный список криптовалют"""
    global FULL_CRYPTO_LIST, CRYPTO_LOADING, CRYPTO_LOADED, POPULAR_CRYPTOS
    
    if CRYPTO_LOADING or CRYPTO_LOADED:
        return
    
    CRYPTO_LOADING = True
    
    def load_thread():
        global FULL_CRYPTO_LIST, CRYPTO_LOADING, CRYPTO_LOADED
        try:
            logging.info("Starting async full crypto list loading...")
            response = requests.get(
                "https://api.coingecko.com/api/v3/coins/list",
                timeout=30
            )
            response.raise_for_status()
            
            all_crypto = [c['id'] for c in response.json()]
            
            # Объединяем популярные + полный список (убираем дубликаты)
            combined_list = POPULAR_CRYPTOS.copy()
            for crypto in all_crypto:
                if crypto not in combined_list:
                    combined_list.append(crypto)
            
            FULL_CRYPTO_LIST = combined_list
            CRYPTO_LOADED = True
            CRYPTO_LOADING = False
            
            logging.info(f"Full crypto list loaded: {len(FULL_CRYPTO_LIST)} items")
            
        except Exception as e:
            logging.error(f"Async crypto list loading failed: {e}")
            CRYPTO_LOADING = False
    
    # Запускаем в отдельном потоке
    thread = threading.Thread(target=load_thread)
    thread.daemon = True
    thread.start()

@cached(crypto_list_cache)
def load_crypto_list():
    """Возвращает список криптовалют (сначала популярные, потом полный список)"""
    # Запускаем асинхронную загрузку полного списка
    load_full_crypto_list_async()
    
    # Всегда возвращаем актуальный список
    return FULL_CRYPTO_LIST

# Запускаем асинхронную загрузку при импорте модуля
load_full_crypto_list_async()

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


def get_user_from_db(telegram_id: int) -> Optional[TelegramUser]:
    """Получает пользователя из базы данных по telegram_id"""
    try:
        with get_db() as db:
            user = db.query(TelegramUser).filter(
                TelegramUser.telegram_id == telegram_id,
                TelegramUser.is_active == True
            ).first()
            return user
    except SQLAlchemyError as e:
        log_message(f"Error getting user from DB: {e}", 'error')
        return None

def update_user_last_login(telegram_id: int):
    """Обновляет время последнего входа пользователя"""
    try:
        with get_db() as db:
            user = db.query(TelegramUser).filter(
                TelegramUser.telegram_id == telegram_id
            ).first()
            if user:
                user.last_login = int(time.time())
                db.commit()
    except SQLAlchemyError as e:
        log_message(f"Error updating user last login: {e}", 'error')


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


@app.route('/telegram-auth', methods=['POST'])
def telegram_auth():
    """
    Обрабатывает callback от Telegram Login Widget
    """
    try:
        user_data = request.get_json()
        
        print("=" * 50)
        print("TELEGRAM AUTH DEBUG INFO:")
        print(f"Received data: {user_data}")
        print("=" * 50)
        
        if not user_data:
            print("ERROR: No user data received")
            return jsonify({'success': False, 'error': 'No data received'})
        
        # Проверяем подлинность данных
        if not verify_telegram_authentication(user_data, BOT_TOKEN):
            print("ERROR: Telegram authentication failed")
            return jsonify({'success': False, 'error': 'Invalid authentication data'})
        
        # Сохраняем/обновляем пользователя в БД
        try:
            with get_db() as db:
                # Проверяем существует ли пользователь
                existing_user = db.query(TelegramUser).filter(
                    TelegramUser.telegram_id == user_data['id']
                ).first()
                
                current_time = int(time.time())
                
                if existing_user:
                    # Обновляем существующего пользователя
                    existing_user.first_name = user_data['first_name']
                    existing_user.last_name = user_data.get('last_name')
                    existing_user.username = user_data.get('username')
                    existing_user.photo_url = user_data.get('photo_url')
                    existing_user.auth_date = user_data['auth_date']
                    existing_user.last_login = current_time
                    existing_user.is_active = True
                    
                    db.commit()
                    user_id = existing_user.id
                    action = "updated"
                else:
                    # Создаем нового пользователя
                    new_user = TelegramUser(
                        telegram_id=user_data['id'],
                        first_name=user_data['first_name'],
                        last_name=user_data.get('last_name'),
                        username=user_data.get('username'),
                        photo_url=user_data.get('photo_url'),
                        auth_date=user_data['auth_date'],
                        created_at=current_time,
                        last_login=current_time,
                        is_active=True
                    )
                    
                    db.add(new_user)
                    db.commit()
                    user_id = new_user.id
                    action = "created"
                
                print(f"SUCCESS: User {user_data['id']} {action} in database")
                
        except SQLAlchemyError as e:
            print(f"ERROR: Database operation failed: {e}")
            log_message(f"Database error during user save: {e}", 'error')
            return jsonify({'success': False, 'error': 'Database error'})
        
        # Сохраняем пользователя в сессии
        session['user'] = {
            'id': user_data['id'],
            'first_name': user_data['first_name'],
            'username': user_data.get('username'),
            'photo_url': user_data.get('photo_url'),
            'auth_date': user_data['auth_date'],
            'db_user_id': user_id
        }
        
        log_message(f"User {user_data['id']} successfully authenticated via Telegram (DB ID: {user_id})", 
                   'info', user_id=str(user_data['id']))
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"ERROR: Telegram auth exception: {e}")
        log_message(f"Telegram auth error: {e}", 'error')
        return jsonify({'success': False, 'error': str(e)})


@app.route('/logout')
def logout():
    """
    Выход пользователя
    """
    user_id = session.get('user', {}).get('id')
    session.pop('user', None)
    flash('Вы успешно вышли из системы', 'success')
    log_message("User logged out", 'info', user_id=str(user_id) if user_id else None)
    return redirect(url_for('index'))


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
    user_session = session.get('user')
    user_from_db = None
    
    # Если пользователь авторизован в сессии, загружаем данные из БД
    if user_session:
        user_from_db = get_user_from_db(user_session['id'])
        if user_from_db:
            # Обновляем время последнего входа
            update_user_last_login(user_session['id'])
    
    if request.method == 'POST':
        crypto = request.form.get('crypto', 'bitcoin')
        currency = request.form.get('currency', 'usd')

        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            log_message("Не выбрана криптовалюта или валюта", 'warning', 
                       user_id=str(user_session['id']) if user_session else None)
            return redirect('/')

        rate_data = get_crypto_rate(crypto, currency)

        if not rate_data['success']:
            flash(rate_data['error'], 'error')
            log_message(rate_data['error'], 'error', 
                       user_id=str(user_session['id']) if user_session else None)
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
            log_message(msg, 'info', 
                       user_id=str(user_session['id']) if user_session else None)

        except SQLAlchemyError as e:
            error_msg = f"Ошибка БД: {str(e)}"
            flash(error_msg, 'error')
            log_message(error_msg, 'error', traceback=str(e), 
                       user_id=str(user_session['id']) if user_session else None)

    cryptos = load_crypto_list()
    return render_template('index.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=PERIODS,
                           rate=rate_data.get('rate') if request.method == 'POST' else None,
                           db_connected=db_connection_active,
                           user=user_from_db,  # Передаем пользователя из БД
                           bot_username=BOT_USERNAME)


@app.route('/auth', methods=['GET', 'POST'])
def auth():
    """Страница авторизации через telegram"""
    user = session.get('user')
    
    if user:
        flash("Вы уже авторизованы!", 'info')
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('auth.html', user=user, bot_username=BOT_USERNAME)
    
    # POST обработка
    phone_number = request.form.get('phone_number')
    
    # Проверка наличия номера
    if not phone_number:
        flash("Введите номер телефона", 'error')
        return redirect(url_for('auth'))
    
    # Добавляем + если его нет
    if not phone_number.startswith('+'):
        phone_number = '+' + phone_number
    
    """
    Проверка с использованием библиотеки phonenumbers
    """
    try:
        parsed = phonenumbers.parse(phone_number, None)
        if not phonenumbers.is_valid_number(parsed):
            flash("Неверный формат номера телефона", 'error')
            return redirect(url_for('auth'))
    except phonenumbers.NumberParseException:
        flash("Введите номер телефона в международном формате (например: +79161234567)", 'error')
        return redirect(url_for('auth'))

    # Отправка сообщения через Telegram
    try:
        response = t_a.send_message(phone_number)
        res = response.get('delivery_status')
        if res.get("status") == "sent":
            t_a.last_request_response = response
            flash("Код подтверждения отправлен в Telegram! Проверьте ваши сообщения.", 'success')
            log_message(f"Успешная отправка кода для номера: {phone_number}", 'info')
            return redirect(url_for('auth'))
            
        else:
            flash("Ошибка отправки сообщения. Проверьте номер и попробуйте снова.", 'error')
            log_message(f"Ошибка отправки кода для номера: {phone_number}", 'error')
            return redirect(url_for('auth'))
            
    except Exception as e:
        error_msg = f"Ошибка при отправке сообщения: {str(e)}"
        flash(error_msg, 'error')
        log_message(f"Исключение при отправке Telegram сообщения: {e}", 'error')
        return redirect(url_for('auth'))


@app.route('/verify_code', methods=['POST'])
def verify_code():
    """Проверка кода подтверждения из Telegram"""
    verification_code = request.form.get('verification_code')
    
    if not verification_code or len(verification_code) != 6:
        flash("Введите 6-значный код подтверждения", 'error')
        return redirect(url_for('auth'))
    
    # Проверка кода авторнизации через модуль telegram_auth
    try:
        if t_a.verify_code(t_a.last_request_response, verification_code):  # Предполагаем, что такая функция есть
            flash("Авторизация успешно завершена! Добро пожаловать!", 'success')
            log_message("Успешная авторизация через Telegram", 'info')
            return redirect(url_for('index'))
        else:
            flash("Неверный код подтверждения. Попробуйте снова.", 'error')
            return redirect(url_for('auth'))
            
    except Exception as e:
        flash(f"Ошибка при проверке кода: {str(e)}", 'error')
        log_message(f"Ошибка проверки кода: {e}", 'error')
        return redirect(url_for('auth'))


@app.route('/chart', methods=['GET', 'POST'])
def chart():
    """Страница с графиком курса криптовалюты"""
    plot_url = None
    error = None
    user = session.get('user')
    
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
                    log_message(f"Сгенерирован график для {crypto}/{currency} за {period} дней", 'info', user_id=str(user['id']) if user else None)
                else:
                    error = "Ошибка генерации графика"
            else:
                error = "Не удалось получить данные для построения графика"
                
        except Exception as e:
            error = f"Ошибка: {str(e)}"
            log_message(f"Ошибка построения графика: {e}", 'error', user_id=str(user['id']) if user else None)
    
    cryptos = load_crypto_list()
    return render_template('chart.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=PERIODS,
                           plot_url=plot_url,
                           error=error,
                           user=user)


@app.route('/crypto_table')
def show_crypto_table():
    """Отображает таблицу курсов криптовалют"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')

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
                               columns=['crypto', 'currency', 'rate', 'date_time', 'source'],
                               user=user)
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        return redirect(url_for('index'))


@app.route('/log_table')
def show_log_table():
    """Отображает таблицу логов"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')

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
                                        'traceback'],
                               user=user)
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        return redirect(url_for('index'))


@app.route('/users_table')
def show_users_table():
    """Отображает таблицу пользователей"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    try:
        with get_db() as db:
            users = db.query(TelegramUser).order_by(TelegramUser.last_login.desc()).limit(100).all()

        users_data = [{
            'id': u.id,
            'telegram_id': u.telegram_id,
            'first_name': u.first_name,
            'last_name': u.last_name,
            'username': u.username,
            'created_at': datetime.fromtimestamp(u.created_at).strftime('%Y-%m-%d %H:%M:%S'),
            'last_login': datetime.fromtimestamp(u.last_login).strftime('%Y-%m-%d %H:%M:%S'),
            'is_active': 'Да' if u.is_active else 'Нет'
        } for u in users]

        return render_template('data_table.html',
                               title='Пользователи Telegram',
                               data=users_data,
                               columns=['id', 'telegram_id', 'first_name', 'last_name', 'username', 'created_at', 'last_login', 'is_active'])
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/candlestick', methods=['GET', 'POST'])
def candlestick_chart():
    """Страница со свечным графиком"""
    plot_url = None
    error = None
    chart_data = None
    
    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')
        period = request.form.get('period', '7')
        chart_type = request.form.get('chart_type', 'candlestick')
        
        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            return redirect(url_for('candlestick_chart'))
        
        try:
            # Получаем OHLC данные для свечного графика
            if chart_type == 'candlestick':
                data = crypto_chart_api.get_ohlc_data(crypto, currency, period)
                if data is not None:
                    plot_url = crypto_chart_api.create_candlestick_chart(data, crypto, currency, period)
                    chart_data = data
                else:
                    error = "Не удалось получить данные для свечного графика"
            else:
                # Простой линейный график
                data = crypto_chart_api.get_historical_data(crypto, currency, period)
                if data is not None:
                    plot_url = crypto_chart_api.create_simple_price_chart(data, crypto, currency, period, chart_type)
                    chart_data = data
                else:
                    error = "Не удалось получить данные для графика"
            
            if plot_url:
                log_message(f"Сгенерирован {chart_type} график для {crypto}/{currency} за {period} дней", 'info')
            elif not error:
                error = "Ошибка генерации графика"
                
        except Exception as e:
            error = f"Ошибка: {str(e)}"
            log_message(f"Ошибка построения графика: {e}", 'error')
    
    cryptos = load_crypto_list()
    periods = crypto_chart_api.get_available_periods()
    
    return render_template('candlestick.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=periods,
                           plot_url=plot_url,
                           chart_data=chart_data,
                           error=error)

@app.route('/chart-data/<crypto>/<currency>/<period>')
def get_chart_data(crypto: str, currency: str, period: str):
    """API endpoint для получения данных графика в JSON"""
    try:
        data = crypto_chart_api.get_ohlc_data(crypto, currency, period)
        if data is not None:
            # Конвертируем DataFrame в JSON
            chart_data = []
            for idx, row in data.iterrows():
                chart_data.append({
                    'timestamp': idx.isoformat(),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                })
            return jsonify({'success': True, 'data': chart_data})
        else:
            return jsonify({'success': False, 'error': 'No data available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})               


if __name__ == '__main__':
    init_db_connection()
    init_db()
    log_message("Приложение запущено", 'info')
    app.run(debug=True)
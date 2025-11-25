# app.py
import threading

from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, BigInteger, Boolean
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
import json

matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import io
import base64
import phonenumbers
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional

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
db_connection_active = False

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
class TelegramUserData:
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


class CryptoRequest(Base):
    __tablename__ = 'crypto_requests'

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)  # ID пользователя из сессии или 0 для анонимных
    crypto = Column(String(50), nullable=False)
    currency = Column(String(10), nullable=False)
    status = Column(String(20), default='pending')  # pending, processing, finished, error
    response_data = Column(Text)  # JSON ответ от API
    created_at = Column(BigInteger, nullable=False)
    finished_at = Column(BigInteger)  # Время завершения обработки


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

    # Инициализируем подключение, если оно еще не создано
    if SessionLocal is None:
        init_db_connection()

    # Создаем сессию с проверкой на None
    if SessionLocal is not None:
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            log_message(f"Ошибка БД: {e}", 'error')
            raise
        finally:
            db.close()
    else:
        raise RuntimeError("Не удалось инициализировать подключение к базе данных")


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


# Функции для работы с очередью запросов
def create_crypto_request(user_id: int, crypto: str, currency: str) -> int:
    """Создает новый запрос в очереди и сразу помечает как processing"""
    try:
        with get_db() as db:
            # Проверяем, нет ли уже активного запроса для этой пары
            existing_request = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id,
                CryptoRequest.crypto == crypto,
                CryptoRequest.currency == currency,
                CryptoRequest.status.in_(['pending', 'processing'])
            ).first()

            if existing_request:
                log_message(f"Active request already exists for {crypto}/{currency}, returning existing ID", 'info',
                            user_id=str(user_id))
                return existing_request.id

            # Создаем новый запрос и сразу помечаем как processing
            request = CryptoRequest(
                user_id=user_id,
                crypto=crypto,
                currency=currency,
                status='processing',  # Сразу processing для worker
                created_at=int(time.time())
            )
            db.add(request)
            db.commit()
            # Получаем ID напрямую из объекта после коммита
            request_id = request.id
            log_message(f"Created crypto request {request_id} for {crypto}/{currency}", 'info', user_id=str(user_id))
            return request_id
    except SQLAlchemyError as e:
        log_message(f"Error creating crypto request: {e}", 'error')
        return -1


def mark_request_as_error(request_id: int, error_message: str):
    """Помечает запрос как ошибочный"""
    try:
        with get_db() as db:
            request = db.query(CryptoRequest).filter(CryptoRequest.id == request_id).first()
            if request:
                request.status = 'error'
                request.response_data = json.dumps({'error': error_message})
                db.commit()
                log_message(f"Marked request {request_id} as error: {error_message}", 'error')
    except Exception as e:
        log_message(f"Failed to mark request {request_id} as error: {e}", 'error')


def get_pending_requests_count() -> int:
    """Получает количество pending запросов"""
    try:
        with get_db() as db:
            count = db.query(CryptoRequest).filter(
                CryptoRequest.status == 'pending'
            ).count()
            return count
    except SQLAlchemyError as e:
        log_message(f"Error getting pending requests count: {e}", 'error')
        return 0


def get_processing_requests_count() -> int:
    """Получает количество processing запросов (ожидающих worker)"""
    try:
        with get_db() as db:
            count = db.query(CryptoRequest).filter(
                CryptoRequest.status == 'processing'
            ).count()
            return count
    except SQLAlchemyError as e:
        log_message(f"Error getting processing requests count: {e}", 'error')
        return 0


def get_latest_finished_request(user_id: int) -> Optional[dict]:
    """Получает последний завершенный запрос для пользователя"""
    try:
        with get_db() as db:
            request = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id,
                CryptoRequest.status == 'finished'
            ).order_by(CryptoRequest.finished_at.desc()).first()

            if request and request.response_data:
                try:
                    response = json.loads(request.response_data)
                    return {
                        'id': request.id,
                        'crypto': request.crypto,
                        'currency': request.currency,
                        'rate': response.get('rate'),
                        'status': request.status,
                        'created_at': request.created_at,
                        'finished_at': request.finished_at,
                        'response_data': response
                    }
                except json.JSONDecodeError:
                    log_message(f"Error decoding JSON for request {request.id}", 'error')
                    return None
            return None
    except SQLAlchemyError as e:
        log_message(f"Error getting latest finished request: {e}", 'error')
        return None


def get_user_requests_history(user_id: int, limit: int = 10) -> list:
    """Получает историю запросов пользователя"""
    try:
        with get_db() as db:
            requests = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id
            ).order_by(CryptoRequest.created_at.desc()).limit(limit).all()

            result = []
            for req in requests:
                # Обрабатываем данные внутри сессии
                response_data = {}
                if req.response_data:
                    try:
                        response_data = json.loads(req.response_data)
                    except json.JSONDecodeError:
                        pass

                result.append({
                    'id': req.id,
                    'crypto': req.crypto,
                    'currency': req.currency,
                    'status': req.status,
                    'rate': response_data.get('rate') if response_data else None,
                    'created_at': req.created_at,
                    'finished_at': req.finished_at,
                    'error': response_data.get('error') if response_data else None
                })
            return result
    except SQLAlchemyError as e:
        log_message(f"Error getting user requests history: {e}", 'error')
        return []


def process_pending_requests():
    """Обрабатывает ТОЛЬКО СТАРЫЕ pending запросы (для backward compatibility)"""
    try:
        if not db_connection_active:
            return

        with get_db() as db:
            # Получаем только СТАРЫЕ pending запросы (созданные более 1 минуты назад)
            # Это для запросов, которые могли остаться с предыдущих версий
            one_minute_ago = int(time.time()) - 60
            old_pending_requests = db.query(CryptoRequest).filter(
                CryptoRequest.status == 'pending',
                CryptoRequest.created_at < one_minute_ago
            ).order_by(CryptoRequest.created_at.asc()).limit(2).all()

            processed_count = 0
            for request in old_pending_requests:
                # Помечаем как processing - worker заберет их
                request.status = 'processing'
                processed_count += 1

            if processed_count > 0:
                db.commit()
                log_message(f"Marked {processed_count} OLD pending requests as processing for worker", 'info')

    except SQLAlchemyError as e:
        log_message(f"Error processing OLD pending requests: {e}", 'error')


class CoinGeckoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_ohlc(self, coin_id, vs_currency, days):
        """
        Получение OHLC данных (Open, High, Low, Close)
        Только для графиков - не для основных курсов
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
            # Создаем свечной график
            fig, ax = plt.subplots(figsize=(12, 6))

            # Определяем цвета для свечей (зеленые - рост, красные - падение)
            colors = []
            for i in range(len(data)):
                if data['close'].iloc[i] >= data['open'].iloc[i]:
                    colors.append('green')  # Рост
                else:
                    colors.append('red')  # Падение

            # Рисуем свечи
            for i in range(len(data)):
                # Тело свечи
                ax.fill_between([i - 0.3, i + 0.3],
                                [data['open'].iloc[i], data['open'].iloc[i]],
                                [data['close'].iloc[i], data['close'].iloc[i]],
                                color=colors[i], alpha=0.7)

                # Верхняя тень
                ax.plot([i, i], [data['high'].iloc[i], max(data['open'].iloc[i], data['close'].iloc[i])],
                        color=colors[i], linewidth=1)

                # Нижняя тень
                ax.plot([i, i], [min(data['open'].iloc[i], data['close'].iloc[i]), data['low'].iloc[i]],
                        color=colors[i], linewidth=1)

            # Настройки графика
            ax.set_title(f'{crypto.upper()}/{currency.upper()} Свечной график ({period} дней)',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Временной период', fontsize=12)
            ax.set_ylabel(f'Цена ({currency.upper()})', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Упрощаем оси X (показываем только некоторые метки)
            n = len(data)
            step = max(1, n // 10)  # Показываем примерно 10 меток
            ax.set_xticks(range(0, n, step))
            ax.set_xticklabels([data.index[i].strftime('%m-%d') for i in range(0, n, step)], rotation=45)

            plt.tight_layout()

            # Конвертируем график в base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plt.close()

            plot_url = base64.b64encode(img.getvalue()).decode()
            return f"data:image/png;base64,{plot_url}"

        except Exception as e:
            logging.error(f"Ошибка генерации свечного графика: {e}")
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


@app.route('/auth', methods=['GET'])
def auth():
    """Страница авторизации через Telegram"""
    user = session.get('user')

    if user:
        flash("Вы уже авторизованы!", 'info')
        return redirect(url_for('index'))

    return render_template('auth.html', user=user, bot_username=BOT_USERNAME)


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
    if engine and hasattr(engine, 'dispose'):
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


@app.route('/request-status/<int:request_id>')
def get_request_status(request_id):
    """Проверяет статус запроса"""
    try:
        with get_db() as db:
            request = db.query(CryptoRequest).filter(
                CryptoRequest.id == request_id
            ).first()

            if not request:
                return jsonify({'success': False, 'error': 'Request not found'})

            response_data = {}
            if request.response_data:
                try:
                    response_data = json.loads(request.response_data)
                except json.JSONDecodeError:
                    pass

            return jsonify({
                'success': True,
                'status': request.status,
                'rate': response_data.get('rate'),
                'error': response_data.get('error'),
                'finished_at': request.finished_at,
                'crypto': request.crypto,
                'currency': request.currency
            })
    except SQLAlchemyError as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/', methods=['GET', 'POST'])
def index():
    current_rate = None
    current_crypto = None
    current_currency = None
    current_request_id = None
    requests_history = []
    pending_requests_count = 0
    processing_requests_count = 0
    user_session = session.get('user')
    user_from_db = None
    user_id = user_session['id'] if user_session else 0

    # Если пользователь авторизован в сессии, загружаем данные из БД
    if user_session:
        user_from_db = get_user_from_db(user_session['id'])
        if user_from_db:
            # Обновляем время последнего входа
            update_user_last_login(user_session['id'])

        # Получаем историю запросов
        requests_history = get_user_requests_history(user_session['id'])

        # Получаем количество pending и processing запросов
        pending_requests_count = get_pending_requests_count()
        processing_requests_count = get_processing_requests_count()

    if request.method == 'POST':
        crypto = request.form.get('crypto', 'bitcoin')
        currency = request.form.get('currency', 'usd')

        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            log_message("Не выбрана криптовалюта или валюта", 'warning',
                        user_id=str(user_session['id']) if user_session else None)
            return redirect('/')

        # Сохраняем выбранные значения для отображения
        current_crypto = crypto
        current_currency = currency

        # Создаем запрос в очереди (автоматически помечается как processing)
        request_id = create_crypto_request(user_id, crypto, currency)

        if request_id > 0:
            # Сохраняем ID последнего запроса в сессии для отслеживания
            session['last_request_id'] = request_id
            session['current_crypto'] = crypto
            session['current_currency'] = currency
            session.modified = True

            flash(
                f"Запрос на получение курса {crypto.upper()}/{currency.upper()} отправлен в обработку. ID: {request_id}",
                'info')
            log_message(f"Request created for {crypto}/{currency} (ID: {request_id})", 'info',
                        user_id=str(user_session['id']) if user_session else None)

            # Обновляем счетчики и историю
            pending_requests_count = get_pending_requests_count()
            processing_requests_count = get_processing_requests_count()
            if user_session:
                requests_history = get_user_requests_history(user_session['id'])
        else:
            flash("Ошибка при создании запроса", 'error')
            log_message("Error creating request in queue", 'error',
                        user_id=str(user_session['id']) if user_session else None)

    else:
        # GET запрос - пытаемся получить текущий курс
        current_crypto = session.get('current_crypto')
        current_currency = session.get('current_currency')

        if user_session and current_crypto and current_currency:
            # Получаем последний завершенный запрос для этой пары
            latest_request = get_latest_finished_request(user_session['id'])
            if latest_request and latest_request.get('rate'):
                current_rate = latest_request['rate']
                current_crypto = latest_request['crypto']
                current_currency = latest_request['currency']
                current_request_id = latest_request['id']

    # Обрабатываем ТОЛЬКО СТАРЫЕ pending запросы (для совместимости)
    process_pending_requests()

    cryptos = load_crypto_list()
    return render_template('index.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=PERIODS,
                           current_rate=current_rate,
                           current_crypto=current_crypto,
                           current_currency=current_currency,
                           current_request_id=current_request_id,
                           db_connected=db_connection_active,
                           user=user_from_db,
                           requests_history=requests_history,
                           pending_requests_count=pending_requests_count,
                           processing_requests_count=processing_requests_count,
                           bot_username=BOT_USERNAME)


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
                    log_message(f"Сгенерирован график для {crypto}/{currency} за {period} дней", 'info',
                                user_id=str(user['id']) if user else None)
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


@app.route('/candlestick', methods=['GET', 'POST'])
def candlestick_chart():
    """Страница со свечным графиком"""
    plot_url = None
    error = None
    user = session.get('user')

    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')
        period = request.form.get('period', '7')

        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            return redirect(url_for('candlestick_chart'))

        try:
            api = CoinGeckoAPI()
            data = api.get_ohlc(crypto, currency, period)

            if data is not None:
                plot_url = api.generate_plot(data, crypto, currency, period)
                if plot_url:
                    log_message(f"Сгенерирован свечной график для {crypto}/{currency} за {period} дней", 'info',
                                user_id=str(user['id']) if user else None)
                else:
                    error = "Ошибка генерации графика"
            else:
                error = "Не удалось получить данные для свечного графика"

        except Exception as e:
            error = f"Ошибка: {str(e)}"
            log_message(f"Ошибка построения свечного графика: {e}", 'error', user_id=str(user['id']) if user else None)

    cryptos = load_crypto_list()
    return render_template('candlestick.html',
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
                               columns=['id', 'telegram_id', 'first_name', 'last_name', 'username', 'created_at',
                                        'last_login', 'is_active'])
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        return redirect(url_for('index'))


@app.route('/requests_table')
def show_requests_table():
    """Отображает таблицу запросов"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')

    try:
        with get_db() as db:
            requests = db.query(CryptoRequest).order_by(CryptoRequest.created_at.desc()).limit(100).all()

            requests_data = []
            for req in requests:
                # Обрабатываем данные внутри сессии
                response_data = {}
                if req.response_data:
                    try:
                        response_data = json.loads(req.response_data)
                    except:
                        response_data = {}

                # Форматируем данные внутри сессии
                requests_data.append({
                    'id': req.id,
                    'user_id': req.user_id,
                    'crypto': req.crypto,
                    'currency': req.currency,
                    'status': req.status,
                    'rate': response_data.get('rate', 'N/A'),
                    'created_at': datetime.fromtimestamp(req.created_at).strftime('%Y-%m-%d %H:%M:%S'),
                    'finished_at': datetime.fromtimestamp(req.finished_at).strftime(
                        '%Y-%m-%d %H:%M:%S') if req.finished_at else 'N/A',
                    'error': response_data.get('error', '')
                })

        return render_template('data_table.html',
                               title='Очередь запросов',
                               data=requests_data,
                               columns=['id', 'user_id', 'crypto', 'currency', 'status', 'rate', 'created_at',
                                        'finished_at', 'error'],
                               user=user)
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        return redirect(url_for('index'))


@app.route('/verify_code', methods=['POST'])
def verify_code():
    """Проверка кода подтверждения из Telegram"""
    verification_code = request.form.get('verification_code')

    if not verification_code or len(verification_code) != 6:
        flash("Введите 6-значный код подтверждения", 'error')
        return redirect(url_for('auth'))

    # Здесь должна быть логика проверки кода
    # В текущей реализации используется только Telegram Widget авторизация
    flash("Функция проверки кода временно недоступна. Используйте Telegram Widget авторизацию.", 'warning')
    return redirect(url_for('auth'))


# Инициализация базы данных при импорте модуля
try:
    init_db_connection()
    init_db()
    log_message("Приложение инициализировано", 'info')
except Exception as e:
    log_message(f"Ошибка инициализации приложения: {e}", 'error')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
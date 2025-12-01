# app.py
import threading
import time
import requests
import os
import logging
import json
from datetime import datetime, timedelta
from contextlib import contextmanager

from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, BigInteger, Boolean
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from cachetools import cached, TTLCache
import pandas as pd
import matplotlib
import io
import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

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


# Конфигурация приложения
class Config:
    BOT_USERNAME = os.getenv('BOT_USERNAME', '@romanov_crypto_currency_bot')
    BOT_TOKEN = os.getenv('BOT_TOKEN', '8264247176:AAFByVrbcY8K-aanicYu2QK-tYRaFNq0lxY')
    DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"


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


@dataclass
class TelegramUserData:
    id: int
    first_name: str
    auth_date: int
    hash: str
    username: Optional[str] = None
    photo_url: Optional[str] = None
    last_name: Optional[str] = None


# Модели данных
class CryptoRate(Base):
    __tablename__ = 'crypto_rates'

    id = Column(Integer, primary_key=True)
    crypto = Column(String(50), nullable=False, index=True)
    currency = Column(String(10), nullable=False, index=True)
    rate = Column(Float(precision=8), nullable=False)
    source = Column(String(100), nullable=False)
    timestamp = Column(BigInteger, nullable=False, index=True)


class AppLog(Base):
    __tablename__ = 'app_logs'

    id = Column(Integer, primary_key=True)
    service = Column(String(50), default='crypto_api')
    component = Column(String(50), default='backend')
    message = Column(Text, nullable=False)
    level = Column(String(20), nullable=False)
    traceback = Column(Text)
    user_id = Column(String(36))
    timestamp = Column(BigInteger, nullable=False, index=True)


class TelegramUser(Base):
    __tablename__ = 'telegram_users'

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
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
    user_id = Column(BigInteger, nullable=False, index=True)
    crypto = Column(String(50), nullable=False)
    currency = Column(String(10), nullable=False)
    status = Column(String(20), default='pending')
    response_data = Column(Text)
    created_at = Column(BigInteger, nullable=False, index=True)
    finished_at = Column(BigInteger)


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
        engine = create_engine(
            Config.DB_URI,
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
        log_message("Попытка доступа к отключенной БД", 'warning')
        raise RuntimeError("Соединение с базой данных отключено")

    if SessionLocal is None:
        init_db_connection()

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
            SessionLocal.remove()
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


def verify_telegram_authentication(data: dict, bot_token: str) -> bool:
    """Проверяет данные авторизации Telegram"""
    try:
        if not isinstance(data, dict):
            return False

        required_fields = ['id', 'first_name', 'auth_date', 'hash']
        for field in required_fields:
            if field not in data or not isinstance(data[field], (str, int)):
                return False

        try:
            auth_date = datetime.fromtimestamp(int(data['auth_date']))
        except (ValueError, TypeError):
            return False

        if datetime.now() - auth_date > timedelta(hours=24):
            return False

        data_check_string = '\n'.join(
            f'{key}={value}'
            for key, value in sorted(data.items())
            if key != 'hash'
        )

        secret_key = hashlib.sha256(bot_token.encode()).digest()
        computed_hash = hmac.new(
            secret_key,
            data_check_string.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(computed_hash, data['hash'])

    except Exception as e:
        log_message(f"Error verifying Telegram auth: {e}", 'error')
        return False


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

    thread = threading.Thread(target=load_thread)
    thread.daemon = True
    thread.start()


@cached(crypto_list_cache)
def load_crypto_list():
    """Возвращает список криптовалют"""
    load_full_crypto_list_async()
    return FULL_CRYPTO_LIST


load_full_crypto_list_async()


def get_user_from_db(telegram_id: int) -> Optional[Dict[str, Any]]:
    """Получает пользователя из базы данных по telegram_id и возвращает словарь"""
    try:
        with get_db() as db:
            user = db.query(TelegramUser).filter(
                TelegramUser.telegram_id == telegram_id,
                TelegramUser.is_active == True
            ).first()

            if user:
                return {
                    'id': user.id,
                    'telegram_id': user.telegram_id,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'username': user.username,
                    'photo_url': user.photo_url,
                    'created_at': user.created_at,
                    'last_login': user.last_login,
                    'is_active': user.is_active
                }
            return None
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


def create_crypto_request(user_id: int, crypto: str, currency: str) -> int:
    """Создает новый запрос в очереди и сразу помечает как processing"""
    try:
        with get_db() as db:
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

            request = CryptoRequest(
                user_id=user_id,
                crypto=crypto,
                currency=currency,
                status='processing',
                created_at=int(time.time())
            )
            db.add(request)
            db.commit()
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
    """Получает количество processing запросов"""
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


def get_user_requests_history(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Получает историю запросов пользователя"""
    try:
        with get_db() as db:
            requests = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id
            ).order_by(CryptoRequest.created_at.desc()).limit(limit).all()

            result = []
            for req in requests:
                response_data = {}
                if req.response_data:
                    try:
                        response_data = json.loads(req.response_data)
                    except json.JSONDecodeError as e:
                        log_message(f"JSON decode error for request {req.id}: {e}", 'warning')
                        continue

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
    """Обрабатывает ТОЛЬКО СТАРЫЕ pending запросы"""
    try:
        if not db_connection_active:
            return

        with get_db() as db:
            one_minute_ago = int(time.time()) - 60
            old_pending_requests = db.query(CryptoRequest).filter(
                CryptoRequest.status == 'pending',
                CryptoRequest.created_at < one_minute_ago
            ).order_by(CryptoRequest.created_at.asc()).limit(2).all()

            processed_count = 0
            for request in old_pending_requests:
                request.status = 'processing'
                processed_count += 1

            if processed_count > 0:
                db.commit()
                log_message(f"Marked {processed_count} OLD pending requests as processing for worker", 'info')

    except SQLAlchemyError as e:
        log_message(f"Error processing OLD pending requests: {e}", 'error')


def get_main_crypto_rates_to_btc(timeout=30, coin_ids_to_fetch=None):
    """
    Запрашивает курсы криптовалют к биткоину (BTC) с API CoinGecko.

    Args:
        timeout (int): Таймаут для HTTP-запросов в секундах. По умолчанию 30.
        coin_ids_to_fetch (list): Список ID криптовалют для запроса. Если None, используются POPULAR_CRYPTOS.

    Returns:
        dict: Словарь, где ключ - ID криптовалюты (например, 'ethereum', 'bitcoin'),
              значение - словарь с курсом к BTC {'btc': float_rate}.
              Пример: {'ethereum': {'btc': 0.0654321}, 'bitcoin': {'btc': 1.0}}
              Возвращает пустой словарь в случае ошибки.
    """
    if coin_ids_to_fetch is None:
        coin_ids_to_fetch = [cid for cid in POPULAR_CRYPTOS if cid != 'bitcoin']
    else:
        # Убедимся, что 'bitcoin' не включён, если он там есть
        coin_ids_to_fetch = [cid for cid in coin_ids_to_fetch if cid != 'bitcoin']

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': ','.join(coin_ids_to_fetch), # Передаём список ID через запятую
        'vs_currencies': 'btc' # Запрашиваем курсы относительно BTC
    }

    try:
        logging.info(f"Запрашиваю курсы криптовалют ({len(coin_ids_to_fetch)} шт.) к BTC...")
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status() # Вызывает исключение для HTTP-ошибок (4xx, 5xx)

        rates = response.json()
        logging.info(f"Получены курсы для {len(rates)} криптовалют к BTC.")

        return rates

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP ошибка при запросе к API CoinGecko: {e}")
        logging.error(f"Статус код: {e.response.status_code}")
        logging.error(f"Текст ответа: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка запроса к API CoinGecko: {e}")
    except ValueError as e: # Ошибка при парсинге JSON
        logging.error(f"Ошибка парсинга JSON ответа от API CoinGecko: {e}")
    except Exception as e:
        logging.error(f"Неизвестная ошибка при запросе курсов: {e}")

    # Возвращаем пустой словарь в случае любой ошибки
    return {}


class CoinGeckoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_ohlc(self, coin_id: str, vs_currency: str, days: str) -> Optional[pd.DataFrame]:
        """Получение OHLC данных"""
        try:
            safe_coin_id = requests.utils.quote(coin_id)
            safe_currency = requests.utils.quote(vs_currency)

            url = f"{self.base_url}/coins/{safe_coin_id}/ohlc"
            params = {
                'vs_currency': safe_currency,
                'days': days
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
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

    def generate_plot(self, data: pd.DataFrame, crypto: str, currency: str, period: str) -> Optional[str]:
        """Генерирует график и возвращает его в base64"""
        if data is None or data.empty:
            return None

        try:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            colors = []
            for i in range(len(data)):
                if data['close'].iloc[i] >= data['open'].iloc[i]:
                    colors.append('green')
                else:
                    colors.append('red')

            for i in range(len(data)):
                ax.fill_between([i - 0.3, i + 0.3],
                                [data['open'].iloc[i], data['open'].iloc[i]],
                                [data['close'].iloc[i], data['close'].iloc[i]],
                                color=colors[i], alpha=0.7)

                ax.plot([i, i], [data['high'].iloc[i], max(data['open'].iloc[i], data['close'].iloc[i])],
                        color=colors[i], linewidth=1)

                ax.plot([i, i], [min(data['open'].iloc[i], data['close'].iloc[i]), data['low'].iloc[i]],
                        color=colors[i], linewidth=1)

            ax.set_title(f'{crypto.upper()}/{currency.upper()} Свечной график ({period} дней)',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Временной период', fontsize=12)
            ax.set_ylabel(f'Цена ({currency.upper()})', fontsize=12)
            ax.grid(True, alpha=0.3)

            n = len(data)
            step = max(1, n // 10)
            ax.set_xticks(range(0, n, step))
            ax.set_xticklabels([data.index[i].strftime('%m-%d') for i in range(0, n, step)], rotation=45)

            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plt.close()

            plot_url = base64.b64encode(img.getvalue()).decode()
            return f"data:image/png;base64,{plot_url}"

        except Exception as e:
            logging.error(f"Ошибка генерации свечного графика: {e}")
            return None


# Flask маршруты
@app.route('/telegram-auth', methods=['POST'])
def telegram_auth():
    """Обрабатывает callback от Telegram Login Widget"""
    try:
        user_data = request.get_json()

        if not user_data:
            return jsonify({'success': False, 'error': 'No data received'})

        if not verify_telegram_authentication(user_data, Config.BOT_TOKEN):
            return jsonify({'success': False, 'error': 'Invalid authentication data'})

        try:
            with get_db() as db:
                existing_user = db.query(TelegramUser).filter(
                    TelegramUser.telegram_id == user_data['id']
                ).first()

                current_time = int(time.time())

                if existing_user:
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

        except SQLAlchemyError as e:
            log_message(f"Database error during user save: {e}", 'error')
            return jsonify({'success': False, 'error': 'Database error'})

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
        log_message(f"Telegram auth error: {e}", 'error')
        return jsonify({'success': False, 'error': str(e)})


@app.route('/auth', methods=['GET'])
def auth():
    """Страница авторизации через Telegram"""
    user = session.get('user')

    if user:
        flash("Вы уже авторизованы!", 'info')
        return redirect(url_for('index'))

    return render_template('auth.html', user=user, bot_username=Config.BOT_USERNAME)


@app.route('/logout')
def logout():
    """Выход пользователя"""
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
    # current_rate больше не используется для отображения последнего завершённого
    # current_rate = None
    current_crypto = None
    current_currency = None
    current_request_id = None # ID текущего/последнего отправленного запроса
    requests_history = []
    pending_requests_count = 0
    processing_requests_count = 0
    user_session = session.get('user')
    user_from_db = None
    user_id = user_session['id'] if user_session else 0

    if user_session:
        user_from_db = get_user_from_db(user_session['id'])
        if user_from_db:
            update_user_last_login(user_session['id'])
        requests_history = get_user_requests_history(user_session['id'])
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

        current_crypto = crypto
        current_currency = currency
        request_id = create_crypto_request(user_id, crypto, currency)
        if request_id > 0:
            session['last_request_id'] = request_id
            session['current_crypto'] = crypto
            session['current_currency'] = currency
            session.modified = True
            flash(f"Запрос на получение курса {crypto.upper()}/{currency.upper()} отправлен в обработку. ID: {request_id}", 'info')
            log_message(f"Request created for {crypto}/{currency} (ID: {request_id})", 'info',
                        user_id=str(user_session['id']) if user_session else None)
            current_request_id = request_id # ID текущего запроса для отображения
            pending_requests_count = get_pending_requests_count()
            processing_requests_count = get_processing_requests_count()
            if user_session:
                requests_history = get_user_requests_history(user_session['id'])
        else:
            flash("Ошибка при создании запроса", 'error')
            log_message("Error creating request in queue", 'error',
                        user_id=str(user_session['id']) if user_session else None)
    else: # GET запрос
        # --- ИЗМЕНЕНИЕ ---
        # Не пытаемся получить последний завершённый курс
        # current_rate = None
        # current_crypto и current_currency могут быть из сессии для отображения в форме
        current_crypto = session.get('current_crypto', 'bitcoin') # Значения по умолчанию
        current_currency = session.get('current_currency', 'usd')
        # current_request_id получаем из сессии - это ID последнего отправленного запроса
        current_request_id = session.get('last_request_id') # Это ключевое изменение
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    process_pending_requests()

    cryptos = load_crypto_list()
    return render_template('index.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=PERIODS,
                           # current_rate=current_rate, # Больше не передаём
                           current_crypto=current_crypto,
                           current_currency=current_currency,
                           current_request_id=current_request_id, # Передаём ID текущего/последнего отправленного запроса
                           db_connected=db_connection_active,
                           user=user_from_db,
                           requests_history=requests_history,
                           pending_requests_count=pending_requests_count,
                           processing_requests_count=processing_requests_count,
                           bot_username=Config.BOT_USERNAME)

# ... (остальной код app.py без изменений) ...



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
            rates = db.query(
                CryptoRate.crypto,
                CryptoRate.currency,
                CryptoRate.rate,
                CryptoRate.source,
                CryptoRate.timestamp
            ).order_by(CryptoRate.timestamp.desc()).limit(100).all()

        rates_data = [{
            'crypto': r.crypto,
            'currency': r.currency,
            'rate': r.rate,
            'date_time': datetime.fromtimestamp(r.timestamp).strftime('%Y-%m-%d %H:%M:%S') if r.timestamp else 'N/A',
            'source': r.source
        } for r in rates]

        return render_template('data_table.html',
                               title='Курсы криптовалют',
                               data=rates_data,
                               columns=['crypto', 'currency', 'rate', 'date_time', 'source'],
                               user=user)
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        log_message(f"Ошибка при получении курсов: {e}", 'error')
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
            'date_time': datetime.fromtimestamp(l.timestamp).strftime('%Y-%m-%d %H:%M:%S') if l.timestamp else 'N/A',
            'level': l.level,
            'message': l.message,
            'service': l.service,
            'component': l.component,
            'user_id': l.user_id or '',
            'traceback': l.traceback or ''
        } for l in logs]

        return render_template('data_table.html',
                               title='Логи приложения',
                               data=logs_data,
                               columns=['date_time', 'level', 'message', 'service', 'component', 'user_id',
                                        'traceback'],
                               user=user)
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        log_message(f"Ошибка при получении логов: {e}", 'error')
        return redirect(url_for('index'))


@app.route('/users_table')
def show_users_table():
    """Отображает таблицу пользователей"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')

    try:
        with get_db() as db:
            users = db.query(TelegramUser).order_by(TelegramUser.last_login.desc()).limit(100).all()

            users_data = []
            for u in users:
                users_data.append({
                    'id': u.id,
                    'telegram_id': u.telegram_id,
                    'first_name': u.first_name,
                    'last_name': u.last_name or '',
                    'username': u.username or '',
                    'created_at': datetime.fromtimestamp(u.created_at).strftime(
                        '%Y-%m-%d %H:%M:%S') if u.created_at else 'N/A',
                    'last_login': datetime.fromtimestamp(u.last_login).strftime(
                        '%Y-%m-%d %H:%M:%S') if u.last_login else 'N/A',
                    'is_active': 'Да' if u.is_active else 'Нет'
                })

        return render_template('data_table.html',
                               title='Пользователи Telegram',
                               data=users_data,
                               columns=['id', 'telegram_id', 'first_name', 'last_name', 'username', 'created_at',
                                        'last_login', 'is_active'],
                               user=user)
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        log_message(f"Ошибка при получении пользователей: {e}", 'error')
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
                response_data = {}
                if req.response_data:
                    try:
                        response_data = json.loads(req.response_data)
                    except json.JSONDecodeError:
                        pass

                requests_data.append({
                    'id': req.id,
                    'user_id': req.user_id,
                    'crypto': req.crypto,
                    'currency': req.currency,
                    'status': req.status,
                    'rate': response_data.get('rate', 'N/A'),
                    'created_at': datetime.fromtimestamp(req.created_at).strftime(
                        '%Y-%m-%d %H:%M:%S') if req.created_at else 'N/A',
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
        log_message(f"Ошибка при получении запросов: {e}", 'error')
        return redirect(url_for('index'))


@app.route('/verify_code', methods=['POST'])
def verify_code():
    """Проверка кода подтверждения из Telegram"""
    verification_code = request.form.get('verification_code')

    if not verification_code or len(verification_code) != 6:
        flash("Введите 6-значный код подтверждения", 'error')
        return redirect(url_for('auth'))

    flash("Функция проверки кода временно недоступна. Используйте Telegram Widget авторизацию.", 'warning')
    return redirect(url_for('auth'))


@app.route('/main_crypto_rates_to_btc')
def show_main_crypto_rates_to_btc():
    """Отображает таблицу с курсами основных криптовалют (POPULAR_CRYPTOS) к биткоину"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')

    # Получаем курсы для основных 50 криптовалют
    rates_data = get_main_crypto_rates_to_btc(coin_ids_to_fetch=[cid for cid in POPULAR_CRYPTOS if cid != 'bitcoin'])

    if not rates_data:
        flash("Не удалось получить курсы криптовалют к BTC. Проверьте логи.", 'error')
        log_message("Не удалось получить курсы основных криптовалют к BTC", 'error', user_id=str(user['id']) if user else None)
        return redirect(url_for('index'))

    # Подготовим данные для шаблона
    # Сортируем по названию криптовалюты
    sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
    table_data = []
    for crypto_id, rate_dict in sorted_rates:
        rate = rate_dict.get('btc', 'N/A')
        # Преобразуем float_rate в строку с 8 знаками после запятой
        formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
        table_data.append({
            'crypto': crypto_id,
            'rate_to_btc': formatted_rate
        })

    # Подготовим информацию о просмотренных/оставшихся
    all_crypto_ids = load_crypto_list() # Загружаем полный список
    viewed_ids = set(rates_data.keys()) # Уже полученные ID
    remaining_ids = [cid for cid in all_crypto_ids if cid != 'bitcoin' and cid not in viewed_ids]
    remaining_count = len(remaining_ids)
    viewed_count = len(viewed_ids)

    log_message("Таблица курсов основных криптовалют к BTC отображена", 'info', user_id=str(user['id']) if user else None)
    return render_template('main_crypto_rates_table.html',
                           title='Курсы основных криптовалют к BTC',
                           data=table_data,
                           columns=['crypto', 'rate_to_btc'],
                           user=user,
                           viewed_count=viewed_count,
                           remaining_count=remaining_count,
                           has_next=True) # Показываем кнопку "Следующие пары..."


@app.route('/main_crypto_rates_to_btc/next')
def show_next_crypto_rates_to_btc():
    """Отображает таблицу с курсами следующих 50 криптовалют к биткоину в алфавитном порядке"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')

    # Получаем полный список криптовалют
    all_crypto_ids = load_crypto_list()
    # Исключаем 'bitcoin' и сортируем по алфавиту
    all_crypto_ids_sorted = sorted([cid for cid in all_crypto_ids if cid != 'bitcoin'])

    # Найдем ID, которые уже были отображены (всё, что не в POPULAR_CRYPTOS)
    # Или, более обобщенно, возьмем следующие 50 после POPULAR_CRYPTOS, отсортированные по алфавиту
    # Для простоты, возьмем первые 50 из отсортированного списка, исключая POPULAR_CRYPTOS
    # Но для "следующих" нужно исключить уже полученные.
    # В предыдущем запросе были получены только POPULAR_CRYPTOS.
    # Поэтому "следующие" будут первые 50 из отсортированного списка, исключая POPULAR_CRYPTOS.
    popular_set = set(POPULAR_CRYPTOS)
    next_batch_ids = []
    for cid in all_crypto_ids_sorted:
        if cid not in popular_set:
            next_batch_ids.append(cid)
        if len(next_batch_ids) == 50: # Берем следующие 50
            break

    if not next_batch_ids:
        flash("Больше нет криптовалют для отображения.", 'info')
        return redirect(url_for('show_main_crypto_rates_to_btc'))

    # Получаем курсы для следующей партии
    rates_data = get_main_crypto_rates_to_btc(coin_ids_to_fetch=next_batch_ids)

    if not rates_data:
        flash("Не удалось получить курсы следующих криптовалют к BTC. Проверьте логи.", 'error')
        log_message("Не удалось получить курсы следующих криптовалют к BTC", 'error', user_id=str(user['id']) if user else None)
        return redirect(url_for('show_main_crypto_rates_to_btc'))

    # Подготовим данные для шаблона
    # Сортируем по названию криптовалюты (уже должны быть отсортированы, но перестрахуемся)
    sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
    table_data = []
    for crypto_id, rate_dict in sorted_rates:
        rate = rate_dict.get('btc', 'N/A')
        # Преобразуем float_rate в строку с 8 знаками после запятой
        formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
        table_data.append({
            'crypto': crypto_id,
            'rate_to_btc': formatted_rate
        })

    # Подготовим информацию о просмотренных/оставшихся
    # Теперь просмотренные - это POPULAR_CRYPTOS + полученные сейчас
    viewed_ids = set(POPULAR_CRYPTOS) | set(rates_data.keys())
    remaining_ids = [cid for cid in all_crypto_ids_sorted if cid not in viewed_ids]
    remaining_count = len(remaining_ids)
    viewed_count = len(viewed_ids)

    log_message("Таблица курсов следующих криптовалют к BTC отображена", 'info', user_id=str(user['id']) if user else None)
    return render_template('main_crypto_rates_table.html',
                           title='Курсы следующих криптовалют к BTC',
                           data=table_data,
                           columns=['crypto', 'rate_to_btc'],
                           user=user,
                           viewed_count=viewed_count,
                           remaining_count=remaining_count,
                           has_next=bool(remaining_count)) # Показываем кнопку "Следующие пары...", если есть что показывать


# app.py - Добавить после существующих маршрутов

@app.route('/historical', methods=['GET', 'POST'])
def historical_data():
    """Страница с историческими данными по выбранной паре"""
    plot_url = None
    error = None
    user = session.get('user')
    df_stats = None

    # Значения по умолчанию из сессии или формы
    default_crypto = 'bitcoin'
    default_currency = 'usd'
    default_start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    default_end_date = datetime.now().strftime('%Y-%m-%d')

    # Получаем данные из формы
    if request.method == 'POST':
        crypto = request.form.get('crypto', default_crypto)
        currency = request.form.get('currency', default_currency)
        start_date = request.form.get('start_date', default_start_date)
        end_date = request.form.get('end_date', default_end_date)

        # Сохраняем в сессию для запоминания выбора
        session['historical_crypto'] = crypto
        session['historical_currency'] = currency
        session['historical_start_date'] = start_date
        session['historical_end_date'] = end_date

    else:
        # GET запрос - берем из сессии или используем значения по умолчанию
        crypto = session.get('historical_crypto', default_crypto)
        currency = session.get('historical_currency', default_currency)
        start_date = session.get('historical_start_date', default_start_date)
        end_date = session.get('historical_end_date', default_end_date)

    # Если получены данные из формы, строим график
    if request.method == 'POST' and crypto and currency and start_date and end_date:
        try:
            # Используем функцию get_historical_price_range
            df = get_historical_price_range(
                coin_id=crypto,
                vs_currency=currency,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                # Рассчитываем статистику
                df_stats = {
                    'start_price': float(df['price'].iloc[0]),
                    'end_price': float(df['price'].iloc[-1]),
                    'max_price': float(df['price'].max()),
                    'min_price': float(df['price'].min()),
                    'avg_price': float(df['price'].mean()),
                    'change_pct': ((df['price'].iloc[-1] / df['price'].iloc[0]) - 1) * 100
                }

                plot_url = generate_historical_plot(df, crypto, currency, start_date, end_date)
                if plot_url:
                    log_message(
                        f"Сгенерирован исторический график для {crypto}/{currency} за период {start_date} - {end_date}",
                        'info', user_id=str(user['id']) if user else None)
                else:
                    error = "Ошибка генерации графика"
            else:
                error = "Не удалось получить исторические данные для выбранного периода"

        except Exception as e:
            error = f"Ошибка: {str(e)}"
            log_message(f"Ошибка получения исторических данных: {e}", 'error',
                        user_id=str(user['id']) if user else None)

    cryptos = load_crypto_list()
    return render_template('historical.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           current_crypto=crypto,
                           current_currency=currency,
                           start_date=start_date,
                           end_date=end_date,
                           plot_url=plot_url,
                           error=error,
                           df_stats=df_stats,
                           user=user)


def generate_historical_plot(df, crypto, currency, start_date, end_date):
    """Генерирует график исторических данных и возвращает base64 строку"""
    try:
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # График цены
        ax1.plot(df['timestamp'], df['price'], 'b-', linewidth=2, label='Цена')
        ax1.set_title(f'{crypto.upper()}/{currency.upper()} - Исторические данные ({start_date} - {end_date})',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'Цена ({currency.upper()})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Форматирование оси времени
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # График процентных изменений
        if 'returns_pct' in df.columns:
            colors = ['green' if x >= 0 else 'red' for x in df['returns_pct']]
            ax2.bar(df['timestamp'], df['returns_pct'], color=colors, alpha=0.7, width=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Дата', fontsize=12)
            ax2.set_ylabel('Изменение (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Форматирование оси времени для второго графика
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Сохраняем в буфер
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()

        # Конвертируем в base64
        plot_url = base64.b64encode(img.getvalue()).decode()
        return f"data:image/png;base64,{plot_url}"

    except Exception as e:
        logging.error(f"Ошибка генерации исторического графика: {e}")
        return None


# Обновленная функция get_historical_price_range (добавить в app.py, если её нет)
def get_historical_price_range(coin_id='bitcoin', vs_currency='usd',
                               start_date='2024-01-01', end_date=None):
    """Получить исторические цены для выбранного периода"""
    import requests
    import pandas as pd
    from datetime import datetime

    # Если конечная дата не указана - берем сегодня
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Преобразуем строки в datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Проверяем, что даты корректны
    if start_dt >= end_dt:
        raise ValueError("Дата начала должна быть раньше даты окончания")

    # Рассчитываем количество дней между датами
    days_diff = (end_dt - start_dt).days

    if days_diff < 1:
        raise ValueError("Период должен быть хотя бы 1 день")

    # Определяем интервал в зависимости от периода
    if days_diff <= 90:
        days_param = days_diff
        interval = 'daily'
    else:
        days_param = days_diff
        interval = 'daily'
        logging.info(f"Для периода >90 дней данные будут агрегированными")

    # URL для запроса
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    params = {
        'vs_currency': vs_currency,
        'days': days_param,
        'interval': interval
    }

    try:
        logging.info(f"Загрузка данных {coin_id.upper()}/{vs_currency.upper()}...")
        logging.info(f"Период: {start_date} - {end_date} ({days_diff} дней)")

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        # Преобразуем данные в DataFrame
        timestamps = [pd.to_datetime(x[0], unit='ms') for x in data['prices']]
        prices = [x[1] for x in data['prices']]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })

        # Фильтруем по нашему диапазону дат
        mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
        df = df.loc[mask].copy()

        if len(df) == 0:
            logging.warning("Нет данных для указанного периода")
            return None

        # Сортируем по дате
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Добавляем дополнительные колонки
        df['date'] = df['timestamp'].dt.date
        df['returns_pct'] = df['price'].pct_change() * 100

        logging.info(f"Успешно загружено {len(df)} записей")
        logging.info(f"Диапазон цен: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка запроса: {e}")
        raise
    except Exception as e:
        logging.error(f"Ошибка обработки данных: {e}")
        raise


# Инициализация базы данных при импорте модуля
try:
    init_db_connection()
    init_db()
    log_message("Приложение инициализировано", 'info')
except Exception as e:
    log_message(f"Ошибка инициализации приложения: {e}", 'error')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

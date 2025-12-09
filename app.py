# app.py - ПОЛНЫЙ ИСПРАВЛЕННЫЙ КОД
import threading
import time
import requests
import os
import logging
import json
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import socket
import psutil
import platform

from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, BigInteger, Boolean, text
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

import numpy as np
from scipy import stats
import plotly.graph_objs as go
import plotly.utils

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
    host = Column(String(100))


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
    host = Column(String(100))


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
    host = Column(String(100))


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
    host = Column(String(100))


# Вспомогательные функции
def get_host_id():
    """Возвращает идентификатор хоста"""
    try:
        return socket.gethostname()[:50]
    except:
        return "localhost"


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
        print("✅ Инициализировано подключение к базе данных")
    except SQLAlchemyError as e:
        db_connection_active = False
        print(f"❌ Ошибка подключения к базе данных: {e}")
        raise


@contextmanager
def get_db():
    """Контекстный менеджер для работы с сессией"""
    # Проверяем соединение
    if not db_connection_active:
        print("⚠️ База данных не подключена")
        raise RuntimeError("Соединение с базой данных отключено")

    # Убеждаемся, что SessionLocal инициализирован
    global SessionLocal
    if SessionLocal is None:
        print("⚠️ SessionLocal не инициализирован, пытаюсь инициализировать...")
        try:
            init_db_connection()
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            raise RuntimeError(f"Ошибка инициализации БД: {e}")

    # Теперь точно есть SessionLocal
    if SessionLocal is None:
        raise RuntimeError("SessionLocal все еще None после инициализации")

    # Создаем сессию
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Ошибка БД: {e}")
        raise
    except Exception as e:
        db.rollback()
        print(f"Неизвестная ошибка в сессии БД: {e}")
        raise
    finally:
        try:
            db.close()
        except:
            pass
        try:
            SessionLocal.remove()
        except:
            pass


def migrate_database_safe():
    """Безопасная миграция базы данных"""
    if not db_connection_active:
        return

    try:
        with get_db() as db:
            # Проверяем существование колонки host в app_logs
            result = db.execute(text("""
                SELECT COUNT(*) as count 
                FROM information_schema.columns 
                WHERE table_schema = DATABASE() 
                AND table_name = 'app_logs' 
                AND column_name = 'host'
            """)).fetchone()

            if result and result.count == 0:
                # Добавляем колонку host
                db.execute(text("""
                    ALTER TABLE app_logs 
                    ADD COLUMN host VARCHAR(100) DEFAULT NULL
                """))
                print("✅ Колонка 'host' добавлена в таблицу app_logs")

            # Проверяем другие таблицы
            tables = ['crypto_rates', 'crypto_requests', 'telegram_users']
            for table in tables:
                try:
                    result = db.execute(text(f"""
                        SELECT COUNT(*) as count 
                        FROM information_schema.columns 
                        WHERE table_schema = DATABASE() 
                        AND table_name = '{table}' 
                        AND column_name = 'host'
                    """)).fetchone()

                    if result and result.count == 0:
                        db.execute(text(f"""
                            ALTER TABLE {table} 
                            ADD COLUMN host VARCHAR(100) DEFAULT NULL
                        """))
                        print(f"✅ Колонка 'host' добавлена в таблицу {table}")
                except:
                    continue

    except Exception as e:
        print(f"❌ Ошибка миграции: {e}")


def log_message(
        message: str,
        level: str = 'info',
        service: str = 'crypto_api',
        component: str = 'backend',
        traceback: str = None,
        user_id: str = None
):
    """Запись лога в файл и БД - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    # Всегда пишем в файловый лог
    logging.log(
        getattr(logging, level.upper()),
        f"[{service}.{component}] User={user_id} | {message}",
        exc_info=traceback is not None
    )

    # Если БД не подключена - выходим
    if not db_connection_active:
        return

    try:
        # Создаем новую сессию для каждой записи лога
        if SessionLocal is None:
            return

        db = SessionLocal()

        try:
            # Ограничиваем длину полей
            message_short = str(message)[:500] if message else ""
            traceback_short = str(traceback)[:1000] if traceback else None

            # Пытаемся записать лог
            # ВАЖНО: не используем объект AppLog после закрытия сессии
            db.execute(
                text("""
                    INSERT INTO app_logs 
                    (service, component, message, level, traceback, user_id, timestamp, host) 
                    VALUES (:service, :component, :message, :level, :traceback, :user_id, :timestamp, :host)
                """),
                {
                    'service': service,
                    'component': component,
                    'message': message_short,
                    'level': level,
                    'traceback': traceback_short,
                    'user_id': user_id,
                    'timestamp': int(time.time()),
                    'host': get_host_id()
                }
            )
            db.commit()

        except SQLAlchemyError as e:
            db.rollback()

            # Если ошибка из-за отсутствия колонки host
            if "Unknown column 'host'" in str(e):
                try:
                    # Пробуем без host
                    db.execute(
                        text("""
                            INSERT INTO app_logs 
                            (service, component, message, level, traceback, user_id, timestamp) 
                            VALUES (:service, :component, :message, :level, :traceback, :user_id, :timestamp)
                        """),
                        {
                            'service': service,
                            'component': component,
                            'message': message_short,
                            'level': level,
                            'traceback': traceback_short,
                            'user_id': user_id,
                            'timestamp': int(time.time())
                        }
                    )
                    db.commit()

                    # Запускаем миграцию в фоне
                    if not hasattr(log_message, '_migration_started'):
                        log_message._migration_started = True

                        def run_migration():
                            try:
                                time.sleep(2)
                                migrate_database_safe()
                            except:
                                pass

                        thread = threading.Thread(target=run_migration)
                        thread.daemon = True
                        thread.start()

                except Exception:
                    pass
            else:
                # Другие ошибки SQL - игнорируем
                pass

        finally:
            # Всегда закрываем сессию
            db.close()
            SessionLocal.remove()

    except Exception:
        # Игнорируем все другие ошибки при записи логов
        pass


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
                user.host = get_host_id()
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
                created_at=int(time.time()),
                host=get_host_id()
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
                request.host = get_host_id()
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
    """Запрашивает курсы криптовалют к биткоину (BTC) с API CoinGecko"""
    if coin_ids_to_fetch is None:
        coin_ids_to_fetch = [cid for cid in POPULAR_CRYPTOS if cid != 'bitcoin']
    else:
        coin_ids_to_fetch = [cid for cid in coin_ids_to_fetch if cid != 'bitcoin']

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': ','.join(coin_ids_to_fetch),
        'vs_currencies': 'btc'
    }

    try:
        logging.info(f"Запрашиваю курсы криптовалют ({len(coin_ids_to_fetch)} шт.) к BTC...")
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        rates = response.json()
        logging.info(f"Получены курсы для {len(rates)} криптовалют к BTC.")

        return rates

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP ошибка при запросе к API CoinGecko: {e}")
        logging.error(f"Статус код: {e.response.status_code}")
        logging.error(f"Текст ответа: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка запроса к API CoinGecko: {e}")
    except ValueError as e:
        logging.error(f"Ошибка парсинга JSON ответа от API CoinGecko: {e}")
    except Exception as e:
        logging.error(f"Неизвестная ошибка при запросе курсов: {e}")

    return {}


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


def get_historical_price_range(coin_id='bitcoin', vs_currency='usd',
                               start_date='2024-01-01', end_date=None):
    """Получить исторические цены для выбранного периода"""
    import requests
    import pandas as pd
    from datetime import datetime

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    if start_dt >= end_dt:
        raise ValueError("Дата начала должна быть раньше даты окончания")

    days_diff = (end_dt - start_dt).days

    if days_diff < 1:
        raise ValueError("Период должен быть хотя бы 1 день")

    if days_diff <= 90:
        days_param = days_diff
        interval = 'daily'
    else:
        days_param = days_diff
        interval = 'daily'
        logging.info(f"Для периода >90 дней данные будут агрегированными")

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

        timestamps = [pd.to_datetime(x[0], unit='ms') for x in data['prices']]
        prices = [x[1] for x in data['prices']]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })

        mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
        df = df.loc[mask].copy()

        if len(df) == 0:
            logging.warning("Нет данных для указанного периода")
            return None

        df = df.sort_values('timestamp').reset_index(drop=True)

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


def calculate_correlation_with_btc(coin_id, vs_currency='usd', days=30, timeframe='1d'):
    """Рассчитывает корреляцию выбранной криптовалюты с Bitcoin с обработкой ошибок"""
    try:
        if coin_id.lower() == 'bitcoin':
            logging.info("Пропускаем Bitcoin (корреляция с самим собой = 1)")
            return {
                'coin_id': 'bitcoin',
                'correlation': 1.0,
                'r_squared': 1.0,
                'p_value': 0.0,
                'beta': 1.0,
                'alpha': 0.0,
                'btc_std': 0.0,
                'coin_std': 0.0,
                'n_observations': 0,
                'significant': True,
                'correlation_strength': 'perfect',
                'correlation_direction': 'positive'
            }

        logging.info(f"Начинаю расчет корреляции для {coin_id}...")

        btc_data = get_historical_price_data('bitcoin', vs_currency, days)
        if btc_data is None or btc_data.empty:
            logging.error(f"Не удалось получить данные для Bitcoin")
            return None

        coin_data = get_historical_price_data(coin_id, vs_currency, days)
        if coin_data is None or coin_data.empty:
            logging.error(f"Не удалось получить данные для {coin_id}")
            return None

        common_dates = set(btc_data['timestamp']).intersection(set(coin_data['timestamp']))
        if len(common_dates) < 2:
            logging.error(f"Недостаточно общих дат для {coin_id}: {len(common_dates)}")
            return None

        btc_filtered = btc_data[btc_data['timestamp'].isin(common_dates)].sort_values('timestamp')
        coin_filtered = coin_data[coin_data['timestamp'].isin(common_dates)].sort_values('timestamp')

        if len(btc_filtered) != len(coin_filtered):
            logging.error(f"Разное количество точек: BTC={len(btc_filtered)}, {coin_id}={len(coin_filtered)}")
            return None

        if timeframe == '1d':
            btc_prices = btc_filtered['price'].values
            coin_prices = coin_filtered['price'].values
        elif timeframe == '1w':
            btc_filtered['week'] = btc_filtered['timestamp'].dt.isocalendar().week
            btc_filtered['year'] = btc_filtered['timestamp'].dt.isocalendar().year
            coin_filtered['week'] = coin_filtered['timestamp'].dt.isocalendar().week
            coin_filtered['year'] = coin_filtered['timestamp'].dt.isocalendar().year

            btc_weekly = btc_filtered.groupby(['year', 'week'])['price'].last()
            coin_weekly = coin_filtered.groupby(['year', 'week'])['price'].last()

            common_indices = set(btc_weekly.index).intersection(set(coin_weekly.index))
            if len(common_indices) < 2:
                logging.error(f"Недостаточно общих недель для {coin_id}")
                return None

            btc_prices = btc_weekly.loc[list(common_indices)].values
            coin_prices = coin_weekly.loc[list(common_indices)].values
        elif timeframe == '1M':
            btc_filtered['month'] = btc_filtered['timestamp'].dt.strftime('%Y-%m')
            coin_filtered['month'] = coin_filtered['timestamp'].dt.strftime('%Y-%m')

            btc_monthly = btc_filtered.groupby('month')['price'].last()
            coin_monthly = coin_filtered.groupby('month')['price'].last()

            common_months = set(btc_monthly.index).intersection(set(coin_monthly.index))
            if len(common_months) < 2:
                logging.error(f"Недостаточно общих месяцев для {coin_id}")
                return None

            btc_prices = btc_monthly.loc[list(common_months)].values
            coin_prices = coin_monthly.loc[list(common_months)].values
        else:
            logging.error(f"Неизвестный таймфрейм: {timeframe}")
            return None

        if len(btc_prices) < 5 or len(coin_prices) < 5:
            logging.error(f"Недостаточно данных после агрегации: {len(btc_prices)} точек")
            return None

        btc_returns = np.diff(btc_prices) / btc_prices[:-1]
        coin_returns = np.diff(coin_prices) / coin_prices[:-1]

        correlation, p_value = stats.pearsonr(btc_returns, coin_returns)

        if np.isnan(correlation) or np.isnan(p_value):
            logging.error(f"Результат корреляции содержит NaN для {coin_id}")
            return None

        r_squared = correlation ** 2

        covariance = np.cov(btc_returns, coin_returns)[0, 1]
        btc_variance = np.var(btc_returns)
        beta = covariance / btc_variance if btc_variance != 0 else 0

        coin_mean_return = np.mean(coin_returns)
        btc_mean_return = np.mean(btc_returns)
        alpha = coin_mean_return - beta * btc_mean_return

        btc_std = np.std(btc_returns)
        coin_std = np.std(coin_returns)

        results = {
            'coin_id': coin_id,
            'vs_currency': vs_currency,
            'timeframe': timeframe,
            'days': days,
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'beta': float(beta),
            'alpha': float(alpha),
            'btc_std': float(btc_std),
            'coin_std': float(coin_std),
            'n_observations': len(btc_returns),
            'significant': p_value < 0.05,
            'correlation_strength': get_correlation_strength(abs(correlation)),
            'correlation_direction': 'positive' if correlation > 0 else 'negative',
            'btc_mean_return': float(btc_mean_return),
            'coin_mean_return': float(coin_mean_return)
        }

        logging.info(f"Успешно рассчитана корреляция для {coin_id}: {correlation:.3f}")

        return results

    except Exception as e:
        logging.error(f"Ошибка расчета корреляции для {coin_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def get_correlation_strength(corr_value):
    """Определяет силу корреляции"""
    abs_corr = abs(corr_value)
    if abs_corr >= 0.9:
        return 'very strong'
    elif abs_corr >= 0.7:
        return 'strong'
    elif abs_corr >= 0.5:
        return 'moderate'
    elif abs_corr >= 0.3:
        return 'weak'
    else:
        return 'very weak'


def get_historical_price_data(coin_id, vs_currency='usd', days=30):
    """Получает исторические данные для расчета корреляции с обработкой ошибок"""
    try:
        ping_url = "https://api.coingecko.com/api/v3/ping"
        ping_response = requests.get(ping_url, timeout=5)

        if ping_response.status_code != 200:
            logging.error(f"CoinGecko API недоступен. Статус: {ping_response.status_code}")
            return None

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }

        logging.info(f"Запрашиваю данные для {coin_id}/{vs_currency} за {days} дней...")

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            logging.error(f"Превышен лимит запросов для {coin_id}. Подождите минуту.")
            return None
        elif response.status_code == 404:
            logging.error(f"Криптовалюта {coin_id} не найдена")
            return None
        elif response.status_code != 200:
            logging.error(f"Ошибка API для {coin_id}: {response.status_code}")
            logging.error(f"Ответ: {response.text[:200]}")
            return None

        data = response.json()

        if 'prices' not in data or not data['prices']:
            logging.error(f"Нет данных о ценах для {coin_id}")
            return None

        timestamps = []
        prices = []

        for item in data['prices']:
            if len(item) >= 2:
                timestamps.append(pd.to_datetime(item[0], unit='ms'))
                prices.append(item[1])

        if len(prices) < 2:
            logging.error(f"Недостаточно данных для {coin_id}: {len(prices)} точек")
            return None

        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })

        df = df.sort_values('timestamp').reset_index(drop=True)

        logging.info(f"Получено {len(df)} записей для {coin_id}")

        return df

    except requests.exceptions.Timeout:
        logging.error(f"Таймаут при запросе данных для {coin_id}")
        return None
    except requests.exceptions.ConnectionError:
        logging.error(f"Ошибка соединения при запросе {coin_id}")
        return None
    except Exception as e:
        logging.error(f"Ошибка получения данных для {coin_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def calculate_multiple_correlations(coin_ids, vs_currency='usd', days=30, timeframe='1d'):
    """Рассчитывает корреляции для нескольких криптовалют"""
    results = {}

    for coin_id in coin_ids:
        if coin_id == 'bitcoin':
            continue

        logging.info(f"Рассчитываю корреляцию для {coin_id}...")
        correlation_result = calculate_correlation_with_btc(
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
            timeframe=timeframe
        )

        if correlation_result:
            results[coin_id] = correlation_result

    sorted_results = dict(sorted(
        results.items(),
        key=lambda x: abs(x[1]['correlation']),
        reverse=True
    ))

    return sorted_results


def generate_correlation_plot(results):
    """Генерирует график корреляций"""
    try:
        if not results:
            return None

        coin_names = []
        correlations = []
        colors = []
        strengths = []

        for coin_id, data in results.items():
            coin_names.append(coin_id.upper())
            correlations.append(data['correlation'])

            if data['correlation'] > 0:
                colors.append('rgba(46, 204, 113, 0.7)')
            else:
                colors.append('rgba(231, 76, 60, 0.7)')

            strengths.append(data['correlation_strength'])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=coin_names,
            y=correlations,
            marker_color=colors,
            text=[f"{corr:.3f}" for corr in correlations],
            textposition='outside',
            hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Корреляция: %{y:.3f}<br>" +
                    "Сила: %{customdata}<br>" +
                    "<extra></extra>"
            ),
            customdata=strengths,
            name='Корреляция с BTC'
        ))

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )

        fig.update_layout(
            title={
                'text': 'Корреляция криптовалют с Bitcoin',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Криптовалюта",
            yaxis_title="Коэффициент корреляции",
            yaxis=dict(
                range=[-1.1, 1.1],
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            plot_bgcolor='white',
            showlegend=False,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON

    except Exception as e:
        logging.error(f"Ошибка генерации графика корреляции: {e}")
        return None


def calculate_multiple_correlations_with_retry(coin_ids, vs_currency='usd', days=30, timeframe='1d', max_retries=3):
    """Рассчитывает корреляции с повторными попытками"""
    results = {}

    for coin_id in coin_ids:
        if coin_id == 'bitcoin':
            continue

        logging.info(f"Обрабатываю {coin_id}...")

        for attempt in range(max_retries):
            try:
                correlation_result = calculate_correlation_with_btc(
                    coin_id=coin_id,
                    vs_currency=vs_currency,
                    days=days,
                    timeframe=timeframe
                )

                if correlation_result:
                    results[coin_id] = correlation_result
                    logging.info(f"Успешно: {coin_id} (попытка {attempt + 1})")
                    break
                else:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logging.warning(f"Повторная попытка для {coin_id} через {wait_time} сек...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Не удалось рассчитать корреляцию для {coin_id} после {max_retries} попыток")

            except Exception as e:
                logging.error(f"Ошибка при попытке {attempt + 1} для {coin_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)

    if results:
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        ))
        return sorted_results

    return None


class BinanceAPI:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbols_cache = {}

    def get_all_symbols(self):
        """Получает все доступные торговые пары на Binance"""
        try:
            if self.symbols_cache:
                return self.symbols_cache

            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url, timeout=10)
            data = response.json()

            symbols = []
            for symbol_info in data['symbols']:
                if symbol_info['status'] == 'TRADING':
                    symbols.append(symbol_info['symbol'])

            self.symbols_cache = symbols
            return symbols

        except Exception as e:
            logging.error(f"Ошибка получения символов Binance: {e}")
            return []

    def get_historical_klines(self, symbol, interval='1d', limit=100, start_time=None, end_time=None):
        """Получает исторические данные свечей (klines) с Binance"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            return df

        except Exception as e:
            logging.error(f"Ошибка получения данных для {symbol}: {e}")
            return None

    def get_daily_returns(self, symbol, days=30):
        """Получает дневные доходности для символа"""
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)

            df = self.get_historical_klines(
                symbol=symbol,
                interval='1d',
                limit=days,
                start_time=start_time,
                end_time=end_time
            )

            if df is None or df.empty:
                return None

            prices = df['close'].values
            if len(prices) < 2:
                return None

            returns = np.diff(prices) / prices[:-1] * 100
            return returns

        except Exception as e:
            logging.error(f"Ошибка расчета доходностей для {symbol}: {e}")
            return None

    def get_available_crypto_pairs(self, vs_currency='USDT'):
        """Получает список криптовалютных пар с указанной валютой"""
        try:
            symbols = self.get_all_symbols()
            crypto_pairs = {}

            for symbol in symbols:
                if symbol.endswith(vs_currency):
                    crypto = symbol.replace(vs_currency, '')
                    crypto_pairs[crypto.lower()] = symbol

            return crypto_pairs

        except Exception as e:
            logging.error(f"Ошибка получения пар: {e}")
            return {}


binance_api = BinanceAPI()


def calculate_correlation_with_btc_binance(coin_symbol, vs_currency='USDT', days=30):
    """Рассчитывает корреляцию криптовалюты с Bitcoin через Binance API"""
    try:
        btc_symbol = f"BTC{vs_currency}"
        coin_symbol_full = f"{coin_symbol}{vs_currency}"

        logging.info(f"Рассчитываю корреляцию для {coin_symbol_full} с {btc_symbol}")

        btc_returns = binance_api.get_daily_returns(btc_symbol, days)
        if btc_returns is None or len(btc_returns) < 5:
            logging.error(f"Не удалось получить данные для {btc_symbol}")
            return None

        coin_returns = binance_api.get_daily_returns(coin_symbol_full, days)
        if coin_returns is None or len(coin_returns) < 5:
            logging.error(f"Не удалось получить данные для {coin_symbol_full}")
            return None

        min_len = min(len(btc_returns), len(coin_returns))
        btc_returns_aligned = btc_returns[:min_len]
        coin_returns_aligned = coin_returns[:min_len]

        if min_len < 5:
            logging.error(f"Недостаточно данных после выравнивания: {min_len}")
            return None

        correlation, p_value = stats.pearsonr(btc_returns_aligned, coin_returns_aligned)

        r_squared = correlation ** 2

        covariance = np.cov(btc_returns_aligned, coin_returns_aligned)[0, 1]
        btc_variance = np.var(btc_returns_aligned)
        beta = covariance / btc_variance if btc_variance != 0 else 0

        coin_mean_return = np.mean(coin_returns_aligned)
        btc_mean_return = np.mean(btc_returns_aligned)
        alpha = coin_mean_return - beta * btc_mean_return

        btc_std = np.std(btc_returns_aligned)
        coin_std = np.std(coin_returns_aligned)

        correlation_strength = get_correlation_strength(abs(correlation))

        results = {
            'coin_symbol': coin_symbol,
            'vs_currency': vs_currency,
            'days': days,
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'beta': float(beta),
            'alpha': float(alpha),
            'btc_std': float(btc_std),
            'coin_std': float(coin_std),
            'n_observations': min_len,
            'significant': p_value < 0.05,
            'correlation_strength': correlation_strength,
            'correlation_direction': 'positive' if correlation > 0 else 'negative',
            'btc_mean_return': float(btc_mean_return),
            'coin_mean_return': float(coin_mean_return),
            'btc_symbol': btc_symbol,
            'coin_symbol_full': coin_symbol_full
        }

        logging.info(f"Успешно рассчитана корреляция: {correlation:.3f} для {coin_symbol}")

        return results

    except Exception as e:
        logging.error(f"Ошибка расчета корреляции через Binance: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def calculate_multiple_correlations_binance(coin_symbols, vs_currency='USDT', days=30):
    """Рассчитывает корреляции для нескольких криптовалют через Binance"""
    results = {}
    successful = 0
    failed = []

    available_pairs = binance_api.get_available_crypto_pairs(vs_currency)

    for coin_symbol in coin_symbols:
        coin_lower = coin_symbol.lower()
        if coin_lower not in available_pairs:
            logging.warning(f"Пара {coin_symbol}{vs_currency} не найдена на Binance")
            failed.append(coin_symbol)
            continue

        logging.info(f"Обрабатываю {coin_symbol}...")

        correlation_result = calculate_correlation_with_btc_binance(
            coin_symbol=coin_symbol.upper(),
            vs_currency=vs_currency,
            days=days
        )

        if correlation_result:
            results[coin_symbol] = correlation_result
            successful += 1
            logging.info(f"✓ Успешно: {coin_symbol}")
            time.sleep(0.1)
        else:
            failed.append(coin_symbol)
            logging.warning(f"✗ Не удалось: {coin_symbol}")

    if results:
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        ))
        return sorted_results, successful, failed

    return None, successful, failed


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
                    existing_user.host = get_host_id()
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
                        is_active=True,
                        host=get_host_id()
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
    current_crypto = None
    current_currency = None
    current_request_id = None
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
            flash(
                f"Запрос на получение курса {crypto.upper()}/{currency.upper()} отправлен в обработку. ID: {request_id}",
                'info')
            log_message(f"Request created for {crypto}/{currency} (ID: {request_id})", 'info',
                        user_id=str(user_session['id']) if user_session else None)
            current_request_id = request_id
            pending_requests_count = get_pending_requests_count()
            processing_requests_count = get_processing_requests_count()
            if user_session:
                requests_history = get_user_requests_history(user_session['id'])
        else:
            flash("Ошибка при создании запроса", 'error')
            log_message("Error creating request in queue", 'error',
                        user_id=str(user_session['id']) if user_session else None)
    else:
        current_crypto = session.get('current_crypto', 'bitcoin')
        current_currency = session.get('current_currency', 'usd')
        current_request_id = session.get('last_request_id')

    process_pending_requests()

    cryptos = load_crypto_list()
    return render_template('index.html',
                           cryptos=cryptos,
                           currencies=CURRENCIES,
                           periods=PERIODS,
                           current_crypto=current_crypto,
                           current_currency=current_currency,
                           current_request_id=current_request_id,
                           db_connected=db_connection_active,
                           user=user_from_db,
                           requests_history=requests_history,
                           pending_requests_count=pending_requests_count,
                           processing_requests_count=processing_requests_count,
                           bot_username=Config.BOT_USERNAME)


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
    """Отображает таблицу логов с фильтрацией по уровню"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')
    level_filter = request.args.get('level', '').lower()

    try:
        with get_db() as db:
            query = db.query(AppLog)

            if level_filter and level_filter != 'all':
                query = query.filter(AppLog.level == level_filter)

            logs = query.order_by(AppLog.timestamp.desc()).limit(200).all()

        logs_data = []
        moscow_tz = timezone(timedelta(hours=3))

        for l in logs:
            if l.timestamp:
                utc_time = datetime.fromtimestamp(l.timestamp, tz=timezone.utc)
                moscow_time = utc_time.astimezone(moscow_tz)
                date_time_str = moscow_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_time_str = 'N/A'

            log_entry = {
                'date_time': date_time_str,
                'level': l.level,
                'message': l.message[:200] + '...' if len(l.message) > 200 else l.message,
                'service': l.service,
                'component': l.component,
                'user_id': l.user_id or '',
                'traceback': '',
                'host': l.host or 'N/A'
            }

            if l.level.lower() == 'error' and l.traceback:
                log_entry['traceback'] = l.traceback[:500] + '...' if len(l.traceback) > 500 else l.traceback

            logs_data.append(log_entry)

        if level_filter and level_filter != 'all' and not logs_data:
            flash(f"Логи с уровнем '{level_filter.upper()}' не найдены", 'warning')

        available_levels = ['all', 'debug', 'info', 'warning', 'error', 'critical']

        return render_template('data_table_logs.html',
                               title='Логи приложения',
                               data=logs_data,
                               columns=['date_time', 'level', 'message', 'service', 'component', 'user_id',
                                        'traceback', 'host'],
                               user=user,
                               current_level=level_filter,
                               available_levels=available_levels)
    except SQLAlchemyError as e:
        flash(f"Ошибка БД: {str(e)}", 'error')
        print(f"Ошибка при получении логов: {e}")
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
                    'is_active': 'Да' if u.is_active else 'Нет',
                    'host': u.host or 'N/A'
                })

        return render_template('data_table.html',
                               title='Пользователи Telegram',
                               data=users_data,
                               columns=['id', 'telegram_id', 'first_name', 'last_name', 'username', 'created_at',
                                        'last_login', 'is_active', 'host'],
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
                    'error': response_data.get('error', ''),
                    'host': req.host or 'N/A'
                })

        return render_template('data_table.html',
                               title='Очередь запросов',
                               data=requests_data,
                               columns=['id', 'user_id', 'crypto', 'currency', 'status', 'rate', 'created_at',
                                        'finished_at', 'error', 'host'],
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

    rates_data = get_main_crypto_rates_to_btc(coin_ids_to_fetch=[cid for cid in POPULAR_CRYPTOS if cid != 'bitcoin'])

    if not rates_data:
        flash("Не удалось получить курсы криптовалют к BTC. Проверьте логи.", 'error')
        log_message("Не удалось получить курсы основных криптовалют к BTC", 'error',
                    user_id=str(user['id']) if user else None)
        return redirect(url_for('index'))

    sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
    table_data = []
    for crypto_id, rate_dict in sorted_rates:
        rate = rate_dict.get('btc', 'N/A')
        formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
        table_data.append({
            'crypto': crypto_id,
            'rate_to_btc': formatted_rate
        })

    all_crypto_ids = load_crypto_list()
    viewed_ids = set(rates_data.keys())
    remaining_ids = [cid for cid in all_crypto_ids if cid != 'bitcoin' and cid not in viewed_ids]
    remaining_count = len(remaining_ids)
    viewed_count = len(viewed_ids)

    log_message("Таблица курсов основных криптовалют к BTC отображена", 'info',
                user_id=str(user['id']) if user else None)
    return render_template('main_crypto_rates_table.html',
                           title='Курсы основных криптовалют к BTC',
                           data=table_data,
                           columns=['crypto', 'rate_to_btc'],
                           user=user,
                           viewed_count=viewed_count,
                           remaining_count=remaining_count,
                           has_next=True)


@app.route('/main_crypto_rates_to_btc/next')
def show_next_crypto_rates_to_btc():
    """Отображает таблицу с курсами следующих 50 криптовалют к биткоину в алфавитном порядке"""
    if not db_connection_active:
        flash("Соединение с базой данных отключено", 'error')
        return redirect(url_for('index'))

    user = session.get('user')

    all_crypto_ids = load_crypto_list()
    all_crypto_ids_sorted = sorted([cid for cid in all_crypto_ids if cid != 'bitcoin'])

    popular_set = set(POPULAR_CRYPTOS)
    next_batch_ids = []
    for cid in all_crypto_ids_sorted:
        if cid not in popular_set:
            next_batch_ids.append(cid)
        if len(next_batch_ids) == 50:
            break

    if not next_batch_ids:
        flash("Больше нет криптовалют для отображения.", 'info')
        return redirect(url_for('show_main_crypto_rates_to_btc'))

    rates_data = get_main_crypto_rates_to_btc(coin_ids_to_fetch=next_batch_ids)

    if not rates_data:
        flash("Не удалось получить курсы следующих криптовалют к BTC. Проверьте логи.", 'error')
        log_message("Не удалось получить курсы следующих криптовалют к BTC", 'error',
                    user_id=str(user['id']) if user else None)
        return redirect(url_for('show_main_crypto_rates_to_btc'))

    sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
    table_data = []
    for crypto_id, rate_dict in sorted_rates:
        rate = rate_dict.get('btc', 'N/A')
        formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
        table_data.append({
            'crypto': crypto_id,
            'rate_to_btc': formatted_rate
        })

    viewed_ids = set(POPULAR_CRYPTOS) | set(rates_data.keys())
    remaining_ids = [cid for cid in all_crypto_ids_sorted if cid not in viewed_ids]
    remaining_count = len(remaining_ids)
    viewed_count = len(viewed_ids)

    log_message("Таблица курсов следующих криптовалют к BTC отображена", 'info',
                user_id=str(user['id']) if user else None)
    return render_template('main_crypto_rates_table.html',
                           title='Курсы следующих криптовалют к BTC',
                           data=table_data,
                           columns=['crypto', 'rate_to_btc'],
                           user=user,
                           viewed_count=viewed_count,
                           remaining_count=remaining_count,
                           has_next=bool(remaining_count))


@app.route('/historical', methods=['GET', 'POST'])
def historical_data():
    """Страница с историческими данными по выбранной паре"""
    plot_url = None
    error = None
    user = session.get('user')
    df_stats = None

    default_crypto = 'bitcoin'
    default_currency = 'usd'
    default_start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    default_end_date = datetime.now().strftime('%Y-%m-%d')

    if request.method == 'POST':
        crypto = request.form.get('crypto', default_crypto)
        currency = request.form.get('currency', default_currency)
        start_date = request.form.get('start_date', default_start_date)
        end_date = request.form.get('end_date', default_end_date)

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_dt > end_dt:
            start_date, end_date = end_date, start_date
            flash("Даты были автоматически переставлены, так как дата начала была позже даты окончания", 'info')

        session['historical_crypto'] = crypto
        session['historical_currency'] = currency
        session['historical_start_date'] = start_date
        session['historical_end_date'] = end_date

    else:
        crypto = session.get('historical_crypto', default_crypto)
        currency = session.get('historical_currency', default_currency)
        start_date = session.get('historical_start_date', default_start_date)
        end_date = session.get('historical_end_date', default_end_date)

    if request.method == 'POST' and crypto and currency and start_date and end_date:
        try:
            df = get_historical_price_range(
                coin_id=crypto,
                vs_currency=currency,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
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


@app.route('/correlation', methods=['GET', 'POST'])
def correlation_analysis():
    """Страница анализа корреляции с Bitcoin"""
    user = session.get('user')
    correlation_results = None
    plot_json = None
    error = None
    summary_stats = None
    successful_calculations = 0
    failed_calculations = []

    default_cryptos = ['ethereum', 'binancecoin', 'solana', 'cardano', 'ripple']
    default_days = 30
    default_timeframe = '1d'
    default_currency = 'usd'

    if request.method == 'POST':
        selected_cryptos = request.form.getlist('cryptos')
        days = int(request.form.get('days', default_days))
        timeframe = request.form.get('timeframe', default_timeframe)
        currency = request.form.get('currency', default_currency)

        if not selected_cryptos:
            selected_cryptos = default_cryptos

        if len(selected_cryptos) > 10:
            selected_cryptos = selected_cryptos[:10]
            flash(f"Выбрано слишком много криптовалют. Анализ будет проведен для первых 10.", 'warning')

        session['correlation_cryptos'] = selected_cryptos
        session['correlation_days'] = days
        session['correlation_timeframe'] = timeframe
        session['correlation_currency'] = currency

        try:
            logging.info(f"Начинаю расчет корреляций для {len(selected_cryptos)} криптовалют...")

            correlation_results = calculate_multiple_correlations_with_retry(
                coin_ids=selected_cryptos,
                vs_currency=currency,
                days=days,
                timeframe=timeframe,
                max_retries=2
            )

            if correlation_results:
                successful_calculations = len(correlation_results)
                failed_calculations = [c for c in selected_cryptos if c not in correlation_results and c != 'bitcoin']

                if successful_calculations > 0:
                    plot_json = generate_correlation_plot(correlation_results)

                    if plot_json:
                        correlations = [data['correlation'] for data in correlation_results.values()]
                        if correlations:
                            summary_stats = {
                                'average_correlation': np.mean(correlations),
                                'median_correlation': np.median(correlations),
                                'max_correlation': max(correlations),
                                'min_correlation': min(correlations),
                                'positive_count': sum(1 for c in correlations if c > 0),
                                'negative_count': sum(1 for c in correlations if c < 0),
                                'total_count': len(correlations)
                            }

                    log_message(f"Успешно рассчитано {successful_calculations} корреляций из {len(selected_cryptos)}",
                                'info', user_id=str(user['id']) if user else None)

                    if failed_calculations:
                        error = f"Не удалось рассчитать корреляцию для: {', '.join(failed_calculations)}"
                        flash(f"Успешно: {successful_calculations}, Не удалось: {len(failed_calculations)}", 'warning')
                else:
                    error = "Не удалось рассчитать ни одной корреляции."
            else:
                error = "Не удалось рассчитать корреляции. Возможные причины: недоступность API, отсутствие данных или проблемы с соединением."

        except Exception as e:
            error = f"Ошибка при расчете корреляций: {str(e)}"
            logging.error(f"Ошибка анализа корреляции: {e}", exc_info=True)

    else:
        selected_cryptos = session.get('correlation_cryptos', default_cryptos)
        days = session.get('correlation_days', default_days)
        timeframe = session.get('correlation_timeframe', default_timeframe)
        currency = session.get('correlation_currency', default_currency)

    all_cryptos = load_crypto_list()
    available_cryptos = [c for c in all_cryptos if c != 'bitcoin']

    days_options = [
        {'value': '30', 'label': '30 дней (быстрее)'},
        {'value': '90', 'label': '90 дней'},
        {'value': '180', 'label': '180 дней'},
        {'value': '365', 'label': '1 год (медленнее)'}
    ]

    timeframe_options = [
        {'value': '1d', 'label': 'Дневной (1D)'},
        {'value': '1w', 'label': 'Недельный (1W)'},
        {'value': '1M', 'label': 'Месячный (1M)'}
    ]

    return render_template('correlation.html',
                           all_cryptos=available_cryptos[:50],
                           selected_cryptos=selected_cryptos,
                           days=days,
                           days_options=days_options,
                           timeframe=timeframe,
                           timeframe_options=timeframe_options,
                           currency=currency,
                           currencies=CURRENCIES,
                           correlation_results=correlation_results,
                           plot_json=plot_json,
                           error=error,
                           summary_stats=summary_stats,
                           successful_calculations=successful_calculations,
                           failed_calculations=failed_calculations,
                           user=user)


@app.route('/correlation_binance', methods=['GET', 'POST'])
def correlation_binance():
    """Страница анализа корреляции через Binance API"""
    user = session.get('user')
    correlation_results = None
    plot_json = None
    error = None
    summary_stats = None
    successful = 0
    failed = []

    default_cryptos = ['ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'MATIC', 'AVAX', 'LINK']
    default_days = 30
    default_currency = 'USDT'

    if request.method == 'POST':
        selected_cryptos = request.form.get('cryptos', '').upper().split(',')
        if not selected_cryptos or selected_cryptos[0] == '':
            selected_cryptos = default_cryptos
        else:
            selected_cryptos = [c.strip() for c in selected_cryptos if c.strip()]

        days = int(request.form.get('days', default_days))
        currency = request.form.get('currency', default_currency)

        if len(selected_cryptos) > 15:
            selected_cryptos = selected_cryptos[:15]
            flash("Ограничено 15 криптовалютами для производительности", 'warning')

        session['correlation_binance_cryptos'] = selected_cryptos
        session['correlation_binance_days'] = days
        session['correlation_binance_currency'] = currency

        try:
            logging.info("Проверяю доступность Binance API...")
            test_response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            if test_response.status_code != 200:
                error = "Binance API временно недоступен. Попробуйте позже."
                flash(error, 'error')
            else:
                correlation_results, successful, failed = calculate_multiple_correlations_binance(
                    coin_symbols=selected_cryptos,
                    vs_currency=currency,
                    days=days
                )

                if correlation_results:
                    plot_json = generate_correlation_plot(correlation_results)

                    correlations = [data['correlation'] for data in correlation_results.values()]
                    if correlations:
                        summary_stats = {
                            'average_correlation': np.mean(correlations),
                            'median_correlation': np.median(correlations),
                            'max_correlation': max(correlations),
                            'min_correlation': min(correlations),
                            'positive_count': sum(1 for c in correlations if c > 0),
                            'negative_count': sum(1 for c in correlations if c < 0),
                            'total_count': len(correlations)
                        }

                    if successful > 0:
                        flash(f"Успешно рассчитано: {successful} корреляций", 'success')
                    if failed:
                        flash(f"Не удалось: {len(failed)} криптовалют", 'warning')
                else:
                    error = "Не удалось рассчитать корреляции. Проверьте символы криптовалют."

        except Exception as e:
            error = f"Ошибка: {str(e)}"
            logging.error(f"Ошибка в correlation_binance: {e}", exc_info=True)

    else:
        selected_cryptos = session.get('correlation_binance_cryptos', default_cryptos)
        days = session.get('correlation_binance_days', default_days)
        currency = session.get('correlation_binance_currency', default_currency)

    currencies = ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']

    days_options = [
        {'value': '7', 'label': '7 дней'},
        {'value': '30', 'label': '30 дней'},
        {'value': '90', 'label': '90 дней'},
        {'value': '180', 'label': '180 дней'}
    ]

    return render_template('correlation_binance.html',
                           selected_cryptos=selected_cryptos,
                           days=days,
                           days_options=days_options,
                           currency=currency,
                           currencies=currencies,
                           correlation_results=correlation_results,
                           plot_json=plot_json,
                           error=error,
                           summary_stats=summary_stats,
                           successful=successful,
                           failed=failed,
                           user=user)


@app.route('/api/status')
def get_worker_status():
    """Возвращает статус системы и воркеров"""
    try:
        # Информация о системе
        system_info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'host_id': get_host_id()
        }

        # Использование RAM
        ram_info = {
            'total': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'available': round(psutil.virtual_memory().available / (1024 ** 3), 2),
            'percent': psutil.virtual_memory().percent,
            'used': round(psutil.virtual_memory().used / (1024 ** 3), 2)
        }

        # Использование CPU
        cpu_info = {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }

        # Информация о процессах
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
            try:
                if 'nginx' in proc.info['name'].lower() or 'gunicorn' in proc.info['name'].lower():
                    processes.append({
                        'name': proc.info['name'],
                        'pid': proc.info['pid'],
                        'status': proc.info['status'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Статистика nginx
        nginx_processes = [p for p in processes if 'nginx' in p['name'].lower()]
        nginx_status = {
            'count': len(nginx_processes),
            'on': len([p for p in nginx_processes if p['status'] == 'running']),
            'off': len([p for p in nginx_processes if p['status'] != 'running']),
            'processes': nginx_processes[:5]
        }

        # Статистика gunicorn
        gunicorn_processes = [p for p in processes if 'gunicorn' in p['name'].lower()]
        gunicorn_status = {
            'count': len(gunicorn_processes),
            'on': len([p for p in gunicorn_processes if p['status'] == 'running']),
            'off': len([p for p in gunicorn_processes if p['status'] != 'running']),
            'processes': gunicorn_processes[:5]
        }

        # Flask статус
        flask_status = {
            'status': 'running',
            'uptime': time.time() - psutil.Process(os.getpid()).create_time(),
            'requests_processed': 0
        }

        # Статус базы данных
        db_status = {
            'connected': db_connection_active,
            'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Общий статус
        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system': system_info,
            'resources': {
                'ram': ram_info,
                'cpu': cpu_info,
                'disk': {
                    'total': round(psutil.disk_usage('/').total / (1024 ** 3), 2),
                    'used': round(psutil.disk_usage('/').used / (1024 ** 3), 2),
                    'free': round(psutil.disk_usage('/').free / (1024 ** 3), 2),
                    'percent': psutil.disk_usage('/').percent
                }
            },
            'services': {
                'nginx': nginx_status,
                'gunicorn': gunicorn_status,
                'flask': flask_status,
                'database': db_status
            },
            'processes': {
                'total': len(processes),
                'list': processes[:10]
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500


@app.route('/status')
def show_status():
    """Страница для просмотра статуса системы"""
    user = session.get('user')
    return render_template('status.html',
                           title='Статус системы',
                           user=user)


# Инициализация базы данных при импорте модуля
try:
    init_db_connection()
    init_db()
    log_message("Приложение инициализировано", 'info')

    # Запускаем миграцию при старте
    migrate_database_safe()

except Exception as e:
    print(f"❌ Ошибка инициализации приложения: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
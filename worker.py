# worker.py
import time
import requests
import json
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, BigInteger, Boolean, and_
from sqlalchemy.orm import sessionmaker
from app import Base, CryptoRequest, CryptoRate, log_message # Импортируем CryptoRequest
import os
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Dict, Optional

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker.log'),
        logging.StreamHandler()
    ]
)

class Config:
    DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

def get_crypto_rate(crypto: str, currency: str) -> Dict[str, any]:
    """Получает курс криптовалюты к валюте - ТОЛЬКО В WORKER"""
    try:
        # Валидация входных параметров
        if not crypto or not currency:
            return {
                'success': False,
                'error': "Не указана криптовалюта или валюта",
                'source': 'coingecko',
                'timestamp': int(time.time())
            }
        # Защита от инъекций в URL
        safe_crypto = requests.utils.quote(crypto)
        safe_currency = requests.utils.quote(currency)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={safe_crypto}&vs_currencies={safe_currency}"
        logging.info(f"Worker: Calling API for {crypto}/{currency}")
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return {
                'success': False,
                'error': f"Ошибка API: {response.status_code}",
                'source': 'coingecko',
                'timestamp': int(time.time())
            }
        data = response.json()
        # Проверка наличия данных
        if crypto not in data or currency not in data.get(crypto, {}):
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
            'error': f"Ошибка сети: {str(e)}",
            'source': 'coingecko',
            'timestamp': int(time.time())
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Неизвестная ошибка: {str(e)}",
            'source': 'coingecko',
            'timestamp': int(time.time())
        }

class CryptoWorker:
    def __init__(self):
        self.last_api_call_time = 0
        self.min_api_interval = 3  # Минимальный интервал между запросами (секунды)
        self.db_connection_active = False
        self.engine = None
        self.SessionLocal = None
        self.init_db_connection()

    def init_db_connection(self):
        """Инициализация подключения к базе данных"""
        try:
            self.engine = create_engine(
                Config.DB_URI,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            self.db_connection_active = True
            logging.info("Worker: Инициализировано подключение к базе данных")
        except Exception as e:
            logging.error(f"Worker: Ошибка подключения к базе данных: {e}")
            self.db_connection_active = False

    @contextmanager
    def get_db(self):
        """Контекстный менеджер для работы с сессией"""
        if not self.db_connection_active:
            raise RuntimeError("Соединение с базой данных отключено")
        db = self.SessionLocal()
        try:
            yield db
        except Exception as e:
            db.rollback()
            logging.error(f"Worker: Database error: {e}")
            raise
        finally:
            db.close()

    def get_oldest_processing_request(self) -> Optional[CryptoRequest]:
        """Получает самый старый processing запрос"""
        try:
            with self.get_db() as db:
                # Сортировка по ID для более предсказуемого поведения при конкурентности
                request = db.query(CryptoRequest).filter(
                    CryptoRequest.status == 'processing'
                ).order_by(CryptoRequest.id.asc()).first()
                return request
        except Exception as e:
            logging.error(f"Worker: Ошибка получения processing запроса: {e}")
            return None

    def has_duplicate_request(self, db, user_id: int, crypto: str, currency: str, current_request_id: int) -> bool:
        """Проверяет, есть ли другие активные запросы для этого же пользователя и пары crypto/currency"""
        try:
            duplicate_request = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id,
                CryptoRequest.crypto == crypto,
                CryptoRequest.currency == currency,
                CryptoRequest.status.in_(['pending', 'processing']), # Проверяем активные
                CryptoRequest.id != current_request_id # Исключаем текущий
            ).first()
            return duplicate_request is not None
        except Exception as e:
            logging.error(f"Worker: Ошибка проверки дубликата для запроса {current_request_id}: {e}")
            return False # В случае ошибки предполагаем отсутствие дубликата, чтобы не блокировать обработку

    def process_request_with_api(self, request: CryptoRequest):
        """Обрабатывает один запрос с вызовом API"""
        try:
            # Соблюдаем минимальный интервал между запросами
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call_time
            if time_since_last_call < self.min_api_interval:
                sleep_time = self.min_api_interval - time_since_last_call
                logging.info(f"Worker: Waiting {sleep_time:.1f}s before API call")
                time.sleep(sleep_time)

            with self.get_db() as db:
                # Получаем актуальную версию запроса
                current_request = db.query(CryptoRequest).filter(
                    CryptoRequest.id == request.id
                ).first()
                if not current_request or current_request.status != 'processing':
                    logging.info(f"Worker: Request {request.id} already processed or status changed, skipping")
                    return

                # --- НОВАЯ ПРОВЕРКА НА ДУБЛИКАТ ---
                if self.has_duplicate_request(db, current_request.user_id, current_request.crypto, current_request.currency, current_request.id):
                     logging.info(f"Worker: Found duplicate active request for user {current_request.user_id} and {current_request.crypto}/{current_request.currency}. Skipping request {current_request.id}.")
                     # Помечаем текущий запрос как ошибочный или дубликат
                     current_request.status = 'error' # Или 'duplicate', если добавите такой статус
                     current_request.response_data = json.dumps({'error': 'Duplicate request found, skipping.'})
                     current_request.finished_at = int(time.time())
                     db.commit()
                     return
                # --- КОНЕЦ НОВОЙ ПРОВЕРКИ ---

                logging.info(
                    f"Worker: Processing request {current_request.id} for {current_request.crypto}/{current_request.currency}")

                # ВЫЗЫВАЕМ API
                rate_data = get_crypto_rate(current_request.crypto, current_request.currency)
                if rate_data['success']:
                    # Сохраняем в основную таблицу курсов
                    rate_entry = CryptoRate(
                        crypto=current_request.crypto,
                        currency=current_request.currency,
                        rate=round(rate_data['rate'], 8),
                        source=rate_data['source'],
                        timestamp=rate_data['timestamp']
                    )
                    db.add(rate_entry)
                    # Обновляем статус запроса
                    current_request.status = 'finished'
                    current_request.response_data = json.dumps(rate_data)
                    current_request.finished_at = int(time.time())
                    logging.info(
                        f"Worker: Successfully processed request {current_request.id} - {current_request.crypto}/{current_request.currency}: {rate_data['rate']}")
                else:
                    current_request.status = 'error'
                    current_request.response_data = json.dumps(rate_data)
                    current_request.finished_at = int(time.time())
                    logging.error(f"Worker: API error for request {current_request.id}: {rate_data['error']}")

                db.commit()
                self.last_api_call_time = time.time()
        except Exception as e:
            logging.error(f"Worker: Exception processing request {request.id}: {e}")
            try:
                # Пытаемся пометить запрос как ошибочный
                with self.get_db() as db:
                    current_request = db.query(CryptoRequest).filter(
                        CryptoRequest.id == request.id
                    ).first()
                    if current_request:
                        current_request.status = 'error'
                        current_request.response_data = json.dumps({'error': str(e)})
                        current_request.finished_at = int(time.time())
                        db.commit()
            except Exception as retry_error:
                logging.error(f"Worker: Failed to mark request {request.id} as error: {retry_error}")

    def cleanup_old_requests(self):
        """Очищает старые завершенные запросы"""
        try:
            with self.get_db() as db:
                # Удаляем завершенные запросы старше 1 дня
                one_day_ago = int(time.time()) - 24 * 60 * 60
                deleted_count = db.query(CryptoRequest).filter(
                    CryptoRequest.status.in_(['finished', 'error']),
                    CryptoRequest.created_at < one_day_ago
                ).delete()
                if deleted_count > 0:
                    db.commit()
                    logging.info(f"Worker: Cleaned up {deleted_count} old requests")
        except Exception as e:
            logging.error(f"Worker: Error cleaning up old requests: {e}")

    def run(self):
        """Основной цикл воркера"""
        logging.info("Worker: Запуск воркера обработки запросов")
        try:
            while True:
                try:
                    if not self.db_connection_active:
                        logging.warning("Worker: Соединение с БД отключено, попытка переподключения")
                        self.init_db_connection()
                        time.sleep(10)
                        continue
                    # Получаем самый старый processing запрос
                    request = self.get_oldest_processing_request()
                    if request:
                        logging.info(
                            f"Worker: Found processing request: ID={request.id}, {request.crypto}/{request.currency}")
                        self.process_request_with_api(request)
                    else:
                        # Нет запросов для обработки - очистка и пауза
                        self.cleanup_old_requests()
                        time.sleep(5)
                except KeyboardInterrupt:
                    logging.info("Worker: Получен сигнал остановки")
                    break
                except Exception as e:
                    logging.error(f"Worker: Ошибка в основном цикле: {e}")
                    time.sleep(10)
        except Exception as e:
            logging.error(f"Worker: Критическая ошибка: {e}")
        finally:
            logging.info("Worker: Остановка воркера")

if __name__ == '__main__':
    worker = CryptoWorker()
    worker.run()
# modules/app_logging.py
import logging
import time
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from modules.database import get_db, get_db_connection_active, get_session_local, get_host_id
from modules.database import get_host_id

# Настройка глобального логгера
logger = logging.getLogger(__name__)


def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Добавляем файловый обработчик
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def log_message(
        message: str,
        level: str = 'info',
        service: str = 'crypto_api',
        component: str = 'backend',
        traceback: str = None,
        user_id: str = None
):
    """Запись лога в файл и БД"""
    # Логируем в файл
    log_level = getattr(logging, level.upper())
    logger.log(log_level, f"[{service}.{component}] User={user_id} | {message}")

    # Если БД не подключена - выходим
    if not get_db_connection_active():
        return

    try:
        # Создаем новую сессию для каждой записи лога
        if get_session_local() is None:
            return

        db = get_session_local()()

        try:
            # Ограничиваем длину полей
            message_short = str(message)[:500] if message else ""
            traceback_short = str(traceback)[:1000] if traceback else None

            # Пытаемся записать лог в БД
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

                except Exception:
                    pass
            else:
                # Другие ошибки SQL - игнорируем
                pass

        finally:
            # Всегда закрываем сессию
            db.close()
            db.remove()

    except Exception:
        # Игнорируем все другие ошибки при записи логов
        pass


def log_message_test(
        message: str,
        level: str = 'info',
        service: str = 'crypto_api',
        component: str = 'backend',
        traceback: str = None,
        user_id: str = None
):
    """Только файловый лог - временное решение"""
    logger.log(
        getattr(logging, level.upper()),
        f"[{service}.{component}] User={user_id} | {message}"
    )
# modules/database.py
import time
import socket
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager

# Настройка логирования для модуля
logger = logging.getLogger(__name__)

# Пытаемся импортировать конфигурацию и модели
try:
    from modules.models import Base
    from modules.config import Config

    MODULES_IMPORTED = True
    logger.info("✅ Модули успешно импортированы")
except ImportError as e:
    logger.warning(f"⚠️ Не удалось импортировать модули: {e}")
    logger.warning("Режим автономной работы. Используются заглушки.")
    MODULES_IMPORTED = False

    # Создаем заглушки для автономной работы
    from sqlalchemy.orm import declarative_base  # Обновлённый импорт

    Base = declarative_base()


    # Заглушка для Config
    class Config:
        DB_URI = "mysql+pymysql://test:test@localhost/test_db"


# ============================================
# КЛАСС ДЛЯ УПРАВЛЕНИЯ ПОДКЛЮЧЕНИЕМ К БД
# ============================================

class DatabaseManager:
    """Класс для управления подключением к базе данных"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.engine = None
            self.SessionLocal = None
            self.db_connection_active = False
            self.last_connection_error = None
            self._initialized = True

    def get_host_id(self) -> str:
        """Возвращает идентификатор хоста"""
        try:
            return socket.gethostname()[:50]
        except Exception:
            return "localhost"

    def _log(self, message: str, level: str = 'info'):
        """Внутренний метод логирования"""
        level_method = getattr(logger, level, logger.info)
        level_method(message)

    def test_db_connection(self, db_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        Простая проверка подключения к БД
        Возвращает словарь с результатом проверки
        """
        self._log("🔍 Проверка подключения к БД...")

        result = {
            'success': False,
            'error': None,
            'db_version': None,
            'db_name': None,
            'response_time_ms': 0
        }

        # Используем переданную строку или из конфига
        if db_uri is None:
            if not MODULES_IMPORTED:
                result['error'] = "Конфигурация не загружена"
                return result
            db_uri = Config.DB_URI

        try:
            # Создаем временный движок
            temp_engine = create_engine(db_uri, connect_args={'connect_timeout': 5})

            start_time = time.time()

            # Пробуем подключиться
            with temp_engine.connect() as conn:
                # Самый простой запрос
                row = conn.execute(text("SELECT 1 as test, VERSION() as version, DATABASE() as db")).fetchone()

                end_time = time.time()

                if row and row.test == 1:
                    result['success'] = True
                    result['db_version'] = row.version
                    result['db_name'] = row.db
                    result['response_time_ms'] = round((end_time - start_time) * 1000, 2)
                    self._log("✅ Подключение успешно")
                else:
                    result['error'] = "Неверный ответ от БД"

            temp_engine.dispose()

        except OperationalError as e:
            result['error'] = f"Ошибка подключения: {str(e)}"
        except Exception as e:
            result['error'] = f"Неизвестная ошибка: {str(e)}"

        return result

    def init_db_connection(self) -> bool:
        """Инициализация подключения к БД"""
        self._log("🔄 Инициализация подключения к БД...")

        if not MODULES_IMPORTED:
            self._log("❌ Конфигурация не загружена", 'error')
            return False

        try:
            self.engine = create_engine(
                Config.DB_URI,
                pool_pre_ping=True,
                echo=False,
                connect_args={'connect_timeout': 10}
            )

            self.SessionLocal = scoped_session(
                sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine
                )
            )

            # Проверяем подключение
            test_result = self.test_db_connection()

            if test_result['success']:
                self.db_connection_active = True
                self._log(f"✅ Подключение установлено: {test_result.get('db_name')}")
                self._log(f"   Версия: {test_result.get('db_version')}")
                self._log(f"   Время ответа: {test_result.get('response_time_ms')} мс")
                return True
            else:
                self.last_connection_error = test_result['error']
                self._log(f"❌ Не удалось подключиться: {self.last_connection_error}", 'error')
                return False

        except Exception as e:
            self.last_connection_error = str(e)
            self._log(f"❌ Ошибка инициализации: {e}", 'error')
            return False

    def init_db(self):
        """Создает таблицы, если их нет"""
        if not self.db_connection_active:
            self._log("Невозможно инициализировать БД: подключение не активно", 'warning')
            return False

        try:
            from modules.models import Base
            Base.metadata.create_all(bind=self.engine)
            self._log("Таблицы БД созданы/проверены")
            return True
        except Exception as e:
            self._log(f"Ошибка создания таблиц: {e}", 'error')
            return False

    def migrate_database_safe(self):
        """Безопасная миграция БД"""
        if not self.db_connection_active:
            self._log("Невозможно выполнить миграцию: подключение не активно", 'warning')
            return

        try:
            with self.get_db() as db:
                # Проверяем наличие колонки host в таблице app_logs
                result = db.execute(text("""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'app_logs' 
                    AND COLUMN_NAME = 'host'
                """)).fetchone()

                if not result:
                    self._log("Добавляем колонку host в таблицу app_logs")
                    db.execute(text("ALTER TABLE app_logs ADD COLUMN host VARCHAR(100)"))
                    db.commit()
                    self._log("Миграция БД выполнена успешно")
                else:
                    self._log("БД уже имеет актуальную структуру")

        except Exception as e:
            self._log(f"Ошибка миграции БД: {e}", 'error')

    def get_connection_status(self) -> Dict[str, Any]:
        """Возвращает текущий статус подключения"""
        return {
            'active': self.db_connection_active,
            'engine_exists': self.engine is not None,
            'session_exists': self.SessionLocal is not None,
            'config_loaded': MODULES_IMPORTED,
            'last_error': self.last_connection_error,
            'host': self.get_host_id(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    @contextmanager
    def get_db(self):
        """Контекстный менеджер для работы с сессией"""
        if not self.db_connection_active:
            raise RuntimeError("Соединение с БД не активно")

        if self.SessionLocal is None:
            raise RuntimeError("Сессия не инициализирована")

        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
            self.SessionLocal.remove()

    def get_table_list(self) -> Dict[str, Any]:
        """Получает список таблиц из БД"""
        result = {
            'success': False,
            'tables': [],
            'error': None,
            'count': 0
        }

        if not self.db_connection_active:
            result['error'] = "Подключение не активно"
            return result

        try:
            with self.get_db() as db:
                tables = db.execute(text("""
                    SELECT TABLE_NAME, TABLE_ROWS 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = DATABASE()
                    ORDER BY TABLE_NAME
                """)).fetchall()

                result['success'] = True
                result['tables'] = [{'name': t[0], 'rows': t[1] or 0} for t in tables]
                result['count'] = len(tables)

        except Exception as e:
            result['error'] = str(e)

        return result


# ============================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ДЛЯ СОВМЕСТИМОСТИ
# ============================================

# Создаем глобальный экземпляр
db_manager = DatabaseManager()

# Экспортируем методы для обратной совместимости
get_host_id = db_manager.get_host_id
test_db_connection = db_manager.test_db_connection
init_db_connection = db_manager.init_db_connection
get_connection_status = db_manager.get_connection_status
get_db = db_manager.get_db
get_table_list = db_manager.get_table_list
init_db = db_manager.init_db
migrate_database_safe = db_manager.migrate_database_safe

# Экспортируем атрибуты для обратной совместимости
def get_db_connection_active():
    return db_manager.db_connection_active

def get_engine():
    return db_manager.engine

def get_session_local():
    return db_manager.SessionLocal

def get_last_connection_error():
    return db_manager.last_connection_error

def is_db_connected():
    """Проверяет, активно ли подключение к БД"""
    return db_manager.db_connection_active


# Для обратной совместимости оставляем переменные как свойства
db_connection_active = property(lambda self: db_manager.db_connection_active)
engine = property(lambda self: db_manager.engine)
SessionLocal = property(lambda self: db_manager.SessionLocal)
last_connection_error = property(lambda self: db_manager.last_connection_error)


# ============================================
# ТОЧКА ВХОДА ДЛЯ САМОСТОЯТЕЛЬНОГО ЗАПУСКА
# ============================================

def main():
    """
    Главная функция для самостоятельного запуска модуля
    Запуск: python database.py
    """
    print("\n" + "=" * 60)
    print("🚀 ТЕСТ МОДУЛЯ DATABASE.PY")
    print("=" * 60)

    print(f"\n📋 ИНФОРМАЦИЯ О СИСТЕМЕ:")
    print(f"   Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Хост: {db_manager.get_host_id()}")
    print(f"   Модули загружены: {'ДА' if MODULES_IMPORTED else 'НЕТ'}")

    if MODULES_IMPORTED:
        print(f"   Строка подключения: {Config.DB_URI[:80]}..." if len(
            Config.DB_URI) > 80 else f"   Строка подключения: {Config.DB_URI}")

    print(f"\n🔍 ТЕСТ 1: Проверка подключения к БД")
    print("-" * 40)

    test_result = db_manager.test_db_connection()

    if test_result['success']:
        print(f"   ✅ УСПЕШНО!")
        print(f"   База данных: {test_result.get('db_name', 'N/A')}")
        print(f"   Версия MySQL: {test_result.get('db_version', 'N/A')}")
        print(f"   Время ответа: {test_result.get('response_time_ms', 0)} мс")
    else:
        print(f"   ❌ ОШИБКА: {test_result.get('error', 'Неизвестная ошибка')}")

    print(f"\n🔧 ТЕСТ 2: Инициализация подключения")
    print("-" * 40)

    if db_manager.init_db_connection():
        print("   ✅ Подключение инициализировано")
    else:
        print(f"   ❌ Не удалось инициализировать: {db_manager.last_connection_error}")

    print(f"\n📊 ТЕСТ 3: Статус подключения")
    print("-" * 40)

    status = db_manager.get_connection_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    if db_manager.db_connection_active:
        print(f"\n🗃️  ТЕСТ 4: Получение списка таблиц")
        print("-" * 40)

        tables_result = db_manager.get_table_list()

        if tables_result['success']:
            print(f"   ✅ Найдено таблиц: {tables_result['count']}")

            if tables_result['tables']:
                print(f"   📋 Список таблиц (первые 10):")
                for table in tables_result['tables'][:10]:
                    print(f"      - {table['name']}: {table['rows']} строк")
                if tables_result['count'] > 10:
                    print(f"      ... и еще {tables_result['count'] - 10} таблиц")
        else:
            print(f"   ❌ Ошибка: {tables_result.get('error')}")

    print(f"\n🧪 ТЕСТ 5: Проверка работы с сессией")
    print("-" * 40)

    if db_manager.db_connection_active:
        try:
            with db_manager.get_db() as db:
                # Простой запрос для проверки
                result = db.execute(text(
                    "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = DATABASE()")).fetchone()
                print(f"   ✅ Сессия работает, таблиц в БД: {result.count if result else 'N/A'}")

                # Пробуем создать временную таблицу и удалить её
                db.execute(text("CREATE TEMPORARY TABLE test_temp (id INT, name VARCHAR(50))"))
                db.execute(text("INSERT INTO test_temp VALUES (1, 'test'), (2, 'example')"))
                test_rows = db.execute(text("SELECT COUNT(*) FROM test_temp")).scalar()
                print(f"   ✅ Временная таблица создана, записей: {test_rows}")
                db.execute(text("DROP TEMPORARY TABLE test_temp"))
                print("   ✅ Временная таблица удалена")

        except Exception as e:
            print(f"   ❌ Ошибка при работе с сессией: {e}")
    else:
        print("   ⚠️  Пропущено (подключение не активно)")

    print(f"\n" + "=" * 60)
    print("📈 ИТОГИ ТЕСТИРОВАНИЯ:")

    if test_result['success'] and db_manager.db_connection_active:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("   Модуль database.py готов к работе в приложении")
    elif test_result['success']:
        print("⚠️  БАЗОВОЕ ПОДКЛЮЧЕНИЕ РАБОТАЕТ, НО ЕСТЬ ПРОБЛЕМЫ")
        print(f"   Ошибка: {db_manager.last_connection_error}")
    else:
        print("❌ ОСНОВНОЕ ПОДКЛЮЧЕНИЕ НЕ РАБОТАЕТ")
        print(f"   Проверьте настройки БД в .env файле")
        print(f"   Текущая строка подключения: {Config.DB_URI if MODULES_IMPORTED else 'не загружена'}")

    print("=" * 60)
    print("\n💡 Для использования в приложении импортируйте функции:")
    print("   from modules.database import get_db, init_db_connection")
    print("   from modules.database import get_connection_status, test_db_connection")


# ============================================
# ЗАПУСК ПРИ ПРЯМОМ ВЫПОЛНЕНИИ ФАЙЛА
# ============================================

if __name__ == "__main__":
    print("\n📦 Запуск автономного теста модуля database.py")
    print("🔧 Проверка основных функций работы с базой данных\n")

    # Запускаем основной тест
    main()

    # Дополнительные команды для тестирования
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "simple":
            # Простой тест без подробностей
            result = db_manager.test_db_connection()
            if result['success']:
                print(f"✅ OK: {result.get('db_name')} ({result.get('response_time_ms')}ms)")
            else:
                print(f"❌ FAIL: {result.get('error')}")
        elif command == "status":
            status = db_manager.get_connection_status()
            import json

            print(json.dumps(status, indent=2, ensure_ascii=False))
        elif command == "tables":
            if db_manager.init_db_connection():
                tables = db_manager.get_table_list()
                if tables['success']:
                    for table in tables['tables']:
                        print(f"{table['name']}: {table['rows']} строк")

    print("\n🏁 Тест завершен")
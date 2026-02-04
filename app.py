# app.py - ОСНОВНОЙ ФАЙЛ ПРИЛОЖЕНИЯ
import os
import sys
from flask import Flask

# Убедимся, что modules в пути Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем модули
from modules.config import Config
from modules.database import init_db_connection, init_db, migrate_database_safe
from modules.routes import register_routes
from modules.utils import load_full_crypto_list_async

# Убедимся, что используем правильный модуль логирования
# Проверяем, существует ли app_logging
try:
    from modules.app_logging import setup_logging, log_message

    HAS_APP_LOGGING = True
except ImportError:
    # Если файл еще не переименован, создаем простую заглушку
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    def setup_logging():
        pass


    def log_message(message, level='info', **kwargs):
        level_method = getattr(logger, level, logger.info)
        level_method(message)


    HAS_APP_LOGGING = False
    print("⚠️  modules/app_logging.py не найден, используется базовое логирование")

# Настройка логирования
if HAS_APP_LOGGING:
    setup_logging()

# Создаем приложение Flask
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Регистрируем маршруты
register_routes(app)


# Инициализация приложения
def initialize_app():
    """Инициализирует приложение"""
    try:
        # Инициализация базы данных
        db_initialized = init_db_connection()

        if db_initialized:
            # Инициализируем таблицы
            init_db()

            # Выполняем миграцию
            migrate_database_safe()

            log_message("База данных успешно инициализирована", 'info')
        else:
            log_message("Подключение к БД не удалось, приложение работает без БД", 'warning')

        # Запускаем асинхронную загрузку списка криптовалют
        load_full_crypto_list_async()

        log_message("Приложение успешно инициализировано", 'info')

    except Exception as e:
        log_message(f"Ошибка инициализации приложения: {e}", 'error')


# Инициализируем приложение
initialize_app()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК CRYPTO CONVERTER APP")
    print("=" * 60)
    print(f"📊 Режим отладки: {'ВКЛЮЧЕН' if app.debug else 'ВЫКЛЮЧЕН'}")
    print(f"🌐 Сервер: http://localhost:5000")
    print(f"🤖 Бот: {Config.BOT_USERNAME}")
    print(f"💎 Популярных криптовалют: {len(Config.POPULAR_CRYPTOS)}")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
# modules/__init__.py
"""
Модули для Crypto Converter App
Версия 1.0.0
"""

# Основные экспорты для удобства
from .config import Config
from .database import init_db_connection, get_db
from .app_logging import log_message

# Автоматически запускаем загрузку криптовалют при импорте
try:
    from .utils import load_full_crypto_list_async
    load_full_crypto_list_async()
except ImportError:
    pass

__version__ = '1.0.0'
__author__ = 'Romanov Crypto App'
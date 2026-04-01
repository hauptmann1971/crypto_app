from modules.app_logging import log_message
from modules.database import init_db, init_db_connection, migrate_database_safe
from modules.utils import load_full_crypto_list_async


def bootstrap_application() -> None:
    """Initialize infrastructure required for app startup."""
    db_initialized = init_db_connection()

    if db_initialized:
        init_db()
        migrate_database_safe()
        log_message("База данных успешно инициализирована", "info")
    else:
        log_message("Подключение к БД не удалось, приложение работает без БД", "warning")

    load_full_crypto_list_async()
    log_message("Приложение успешно инициализировано", "info")

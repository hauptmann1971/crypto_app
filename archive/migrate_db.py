# migrate_db.py
import sys
import os
from app import migrate_database, init_db_connection, db_connection_active


def main():
    """Ручная миграция базы данных"""
    print("🔄 Запуск миграции базы данных...")

    try:
        # Инициализируем соединение
        init_db_connection()

        if db_connection_active:
            migrate_database()
            print("✅ Миграция успешно завершена!")
        else:
            print("❌ Не удалось подключиться к базе данных")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Ошибка при миграции: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
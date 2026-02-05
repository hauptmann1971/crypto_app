# initial_migration.py
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()


def add_host_column():
    """Добавляет колонку host в таблицы если её нет"""
    try:
        # Конфигурация SQLAlchemy
        DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

        engine = create_engine(DB_URI)

        with engine.connect() as conn:
            # Таблицы для проверки
            tables = ['app_logs', 'crypto_rates', 'crypto_requests', 'telegram_users']

            for table in tables:
                try:
                    # Проверяем существование колонки host
                    result = conn.execute(text(f"""
                        SELECT COUNT(*) 
                        FROM information_schema.columns 
                        WHERE table_schema = DATABASE() 
                        AND table_name = '{table}' 
                        AND column_name = 'host'
                    """)).fetchone()

                    if result[0] == 0:
                        # Добавляем колонку host
                        conn.execute(text(f"""
                            ALTER TABLE {table} 
                            ADD COLUMN host VARCHAR(100) DEFAULT NULL
                        """))
                        conn.commit()
                        print(f"✅ Колонка 'host' добавлена в таблицу {table}")
                    else:
                        print(f"✅ Колонка 'host' уже существует в таблице {table}")

                except SQLAlchemyError as e:
                    print(f"⚠️ Ошибка при проверке таблицы {table}: {e}")
                    continue

        print("\n🎉 Миграция завершена успешно!")

    except Exception as e:
        print(f"❌ Ошибка подключения к базе данных: {e}")
        sys.exit(1)


if __name__ == '__main__':
    print("🔄 Запуск первоначальной миграции базы данных...")
    add_host_column()
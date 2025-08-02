from celery import Celery
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

app = Celery('tasks', broker='redis://localhost:6379/0')

# Конфиг БД (должен совпадать с app.py)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'crypto_db')
}


@app.task
def cleanup_old_logs():
    """Удаляет логи старше 2 дней"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        threshold = datetime.now() - timedelta(days=2)
        cursor.execute("""
            DELETE FROM app_logs 
            WHERE timestamp < %s
        """, (threshold,))

        conn.commit()
        print(f"[Cleanup] Удалено {cursor.rowcount} записей")

    except Error as e:
        print(f"[Cleanup Error] {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()


# Планировщик (выполнять каждый час)
app.conf.beat_schedule = {
    'cleanup-logs-hourly': {
        'task': 'celery_worker.cleanup_old_logs',
        'schedule': 90,  # Каждый час
    },
}
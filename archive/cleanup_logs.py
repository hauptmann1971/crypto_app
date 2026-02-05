import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'crypto_db')
}

def cleanup_old_logs():
    """Удаляет логи старше 2 дней"""
    conn = mysql.connector.connect(**DB_CONFIG)
    if conn:
        try:
            cursor = conn.cursor()
            threshold = datetime.now() - timedelta(seconds=120)
            cursor.execute("""
                DELETE FROM app_logs 
                WHERE timestamp < %s
            """, (threshold,))
            conn.commit()
            print(f"Удалено {cursor.rowcount} старых логов")
        except Error as e:
            print(f"Ошибка при очистке логов: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

if __name__ == '__main__':
    cleanup_old_logs()
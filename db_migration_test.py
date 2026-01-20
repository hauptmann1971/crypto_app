import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()


conn = mysql.connector.connect(
    host = 'kalmyk3j.beget.tech',
    user = 'kalmyk3j_romanov',
    password = '6xkGG33NX9%p',
    database = 'kalmyk3j_romanov'
)
cursor = conn.cursor()
try:
    cursor.execute('ALTER TABLE app_logs ADD COLUMN host VARCHAR(100) DEFAULT NULL')
    print('Колонка host добавлена')
except Exception as e:
    print(f'Ошибка: {e}')
conn.close()
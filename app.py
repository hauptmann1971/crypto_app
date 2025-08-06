from flask import Flask, render_template, request, flash, redirect
import mysql.connector
from mysql.connector import Error, pooling
import time
import requests
import os
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
from datetime import datetime
from cachetools import cached, TTLCache

# Инициализация
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Конфигурация БД
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'pool_name': 'crypto_pool',
    'pool_size': 5
}

db_pool = pooling.MySQLConnectionPool(**DB_CONFIG)
TOP_CRYPTO = ['bitcoin', 'ethereum', 'binancecoin']
CURRENCIES = ['usd', 'eur', 'gbp', 'jpy', 'cny', 'rub']
crypto_list_cache = TTLCache(maxsize=1, ttl=3600)


@contextmanager
def db_connection():
    conn = db_pool.get_connection()
    try:
        yield conn
    except Error as e:
        conn.rollback()
        log_message(f"DB Error: {e}", 'error')
        raise
    finally:
        conn.close()


def log_message(message: str, level: str = 'info'):
    """Запись лога в БД и файл"""
    logging.log(getattr(logging, level.upper()), message)

    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO app_logs (message, level, timestamp)
                VALUES (%s, %s, %s)
            """, (message, level, int(time.time())))
            conn.commit()
    except Error as e:
        logging.error(f"Failed to write log to DB: {e}")


def init_db():
    """Инициализация структуры БД"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Таблица курсов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_rates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                crypto VARCHAR(50) NOT NULL,
                currency VARCHAR(10) NOT NULL,
                rate DECIMAL(20,8) NOT NULL,
                source VARCHAR(100) NOT NULL,
                timestamp BIGINT NOT NULL,
                INDEX idx_crypto_currency (crypto, currency)
            )
        """)

        # Таблица логов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                message TEXT NOT NULL,
                level VARCHAR(20) NOT NULL,
                timestamp BIGINT NOT NULL,
                INDEX idx_timestamp (timestamp)
            )
        """)
        conn.commit()
    log_message("Database initialized", 'info')


@cached(crypto_list_cache)
def load_crypto_list():
    """Загружает список криптовалют, сохраняя TOP_CRYPTO в начале"""
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/list",
            timeout=5
        )
        response.raise_for_status()

        # Получаем все криптовалюты из API
        all_crypto = [c['id'] for c in response.json()]

        # Фильтруем, чтобы TOP_CRYPTO были в начале
        sorted_crypto = TOP_CRYPTO.copy()  # Стартуем с основных криптовалют

        # Добавляем остальные, исключая дубликаты
        for crypto in all_crypto:
            if crypto not in TOP_CRYPTO:
                sorted_crypto.append(crypto)

        return sorted_crypto

    except requests.RequestException as e:
        logging.warning(f"Ошибка загрузки списка: {e}. Используем базовый набор.")
        return TOP_CRYPTO  # Возвращаем хотя бы основные, если API недоступно


@app.route('/', methods=['GET', 'POST'])
def index():
    rate = None
    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')

        if not crypto or not currency:
            flash("Select crypto and currency", 'error')
            log_message("No crypto/currency selected", 'warning')
            return redirect('/')

        try:
            # Получаем курс
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies={currency}"
            log_message(f"API request: {url}", 'debug')
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if currency not in data.get(crypto, {}):
                msg = f"Rate not found for {crypto}/{currency}"
                flash(msg, 'error')
                log_message(msg, 'error')
                return redirect('/')

            rate = data[crypto][currency]
            epoch_time = int(time.time())

            # Запись в БД
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO crypto_rates
                    (crypto, currency, rate, source, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (crypto, currency, rate, "coingecko", epoch_time))
                conn.commit()

            msg = f"Rate saved: {crypto.upper()}/{currency.upper()} = {rate:.4f}"
            flash(msg, 'success')
            log_message(msg, 'info')

        except requests.RequestException as e:
            msg = f"API Error: {str(e)}"
            flash(msg, 'error')
            log_message(msg, 'error')
        except Error as e:
            msg = f"DB Error: {str(e)}"
            flash(msg, 'error')
            log_message(msg, 'error')

    cryptos = load_crypto_list()
    return render_template('index.html', cryptos=cryptos, currencies=CURRENCIES, rate=rate)


@app.route('/crypto_table')
def show_crypto_table():
    """Отображает содержимое таблицы crypto_rates"""
    with db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                crypto, 
                currency, 
                rate, 
                FROM_UNIXTIME(timestamp) as date_time,
                source
            FROM crypto_rates
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        rates = cursor.fetchall()

    return render_template('data_table.html',
                           title='Курсы криптовалют',
                           data=rates,
                           columns=['crypto', 'currency', 'rate', 'date_time', 'source'])


@app.route('/log_table')
def show_log_table():
    """Отображает содержимое таблицы app_logs"""
    with db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                message, 
                level, 
                FROM_UNIXTIME(timestamp) as date_time
            FROM app_logs
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        logs = cursor.fetchall()

    return render_template('data_table.html',
                           title='Логи приложения',
                           data=logs,
                           columns=['date_time', 'level', 'message'])

if __name__ == '__main__':
    init_db()
    log_message("Application started", 'info')
    app.run(debug=True)
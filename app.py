from flask import Flask, render_template, request, flash
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# Конфигурация подключения к БД
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'crypto_db')
}

TOP_CRYPTO = ['bitcoin', 'ethereum', 'binancecoin']
CURRENCIES = ['usd', 'eur', 'gbp', 'jpy', 'cny', 'rub']


def get_db_connection():
    """Устанавливает соединение с MySQL"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        log_message(f"Ошибка подключения к MySQL: {e}", 'error')
        return None


def log_message(message, level='info'):
    """Записывает сообщение в таблицу app_logs"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO app_logs (message, level)
                VALUES (%s, %s)
            """, (message, level))
            conn.commit()
        except Error as e:
            print(f"Критическая ошибка при записи лога: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()


def check_and_create_tables():
    """Проверяет существование таблиц и создает их при необходимости"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()

            # Проверяем существование таблицы crypto_rates
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'crypto_rates'
            """, (DB_CONFIG['database'],))

            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    CREATE TABLE crypto_rates (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        crypto VARCHAR(50) NOT NULL,
                        currency VARCHAR(10) NOT NULL,
                        rate FLOAT NOT NULL,
                        source VARCHAR(200) NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                log_message("Создана таблица crypto_rates", 'info')

            # Проверяем существование таблицы app_logs
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'app_logs'
            """, (DB_CONFIG['database'],))

            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    CREATE TABLE app_logs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        message VARCHAR(500) NOT NULL,
                        level VARCHAR(20) NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                log_message("Создана таблица app_logs", 'info')

            conn.commit()
        except Error as e:
            log_message(f"Ошибка при работе с таблицами: {e}", 'error')
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()


@app.before_first_request
def initialize():
    """Инициализация приложения перед первым запросом"""
    log_message("Запуск приложения", 'info')
    check_and_create_tables()


@app.route('/', methods=['GET', 'POST'])
def index():
    crypto_rate = None

    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')

        if not crypto or not currency:
            msg = "Не выбрана криптовалюта или валюта"
            log_message(msg, 'warning')
            flash(msg, 'error')
        else:
            try:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies={currency}"
                log_message(f"Запрос к API: {url}", 'debug')
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if crypto in data and currency in data[crypto]:
                    crypto_rate = data[crypto][currency]
                    source = "https://www.coingecko.com"
                    log_message(f"Получен курс: {crypto} -> {currency} = {crypto_rate}", 'info')

                    conn = get_db_connection()
                    if conn:
                        try:
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO crypto_rates (crypto, currency, rate, source)
                                VALUES (%s, %s, %s, %s)
                            """, (crypto, currency, crypto_rate, source))
                            conn.commit()
                            log_message("Данные сохранены в crypto_rates", 'info')

                            formatted_rate = f"{crypto_rate:.2f}" if isinstance(crypto_rate, float) else crypto_rate
                            flash_msg = f"Курс {crypto.upper()} к {currency.upper()}: {formatted_rate}"
                            flash(flash_msg, 'success')

                        except Error as e:
                            error_msg = f"Ошибка БД: {str(e)}"
                            log_message(error_msg, 'error')
                            flash(error_msg, 'error')
                        finally:
                            if conn.is_connected():
                                cursor.close()
                                conn.close()
                else:
                    error_msg = f"API не вернул курс для пары {crypto}/{currency}"
                    log_message(error_msg, 'error')
                    flash(error_msg, 'error')

            except requests.RequestException as e:
                error_msg = f"Ошибка API: {str(e)}"
                log_message(error_msg, 'error')
                flash(error_msg, 'error')

    # Получаем список криптовалют
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/list")
        response.raise_for_status()
        all_crypto = response.json()
        crypto_list = sorted([crypto['id'] for crypto in all_crypto])
        sorted_crypto = TOP_CRYPTO + [c for c in crypto_list if c not in TOP_CRYPTO]
        log_message("Список криптовалют успешно загружен", 'info')
    except requests.RequestException as e:
        sorted_crypto = TOP_CRYPTO
        error_msg = f"Ошибка загрузки списка криптовалют: {str(e)}"
        log_message(error_msg, 'error')
        flash("Не удалось загрузить полный список криптовалют. Используем базовый набор.", 'error')

    return render_template(
        'index.html',
        cryptos=sorted_crypto,
        currencies=CURRENCIES,
        rate=crypto_rate
    )


if __name__ == '__main__':
    check_and_create_tables()
    app.run(debug=True)
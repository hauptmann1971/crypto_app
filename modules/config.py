# modules/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    BOT_USERNAME = os.getenv('BOT_USERNAME', '@romanov_crypto_currency_bot')
    BOT_TOKEN = os.getenv('BOT_TOKEN', '')
    DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    SECRET_KEY = os.getenv('SECRET_KEY')
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() in {'1', 'true', 'yes', 'on'}

    # Таймауты API
    API_TIMEOUT_SHORT = int(os.getenv('API_TIMEOUT_SHORT', '10'))
    API_TIMEOUT_LONG = int(os.getenv('API_TIMEOUT_LONG', '30'))

    # Константы
    POPULAR_CRYPTOS = [
        'bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano', 'solana',
        'polkadot', 'dogecoin', 'matic-network', 'stellar', 'litecoin', 'chainlink',
        'bitcoin-cash', 'ethereum-classic', 'monero', 'eos', 'tezos', 'aave',
        'cosmos', 'uniswap', 'tron', 'neo', 'vechain', 'theta-token', 'filecoin',
        'algorand', 'maker', 'compound-governance-token', 'dash', 'zcash',
        'decred', 'waves', 'ontology', 'icon', 'zilliqa', 'bittorrent',
        'pancakeswap-token', 'sushi', 'curve-dao-token', 'yearn-finance',
        'balancer', 'uma', 'renbtc', 'helium', 'chiliz', 'enjincoin',
        'axie-infinity', 'the-sandbox', 'decentraland', 'gala'
    ]

    TOP_CRYPTO = ['bitcoin', 'ethereum', 'binancecoin']
    CURRENCIES = ['usd', 'eur', 'gbp', 'jpy', 'cny', 'rub']
    PERIODS = [
        {'value': '1', 'label': '1 день'},
        {'value': '7', 'label': '7 дней'},
        {'value': '30', 'label': '30 дней'},
        {'value': '90', 'label': '90 дней'},
        {'value': '180', 'label': '180 дней'},
        {'value': '365', 'label': '1 год'}
    ]

    @classmethod
    def validate(cls):
        """Проверяет критические настройки при старте"""
        errors = []

        if not cls.SECRET_KEY:
            errors.append("SECRET_KEY не установлен в переменных окружения!")

        if not cls.BOT_TOKEN:
            errors.append("BOT_TOKEN не установлен!")

        db_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_NAME']
        missing_db = [var for var in db_vars if not os.getenv(var)]
        if missing_db:
            errors.append(f"Отсутствуют переменные БД: {', '.join(missing_db)}")

        if errors and not cls.DEBUG:
            raise ValueError("Ошибки конфигурации:\n" + "\n".join(f"  - {e}" for e in errors))
        elif errors:
            import logging
            for error in errors:
                logging.warning(f"⚠️ Конфигурация: {error}")
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Основные настройки Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

    # Настройки MySQL
    DB_HOST = os.getenv('DB_HOST', 'kalmyk3j.beget.tech')
    DB_USER = os.getenv('DB_USER', 'kalmyk3j_romanov')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '6xkGG33NX9%p')
    DB_NAME = os.getenv('DB_NAME', 'kalmyk3j_romanov')

    # Настройки Redis для Celery (если используется)
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = os.getenv('REDIS_PORT', '6379')
    REDIS_DB = os.getenv('REDIS_DB', '0')

    # Настройки API CoinGecko
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

    # Другие настройки
    TOP_CRYPTO = ['bitcoin', 'ethereum', 'binancecoin']  # Основные криптовалюты
    CURRENCIES = ['usd', 'eur', 'gbp', 'jpy', 'cny', 'rub']  # Поддерживаемые валюты

    @property
    def DB_CONFIG(self):
        """Возвращает конфиг для подключения к MySQL"""
        return {
            'host': self.DB_HOST,
            'user': self.DB_USER,
            'password': self.DB_PASSWORD,
            'database': self.DB_NAME
        }

    @property
    def CELERY_BROKER_URL(self):
        """URL для подключения Celery к Redis"""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
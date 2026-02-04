# modules/utils.py (обновленная версия)
import logging
import time
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from cachetools import cached, TTLCache
from modules.config import Config

# Кэш для списка криптовалют
crypto_list_cache = TTLCache(maxsize=1, ttl=3600)

# Глобальные переменные для асинхронной загрузки
FULL_CRYPTO_LIST = Config.POPULAR_CRYPTOS.copy()
CRYPTO_LOADING = False
CRYPTO_LOADED = False


@dataclass
class TelegramUserData:
    id: int
    first_name: str
    auth_date: int
    hash: str
    username: Optional[str] = None
    photo_url: Optional[str] = None
    last_name: Optional[str] = None


def verify_telegram_authentication(data: dict, bot_token: str) -> bool:
    """Проверяет данные авторизации Telegram"""
    try:
        if not isinstance(data, dict):
            return False

        required_fields = ['id', 'first_name', 'auth_date', 'hash']
        for field in required_fields:
            if field not in data or not isinstance(data[field], (str, int)):
                return False

        try:
            auth_date = datetime.fromtimestamp(int(data['auth_date']))
        except (ValueError, TypeError):
            return False

        if datetime.now() - auth_date > timedelta(hours=24):
            return False

        data_check_string = '\n'.join(
            f'{key}={value}'
            for key, value in sorted(data.items())
            if key != 'hash'
        )

        secret_key = hashlib.sha256(bot_token.encode()).digest()
        computed_hash = hmac.new(
            secret_key,
            data_check_string.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(computed_hash, data['hash'])

    except Exception as e:
        logging.error(f"Error verifying Telegram auth: {e}")
        return False


def load_full_crypto_list_async():
    """Асинхронно загружает полный список криптовалют"""
    global FULL_CRYPTO_LIST, CRYPTO_LOADING, CRYPTO_LOADED

    if CRYPTO_LOADING or CRYPTO_LOADED:
        return

    CRYPTO_LOADING = True

    def load_thread():
        global FULL_CRYPTO_LIST, CRYPTO_LOADING, CRYPTO_LOADED
        try:
            import requests
            logging.info("Starting async full crypto list loading...")
            response = requests.get(
                "https://api.coingecko.com/api/v3/coins/list",
                timeout=30
            )
            response.raise_for_status()

            all_crypto = [c['id'] for c in response.json()]

            combined_list = Config.POPULAR_CRYPTOS.copy()
            for crypto in all_crypto:
                if crypto not in combined_list:
                    combined_list.append(crypto)

            FULL_CRYPTO_LIST = combined_list
            CRYPTO_LOADED = True
            CRYPTO_LOADING = False
            logging.info(f"Full crypto list loaded: {len(FULL_CRYPTO_LIST)} items")

        except Exception as e:
            logging.error(f"Async crypto list loading failed: {e}")
            CRYPTO_LOADING = False

    import threading
    thread = threading.Thread(target=load_thread)
    thread.daemon = True
    thread.start()


@cached(crypto_list_cache)
def load_crypto_list():
    """Возвращает список криптовалют"""
    load_full_crypto_list_async()
    return FULL_CRYPTO_LIST


def get_correlation_strength(corr_value):
    """Определяет силу корреляции"""
    abs_corr = abs(corr_value)
    if abs_corr >= 0.9:
        return 'very strong'
    elif abs_corr >= 0.7:
        return 'strong'
    elif abs_corr >= 0.5:
        return 'moderate'
    elif abs_corr >= 0.3:
        return 'weak'
    else:
        return 'very weak'
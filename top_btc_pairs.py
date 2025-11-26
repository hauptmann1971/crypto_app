import requests
import logging

# Настройка логирования (опционально, но рекомендуется)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Словарь популярных криптовалют (50 штук) - взято из app.py
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

def get_main_crypto_rates_to_btc(timeout=30):
    """
    Запрашивает курсы основных криптовалют к биткоину (BTC) с API CoinGecko.

    Args:
        timeout (int): Таймаут для HTTP-запросов в секундах. По умолчанию 30.

    Returns:
        dict: Словарь, где ключ - ID криптовалюты (например, 'ethereum', 'bitcoin'),
              значение - словарь с курсом к BTC {'btc': float_rate}.
              Пример: {'ethereum': {'btc': 0.0654321}, 'bitcoin': {'btc': 1.0}}
              Возвращает пустой словарь в случае ошибки.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    # Подготовим список ID, исключив 'bitcoin', чтобы не запрашивать курс к себе самому
    # (хотя API обычно корректно возвращает 1.0 для 'bitcoin' к 'btc')
    main_coin_ids = [cid for cid in POPULAR_CRYPTOS if cid != 'bitcoin']
    params = {
        'ids': ','.join(main_coin_ids), # Передаём список ID через запятую
        'vs_currencies': 'btc' # Запрашиваем курсы относительно BTC
    }

    try:
        logger.info(f"Запрашиваю курсы основных криптовалют ({len(main_coin_ids)} шт.) к BTC...")
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status() # Вызывает исключение для HTTP-ошибок (4xx, 5xx)

        rates = response.json()
        logger.info(f"Получены курсы для {len(rates)} криптовалют к BTC.")

        # Добавим курс самого биткоина к себе (если нужно)
        # rates['bitcoin'] = {'btc': 1.0}

        return rates

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP ошибка при запросе к API CoinGecko: {e}")
        logger.error(f"Статус код: {e.response.status_code}")
        logger.error(f"Текст ответа: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к API CoinGecko: {e}")
    except ValueError as e: # Ошибка при парсинге JSON
        logger.error(f"Ошибка парсинга JSON ответа от API CoinGecko: {e}")
    except Exception as e:
        logger.error(f"Неизвестная ошибка при запросе курсов: {e}")

    # Возвращаем пустой словарь в случае любой ошибки
    return {}

# Пример использования
if __name__ == "__main__":
    rates_to_btc = get_main_crypto_rates_to_btc()
    if rates_to_btc:
        # Выведем все пары
        for crypto_id, rate_data in rates_to_btc.items():
            print(f"{crypto_id}: {rate_data.get('btc', 'N/A')} BTC")
    else:
        print("Не удалось получить курсы основных криптовалют к BTC.")
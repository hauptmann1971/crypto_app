# modules/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    BOT_USERNAME = os.getenv('BOT_USERNAME', '@romanov_crypto_currency_bot')
    BOT_TOKEN = os.getenv('BOT_TOKEN', '8264247176:AAFByVrbcY8K-aanicYu2QK-tYRaFNq0lxY')
    DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

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
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class CoinGeckoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
    
    def get_ohlc(self, coin_id, vs_currency, days):
        """
        Получение OHLC данных (Open, High, Low, Close)
        
        Args:
            coin_id: идентификатор монеты (e.g., 'bitcoin')
            vs_currency: валюта (e.g., 'usd')
            days: период (1, 7, 14, 30, 90, 180, 365, max)
        """
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Конвертируем в DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        else:
            print(f"Ошибка: {response.status_code}")
            return None

# Пример использования
api = CoinGeckoAPI()

# Получаем данные по Bitcoin за 30 дней
btc_data = api.get_ohlc('bitcoin', 'eur', 1)

if btc_data is not None:
    print(btc_data.head())
    
    # Построение свечного графика
    plt.figure(figsize=(12, 6))
    
    # Простой линейный график для начала
    print(btc_data['close'])
    plt.plot(btc_data.index, btc_data['close'])
    plt.title('BTC/USD Price (1 day(s))')
    plt.xlabel('Date')
    plt.ylabel('Price (EUR)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
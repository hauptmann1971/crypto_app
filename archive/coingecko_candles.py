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
            print(data)
            # Конвертируем в DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        else:
            print(f"Ошибка: {response.status_code}")
            return None


    def get_plot(self, data):
        
        if data is not None:
            print(data.head())
            
            # Построение графика
            plt.figure(figsize=(12, 6))
            
            # Линейный график
            
            plt.plot(data.index, data['close'])
            plt.title('BTC/USD Price (1 day(s))')
            plt.xlabel('Date')
            plt.ylabel('Price (EUR)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()      

# Пример использования

api = CoinGeckoAPI()

# Получаем данные по Bitcoin за 30 дней
btc_data = api.get_ohlc('bitcoin', 'rub', 1)
api.get_plot(btc_data)

    
    
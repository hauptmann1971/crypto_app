# crypto_chart.py
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import io
import base64
from datetime import datetime
import logging
from typing import Optional, Dict, List, Tuple

class CryptoChartAPI:
    """
    Пакет для работы с CoinGecko API и построения свечных графиков
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoApp/1.0',
            'Accept': 'application/json'
        })
    
    def get_ohlc_data(self, coin_id: str, vs_currency: str = 'usd', days: int = 7) -> Optional[pd.DataFrame]:
        """
        Получает OHLC данные (Open, High, Low, Close) для построения свечного графика
        
        Args:
            coin_id: идентификатор монеты (e.g., 'bitcoin')
            vs_currency: валюта (e.g., 'usd')
            days: период (1, 7, 14, 30, 90, 180, 365)
        
        Returns:
            DataFrame с колонками: timestamp, open, high, low, close
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logging.warning(f"No OHLC data received for {coin_id}/{vs_currency}")
                return None
            
            # Конвертируем в DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Конвертируем цены в float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            
            logging.info(f"Retrieved {len(df)} OHLC records for {coin_id}/{vs_currency}")
            return df
            
        except requests.RequestException as e:
            logging.error(f"API request error: {e}")
            return None
        except Exception as e:
            logging.error(f"Error processing OHLC data: {e}")
            return None
    
    def get_historical_data(self, coin_id: str, vs_currency: str = 'usd', days: int = 30) -> Optional[pd.DataFrame]:
        """
        Альтернативный метод получения исторических данных
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            prices = data.get('prices', [])
            
            if not prices:
                return None
            
            # Создаем DataFrame из цен
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Для простого графика используем только цены закрытия
            df = df.rename(columns={'price': 'close'})
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting historical data: {e}")
            return None
    
    def create_candlestick_chart(self, 
                               data: pd.DataFrame, 
                               crypto: str, 
                               currency: str, 
                               period: int,
                               figsize: Tuple[int, int] = (12, 6)) -> Optional[str]:
        """
        Создает свечной график и возвращает его в base64
        
        Args:
            data: DataFrame с OHLC данными
            crypto: название криптовалюты
            currency: валюта
            period: период в днях
            figsize: размер графика
        
        Returns:
            base64 строка с изображением графика или None при ошибке
        """
        try:
            if data is None or data.empty:
                logging.warning("No data provided for chart creation")
                return None
            
            # Создаем график
            fig, ax = plt.subplots(figsize=figsize)
            
            # Упрощенный свечной график
            for idx, row in data.iterrows():
                open_price = row['open']
                close_price = row['close']
                high_price = row['high']
                low_price = row['low']
                
                # Определяем цвет свечи (зеленая - рост, красная - падение)
                color = 'green' if close_price >= open_price else 'red'
                
                # Рисуем тело свечи
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                # Рисуем тело свечи (прямоугольник)
                rect = Rectangle((mdates.date2num(idx) - 0.3, body_bottom), 
                               0.6, body_height, 
                               facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
                
                # Рисуем тени (высшая и низшая точки)
                ax.plot([mdates.date2num(idx), mdates.date2num(idx)], 
                       [low_price, high_price], 
                       color='black', linewidth=1)
            
            # Настройки графика
            ax.set_title(f'{crypto.upper()}/{currency.upper()} Candlestick Chart ({period} days)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(f'Price ({currency.upper()})', fontsize=12)
            
            # Форматируем оси
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Конвертируем график в base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plt.close(fig)
            
            plot_url = base64.b64encode(img.getvalue()).decode()
            return f"data:image/png;base64,{plot_url}"
            
        except Exception as e:
            logging.error(f"Error creating candlestick chart: {e}")
            plt.close('all')
            return None
    
    def create_simple_price_chart(self, 
                                data: pd.DataFrame, 
                                crypto: str, 
                                currency: str, 
                                period: int,
                                chart_type: str = 'line') -> Optional[str]:
        """
        Создает простой линейный график цен
        
        Args:
            data: DataFrame с ценовыми данными
            crypto: название криптовалюты
            currency: валюта
            period: период в днях
            chart_type: тип графика ('line' или 'area')
        
        Returns:
            base64 строка с изображением графика
        """
        try:
            if data is None or data.empty:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if 'close' in data.columns:
                prices = data['close']
            else:
                # Если есть только OHLC данные, используем close
                prices = data['close']
            
            if chart_type == 'area':
                ax.fill_between(prices.index, prices.values, alpha=0.3)
                ax.plot(prices.index, prices.values, linewidth=2)
            else:
                ax.plot(prices.index, prices.values, linewidth=2)
            
            ax.set_title(f'{crypto.upper()}/{currency.upper()} Price Chart ({period} days)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(f'Price ({currency.upper()})', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Конвертируем в base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plt.close(fig)
            
            plot_url = base64.b64encode(img.getvalue()).decode()
            return f"data:image/png;base64,{plot_url}"
            
        except Exception as e:
            logging.error(f"Error creating simple chart: {e}")
            plt.close('all')
            return None
    
    def get_available_periods(self) -> List[Dict[str, str]]:
        """
        Возвращает список доступных периодов для графиков
        """
        return [
            {'value': '1', 'label': '1 день'},
            {'value': '7', 'label': '7 дней'},
            {'value': '14', 'label': '14 дней'},
            {'value': '30', 'label': '30 дней'},
            {'value': '90', 'label': '90 дней'},
            {'value': '180', 'label': '180 дней'},
            {'value': '365', 'label': '1 год'}
        ]
    
    def validate_coin_id(self, coin_id: str) -> bool:
        """
        Проверяет валидность идентификатора монеты
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

# Создаем глобальный экземпляр для использования
crypto_chart_api = CryptoChartAPI()
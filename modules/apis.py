# modules/apis.py
import requests
import pandas as pd
import numpy as np
from scipy import stats
import logging
import time
from typing import Optional, Dict, List
from datetime import datetime
import io
import base64
import plotly.graph_objs as go
import plotly.utils
import json
import logging

from modules.utils import get_correlation_strength, load_crypto_list
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class CoinGeckoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_ohlc(self, coin_id: str, vs_currency: str, days: str) -> Optional[pd.DataFrame]:
        """Получение OHLC данных"""
        try:
            safe_coin_id = requests.utils.quote(coin_id)
            safe_currency = requests.utils.quote(vs_currency)

            url = f"{self.base_url}/coins/{safe_coin_id}/ohlc"
            params = {
                'vs_currency': safe_currency,
                'days': days
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                logging.error(f"Ошибка API CoinGecko: {response.status_code}")
                return None
        except requests.RequestException as e:
            logging.error(f"Ошибка запроса к CoinGecko: {e}")
            return None

    def generate_plot(self, data: pd.DataFrame, crypto: str, currency: str, period: str) -> Optional[str]:
        """Генерирует график и возвращает его в base64"""
        if data is None or data.empty:
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            colors = []
            for i in range(len(data)):
                if data['close'].iloc[i] >= data['open'].iloc[i]:
                    colors.append('green')
                else:
                    colors.append('red')

            for i in range(len(data)):
                ax.fill_between([i - 0.3, i + 0.3],
                                [data['open'].iloc[i], data['open'].iloc[i]],
                                [data['close'].iloc[i], data['close'].iloc[i]],
                                color=colors[i], alpha=0.7)

                ax.plot([i, i], [data['high'].iloc[i], max(data['open'].iloc[i], data['close'].iloc[i])],
                        color=colors[i], linewidth=1)

                ax.plot([i, i], [min(data['open'].iloc[i], data['close'].iloc[i]), data['low'].iloc[i]],
                        color=colors[i], linewidth=1)

            ax.set_title(f'{crypto.upper()}/{currency.upper()} Свечной график ({period} дней)',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Временной период', fontsize=12)
            ax.set_ylabel(f'Цена ({currency.upper()})', fontsize=12)
            ax.grid(True, alpha=0.3)

            n = len(data)
            step = max(1, n // 10)
            ax.set_xticks(range(0, n, step))
            ax.set_xticklabels([data.index[i].strftime('%m-%d') for i in range(0, n, step)], rotation=45)

            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plt.close()

            plot_url = base64.b64encode(img.getvalue()).decode()
            return f"data:image/png;base64,{plot_url}"

        except Exception as e:
            logging.error(f"Ошибка генерации свечного графика: {e}")
            return None

    def get_historical_price_data(self, coin_id, vs_currency='usd', days=30):
        """Получает исторические данные для расчета корреляции с обработкой ошибок"""
        try:
            ping_url = "https://api.coingecko.com/api/v3/ping"
            ping_response = requests.get(ping_url, timeout=5)

            if ping_response.status_code != 200:
                logging.error(f"CoinGecko API недоступен. Статус: {ping_response.status_code}")
                return None

            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily'
            }

            logging.info(f"Запрашиваю данные для {coin_id}/{vs_currency} за {days} дней...")

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                logging.error(f"Превышен лимит запросов для {coin_id}. Подождите минуту.")
                return None
            elif response.status_code == 404:
                logging.error(f"Криптовалюта {coin_id} не найдена")
                return None
            elif response.status_code != 200:
                logging.error(f"Ошибка API для {coin_id}: {response.status_code}")
                logging.error(f"Ответ: {response.text[:200]}")
                return None

            data = response.json()

            if 'prices' not in data or not data['prices']:
                logging.error(f"Нет данных о ценах для {coin_id}")
                return None

            timestamps = []
            prices = []

            for item in data['prices']:
                if len(item) >= 2:
                    timestamps.append(pd.to_datetime(item[0], unit='ms'))
                    prices.append(item[1])

            if len(prices) < 2:
                logging.error(f"Недостаточно данных для {coin_id}: {len(prices)} точек")
                return None

            df = pd.DataFrame({
                'timestamp': timestamps,
                'price': prices
            })

            df = df.sort_values('timestamp').reset_index(drop=True)

            logging.info(f"Получено {len(df)} записей для {coin_id}")

            return df

        except requests.exceptions.Timeout:
            logging.error(f"Таймаут при запросе данных для {coin_id}")
            return None
        except requests.exceptions.ConnectionError:
            logging.error(f"Ошибка соединения при запросе {coin_id}")
            return None
        except Exception as e:
            logging.error(f"Ошибка получения данных для {coin_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None


class BinanceAPI:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbols_cache = {}

    def get_all_symbols(self):
        """Получает все доступные торговые пары на Binance"""
        try:
            if self.symbols_cache:
                return self.symbols_cache

            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url, timeout=10)
            data = response.json()

            symbols = []
            for symbol_info in data['symbols']:
                if symbol_info['status'] == 'TRADING':
                    symbols.append(symbol_info['symbol'])

            self.symbols_cache = symbols
            return symbols

        except Exception as e:
            logging.error(f"Ошибка получения символов Binance: {e}")
            return []

    def get_historical_klines(self, symbol, interval='1d', limit=100, start_time=None, end_time=None):
        """Получает исторические данные свечей (klines) с Binance"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            return df

        except Exception as e:
            logging.error(f"Ошибка получения данных для {symbol}: {e}")
            return None

    def get_daily_returns(self, symbol, days=30):
        """Получает дневные доходности для символа"""
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)

            df = self.get_historical_klines(
                symbol=symbol,
                interval='1d',
                limit=days,
                start_time=start_time,
                end_time=end_time
            )

            if df is None or df.empty:
                return None

            prices = df['close'].values
            if len(prices) < 2:
                return None

            returns = np.diff(prices) / prices[:-1] * 100
            return returns

        except Exception as e:
            logging.error(f"Ошибка расчета доходностей для {symbol}: {e}")
            return None

    def get_available_crypto_pairs(self, vs_currency='USDT'):
        """Получает список криптовалютных пар с указанной валютой"""
        try:
            symbols = self.get_all_symbols()
            crypto_pairs = {}

            for symbol in symbols:
                if symbol.endswith(vs_currency):
                    crypto = symbol.replace(vs_currency, '')
                    crypto_pairs[crypto.lower()] = symbol

            return crypto_pairs

        except Exception as e:
            logging.error(f"Ошибка получения пар: {e}")
            return {}


# Создаем экземпляры API
coingecko_api = CoinGeckoAPI()
binance_api = BinanceAPI()


# Функции для работы с API
def get_main_crypto_rates_to_btc(timeout=30, coin_ids_to_fetch=None):
    """Запрашивает курсы криптовалют к биткоину (BTC) с API CoinGecko"""
    from modules.config import Config

    if coin_ids_to_fetch is None:
        coin_ids_to_fetch = [cid for cid in Config.POPULAR_CRYPTOS if cid != 'bitcoin']
    else:
        coin_ids_to_fetch = [cid for cid in coin_ids_to_fetch if cid != 'bitcoin']

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': ','.join(coin_ids_to_fetch),
        'vs_currencies': 'btc'
    }

    try:
        logging.info(f"Запрашиваю курсы криптовалют ({len(coin_ids_to_fetch)} шт.) к BTC...")
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        rates = response.json()
        logging.info(f"Получены курсы для {len(rates)} криптовалют к BTC.")

        return rates

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP ошибка при запросе к API CoinGecko: {e}")
        logging.error(f"Статус код: {e.response.status_code}")
        logging.error(f"Текст ответа: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка запроса к API CoinGecko: {e}")
    except ValueError as e:
        logging.error(f"Ошибка парсинга JSON ответа от API CoinGecko: {e}")
    except Exception as e:
        logging.error(f"Неизвестная ошибка при запросе курсов: {e}")

    return {}


def generate_historical_plot(df, crypto, currency, start_date, end_date):
    """Генерирует график исторических данных и возвращает base64 строку"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # График цены
        ax1.plot(df['timestamp'], df['price'], 'b-', linewidth=2, label='Цена')
        ax1.set_title(f'{crypto.upper()}/{currency.upper()} - Исторические данные ({start_date} - {end_date})',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'Цена ({currency.upper()})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Форматирование оси времени
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # График процентных изменений
        if 'returns_pct' in df.columns:
            colors = ['green' if x >= 0 else 'red' for x in df['returns_pct']]
            ax2.bar(df['timestamp'], df['returns_pct'], color=colors, alpha=0.7, width=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Дата', fontsize=12)
            ax2.set_ylabel('Изменение (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)

            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Сохраняем в буфер
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()

        # Конвертируем в base64
        plot_url = base64.b64encode(img.getvalue()).decode()
        return f"data:image/png;base64,{plot_url}"

    except Exception as e:
        logging.error(f"Ошибка генерации исторического графика: {e}")
        return None


def get_historical_price_range(coin_id='bitcoin', vs_currency='usd',
                               start_date='2024-01-01', end_date=None):
    """Получить исторические цены для выбранного периода"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    if start_dt >= end_dt:
        raise ValueError("Дата начала должна быть раньше даты окончания")

    days_diff = (end_dt - start_dt).days

    if days_diff < 1:
        raise ValueError("Период должен быть хотя бы 1 день")

    if days_diff <= 90:
        days_param = days_diff
        interval = 'daily'
    else:
        days_param = days_diff
        interval = 'daily'
        logging.info(f"Для периода >90 дней данные будут агрегированными")

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    params = {
        'vs_currency': vs_currency,
        'days': days_param,
        'interval': interval
    }

    try:
        logging.info(f"Загрузка данных {coin_id.upper()}/{vs_currency.upper()}...")
        logging.info(f"Период: {start_date} - {end_date} ({days_diff} дней)")

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        timestamps = [pd.to_datetime(x[0], unit='ms') for x in data['prices']]
        prices = [x[1] for x in data['prices']]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })

        mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
        df = df.loc[mask].copy()

        if len(df) == 0:
            logging.warning("Нет данных для указанного периода")
            return None

        df = df.sort_values('timestamp').reset_index(drop=True)

        df['date'] = df['timestamp'].dt.date
        df['returns_pct'] = df['price'].pct_change() * 100

        logging.info(f"Успешно загружено {len(df)} записей")
        logging.info(f"Диапазон цен: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка запроса: {e}")
        raise
    except Exception as e:
        logging.error(f"Ошибка обработки данных: {e}")
        raise


def calculate_correlation_with_btc(coin_id, vs_currency='usd', days=30, timeframe='1d'):
    """Рассчитывает корреляцию выбранной криптовалюты с Bitcoin с обработкой ошибок"""
    try:
        if coin_id.lower() == 'bitcoin':
            logging.info("Пропускаем Bitcoin (корреляция с самим собой = 1)")
            return {
                'coin_id': 'bitcoin',
                'correlation': 1.0,
                'r_squared': 1.0,
                'p_value': 0.0,
                'beta': 1.0,
                'alpha': 0.0,
                'btc_std': 0.0,
                'coin_std': 0.0,
                'n_observations': 0,
                'significant': True,
                'correlation_strength': 'perfect',
                'correlation_direction': 'positive'
            }

        logging.info(f"Начинаю расчет корреляции для {coin_id}...")

        btc_data = coingecko_api.get_historical_price_data('bitcoin', vs_currency, days)
        if btc_data is None or btc_data.empty:
            logging.error(f"Не удалось получить данные для Bitcoin")
            return None

        coin_data = coingecko_api.get_historical_price_data(coin_id, vs_currency, days)
        if coin_data is None or coin_data.empty:
            logging.error(f"Не удалось получить данные для {coin_id}")
            return None

        common_dates = set(btc_data['timestamp']).intersection(set(coin_data['timestamp']))
        if len(common_dates) < 2:
            logging.error(f"Недостаточно общих дат для {coin_id}: {len(common_dates)}")
            return None

        btc_filtered = btc_data[btc_data['timestamp'].isin(common_dates)].sort_values('timestamp')
        coin_filtered = coin_data[coin_data['timestamp'].isin(common_dates)].sort_values('timestamp')

        if len(btc_filtered) != len(coin_filtered):
            logging.error(f"Разное количество точек: BTC={len(btc_filtered)}, {coin_id}={len(coin_filtered)}")
            return None

        if timeframe == '1d':
            btc_prices = btc_filtered['price'].values
            coin_prices = coin_filtered['price'].values
        elif timeframe == '1w':
            btc_filtered['week'] = btc_filtered['timestamp'].dt.isocalendar().week
            btc_filtered['year'] = btc_filtered['timestamp'].dt.isocalendar().year
            coin_filtered['week'] = coin_filtered['timestamp'].dt.isocalendar().week
            coin_filtered['year'] = coin_filtered['timestamp'].dt.isocalendar().year

            btc_weekly = btc_filtered.groupby(['year', 'week'])['price'].last()
            coin_weekly = coin_filtered.groupby(['year', 'week'])['price'].last()

            common_indices = set(btc_weekly.index).intersection(set(coin_weekly.index))
            if len(common_indices) < 2:
                logging.error(f"Недостаточно общих недель для {coin_id}")
                return None

            btc_prices = btc_weekly.loc[list(common_indices)].values
            coin_prices = coin_weekly.loc[list(common_indices)].values
        elif timeframe == '1M':
            btc_filtered['month'] = btc_filtered['timestamp'].dt.strftime('%Y-%m')
            coin_filtered['month'] = coin_filtered['timestamp'].dt.strftime('%Y-%m')

            btc_monthly = btc_filtered.groupby('month')['price'].last()
            coin_monthly = coin_filtered.groupby('month')['price'].last()

            common_months = set(btc_monthly.index).intersection(set(coin_monthly.index))
            if len(common_months) < 2:
                logging.error(f"Недостаточно общих месяцев для {coin_id}")
                return None

            btc_prices = btc_monthly.loc[list(common_months)].values
            coin_prices = coin_monthly.loc[list(common_months)].values
        else:
            logging.error(f"Неизвестный таймфрейм: {timeframe}")
            return None

        if len(btc_prices) < 5 or len(coin_prices) < 5:
            logging.error(f"Недостаточно данных после агрегации: {len(btc_prices)} точек")
            return None

        btc_returns = np.diff(btc_prices) / btc_prices[:-1]
        coin_returns = np.diff(coin_prices) / coin_prices[:-1]

        correlation, p_value = stats.pearsonr(btc_returns, coin_returns)

        if np.isnan(correlation) or np.isnan(p_value):
            logging.error(f"Результат корреляции содержит NaN для {coin_id}")
            return None

        r_squared = correlation ** 2

        covariance = np.cov(btc_returns, coin_returns)[0, 1]
        btc_variance = np.var(btc_returns)
        beta = covariance / btc_variance if btc_variance != 0 else 0

        coin_mean_return = np.mean(coin_returns)
        btc_mean_return = np.mean(btc_returns)
        alpha = coin_mean_return - beta * btc_mean_return

        btc_std = np.std(btc_returns)
        coin_std = np.std(coin_returns)

        results = {
            'coin_id': coin_id,
            'vs_currency': vs_currency,
            'timeframe': timeframe,
            'days': days,
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'beta': float(beta),
            'alpha': float(alpha),
            'btc_std': float(btc_std),
            'coin_std': float(coin_std),
            'n_observations': len(btc_returns),
            'significant': p_value < 0.05,
            'correlation_strength': get_correlation_strength(abs(correlation)),
            'correlation_direction': 'positive' if correlation > 0 else 'negative',
            'btc_mean_return': float(btc_mean_return),
            'coin_mean_return': float(coin_mean_return)
        }

        logging.info(f"Успешно рассчитана корреляция для {coin_id}: {correlation:.3f}")

        return results

    except Exception as e:
        logging.error(f"Ошибка расчета корреляции для {coin_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def calculate_multiple_correlations(coin_ids, vs_currency='usd', days=30, timeframe='1d'):
    """Рассчитывает корреляции для нескольких криптовалют"""
    results = {}

    for coin_id in coin_ids:
        if coin_id == 'bitcoin':
            continue

        logging.info(f"Рассчитываю корреляцию для {coin_id}...")
        correlation_result = calculate_correlation_with_btc(
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
            timeframe=timeframe
        )

        if correlation_result:
            results[coin_id] = correlation_result

    sorted_results = dict(sorted(
        results.items(),
        key=lambda x: abs(x[1]['correlation']),
        reverse=True
    ))

    return sorted_results


def generate_correlation_plot(results):
    """Генерирует график корреляций"""
    try:
        if not results:
            return None

        coin_names = []
        correlations = []
        colors = []
        strengths = []

        for coin_id, data in results.items():
            coin_names.append(coin_id.upper())
            correlations.append(data['correlation'])

            if data['correlation'] > 0:
                colors.append('rgba(46, 204, 113, 0.7)')
            else:
                colors.append('rgba(231, 76, 60, 0.7)')

            strengths.append(data['correlation_strength'])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=coin_names,
            y=correlations,
            marker_color=colors,
            text=[f"{corr:.3f}" for corr in correlations],
            textposition='outside',
            hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Корреляция: %{y:.3f}<br>" +
                    "Сила: %{customdata}<br>" +
                    "<extra></extra>"
            ),
            customdata=strengths,
            name='Корреляция с BTC'
        ))

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )

        fig.update_layout(
            title={
                'text': 'Корреляция криптовалют с Bitcoin',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Криптовалюта",
            yaxis_title="Коэффициент корреляции",
            yaxis=dict(
                range=[-1.1, 1.1],
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            plot_bgcolor='white',
            showlegend=False,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON

    except Exception as e:
        logging.error(f"Ошибка генерации графика корреляции: {e}")
        return None


def calculate_multiple_correlations_with_retry(coin_ids, vs_currency='usd', days=30, timeframe='1d', max_retries=3):
    """Рассчитывает корреляции с повторными попытками"""
    results = {}

    for coin_id in coin_ids:
        if coin_id == 'bitcoin':
            continue

        logging.info(f"Обрабатываю {coin_id}...")

        for attempt in range(max_retries):
            try:
                correlation_result = calculate_correlation_with_btc(
                    coin_id=coin_id,
                    vs_currency=vs_currency,
                    days=days,
                    timeframe=timeframe
                )

                if correlation_result:
                    results[coin_id] = correlation_result
                    logging.info(f"Успешно: {coin_id} (попытка {attempt + 1})")
                    break
                else:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logging.warning(f"Повторная попытка для {coin_id} через {wait_time} сек...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Не удалось рассчитать корреляцию для {coin_id} после {max_retries} попыток")

            except Exception as e:
                logging.error(f"Ошибка при попытке {attempt + 1} для {coin_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)

    if results:
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        ))
        return sorted_results

    return None


def calculate_correlation_with_btc_binance(coin_symbol, vs_currency='USDT', days=30):
    """Рассчитывает корреляцию криптовалюты с Bitcoin через Binance API"""
    try:
        btc_symbol = f"BTC{vs_currency}"
        coin_symbol_full = f"{coin_symbol}{vs_currency}"

        logging.info(f"Рассчитываю корреляцию для {coin_symbol_full} с {btc_symbol}")

        btc_returns = binance_api.get_daily_returns(btc_symbol, days)
        if btc_returns is None or len(btc_returns) < 5:
            logging.error(f"Не удалось получить данные для {btc_symbol}")
            return None

        coin_returns = binance_api.get_daily_returns(coin_symbol_full, days)
        if coin_returns is None or len(coin_returns) < 5:
            logging.error(f"Не удалось получить данные для {coin_symbol_full}")
            return None

        min_len = min(len(btc_returns), len(coin_returns))
        btc_returns_aligned = btc_returns[:min_len]
        coin_returns_aligned = coin_returns[:min_len]

        if min_len < 5:
            logging.error(f"Недостаточно данных после выравнивания: {min_len}")
            return None

        correlation, p_value = stats.pearsonr(btc_returns_aligned, coin_returns_aligned)

        r_squared = correlation ** 2

        covariance = np.cov(btc_returns_aligned, coin_returns_aligned)[0, 1]
        btc_variance = np.var(btc_returns_aligned)
        beta = covariance / btc_variance if btc_variance != 0 else 0

        coin_mean_return = np.mean(coin_returns_aligned)
        btc_mean_return = np.mean(btc_returns_aligned)
        alpha = coin_mean_return - beta * btc_mean_return

        btc_std = np.std(btc_returns_aligned)
        coin_std = np.std(coin_returns_aligned)

        correlation_strength = get_correlation_strength(abs(correlation))

        results = {
            'coin_symbol': coin_symbol,
            'vs_currency': vs_currency,
            'days': days,
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'beta': float(beta),
            'alpha': float(alpha),
            'btc_std': float(btc_std),
            'coin_std': float(coin_std),
            'n_observations': min_len,
            'significant': p_value < 0.05,
            'correlation_strength': correlation_strength,
            'correlation_direction': 'positive' if correlation > 0 else 'negative',
            'btc_mean_return': float(btc_mean_return),
            'coin_mean_return': float(coin_mean_return),
            'btc_symbol': btc_symbol,
            'coin_symbol_full': coin_symbol_full
        }

        logging.info(f"Успешно рассчитана корреляция: {correlation:.3f} для {coin_symbol}")

        return results

    except Exception as e:
        logging.error(f"Ошибка расчета корреляции через Binance: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def calculate_multiple_correlations_binance(coin_symbols, vs_currency='USDT', days=30):
    """Рассчитывает корреляции для нескольких криптовалют через Binance"""
    results = {}
    successful = 0
    failed = []

    available_pairs = binance_api.get_available_crypto_pairs(vs_currency)

    for coin_symbol in coin_symbols:
        coin_lower = coin_symbol.lower()
        if coin_lower not in available_pairs:
            logging.warning(f"Пара {coin_symbol}{vs_currency} не найдена на Binance")
            failed.append(coin_symbol)
            continue

        logging.info(f"Обрабатываю {coin_symbol}...")

        correlation_result = calculate_correlation_with_btc_binance(
            coin_symbol=coin_symbol.upper(),
            vs_currency=vs_currency,
            days=days
        )

        if correlation_result:
            results[coin_symbol] = correlation_result
            successful += 1
            logging.info(f"✓ Успешно: {coin_symbol}")
            time.sleep(0.1)
        else:
            failed.append(coin_symbol)
            logging.warning(f"✗ Не удалось: {coin_symbol}")

    if results:
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        ))
        return sorted_results, successful, failed

    return None, successful, failed
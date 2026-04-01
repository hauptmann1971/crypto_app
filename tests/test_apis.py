# tests/test_apis.py
"""Тесты модуля API"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


class TestCoinGeckoAPI:
    """Тесты CoinGeckoAPI"""

    def test_get_ohlc_success(self, mock_requests_get):
        """Успешное получение OHLC данных"""
        from modules.apis import CoinGeckoAPI

        # Настраиваем мок для OHLC данных
        mock_ohlc_data = [
            [1699999999000, 45000.0, 45500.0, 44500.0, 45200.0],
            [1700086399000, 45200.0, 46000.0, 45000.0, 45800.0],
        ]
        mock_requests_get.return_value.json.return_value = mock_ohlc_data

        api = CoinGeckoAPI()
        result = api.get_ohlc('bitcoin', 'usd', '7')

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'open' in result.columns
        assert 'close' in result.columns

    def test_get_ohlc_handles_error(self):
        """Обработка ошибки API"""
        from modules.apis import CoinGeckoAPI

        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 500

            api = CoinGeckoAPI()
            result = api.get_ohlc('bitcoin', 'usd', '7')

        assert result is None

    def test_get_ohlc_handles_timeout(self):
        """Обработка таймаута"""
        from modules.apis import CoinGeckoAPI
        import requests

        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Timeout")

            api = CoinGeckoAPI()
            result = api.get_ohlc('bitcoin', 'usd', '7')

        assert result is None


class TestBinanceAPI:
    """Тесты BinanceAPI"""

    def test_get_all_symbols(self):
        """Получение списка символов"""
        from modules.apis import BinanceAPI

        mock_response = {
            'symbols': [
                {'symbol': 'BTCUSDT', 'status': 'TRADING'},
                {'symbol': 'ETHUSDT', 'status': 'TRADING'},
                {'symbol': 'OLDCOIN', 'status': 'BREAK'},
            ]
        }

        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_response

            api = BinanceAPI()
            result = api.get_all_symbols()

        assert 'BTCUSDT' in result
        assert 'ETHUSDT' in result
        assert 'OLDCOIN' not in result  # Не торгуется

    def test_get_available_crypto_pairs(self):
        """Получение доступных пар"""
        from modules.apis import BinanceAPI

        with patch.object(BinanceAPI, 'get_all_symbols') as mock_symbols:
            mock_symbols.return_value = ['BTCUSDT', 'ETHUSDT', 'BTCEUR']

            api = BinanceAPI()
            result = api.get_available_crypto_pairs('USDT')

        assert 'btc' in result
        assert 'eth' in result
        assert result['btc'] == 'BTCUSDT'


class TestGetMainCryptoRatesToBtc:
    """Тесты функции get_main_crypto_rates_to_btc"""

    def test_returns_rates(self, mock_requests_get, mock_coingecko_response):
        """Возвращает курсы криптовалют"""
        from modules.apis import get_main_crypto_rates_to_btc

        result = get_main_crypto_rates_to_btc(
            timeout=10,
            coin_ids_to_fetch=['ethereum']
        )

        assert result is not None
        mock_requests_get.assert_called_once()

    def test_returns_empty_on_error(self):
        """Возвращает пустой словарь при ошибке"""
        from modules.apis import get_main_crypto_rates_to_btc
        import requests

        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Error")

            result = get_main_crypto_rates_to_btc(
                timeout=10,
                coin_ids_to_fetch=['bitcoin']
            )

        assert result == {}

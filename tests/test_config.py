# tests/test_config.py
"""Тесты модуля конфигурации"""
import os
import pytest
from unittest.mock import patch


class TestConfig:
    """Тесты класса Config"""

    def test_config_loads_env_variables(self):
        """Проверяет загрузку переменных окружения"""
        from modules.config import Config

        assert Config.SECRET_KEY is not None
        assert Config.HOST == '0.0.0.0'
        assert isinstance(Config.PORT, int)

    def test_config_debug_mode_parsing(self):
        """Проверяет парсинг DEBUG режима"""
        with patch.dict(os.environ, {'FLASK_DEBUG': 'true'}):
            # Перезагружаем модуль для применения изменений
            import importlib
            import modules.config
            importlib.reload(modules.config)

            assert modules.config.Config.DEBUG is True

    def test_config_popular_cryptos_not_empty(self):
        """Проверяет наличие списка популярных криптовалют"""
        from modules.config import Config

        assert len(Config.POPULAR_CRYPTOS) > 0
        assert 'bitcoin' in Config.POPULAR_CRYPTOS
        assert 'ethereum' in Config.POPULAR_CRYPTOS

    def test_config_currencies_list(self):
        """Проверяет список валют"""
        from modules.config import Config

        assert 'usd' in Config.CURRENCIES
        assert 'eur' in Config.CURRENCIES
        assert 'rub' in Config.CURRENCIES

    def test_config_periods_structure(self):
        """Проверяет структуру периодов"""
        from modules.config import Config

        assert len(Config.PERIODS) > 0
        for period in Config.PERIODS:
            assert 'value' in period
            assert 'label' in period

    def test_config_validate_with_valid_config(self):
        """Проверяет валидацию с корректной конфигурацией"""
        from modules.config import Config

        # Не должно вызывать исключение в DEBUG режиме
        Config.validate()

    def test_api_timeouts_are_integers(self):
        """Проверяет что таймауты — целые числа"""
        from modules.config import Config

        assert isinstance(Config.API_TIMEOUT_SHORT, int)
        assert isinstance(Config.API_TIMEOUT_LONG, int)
        assert Config.API_TIMEOUT_SHORT > 0
        assert Config.API_TIMEOUT_LONG > Config.API_TIMEOUT_SHORT

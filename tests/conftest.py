# tests/conftest.py
"""Общие фикстуры для тестов"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Устанавливаем тестовые переменные окружения ДО импорта модулей
os.environ['SECRET_KEY'] = 'test-secret-key-for-testing'
os.environ['BOT_TOKEN'] = 'test-bot-token'
os.environ['DB_USER'] = 'test_user'
os.environ['DB_PASSWORD'] = 'test_pass'
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_NAME'] = 'test_crypto_app'
os.environ['FLASK_DEBUG'] = 'true'


@pytest.fixture(scope='session')
def app():
    """Создаёт тестовое Flask приложение"""
    from modules.app_factory import create_app

    app = create_app()
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False

    yield app


@pytest.fixture
def client(app):
    """Тестовый клиент Flask"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """CLI runner для Flask"""
    return app.test_cli_runner()


@pytest.fixture
def mock_db_session():
    """Мок сессии базы данных"""
    with patch('modules.database.get_db') as mock:
        session = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=session)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield session


@pytest.fixture
def mock_db_connected():
    """Мок активного подключения к БД"""
    with patch('modules.database.get_db_connection_active', return_value=True):
        yield


@pytest.fixture
def sample_user_data():
    """Тестовые данные пользователя Telegram"""
    return {
        'id': 123456789,
        'first_name': 'Test',
        'last_name': 'User',
        'username': 'testuser',
        'photo_url': 'https://example.com/photo.jpg',
        'auth_date': 1699999999
    }


@pytest.fixture
def sample_crypto_request():
    """Тестовые данные запроса криптовалюты"""
    return {
        'crypto': 'bitcoin',
        'currency': 'usd',
        'user_id': 123456789
    }


@pytest.fixture
def mock_coingecko_response():
    """Мок ответа CoinGecko API"""
    return {
        'bitcoin': {'usd': 45000.0, 'btc': 1.0},
        'ethereum': {'usd': 2500.0, 'btc': 0.055}
    }


@pytest.fixture
def mock_requests_get(mock_coingecko_response):
    """Мок requests.get для API запросов"""
    with patch('requests.get') as mock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = mock_coingecko_response
        mock.return_value = response
        yield mock

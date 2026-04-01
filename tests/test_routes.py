# tests/test_routes.py
"""Тесты маршрутов (routes)"""
import pytest
from unittest.mock import patch, MagicMock


class TestMarketRoutes:
    """Тесты маршрутов market blueprint"""

    def test_index_page_loads(self, client, mock_db_connected):
        """Главная страница загружается"""
        with patch('modules.blueprints.market.get_db_connection_active', return_value=True):
            with patch('modules.blueprints.market.load_crypto_list', return_value=['bitcoin', 'ethereum']):
                with patch('modules.blueprints.market.get_pending_requests_count', return_value=0):
                    with patch('modules.blueprints.market.get_processing_requests_count', return_value=0):
                        with patch('modules.blueprints.market.process_pending_requests'):
                            response = client.get('/')

        assert response.status_code == 200

    def test_index_post_without_auth(self, client):
        """POST на главную без авторизации"""
        with patch('modules.blueprints.market.get_db_connection_active', return_value=True):
            with patch('modules.blueprints.market.create_crypto_request', return_value=1):
                with patch('modules.blueprints.market.load_crypto_list', return_value=['bitcoin']):
                    with patch('modules.blueprints.market.get_pending_requests_count', return_value=0):
                        with patch('modules.blueprints.market.get_processing_requests_count', return_value=0):
                            with patch('modules.blueprints.market.process_pending_requests'):
                                response = client.post('/', data={
                                    'crypto': 'bitcoin',
                                    'currency': 'usd'
                                })

        # Должен вернуть страницу (200) или редирект (302)
        assert response.status_code in [200, 302]


class TestAuthRoutes:
    """Тесты маршрутов auth blueprint"""

    def test_auth_page_loads(self, client):
        """Страница авторизации загружается"""
        response = client.get('/auth')

        assert response.status_code == 200

    def test_logout_redirects(self, client):
        """Logout перенаправляет на главную"""
        response = client.get('/logout')

        assert response.status_code == 302
        assert '/' in response.location

    def test_telegram_auth_without_data(self, client):
        """Telegram auth без данных возвращает ошибку"""
        response = client.post('/telegram-auth', json={})

        data = response.get_json()
        assert data['success'] is False


class TestRequestStatus:
    """Тесты статуса запроса"""

    def test_get_request_status_not_found(self, client, mock_db_session):
        """Запрос статуса несуществующего запроса"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        with patch('modules.blueprints.market.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            response = client.get('/request-status/99999')

        data = response.get_json()
        assert data['success'] is False

# tests/test_services.py
"""Тесты модуля сервисов"""
import pytest
from unittest.mock import MagicMock, patch


class TestGetUserFromDb:
    """Тесты функции get_user_from_db"""

    def test_returns_user_dict_when_found(self, mock_db_session):
        """Возвращает словарь пользователя когда найден"""
        from modules.services import get_user_from_db

        # Настраиваем мок
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.telegram_id = 123456789
        mock_user.first_name = 'Test'
        mock_user.last_name = 'User'
        mock_user.username = 'testuser'
        mock_user.photo_url = None
        mock_user.created_at = 1699999999
        mock_user.last_login = 1699999999
        mock_user.is_active = True

        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user

        with patch('modules.services.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            result = get_user_from_db(123456789)

        assert result is not None
        assert result['telegram_id'] == 123456789
        assert result['first_name'] == 'Test'

    def test_returns_none_when_not_found(self, mock_db_session):
        """Возвращает None когда пользователь не найден"""
        from modules.services import get_user_from_db

        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        with patch('modules.services.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            result = get_user_from_db(999999999)

        assert result is None


class TestCreateCryptoRequest:
    """Тесты функции create_crypto_request"""

    def test_returns_existing_request_id(self, mock_db_session):
        """Возвращает ID существующего запроса"""
        from modules.services import create_crypto_request

        existing_request = MagicMock()
        existing_request.id = 42
        mock_db_session.query.return_value.filter.return_value.first.return_value = existing_request

        with patch('modules.services.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            result = create_crypto_request(123, 'bitcoin', 'usd')

        assert result == 42

    def test_creates_new_request(self, mock_db_session):
        """Создаёт новый запрос когда существующий не найден"""
        from modules.services import create_crypto_request

        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        with patch('modules.services.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            with patch('modules.services.get_host_id', return_value='test-host'):
                result = create_crypto_request(123, 'bitcoin', 'usd')

        # Проверяем что add был вызван
        mock_db_session.add.assert_called_once()

    def test_returns_none_on_error(self, mock_db_session):
        """Возвращает None при ошибке"""
        from modules.services import create_crypto_request
        from sqlalchemy.exc import SQLAlchemyError

        mock_db_session.query.side_effect = SQLAlchemyError("Test error")

        with patch('modules.services.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            result = create_crypto_request(123, 'bitcoin', 'usd')

        assert result is None


class TestGetPendingRequestsCount:
    """Тесты функции get_pending_requests_count"""

    def test_returns_count(self, mock_db_session):
        """Возвращает количество pending запросов"""
        from modules.services import get_pending_requests_count

        mock_db_session.query.return_value.filter.return_value.count.return_value = 5

        with patch('modules.services.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            result = get_pending_requests_count()

        assert result == 5

    def test_returns_zero_on_error(self, mock_db_session):
        """Возвращает 0 при ошибке"""
        from modules.services import get_pending_requests_count
        from sqlalchemy.exc import SQLAlchemyError

        mock_db_session.query.side_effect = SQLAlchemyError("Test error")

        with patch('modules.services.get_db') as mock_get_db:
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db_session)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            result = get_pending_requests_count()

        assert result == 0

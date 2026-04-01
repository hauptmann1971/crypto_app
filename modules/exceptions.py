# modules/exceptions.py
"""Кастомные исключения для приложения crypto_app"""


class CryptoAppError(Exception):
    """Базовое исключение для приложения"""
    pass


class DatabaseError(CryptoAppError):
    """Ошибки работы с базой данных"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Ошибка подключения к базе данных"""
    pass


class RequestCreationError(CryptoAppError):
    """Ошибка при создании запроса"""
    pass


class APIError(CryptoAppError):
    """Ошибки при работе с внешними API"""
    pass


class RateLimitError(APIError):
    """Превышен лимит запросов к API"""
    pass


class AuthenticationError(CryptoAppError):
    """Ошибка аутентификации"""
    pass


class ConfigurationError(CryptoAppError):
    """Ошибка конфигурации"""
    pass

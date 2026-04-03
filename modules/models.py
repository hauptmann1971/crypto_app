# modules/models.py
"""
Модели SQLAlchemy с оптимизированными индексами для производительности
"""
from sqlalchemy import Column, Integer, String, Float, Text, BigInteger, Boolean, Index, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CryptoRate(Base):
    """Модель для хранения курсов криптовалют"""
    __tablename__ = 'crypto_rates'
    
    id = Column(Integer, primary_key=True)
    crypto = Column(String(50), nullable=False, index=True)
    currency = Column(String(10), nullable=False, index=True)
    rate = Column(Float(precision=8), nullable=False)
    source = Column(String(100), nullable=False)
    timestamp = Column(BigInteger, nullable=False, index=True)
    host = Column(String(100))
    
    # Составные индексы для ускорения частых запросов
    __table_args__ = (
        Index('ix_crypto_rates_crypto_currency', 'crypto', 'currency'),
        Index('ix_crypto_rates_timestamp_desc', timestamp.desc()),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'crypto': self.crypto,
            'currency': self.currency,
            'rate': self.rate,
            'source': self.source,
            'timestamp': self.timestamp
        }


class AppLog(Base):
    """Модель для логирования событий приложения"""
    __tablename__ = 'app_logs'
    
    id = Column(Integer, primary_key=True)
    service = Column(String(50), default='crypto_api', index=True)
    component = Column(String(50), default='backend')
    message = Column(Text, nullable=False)
    level = Column(String(20), nullable=False, index=True)
    traceback = Column(Text)
    user_id = Column(String(36), index=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    host = Column(String(100))
    
    # Составные индексы для фильтрации логов
    __table_args__ = (
        Index('ix_app_logs_level_timestamp', 'level', timestamp.desc()),
        Index('ix_app_logs_user_timestamp', 'user_id', timestamp.desc()),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'service': self.service,
            'component': self.component,
            'message': self.message,
            'level': self.level,
            'user_id': self.user_id,
            'timestamp': self.timestamp
        }


class TelegramUser(Base):
    """Модель пользователя Telegram"""
    __tablename__ = 'telegram_users'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100))
    username = Column(String(100), index=True)
    photo_url = Column(Text)
    auth_date = Column(BigInteger, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    last_login = Column(BigInteger, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    host = Column(String(100))
    
    # Индексы для поиска пользователей
    __table_args__ = (
        Index('ix_telegram_users_username', 'username', postgresql_using='gin'),
        Index('ix_telegram_users_active_lastlogin', 'is_active', last_login.desc()),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'telegram_id': self.telegram_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'username': self.username,
            'is_active': self.is_active,
            'last_login': self.last_login
        }


class CryptoRequest(Base):
    """Модель запроса курса криптовалюты"""
    __tablename__ = 'crypto_requests'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    crypto = Column(String(50), nullable=False, index=True)
    currency = Column(String(10), nullable=False, index=True)
    status = Column(String(20), default='pending', index=True)
    response_data = Column(Text)
    created_at = Column(BigInteger, nullable=False, index=True)
    finished_at = Column(BigInteger)
    host = Column(String(100))
    
    # Составные индексы для ускорения выборок по статусу и дате
    __table_args__ = (
        Index('ix_crypto_requests_status_created', 'status', created_at.desc()),
        Index('ix_crypto_requests_user_status', 'user_id', 'status'),
        Index('ix_crypto_requests_crypto_currency', 'crypto', 'currency'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'crypto': self.crypto,
            'currency': self.currency,
            'status': self.status,
            'created_at': self.created_at,
            'finished_at': self.finished_at
        }

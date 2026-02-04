# modules/models.py
from sqlalchemy import Column, Integer, String, Float, Text, BigInteger, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CryptoRate(Base):
    __tablename__ = 'crypto_rates'
    id = Column(Integer, primary_key=True)
    crypto = Column(String(50), nullable=False, index=True)
    currency = Column(String(10), nullable=False, index=True)
    rate = Column(Float(precision=8), nullable=False)
    source = Column(String(100), nullable=False)
    timestamp = Column(BigInteger, nullable=False, index=True)
    host = Column(String(100))

class AppLog(Base):
    __tablename__ = 'app_logs'
    id = Column(Integer, primary_key=True)
    service = Column(String(50), default='crypto_api')
    component = Column(String(50), default='backend')
    message = Column(Text, nullable=False)
    level = Column(String(20), nullable=False)
    traceback = Column(Text)
    user_id = Column(String(36))
    timestamp = Column(BigInteger, nullable=False, index=True)
    host = Column(String(100))

class TelegramUser(Base):
    __tablename__ = 'telegram_users'
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100))
    username = Column(String(100))
    photo_url = Column(Text)
    auth_date = Column(BigInteger, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    last_login = Column(BigInteger, nullable=False)
    is_active = Column(Boolean, default=True)
    host = Column(String(100))

class CryptoRequest(Base):
    __tablename__ = 'crypto_requests'
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    crypto = Column(String(50), nullable=False)
    currency = Column(String(10), nullable=False)
    status = Column(String(20), default='pending')
    response_data = Column(Text)
    created_at = Column(BigInteger, nullable=False, index=True)
    finished_at = Column(BigInteger)
    host = Column(String(100))
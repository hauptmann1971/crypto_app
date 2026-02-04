# modules/services.py
import json
import time
from typing import Optional, Dict, List, Any
from sqlalchemy.exc import SQLAlchemyError
from modules.database import get_db
from modules.models import TelegramUser, CryptoRequest
from modules.database import get_host_id
from modules.app_logging import log_message

import logging


def get_user_from_db(telegram_id: int) -> Optional[Dict[str, Any]]:
    """Получает пользователя из базы данных по telegram_id и возвращает словарь"""
    try:
        with get_db() as db:
            user = db.query(TelegramUser).filter(
                TelegramUser.telegram_id == telegram_id,
                TelegramUser.is_active == True
            ).first()

            if user:
                return {
                    'id': user.id,
                    'telegram_id': user.telegram_id,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'username': user.username,
                    'photo_url': user.photo_url,
                    'created_at': user.created_at,
                    'last_login': user.last_login,
                    'is_active': user.is_active
                }
            return None
    except SQLAlchemyError as e:
        logging.error(f"Error getting user from DB: {e}")
        return None


def update_user_last_login(telegram_id: int):
    """Обновляет время последнего входа пользователя"""
    try:
        with get_db() as db:
            user = db.query(TelegramUser).filter(
                TelegramUser.telegram_id == telegram_id
            ).first()
            if user:
                user.last_login = int(time.time())
                user.host = get_host_id()
                db.commit()
    except SQLAlchemyError as e:
        logging.error(f"Error updating user last login: {e}")


def create_crypto_request(user_id: int, crypto: str, currency: str) -> int:
    """Создает новый запрос в очереди и сразу помечает как processing"""
    try:
        with get_db() as db:
            existing_request = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id,
                CryptoRequest.crypto == crypto,
                CryptoRequest.currency == currency,
                CryptoRequest.status.in_(['pending', 'processing'])
            ).first()

            if existing_request:
                return existing_request.id

            request = CryptoRequest(
                user_id=user_id,
                crypto=crypto,
                currency=currency,
                status='processing',
                created_at=int(time.time()),
                host=get_host_id()
            )
            db.add(request)
            db.commit()
            request_id = request.id
            return request_id
    except SQLAlchemyError as e:
        logging.error(f"Error creating crypto request: {e}")
        return -1


def mark_request_as_error(request_id: int, error_message: str):
    """Помечает запрос как ошибочный"""
    try:
        with get_db() as db:
            request = db.query(CryptoRequest).filter(CryptoRequest.id == request_id).first()
            if request:
                request.status = 'error'
                request.response_data = json.dumps({'error': error_message})
                request.host = get_host_id()
                db.commit()
                logging.error(f"Marked request {request_id} as error: {error_message}")
    except Exception as e:
        logging.error(f"Failed to mark request {request_id} as error: {e}")


def get_pending_requests_count() -> int:
    """Получает количество pending запросов"""
    try:
        with get_db() as db:
            count = db.query(CryptoRequest).filter(
                CryptoRequest.status == 'pending'
            ).count()
            return count
    except SQLAlchemyError as e:
        logging.error(f"Error getting pending requests count: {e}")
        return 0


def get_processing_requests_count() -> int:
    """Получает количество processing запросов"""
    try:
        with get_db() as db:
            count = db.query(CryptoRequest).filter(
                CryptoRequest.status == 'processing'
            ).count()
            return count
    except SQLAlchemyError as e:
        logging.error(f"Error getting processing requests count: {e}")
        return 0


def get_latest_finished_request(user_id: int) -> Optional[dict]:
    """Получает последний завершенный запрос для пользователя"""
    try:
        with get_db() as db:
            request = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id,
                CryptoRequest.status == 'finished'
            ).order_by(CryptoRequest.finished_at.desc()).first()

            if request and request.response_data:
                try:
                    response = json.loads(request.response_data)
                    return {
                        'id': request.id,
                        'crypto': request.crypto,
                        'currency': request.currency,
                        'rate': response.get('rate'),
                        'status': request.status,
                        'created_at': request.created_at,
                        'finished_at': request.finished_at,
                        'response_data': response
                    }
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON for request {request.id}")
                    return None
            return None
    except SQLAlchemyError as e:
        logging.error(f"Error getting latest finished request: {e}")
        return None


def get_user_requests_history(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Получает историю запросов пользователя"""
    try:
        with get_db() as db:
            requests = db.query(CryptoRequest).filter(
                CryptoRequest.user_id == user_id
            ).order_by(CryptoRequest.created_at.desc()).limit(limit).all()

            result = []
            for req in requests:
                response_data = {}
                if req.response_data:
                    try:
                        response_data = json.loads(req.response_data)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error for request {req.id}: {e}")
                        continue

                result.append({
                    'id': req.id,
                    'crypto': req.crypto,
                    'currency': req.currency,
                    'status': req.status,
                    'rate': response_data.get('rate') if response_data else None,
                    'created_at': req.created_at,
                    'finished_at': req.finished_at,
                    'error': response_data.get('error') if response_data else None
                })
            return result
    except SQLAlchemyError as e:
        logging.error(f"Error getting user requests history: {e}")
        return []


def process_pending_requests():
    """Обрабатывает ТОЛЬКО СТАРЫЕ pending запросы"""
    try:
        from modules.database import db_connection_active
        if not db_connection_active:
            return

        with get_db() as db:
            one_minute_ago = int(time.time()) - 60
            old_pending_requests = db.query(CryptoRequest).filter(
                CryptoRequest.status == 'pending',
                CryptoRequest.created_at < one_minute_ago
            ).order_by(CryptoRequest.created_at.asc()).limit(2).all()

            processed_count = 0
            for request in old_pending_requests:
                request.status = 'processing'
                processed_count += 1

            if processed_count > 0:
                db.commit()
                logging.info(f"Marked {processed_count} OLD pending requests as processing for worker")

    except SQLAlchemyError as e:
        logging.error(f"Error processing OLD pending requests: {e}")
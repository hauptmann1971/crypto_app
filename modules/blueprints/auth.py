import time

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for

from modules.app_logging import log_message
from modules.config import Config
from modules.database import get_db, get_host_id
from modules.models import TelegramUser
from modules.utils import verify_telegram_authentication

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/telegram-auth', methods=['POST'])
def telegram_auth():
    """Обрабатывает callback от Telegram Login Widget"""
    try:
        user_data = request.get_json()

        if not user_data:
            return jsonify({'success': False, 'error': 'No data received'})

        if not verify_telegram_authentication(user_data, Config.BOT_TOKEN):
            return jsonify({'success': False, 'error': 'Invalid authentication data'})

        try:
            with get_db() as db:
                existing_user = db.query(TelegramUser).filter(
                    TelegramUser.telegram_id == user_data['id']
                ).first()

                current_time = int(time.time())

                if existing_user:
                    existing_user.first_name = user_data['first_name']
                    existing_user.last_name = user_data.get('last_name')
                    existing_user.username = user_data.get('username')
                    existing_user.photo_url = user_data.get('photo_url')
                    existing_user.auth_date = user_data['auth_date']
                    existing_user.last_login = current_time
                    existing_user.is_active = True
                    existing_user.host = get_host_id()
                    db.commit()
                    user_id = existing_user.id
                else:
                    new_user = TelegramUser(
                        telegram_id=user_data['id'],
                        first_name=user_data['first_name'],
                        last_name=user_data.get('last_name'),
                        username=user_data.get('username'),
                        photo_url=user_data.get('photo_url'),
                        auth_date=user_data['auth_date'],
                        created_at=current_time,
                        last_login=current_time,
                        is_active=True,
                        host=get_host_id()
                    )
                    db.add(new_user)
                    db.commit()
                    user_id = new_user.id

        except Exception as exc:
            log_message(f"Database error during user save: {exc}", 'error')
            return jsonify({'success': False, 'error': 'Database error'})

        session['user'] = {
            'id': user_data['id'],
            'first_name': user_data['first_name'],
            'username': user_data.get('username'),
            'photo_url': user_data.get('photo_url'),
            'auth_date': user_data['auth_date'],
            'db_user_id': user_id
        }

        log_message(
            f"User {user_data['id']} successfully authenticated via Telegram (DB ID: {user_id})",
            'info',
            user_id=str(user_data['id'])
        )
        return jsonify({'success': True})

    except Exception as exc:
        log_message(f"Telegram auth error: {exc}", 'error')
        return jsonify({'success': False, 'error': str(exc)})


@auth_bp.route('/auth', methods=['GET'])
def auth():
    """Страница авторизации через Telegram"""
    user = session.get('user')
    if user:
        flash("Вы уже авторизованы!", 'info')
        return redirect(url_for('market.index'))
    return render_template('auth.html', user=user, bot_username=Config.BOT_USERNAME)


@auth_bp.route('/logout')
def logout():
    """Выход пользователя"""
    user_id = session.get('user', {}).get('id')
    session.pop('user', None)
    flash('Вы успешно вышли из системы', 'success')
    log_message("User logged out", 'info', user_id=str(user_id) if user_id else None)
    return redirect(url_for('market.index'))


@auth_bp.route('/verify_code', methods=['POST'])
def verify_code():
    """Проверка кода подтверждения из Telegram"""
    verification_code = request.form.get('verification_code')

    if not verification_code or len(verification_code) != 6:
        flash("Введите 6-значный код подтверждения", 'error')
        return redirect(url_for('auth.auth'))

    flash("Функция проверки кода временно недоступна. Используйте Telegram Widget авторизацию.", 'warning')
    return redirect(url_for('auth.auth'))

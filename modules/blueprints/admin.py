import os
import platform
import time
from datetime import datetime, timedelta, timezone

import psutil
from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for

from modules.app_logging import log_message
from modules.blueprints.helpers import ensure_db_connected, safe_json_loads
from modules.database import get_db, get_db_connection_active, get_host_id
from modules.models import CryptoRequest, TelegramUser

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/log_table')
def show_log_table():
    """Отображает таблицу логов с фильтром по уровню."""
    db_guard = ensure_db_connected()
    if db_guard is not None:
        return db_guard

    user = session.get('user')
    level_filter = request.args.get('level', '').lower()
    logs_data = []

    try:
        with get_db() as db:
            from sqlalchemy import text
            if level_filter and level_filter != 'all':
                result = db.execute(
                    text(
                        """
                        SELECT
                            timestamp, level, message, service, component,
                            user_id, traceback, host
                        FROM app_logs
                        WHERE level = :level
                        ORDER BY timestamp DESC
                        LIMIT 200
                        """
                    ),
                    {'level': level_filter}
                )
            else:
                result = db.execute(text("""
                    SELECT
                        timestamp, level, message, service, component,
                        user_id, traceback, host
                    FROM app_logs
                    ORDER BY timestamp DESC
                    LIMIT 200
                """))

            rows = result.fetchall()
            moscow_tz = timezone(timedelta(hours=3))

            for row in rows:
                if row.timestamp:
                    try:
                        utc_time = datetime.fromtimestamp(row.timestamp, tz=timezone.utc)
                        moscow_time = utc_time.astimezone(moscow_tz)
                        date_time_str = moscow_time.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        date_time_str = str(row.timestamp)
                else:
                    date_time_str = 'N/A'

                log_entry = {
                    'date_time': date_time_str,
                    'level': row.level or '',
                    'message': (row.message[:200] + '...') if row.message and len(row.message) > 200 else (
                        row.message or ''),
                    'service': row.service or '',
                    'component': row.component or '',
                    'user_id': row.user_id or '',
                    'host': row.host or 'N/A',
                    'traceback': ''
                }

                if row.level and row.level.lower() == 'error' and row.traceback:
                    log_entry['traceback'] = (row.traceback[:500] + '...') if len(row.traceback) > 500 else row.traceback

                logs_data.append(log_entry)

        if level_filter and level_filter != 'all' and not logs_data:
            flash(f"Логи с уровнем '{level_filter.upper()}' не найдены", 'warning')
    except Exception as exc:
        flash(f"Ошибка при загрузке логов: {str(exc)}", 'error')
        print(f"Ошибка в show_log_table: {exc}")

    available_levels = ['all', 'debug', 'info', 'warning', 'error', 'critical']
    return render_template('data_table_logs.html',
                           title='Логи приложения',
                           data=logs_data,
                           columns=['date_time', 'level', 'message', 'service', 'component', 'user_id', 'host', 'traceback'],
                           user=user,
                           current_level=level_filter,
                           available_levels=available_levels)


@admin_bp.route('/users_table')
def show_users_table():
    """Отображает таблицу пользователей Telegram."""
    db_guard = ensure_db_connected()
    if db_guard is not None:
        return db_guard

    user = session.get('user')
    try:
        with get_db() as db:
            users = db.query(TelegramUser).order_by(TelegramUser.last_login.desc()).limit(100).all()
            users_data = []
            for item in users:
                users_data.append({
                    'id': item.id,
                    'telegram_id': item.telegram_id,
                    'first_name': item.first_name,
                    'last_name': item.last_name or '',
                    'username': item.username or '',
                    'created_at': datetime.fromtimestamp(item.created_at).strftime('%Y-%m-%d %H:%M:%S') if item.created_at else 'N/A',
                    'last_login': datetime.fromtimestamp(item.last_login).strftime('%Y-%m-%d %H:%M:%S') if item.last_login else 'N/A',
                    'is_active': 'Да' if item.is_active else 'Нет',
                    'host': item.host or 'N/A'
                })

        return render_template('data_table.html',
                               title='Пользователи Telegram',
                               data=users_data,
                               columns=['id', 'telegram_id', 'first_name', 'last_name', 'username', 'created_at',
                                        'last_login', 'is_active', 'host'],
                               user=user)
    except Exception as exc:
        flash(f"Ошибка БД: {str(exc)}", 'error')
        log_message(f"Ошибка при получении пользователей: {exc}", 'error')
        return redirect(url_for('market.index'))


@admin_bp.route('/requests_table')
def show_requests_table():
    """Отображает таблицу запросов в очереди."""
    db_guard = ensure_db_connected()
    if db_guard is not None:
        return db_guard

    user = session.get('user')
    try:
        with get_db() as db:
            requests = db.query(CryptoRequest).order_by(CryptoRequest.created_at.desc()).limit(100).all()
            requests_data = []
            for req in requests:
                response_data = safe_json_loads(req.response_data)

                requests_data.append({
                    'id': req.id,
                    'user_id': req.user_id,
                    'crypto': req.crypto,
                    'currency': req.currency,
                    'status': req.status,
                    'rate': response_data.get('rate', 'N/A'),
                    'created_at': datetime.fromtimestamp(req.created_at).strftime('%Y-%m-%d %H:%M:%S') if req.created_at else 'N/A',
                    'finished_at': datetime.fromtimestamp(req.finished_at).strftime('%Y-%m-%d %H:%M:%S') if req.finished_at else 'N/A',
                    'error': response_data.get('error', ''),
                    'host': req.host or 'N/A'
                })

        return render_template('data_table.html',
                               title='Очередь запросов',
                               data=requests_data,
                               columns=['id', 'user_id', 'crypto', 'currency', 'status', 'rate', 'created_at',
                                        'finished_at', 'error', 'host'],
                               user=user)
    except Exception as exc:
        flash(f"Ошибка БД: {str(exc)}", 'error')
        log_message(f"Ошибка при получении запросов: {exc}", 'error')
        return redirect(url_for('market.index'))


@admin_bp.route('/api/status')
def get_worker_status():
    """Возвращает сводный статус сервисов и ресурсов."""
    try:
        system_info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'host_id': get_host_id()
        }

        ram_info = {
            'total': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'available': round(psutil.virtual_memory().available / (1024 ** 3), 2),
            'percent': psutil.virtual_memory().percent,
            'used': round(psutil.virtual_memory().used / (1024 ** 3), 2)
        }
        cpu_info = {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }

        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
            try:
                if 'nginx' in proc.info['name'].lower() or 'gunicorn' in proc.info['name'].lower():
                    processes.append({
                        'name': proc.info['name'],
                        'pid': proc.info['pid'],
                        'status': proc.info['status'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        nginx_processes = [p for p in processes if 'nginx' in p['name'].lower()]
        nginx_status = {
            'count': len(nginx_processes),
            'on': len([p for p in nginx_processes if p['status'] == 'running']),
            'off': len([p for p in nginx_processes if p['status'] != 'running']),
            'processes': nginx_processes[:5]
        }
        gunicorn_processes = [p for p in processes if 'gunicorn' in p['name'].lower()]
        gunicorn_status = {
            'count': len(gunicorn_processes),
            'on': len([p for p in gunicorn_processes if p['status'] == 'running']),
            'off': len([p for p in gunicorn_processes if p['status'] != 'running']),
            'processes': gunicorn_processes[:5]
        }
        flask_status = {
            'status': 'running',
            'uptime': time.time() - psutil.Process(os.getpid()).create_time(),
            'requests_processed': 0
        }
        db_status = {
            'connected': get_db_connection_active(),
            'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system': system_info,
            'resources': {
                'ram': ram_info,
                'cpu': cpu_info,
                'disk': {
                    'total': round(psutil.disk_usage('/').total / (1024 ** 3), 2),
                    'used': round(psutil.disk_usage('/').used / (1024 ** 3), 2),
                    'free': round(psutil.disk_usage('/').free / (1024 ** 3), 2),
                    'percent': psutil.disk_usage('/').percent
                }
            },
            'services': {
                'nginx': nginx_status,
                'gunicorn': gunicorn_status,
                'flask': flask_status,
                'database': db_status
            },
            'processes': {
                'total': len(processes),
                'list': processes[:10]
            }
        }
        return jsonify(response)
    except Exception as exc:
        return jsonify({
            'error': str(exc),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500


@admin_bp.route('/status')
def show_status():
    """Страница визуального статуса системы."""
    user = session.get('user')
    return render_template('status.html',
                           title='Статус системы',
                           user=user)

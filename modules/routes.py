# modules/routes.py
from flask import render_template, request, flash, redirect, url_for, session, jsonify
import json
import time
from datetime import datetime, timedelta, timezone
import os
import psutil
import platform
import logging

from modules.config import Config
from modules.utils import verify_telegram_authentication, load_crypto_list
from modules.services import (
    get_user_from_db, update_user_last_login, create_crypto_request,
    get_pending_requests_count, get_processing_requests_count,
    get_user_requests_history, process_pending_requests
)
from modules.database import get_db, get_db_connection_active, get_host_id
from modules.models import CryptoRate, AppLog, TelegramUser, CryptoRequest
from modules.apis import (
    coingecko_api, get_main_crypto_rates_to_btc,
    get_historical_price_range, generate_historical_plot,
    calculate_multiple_correlations_with_retry, generate_correlation_plot,
    calculate_multiple_correlations_binance
)
from modules.app_logging import log_message  # ✅ Исправленный импорт


def register_routes(app):
    """Регистрирует все маршруты в приложении Flask"""

    @app.route('/telegram-auth', methods=['POST'])
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
                        action = "updated"
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
                        action = "created"

            except Exception as e:
                log_message(f"Database error during user save: {e}", 'error')
                return jsonify({'success': False, 'error': 'Database error'})

            session['user'] = {
                'id': user_data['id'],
                'first_name': user_data['first_name'],
                'username': user_data.get('username'),
                'photo_url': user_data.get('photo_url'),
                'auth_date': user_data['auth_date'],
                'db_user_id': user_id
            }

            log_message(f"User {user_data['id']} successfully authenticated via Telegram (DB ID: {user_id})",
                        'info', user_id=str(user_data['id']))

            return jsonify({'success': True})

        except Exception as e:
            log_message(f"Telegram auth error: {e}", 'error')
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/auth', methods=['GET'])
    def auth():
        """Страница авторизации через Telegram"""
        user = session.get('user')

        if user:
            flash("Вы уже авторизованы!", 'info')
            return redirect(url_for('index'))

        return render_template('auth.html', user=user, bot_username=Config.BOT_USERNAME)

    @app.route('/logout')
    def logout():
        """Выход пользователя"""
        user_id = session.get('user', {}).get('id')
        session.pop('user', None)
        flash('Вы успешно вышли из системы', 'success')
        log_message("User logged out", 'info', user_id=str(user_id) if user_id else None)
        return redirect(url_for('index'))

    @app.route('/disconnect_db', methods=['POST'])
    def disconnect_db():
        """Отключение подключения к базе данных"""
        from modules.database import db_manager

        db_manager.db_connection_active = False
        if db_manager.engine and hasattr(db_manager.engine, 'dispose'):
            db_manager.engine.dispose()
            db_manager.engine = None
            db_manager.SessionLocal = None

        # Вместо flash используем сообщение в сессии
        session['db_message'] = "Соединение с базой данных отключено"
        session['db_message_type'] = 'warning'

        log_message("Ручное отключение базы данных", 'info')
        return redirect(url_for('index'))


    @app.route('/connect_db', methods=['POST'])
    def connect_db():
        """Включение подключения к базе данных"""
        from modules.database import db_manager

        try:
            success = db_manager.init_db_connection()
            if success:
                session['db_message'] = "Соединение с базой данных восстановлено"
                session['db_message_type'] = 'success'
                log_message("Ручное подключение базы данных", 'info')
            else:
                session['db_message'] = f"Ошибка подключения: {db_manager.last_connection_error}"
                session['db_message_type'] = 'error'
                log_message(f"Ошибка подключения к БД: {db_manager.last_connection_error}", 'error')
        except Exception as e:
            session['db_message'] = f"Ошибка подключения: {str(e)}"
            session['db_message_type'] = 'error'
            log_message(f"Ошибка подключения к БД: {e}", 'error')

        return redirect(url_for('index'))


    @app.route('/request-status/<int:request_id>')
    def get_request_status(request_id):
        """Проверяет статус запроса"""
        try:
            with get_db() as db:
                request = db.query(CryptoRequest).filter(
                    CryptoRequest.id == request_id
                ).first()

                if not request:
                    return jsonify({'success': False, 'error': 'Request not found'})

                response_data = {}
                if request.response_data:
                    try:
                        response_data = json.loads(request.response_data)
                    except json.JSONDecodeError:
                        pass

                return jsonify({
                    'success': True,
                    'status': request.status,
                    'rate': response_data.get('rate'),
                    'error': response_data.get('error'),
                    'finished_at': request.finished_at,
                    'crypto': request.crypto,
                    'currency': request.currency
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/', methods=['GET', 'POST'])
    def index():
        current_crypto = None
        current_currency = None
        current_request_id = None
        requests_history = []
        pending_requests_count = 0
        processing_requests_count = 0
        user_session = session.get('user')
        user_from_db = None
        user_id = user_session['id'] if user_session else 0

        # Проверяем сообщения о состоянии БД
        if 'db_message' in session:
            flash(session.pop('db_message'), session.pop('db_message_type', 'info'))

        if user_session:
            user_from_db = get_user_from_db(user_session['id'])
            if user_from_db:
                update_user_last_login(user_session['id'])
            requests_history = get_user_requests_history(user_session['id'])
            pending_requests_count = get_pending_requests_count()
            processing_requests_count = get_processing_requests_count()

        if request.method == 'POST':
            crypto = request.form.get('crypto', 'bitcoin')
            currency = request.form.get('currency', 'usd')
            if not crypto or not currency:
                flash("Выберите криптовалюту и валюту", 'error')
                log_message("Не выбрана криптовалюта или валюта", 'warning',
                            user_id=str(user_session['id']) if user_session else None)
                return redirect('/')

            current_crypto = crypto
            current_currency = currency
            request_id = create_crypto_request(user_id, crypto, currency)
            if request_id > 0:
                session['last_request_id'] = request_id
                session['current_crypto'] = crypto
                session['current_currency'] = currency
                session.modified = True
                flash(
                    f"Запрос на получение курса {crypto.upper()}/{currency.upper()} отправлен в обработку. ID: {request_id}",
                    'info')
                log_message(f"Request created for {crypto}/{currency} (ID: {request_id})", 'info',
                            user_id=str(user_session['id']) if user_session else None)
                current_request_id = request_id
                pending_requests_count = get_pending_requests_count()
                processing_requests_count = get_processing_requests_count()
                if user_session:
                    requests_history = get_user_requests_history(user_session['id'])
            else:
                flash("Ошибка при создании запроса", 'error')
                log_message("Error creating request in queue", 'error',
                            user_id=str(user_session['id']) if user_session else None)
        else:
            current_crypto = session.get('current_crypto', 'bitcoin')
            current_currency = session.get('current_currency', 'usd')
            current_request_id = session.get('last_request_id')

        process_pending_requests()

        cryptos = load_crypto_list()
        return render_template('index.html',
                               cryptos=cryptos,
                               currencies=Config.CURRENCIES,
                               periods=Config.PERIODS,
                               current_crypto=current_crypto,
                               current_currency=current_currency,
                               current_request_id=current_request_id,
                               db_connected=get_db_connection_active(),
                               user=user_from_db,
                               requests_history=requests_history,
                               pending_requests_count=pending_requests_count,
                               processing_requests_count=processing_requests_count,
                               bot_username=Config.BOT_USERNAME)

    @app.route('/crypto_table')
    def show_crypto_table():
        """Отображает таблицу курсов криптовалют"""
        if not get_db_connection_active():
            flash("Соединение с базой данных отключено", 'error')
            return redirect(url_for('index'))

        try:
            with get_db() as db:
                rates = db.query(
                    CryptoRate.crypto,
                    CryptoRate.currency,
                    CryptoRate.rate,
                    CryptoRate.source,
                    CryptoRate.timestamp
                ).order_by(CryptoRate.timestamp.desc()).limit(100).all()

            rates_data = [{
                'crypto': r.crypto,
                'currency': r.currency,
                'rate': r.rate,
                'date_time': datetime.fromtimestamp(r.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'source': r.source
            } for r in rates]

            return render_template('data_table.html',
                                   title='Курсы криптовалют',
                                   data=rates_data,
                                   columns=['crypto', 'currency', 'rate', 'date_time', 'source'])
        except Exception as e:
            flash(f"Ошибка БД: {str(e)}", 'error')
            return redirect(url_for('index'))

    @app.route('/chart', methods=['GET', 'POST'])
    def chart():
        """Страница с графиком курса криптовалюты"""
        plot_url = None
        error = None
        user = session.get('user')

        if request.method == 'POST':
            crypto = request.form.get('crypto')
            currency = request.form.get('currency')
            period = request.form.get('period', '7')

            if not crypto or not currency:
                flash("Выберите криптовалюту и валюту", 'error')
                return redirect(url_for('chart'))

            try:
                data = coingecko_api.get_ohlc(crypto, currency, period)

                if data is not None:
                    plot_url = coingecko_api.generate_plot(data, crypto, currency, period)
                    if plot_url:
                        log_message(f"Сгенерирован график для {crypto}/{currency} за {period} дней", 'info',
                                    user_id=str(user['id']) if user else None)
                    else:
                        error = "Ошибка генерации графика"
                else:
                    error = "Не удалось получить данные для построения графика"

            except Exception as e:
                error = f"Ошибка: {str(e)}"
                log_message(f"Ошибка построения графика: {e}", 'error', user_id=str(user['id']) if user else None)

        cryptos = load_crypto_list()
        return render_template('chart.html',
                               cryptos=cryptos,
                               currencies=Config.CURRENCIES,
                               periods=Config.PERIODS,
                               plot_url=plot_url,
                               error=error,
                               user=user)

    @app.route('/candlestick', methods=['GET', 'POST'])
    def candlestick_chart():
        """Страница со свечным графиком"""
        plot_url = None
        error = None
        user = session.get('user')

        if request.method == 'POST':
            crypto = request.form.get('crypto')
            currency = request.form.get('currency')
            period = request.form.get('period', '7')

            if not crypto or not currency:
                flash("Выберите криптовалюту и валюту", 'error')
                return redirect(url_for('candlestick_chart'))

            try:
                data = coingecko_api.get_ohlc(crypto, currency, period)

                if data is not None:
                    plot_url = coingecko_api.generate_plot(data, crypto, currency, period)
                    if plot_url:
                        log_message(f"Сгенерирован свечной график для {crypto}/{currency} за {period} дней", 'info',
                                    user_id=str(user['id']) if user else None)
                    else:
                        error = "Ошибка генерации графика"
                else:
                    error = "Не удалось получить данные для свечного графика"

            except Exception as e:
                error = f"Ошибка: {str(e)}"
                log_message(f"Ошибка построения свечного графика: {e}", 'error',
                            user_id=str(user['id']) if user else None)

        cryptos = load_crypto_list()
        return render_template('candlestick.html',
                               cryptos=cryptos,
                               currencies=Config.CURRENCIES,
                               periods=Config.PERIODS,
                               plot_url=plot_url,
                               error=error,
                               user=user)

    @app.route('/log_table')
    def show_log_table():
        """Отображает таблицу логов с фильтрацией по уровню"""
        if not get_db_connection_active():
            flash("Соединение с базой данных отключено", 'error')
            return redirect(url_for('index'))

        user = session.get('user')
        level_filter = request.args.get('level', '').lower()
        logs_data = []

        try:
            with get_db() as db:
                from sqlalchemy import text
                if level_filter and level_filter != 'all':
                    result = db.execute(
                        text("""
                            SELECT 
                                timestamp, level, message, service, component, 
                                user_id, traceback, host
                            FROM app_logs 
                            WHERE level = :level
                            ORDER BY timestamp DESC 
                            LIMIT 200
                        """),
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
                        log_entry['traceback'] = (row.traceback[:500] + '...') if len(
                            row.traceback) > 500 else row.traceback

                    logs_data.append(log_entry)

            if level_filter and level_filter != 'all' and not logs_data:
                flash(f"Логи с уровнем '{level_filter.upper()}' не найдены", 'warning')

        except Exception as e:
            flash(f"Ошибка при загрузке логов: {str(e)}", 'error')
            print(f"Ошибка в show_log_table: {e}")

        available_levels = ['all', 'debug', 'info', 'warning', 'error', 'critical']

        return render_template('data_table_logs.html',
                               title='Логи приложения',
                               data=logs_data,
                               columns=['date_time', 'level', 'message', 'service', 'component',
                                        'user_id', 'host', 'traceback'],
                               user=user,
                               current_level=level_filter,
                               available_levels=available_levels)

    @app.route('/users_table')
    def show_users_table():
        """Отображает таблицу пользователей"""
        if not get_db_connection_active():
            flash("Соединение с базой данных отключено", 'error')
            return redirect(url_for('index'))

        user = session.get('user')

        try:
            with get_db() as db:
                users = db.query(TelegramUser).order_by(TelegramUser.last_login.desc()).limit(100).all()

                users_data = []
                for u in users:
                    users_data.append({
                        'id': u.id,
                        'telegram_id': u.telegram_id,
                        'first_name': u.first_name,
                        'last_name': u.last_name or '',
                        'username': u.username or '',
                        'created_at': datetime.fromtimestamp(u.created_at).strftime(
                            '%Y-%m-%d %H:%M:%S') if u.created_at else 'N/A',
                        'last_login': datetime.fromtimestamp(u.last_login).strftime(
                            '%Y-%m-%d %H:%M:%S') if u.last_login else 'N/A',
                        'is_active': 'Да' if u.is_active else 'Нет',
                        'host': u.host or 'N/A'
                    })

            return render_template('data_table.html',
                                   title='Пользователи Telegram',
                                   data=users_data,
                                   columns=['id', 'telegram_id', 'first_name', 'last_name', 'username', 'created_at',
                                            'last_login', 'is_active', 'host'],
                                   user=user)
        except Exception as e:
            flash(f"Ошибка БД: {str(e)}", 'error')
            log_message(f"Ошибка при получении пользователей: {e}", 'error')
            return redirect(url_for('index'))

    @app.route('/requests_table')
    def show_requests_table():
        """Отображает таблицу запросов"""
        if not get_db_connection_active():
            flash("Соединение с базой данных отключено", 'error')
            return redirect(url_for('index'))

        user = session.get('user')

        try:
            with get_db() as db:
                requests = db.query(CryptoRequest).order_by(CryptoRequest.created_at.desc()).limit(100).all()

                requests_data = []
                for req in requests:
                    response_data = {}
                    if req.response_data:
                        try:
                            response_data = json.loads(req.response_data)
                        except json.JSONDecodeError:
                            pass

                    requests_data.append({
                        'id': req.id,
                        'user_id': req.user_id,
                        'crypto': req.crypto,
                        'currency': req.currency,
                        'status': req.status,
                        'rate': response_data.get('rate', 'N/A'),
                        'created_at': datetime.fromtimestamp(req.created_at).strftime(
                            '%Y-%m-%d %H:%M:%S') if req.created_at else 'N/A',
                        'finished_at': datetime.fromtimestamp(req.finished_at).strftime(
                            '%Y-%m-%d %H:%M:%S') if req.finished_at else 'N/A',
                        'error': response_data.get('error', ''),
                        'host': req.host or 'N/A'
                    })

            return render_template('data_table.html',
                                   title='Очередь запросов',
                                   data=requests_data,
                                   columns=['id', 'user_id', 'crypto', 'currency', 'status', 'rate', 'created_at',
                                            'finished_at', 'error', 'host'],
                                   user=user)
        except Exception as e:
            flash(f"Ошибка БД: {str(e)}", 'error')
            log_message(f"Ошибка при получении запросов: {e}", 'error')
            return redirect(url_for('index'))

    @app.route('/verify_code', methods=['POST'])
    def verify_code():
        """Проверка кода подтверждения из Telegram"""
        verification_code = request.form.get('verification_code')

        if not verification_code or len(verification_code) != 6:
            flash("Введите 6-значный код подтверждения", 'error')
            return redirect(url_for('auth'))

        flash("Функция проверки кода временно недоступна. Используйте Telegram Widget авторизацию.", 'warning')
        return redirect(url_for('auth'))

    @app.route('/main_crypto_rates_to_btc')
    def show_main_crypto_rates_to_btc():
        """Отображает таблицу с курсами основных криптовалют (POPULAR_CRYPTOS) к биткоину"""
        if not get_db_connection_active():
            flash("Соединение с базой данных отключено", 'error')
            return redirect(url_for('index'))

        user = session.get('user')

        rates_data = get_main_crypto_rates_to_btc(
            coin_ids_to_fetch=[cid for cid in Config.POPULAR_CRYPTOS if cid != 'bitcoin'])

        if not rates_data:
            flash("Не удалось получить курсы криптовалют к BTC. Проверьте логи.", 'error')
            log_message("Не удалось получить курсы основных криптовалют к BTC", 'error',
                        user_id=str(user['id']) if user else None)
            return redirect(url_for('index'))

        sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
        table_data = []
        for crypto_id, rate_dict in sorted_rates:
            rate = rate_dict.get('btc', 'N/A')
            formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
            table_data.append({
                'crypto': crypto_id,
                'rate_to_btc': formatted_rate
            })

        all_crypto_ids = load_crypto_list()
        viewed_ids = set(rates_data.keys())
        remaining_ids = [cid for cid in all_crypto_ids if cid != 'bitcoin' and cid not in viewed_ids]
        remaining_count = len(remaining_ids)
        viewed_count = len(viewed_ids)

        log_message("Таблица курсов основных криптовалют к BTC отображена", 'info',
                    user_id=str(user['id']) if user else None)
        return render_template('main_crypto_rates_table.html',
                               title='Курсы основных криптовалют к BTC',
                               data=table_data,
                               columns=['crypto', 'rate_to_btc'],
                               user=user,
                               viewed_count=viewed_count,
                               remaining_count=remaining_count,
                               has_next=True)

    @app.route('/main_crypto_rates_to_btc/next')
    def show_next_crypto_rates_to_btc():
        """Отображает таблицу с курсами следующих 50 криптовалют к биткоину в алфавитном порядке"""
        if not get_db_connection_active():
            flash("Соединение с базой данных отключено", 'error')
            return redirect(url_for('index'))

        user = session.get('user')

        all_crypto_ids = load_crypto_list()
        all_crypto_ids_sorted = sorted([cid for cid in all_crypto_ids if cid != 'bitcoin'])

        popular_set = set(Config.POPULAR_CRYPTOS)
        next_batch_ids = []
        for cid in all_crypto_ids_sorted:
            if cid not in popular_set:
                next_batch_ids.append(cid)
            if len(next_batch_ids) == 50:
                break

        if not next_batch_ids:
            flash("Больше нет криптовалют для отображения.", 'info')
            return redirect(url_for('show_main_crypto_rates_to_btc'))

        rates_data = get_main_crypto_rates_to_btc(coin_ids_to_fetch=next_batch_ids)

        if not rates_data:
            flash("Не удалось получить курсы следующих криптовалют к BTC. Проверьте логи.", 'error')
            log_message("Не удалось получить курсы следующих криптовалют к BTC", 'error',
                        user_id=str(user['id']) if user else None)
            return redirect(url_for('show_main_crypto_rates_to_btc'))

        sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
        table_data = []
        for crypto_id, rate_dict in sorted_rates:
            rate = rate_dict.get('btc', 'N/A')
            formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
            table_data.append({
                'crypto': crypto_id,
                'rate_to_btc': formatted_rate
            })

        viewed_ids = set(Config.POPULAR_CRYPTOS) | set(rates_data.keys())
        remaining_ids = [cid for cid in all_crypto_ids_sorted if cid not in viewed_ids]
        remaining_count = len(remaining_ids)
        viewed_count = len(viewed_ids)

        log_message("Таблица курсов следующих криптовалют к BTC отображена", 'info',
                    user_id=str(user['id']) if user else None)
        return render_template('main_crypto_rates_table.html',
                               title='Курсы следующих криптовалют к BTC',
                               data=table_data,
                               columns=['crypto', 'rate_to_btc'],
                               user=user,
                               viewed_count=viewed_count,
                               remaining_count=remaining_count,
                               has_next=bool(remaining_count))

    @app.route('/historical', methods=['GET', 'POST'])
    def historical_data():
        """Страница с историческими данными по выбранной паре"""
        plot_url = None
        error = None
        user = session.get('user')
        df_stats = None

        default_crypto = 'bitcoin'
        default_currency = 'usd'
        default_start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        default_end_date = datetime.now().strftime('%Y-%m-%d')

        if request.method == 'POST':
            crypto = request.form.get('crypto', default_crypto)
            currency = request.form.get('currency', default_currency)
            start_date = request.form.get('start_date', default_start_date)
            end_date = request.form.get('end_date', default_end_date)

            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            if start_dt > end_dt:
                start_date, end_date = end_date, start_date
                flash("Даты были автоматически переставлены, так как дата начала была позже даты окончания", 'info')

            session['historical_crypto'] = crypto
            session['historical_currency'] = currency
            session['historical_start_date'] = start_date
            session['historical_end_date'] = end_date

        else:
            crypto = session.get('historical_crypto', default_crypto)
            currency = session.get('historical_currency', default_currency)
            start_date = session.get('historical_start_date', default_start_date)
            end_date = session.get('historical_end_date', default_end_date)

        if request.method == 'POST' and crypto and currency and start_date and end_date:
            try:
                df = get_historical_price_range(
                    coin_id=crypto,
                    vs_currency=currency,
                    start_date=start_date,
                    end_date=end_date
                )

                if df is not None and not df.empty:
                    df_stats = {
                        'start_price': float(df['price'].iloc[0]),
                        'end_price': float(df['price'].iloc[-1]),
                        'max_price': float(df['price'].max()),
                        'min_price': float(df['price'].min()),
                        'avg_price': float(df['price'].mean()),
                        'change_pct': ((df['price'].iloc[-1] / df['price'].iloc[0]) - 1) * 100
                    }

                    plot_url = generate_historical_plot(df, crypto, currency, start_date, end_date)
                    if plot_url:
                        log_message(
                            f"Сгенерирован исторический график для {crypto}/{currency} за период {start_date} - {end_date}",
                            'info', user_id=str(user['id']) if user else None)
                    else:
                        error = "Ошибка генерации графика"
                else:
                    error = "Не удалось получить исторические данные для выбранного периода"

            except Exception as e:
                error = f"Ошибка: {str(e)}"
                log_message(f"Ошибка получения исторических данных: {e}", 'error',
                            user_id=str(user['id']) if user else None)

        cryptos = load_crypto_list()
        return render_template('historical.html',
                               cryptos=cryptos,
                               currencies=Config.CURRENCIES,
                               current_crypto=crypto,
                               current_currency=currency,
                               start_date=start_date,
                               end_date=end_date,
                               plot_url=plot_url,
                               error=error,
                               df_stats=df_stats,
                               user=user)

    @app.route('/correlation', methods=['GET', 'POST'])
    def correlation_analysis():
        """Страница анализа корреляции с Bitcoin"""
        user = session.get('user')
        correlation_results = None
        plot_json = None
        error = None
        summary_stats = None
        successful_calculations = 0
        failed_calculations = []

        default_cryptos = ['ethereum', 'binancecoin', 'solana', 'cardano', 'ripple']
        default_days = 30
        default_timeframe = '1d'
        default_currency = 'usd'

        if request.method == 'POST':
            selected_cryptos = request.form.getlist('cryptos')
            days = int(request.form.get('days', default_days))
            timeframe = request.form.get('timeframe', default_timeframe)
            currency = request.form.get('currency', default_currency)

            if not selected_cryptos:
                selected_cryptos = default_cryptos

            if len(selected_cryptos) > 10:
                selected_cryptos = selected_cryptos[:10]
                flash(f"Выбрано слишком много криптовалют. Анализ будет проведен для первых 10.", 'warning')

            session['correlation_cryptos'] = selected_cryptos
            session['correlation_days'] = days
            session['correlation_timeframe'] = timeframe
            session['correlation_currency'] = currency

            try:
                logging.info(f"Начинаю расчет корреляций для {len(selected_cryptos)} криптовалют...")

                correlation_results = calculate_multiple_correlations_with_retry(
                    coin_ids=selected_cryptos,
                    vs_currency=currency,
                    days=days,
                    timeframe=timeframe,
                    max_retries=2
                )

                if correlation_results:
                    successful_calculations = len(correlation_results)
                    failed_calculations = [c for c in selected_cryptos if
                                           c not in correlation_results and c != 'bitcoin']

                    if successful_calculations > 0:
                        plot_json = generate_correlation_plot(correlation_results)

                        if plot_json:
                            import numpy as np
                            correlations = [data['correlation'] for data in correlation_results.values()]
                            if correlations:
                                summary_stats = {
                                    'average_correlation': np.mean(correlations),
                                    'median_correlation': np.median(correlations),
                                    'max_correlation': max(correlations),
                                    'min_correlation': min(correlations),
                                    'positive_count': sum(1 for c in correlations if c > 0),
                                    'negative_count': sum(1 for c in correlations if c < 0),
                                    'total_count': len(correlations)
                                }

                        log_message(
                            f"Успешно рассчитано {successful_calculations} корреляций из {len(selected_cryptos)}",
                            'info', user_id=str(user['id']) if user else None)

                        if failed_calculations:
                            error = f"Не удалось рассчитать корреляцию для: {', '.join(failed_calculations)}"
                            flash(f"Успешно: {successful_calculations}, Не удалось: {len(failed_calculations)}",
                                  'warning')
                    else:
                        error = "Не удалось рассчитать ни одной корреляции."
                else:
                    error = "Не удалось рассчитать корреляции. Возможные причины: недоступность API, отсутствие данных или проблемы с соединением."

            except Exception as e:
                error = f"Ошибка при расчете корреляций: {str(e)}"
                logging.error(f"Ошибка анализа корреляции: {e}", exc_info=True)

        else:
            selected_cryptos = session.get('correlation_cryptos', default_cryptos)
            days = session.get('correlation_days', default_days)
            timeframe = session.get('correlation_timeframe', default_timeframe)
            currency = session.get('correlation_currency', default_currency)

        all_cryptos = load_crypto_list()
        available_cryptos = [c for c in all_cryptos if c != 'bitcoin']

        days_options = [
            {'value': '30', 'label': '30 дней (быстрее)'},
            {'value': '90', 'label': '90 дней'},
            {'value': '180', 'label': '180 дней'},
            {'value': '365', 'label': '1 год (медленнее)'}
        ]

        timeframe_options = [
            {'value': '1d', 'label': 'Дневной (1D)'},
            {'value': '1w', 'label': 'Недельный (1W)'},
            {'value': '1M', 'label': 'Месячный (1M)'}
        ]

        return render_template('correlation.html',
                               all_cryptos=available_cryptos[:50],
                               selected_cryptos=selected_cryptos,
                               days=days,
                               days_options=days_options,
                               timeframe=timeframe,
                               timeframe_options=timeframe_options,
                               currency=currency,
                               currencies=Config.CURRENCIES,
                               correlation_results=correlation_results,
                               plot_json=plot_json,
                               error=error,
                               summary_stats=summary_stats,
                               successful_calculations=successful_calculations,
                               failed_calculations=failed_calculations,
                               user=user)

    @app.route('/correlation_binance', methods=['GET', 'POST'])
    def correlation_binance():
        """Страница анализа корреляции через Binance API"""
        user = session.get('user')
        correlation_results = None
        plot_json = None
        error = None
        summary_stats = None
        successful = 0
        failed = []

        default_cryptos = ['ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'MATIC', 'AVAX', 'LINK']
        default_days = 30
        default_currency = 'USDT'

        if request.method == 'POST':
            selected_cryptos = request.form.get('cryptos', '').upper().split(',')
            if not selected_cryptos or selected_cryptos[0] == '':
                selected_cryptos = default_cryptos
            else:
                selected_cryptos = [c.strip() for c in selected_cryptos if c.strip()]

            days = int(request.form.get('days', default_days))
            currency = request.form.get('currency', default_currency)

            if len(selected_cryptos) > 15:
                selected_cryptos = selected_cryptos[:15]
                flash("Ограничено 15 криптовалютами для производительности", 'warning')

            session['correlation_binance_cryptos'] = selected_cryptos
            session['correlation_binance_days'] = days
            session['correlation_binance_currency'] = currency

            try:
                import requests
                logging.info("Проверяю доступность Binance API...")
                test_response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
                if test_response.status_code != 200:
                    error = "Binance API временно недоступен. Попробуйте позже."
                    flash(error, 'error')
                else:
                    correlation_results, successful, failed = calculate_multiple_correlations_binance(
                        coin_symbols=selected_cryptos,
                        vs_currency=currency,
                        days=days
                    )

                    if correlation_results:
                        plot_json = generate_correlation_plot(correlation_results)

                        import numpy as np
                        correlations = [data['correlation'] for data in correlation_results.values()]
                        if correlations:
                            summary_stats = {
                                'average_correlation': np.mean(correlations),
                                'median_correlation': np.median(correlations),
                                'max_correlation': max(correlations),
                                'min_correlation': min(correlations),
                                'positive_count': sum(1 for c in correlations if c > 0),
                                'negative_count': sum(1 for c in correlations if c < 0),
                                'total_count': len(correlations)
                            }

                        if successful > 0:
                            flash(f"Успешно рассчитано: {successful} корреляций", 'success')
                        if failed:
                            flash(f"Не удалось: {len(failed)} криптовалют", 'warning')
                    else:
                        error = "Не удалось рассчитать корреляции. Проверьте символы криптовалют."

            except Exception as e:
                error = f"Ошибка: {str(e)}"
                logging.error(f"Ошибка в correlation_binance: {e}", exc_info=True)

        else:
            selected_cryptos = session.get('correlation_binance_cryptos', default_cryptos)
            days = session.get('correlation_binance_days', default_days)
            currency = session.get('correlation_binance_currency', default_currency)

        currencies = ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']

        days_options = [
            {'value': '7', 'label': '7 дней'},
            {'value': '30', 'label': '30 дней'},
            {'value': '90', 'label': '90 дней'},
            {'value': '180', 'label': '180 дней'}
        ]

        return render_template('correlation_binance.html',
                               selected_cryptos=selected_cryptos,
                               days=days,
                               days_options=days_options,
                               currency=currency,
                               currencies=currencies,
                               correlation_results=correlation_results,
                               plot_json=plot_json,
                               error=error,
                               summary_stats=summary_stats,
                               successful=successful,
                               failed=failed,
                               user=user)

    @app.route('/api/status')
    def get_worker_status():
        """Возвращает статус системы и воркеров"""
        try:
            # Информация о системе
            system_info = {
                'system': platform.system(),
                'node': platform.node(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'host_id': get_host_id()
            }

            # Использование RAM
            ram_info = {
                'total': round(psutil.virtual_memory().total / (1024 ** 3), 2),
                'available': round(psutil.virtual_memory().available / (1024 ** 3), 2),
                'percent': psutil.virtual_memory().percent,
                'used': round(psutil.virtual_memory().used / (1024 ** 3), 2)
            }

            # Использование CPU
            cpu_info = {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }

            # Информация о процессах
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

            # Статистика nginx
            nginx_processes = [p for p in processes if 'nginx' in p['name'].lower()]
            nginx_status = {
                'count': len(nginx_processes),
                'on': len([p for p in nginx_processes if p['status'] == 'running']),
                'off': len([p for p in nginx_processes if p['status'] != 'running']),
                'processes': nginx_processes[:5]
            }

            # Статистика gunicorn
            gunicorn_processes = [p for p in processes if 'gunicorn' in p['name'].lower()]
            gunicorn_status = {
                'count': len(gunicorn_processes),
                'on': len([p for p in gunicorn_processes if p['status'] == 'running']),
                'off': len([p for p in gunicorn_processes if p['status'] != 'running']),
                'processes': gunicorn_processes[:5]
            }

            # Flask статус
            flask_status = {
                'status': 'running',
                'uptime': time.time() - psutil.Process(os.getpid()).create_time(),
                'requests_processed': 0
            }

            # Статус базы данных
            db_status = {
                'connected': get_db_connection_active(),
                'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Общий статус
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

        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 500

    @app.route('/status')
    def show_status():
        """Страница для просмотра статуса системы"""
        user = session.get('user')
        return render_template('status.html',
                               title='Статус системы',
                               user=user)
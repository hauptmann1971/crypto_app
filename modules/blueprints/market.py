from datetime import datetime

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for

from modules.app_logging import log_message
from modules.config import Config
from modules.database import get_db, get_db_connection_active
from modules.models import CryptoRate, CryptoRequest
from modules.blueprints.helpers import ensure_db_connected, safe_json_loads
from modules.services import (
    create_crypto_request,
    get_pending_requests_count,
    get_processing_requests_count,
    get_user_from_db,
    get_user_requests_history,
    process_pending_requests,
    update_user_last_login,
)
from modules.utils import load_crypto_list
from modules.apis import get_main_crypto_rates_to_btc

market_bp = Blueprint('market', __name__)


@market_bp.route('/disconnect_db', methods=['POST'])
def disconnect_db():
    """Отключение подключения к базе данных"""
    from modules.database import db_manager

    db_manager.db_connection_active = False
    if db_manager.engine and hasattr(db_manager.engine, 'dispose'):
        db_manager.engine.dispose()
        db_manager.engine = None
        db_manager.SessionLocal = None

    session['db_message'] = "Соединение с базой данных отключено"
    session['db_message_type'] = 'warning'

    log_message("Ручное отключение базы данных", 'info')
    return redirect(url_for('market.index'))


@market_bp.route('/connect_db', methods=['POST'])
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
    except Exception as exc:
        session['db_message'] = f"Ошибка подключения: {str(exc)}"
        session['db_message_type'] = 'error'
        log_message(f"Ошибка подключения к БД: {exc}", 'error')

    return redirect(url_for('market.index'))


@market_bp.route('/request-status/<int:request_id>')
def get_request_status(request_id):
    """Проверяет статус запроса"""
    try:
        with get_db() as db:
            db_request = db.query(CryptoRequest).filter(
                CryptoRequest.id == request_id
            ).first()

            if not db_request:
                return jsonify({'success': False, 'error': 'Request not found'})

            response_data = safe_json_loads(db_request.response_data)

            return jsonify({
                'success': True,
                'status': db_request.status,
                'rate': response_data.get('rate'),
                'error': response_data.get('error'),
                'finished_at': db_request.finished_at,
                'crypto': db_request.crypto,
                'currency': db_request.currency
            })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)})


@market_bp.route('/', methods=['GET', 'POST'])
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
            return redirect(url_for('market.index'))

        current_crypto = crypto
        current_currency = currency
        request_id = create_crypto_request(user_id, crypto, currency)
        if request_id is not None:
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


@market_bp.route('/crypto_table')
def show_crypto_table():
    """Отображает таблицу курсов криптовалют"""
    db_guard = ensure_db_connected()
    if db_guard is not None:
        return db_guard

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
    except Exception as exc:
        flash(f"Ошибка БД: {str(exc)}", 'error')
        return redirect(url_for('market.index'))


@market_bp.route('/main_crypto_rates_to_btc')
def show_main_crypto_rates_to_btc():
    """Отображает таблицу с курсами основных криптовалют к BTC"""
    db_guard = ensure_db_connected()
    if db_guard is not None:
        return db_guard

    user = session.get('user')
    rates_data = get_main_crypto_rates_to_btc(
        coin_ids_to_fetch=[cid for cid in Config.POPULAR_CRYPTOS if cid != 'bitcoin'])

    if not rates_data:
        flash("Не удалось получить курсы криптовалют к BTC. Проверьте логи.", 'error')
        log_message("Не удалось получить курсы основных криптовалют к BTC", 'error',
                    user_id=str(user['id']) if user else None)
        return redirect(url_for('market.index'))

    sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
    table_data = []
    for crypto_id, rate_dict in sorted_rates:
        rate = rate_dict.get('btc', 'N/A')
        formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
        table_data.append({'crypto': crypto_id, 'rate_to_btc': formatted_rate})

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


@market_bp.route('/main_crypto_rates_to_btc/next')
def show_next_crypto_rates_to_btc():
    """Отображает таблицу с курсами следующих 50 криптовалют к BTC"""
    db_guard = ensure_db_connected()
    if db_guard is not None:
        return db_guard

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
        return redirect(url_for('market.show_main_crypto_rates_to_btc'))

    rates_data = get_main_crypto_rates_to_btc(coin_ids_to_fetch=next_batch_ids)
    if not rates_data:
        flash("Не удалось получить курсы следующих криптовалют к BTC. Проверьте логи.", 'error')
        log_message("Не удалось получить курсы следующих криптовалют к BTC", 'error',
                    user_id=str(user['id']) if user else None)
        return redirect(url_for('market.show_main_crypto_rates_to_btc'))

    sorted_rates = sorted(rates_data.items(), key=lambda item: item[0])
    table_data = []
    for crypto_id, rate_dict in sorted_rates:
        rate = rate_dict.get('btc', 'N/A')
        formatted_rate = f"{rate:.8f}" if isinstance(rate, float) else rate
        table_data.append({'crypto': crypto_id, 'rate_to_btc': formatted_rate})

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

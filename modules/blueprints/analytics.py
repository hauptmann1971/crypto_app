import logging
from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from modules.apis import (
    calculate_multiple_correlations_binance,
    calculate_multiple_correlations_with_retry,
    coingecko_api,
    generate_correlation_plot,
    generate_historical_plot,
    get_historical_price_range,
)
from modules.app_logging import log_message
from modules.config import Config
from modules.utils import load_crypto_list

analytics_bp = Blueprint('analytics', __name__)


def _build_correlation_summary(correlation_results):
    import numpy as np

    correlations = [data['correlation'] for data in correlation_results.values()]
    if not correlations:
        return None

    return {
        'average_correlation': np.mean(correlations),
        'median_correlation': np.median(correlations),
        'max_correlation': max(correlations),
        'min_correlation': min(correlations),
        'positive_count': sum(1 for c in correlations if c > 0),
        'negative_count': sum(1 for c in correlations if c < 0),
        'total_count': len(correlations)
    }


@analytics_bp.route('/chart', methods=['GET', 'POST'])
def chart():
    plot_url = None
    error = None
    user = session.get('user')

    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')
        period = request.form.get('period', '7')
        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            return redirect(url_for('analytics.chart'))

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
        except Exception as exc:
            error = f"Ошибка: {str(exc)}"
            log_message(f"Ошибка построения графика: {exc}", 'error', user_id=str(user['id']) if user else None)

    cryptos = load_crypto_list()
    return render_template('chart.html',
                           cryptos=cryptos,
                           currencies=Config.CURRENCIES,
                           periods=Config.PERIODS,
                           plot_url=plot_url,
                           error=error,
                           user=user)


@analytics_bp.route('/candlestick', methods=['GET', 'POST'])
def candlestick_chart():
    plot_url = None
    error = None
    user = session.get('user')

    if request.method == 'POST':
        crypto = request.form.get('crypto')
        currency = request.form.get('currency')
        period = request.form.get('period', '7')

        if not crypto or not currency:
            flash("Выберите криптовалюту и валюту", 'error')
            return redirect(url_for('analytics.candlestick_chart'))

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

        except Exception as exc:
            error = f"Ошибка: {str(exc)}"
            log_message(f"Ошибка построения свечного графика: {exc}", 'error',
                        user_id=str(user['id']) if user else None)

    cryptos = load_crypto_list()
    return render_template('candlestick.html',
                           cryptos=cryptos,
                           currencies=Config.CURRENCIES,
                           periods=Config.PERIODS,
                           plot_url=plot_url,
                           error=error,
                           user=user)


@analytics_bp.route('/historical', methods=['GET', 'POST'])
def historical_data():
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
        except Exception as exc:
            error = f"Ошибка: {str(exc)}"
            log_message(f"Ошибка получения исторических данных: {exc}", 'error',
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


@analytics_bp.route('/correlation', methods=['GET', 'POST'])
def correlation_analysis():
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
            flash("Выбрано слишком много криптовалют. Анализ будет проведен для первых 10.", 'warning')

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
                failed_calculations = [c for c in selected_cryptos if c not in correlation_results and c != 'bitcoin']
                if successful_calculations > 0:
                    plot_json = generate_correlation_plot(correlation_results)
                    if plot_json:
                        summary_stats = _build_correlation_summary(correlation_results)

                    log_message(f"Успешно рассчитано {successful_calculations} корреляций из {len(selected_cryptos)}",
                                'info', user_id=str(user['id']) if user else None)
                    if failed_calculations:
                        error = f"Не удалось рассчитать корреляцию для: {', '.join(failed_calculations)}"
                        flash(f"Успешно: {successful_calculations}, Не удалось: {len(failed_calculations)}", 'warning')
                else:
                    error = "Не удалось рассчитать ни одной корреляции."
            else:
                error = "Не удалось рассчитать корреляции. Возможные причины: недоступность API, отсутствие данных или проблемы с соединением."
        except Exception as exc:
            error = f"Ошибка при расчете корреляций: {str(exc)}"
            logging.error(f"Ошибка анализа корреляции: {exc}", exc_info=True)
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


@analytics_bp.route('/correlation_binance', methods=['GET', 'POST'])
def correlation_binance():
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
                    summary_stats = _build_correlation_summary(correlation_results)

                    if successful > 0:
                        flash(f"Успешно рассчитано: {successful} корреляций", 'success')
                    if failed:
                        flash(f"Не удалось: {len(failed)} криптовалют", 'warning')
                else:
                    error = "Не удалось рассчитать корреляции. Проверьте символы криптовалют."

        except Exception as exc:
            error = f"Ошибка: {str(exc)}"
            logging.error(f"Ошибка в correlation_binance: {exc}", exc_info=True)

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

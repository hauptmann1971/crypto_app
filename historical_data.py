import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def get_historical_price_range(coin_id='bitcoin', vs_currency='usd',
                               start_date='2024-01-01', end_date=None):
    """
    Получить исторические цены для выбранного периода

    Args:
        coin_id: ID криптовалюты (например, 'bitcoin', 'ethereum')
        vs_currency: Валюта (например, 'usd', 'eur', 'btc')
        start_date: Дата начала в формате 'YYYY-MM-DD'
        end_date: Дата окончания в формате 'YYYY-MM-DD' (по умолчанию - сегодня)

    Returns:
        DataFrame с колонками: timestamp, price
    """

    # Если конечная дата не указана - берем сегодня
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Преобразуем строки в datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Проверяем, что даты корректны
    if start_dt >= end_dt:
        print("Ошибка: Дата начала должна быть раньше даты окончания")
        return None

    # Рассчитываем количество дней между датами
    days_diff = (end_dt - start_dt).days

    if days_diff < 1:
        print("Ошибка: Период должен быть хотя бы 1 день")
        return None

    # Определяем интервал в зависимости от периода
    if days_diff <= 90:
        # До 90 дней - можем получить дневные данные
        days_param = days_diff
        interval = 'daily'
    else:
        # Более 90 дней - получаем дневные данные (максимально доступные)
        days_param = days_diff
        interval = 'daily'
        print(f"Примечание: Для периода >90 дней данные могут быть агрегированными")

    # URL для запроса
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    params = {
        'vs_currency': vs_currency,
        'days': days_param,
        'interval': interval
    }

    try:
        print(f"Загрузка данных {coin_id.upper()}/{vs_currency.upper()}...")
        print(f"Период: {start_date} - {end_date} ({days_diff} дней)")

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        # Преобразуем данные в DataFrame
        timestamps = [pd.to_datetime(x[0], unit='ms') for x in data['prices']]
        prices = [x[1] for x in data['prices']]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })

        # Фильтруем по нашему диапазону дат
        mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
        df = df.loc[mask].copy()

        if len(df) == 0:
            print("Нет данных для указанного периода")
            return None

        # Сортируем по дате
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Добавляем дополнительные колонки
        df['date'] = df['timestamp'].dt.date
        df['returns_pct'] = df['price'].pct_change() * 100

        print(f"✅ Успешно загружено {len(df)} записей")
        print(f"📊 Диапазон цен: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"📈 Изменение за период: {((df['price'].iloc[-1] / df['price'].iloc[0]) - 1) * 100:+.2f}%")

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка запроса: {e}")
        return None
    except Exception as e:
        print(f"❌ Ошибка обработки данных: {e}")
        return None


def analyze_price_range_with_plot(coin_id='bitcoin', vs_currency='usd',
                                  start_date='2024-01-01', end_date=None):
    """
    Получить данные и построить график
    """
    # Получаем данные
    df = get_historical_price_range(coin_id, vs_currency, start_date, end_date)

    if df is None or len(df) == 0:
        return None

    # Создаем график
    plt.figure(figsize=(12, 6))

    # График цены
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['price'], color='blue', linewidth=2)
    plt.title(f'{coin_id.upper()}/{vs_currency.upper()} - {start_date} до {end_date}')
    plt.ylabel('Цена ($)')
    plt.grid(True, alpha=0.3)

    # График процентных изменений
    plt.subplot(2, 1, 2)
    colors = ['green' if x >= 0 else 'red' for x in df['returns_pct']]
    plt.bar(df['timestamp'], df['returns_pct'], color=colors, alpha=0.7)
    plt.xlabel('Дата')
    plt.ylabel('Изменение (%)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return df


def interactive_price_query():
    """
    Интерактивный запрос исторических данных
    """
    print("📊 ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("-" * 40)

    # Ввод параметров
    coin_id = input("Введите ID криптовалюты (например: bitcoin): ").strip().lower() or 'bitcoin'
    vs_currency = input("Введите валюту (например: usd): ").strip().lower() or 'usd'

    today = datetime.now().strftime('%Y-%m-%d')

    print(f"\nФормат даты: YYYY-MM-DD (например: 2024-01-15)")
    start_date = input(f"Дата начала (до {today}): ").strip() or '2024-01-01'
    end_date = input(f"Дата окончания (по умолчанию {today}): ").strip() or today

    print("\n" + "=" * 40)

    # Загружаем данные
    df = get_historical_price_range(coin_id, vs_currency, start_date, end_date)

    if df is not None:
        # Показываем сводную статистику
        print("\n📈 СВОДНАЯ СТАТИСТИКА:")
        print(f"Период: {df['date'].iloc[0]} - {df['date'].iloc[-1]}")
        print(f"Количество дней: {len(df)}")
        print(f"Начальная цена: ${df['price'].iloc[0]:.2f}")
        print(f"Конечная цена: ${df['price'].iloc[-1]:.2f}")
        print(f"Изменение: {((df['price'].iloc[-1] / df['price'].iloc[0]) - 1) * 100:+.2f}%")
        print(f"Минимальная цена: ${df['price'].min():.2f}")
        print(f"Максимальная цена: ${df['price'].max():.2f}")
        print(f"Средняя цена: ${df['price'].mean():.2f}")

        # Сохраняем в CSV
        save = input("\nСохранить данные в CSV? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"{coin_id}_{vs_currency}_{start_date}_to_{end_date}.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Данные сохранены в {filename}")

    return df

# Запускаем интерактивный режим
df = interactive_price_query()
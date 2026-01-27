# locust_correlation.py - Упрощенная версия
import random
import time
import sys
from locust import HttpUser, task, between


class CorrelationUser(HttpUser):
    """
    Простой пользователь для тестирования эндпоинтов корреляции
    """

    # Базовые настройки
    wait_time = between(5, 15)

    # Тестовые данные прямо в коде
    COINGECKO_CRYPTOS = [
        'bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano',
        'solana', 'polkadot', 'dogecoin', 'matic-network', 'chainlink'
    ]

    BINANCE_CRYPTOS = [
        'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'MATIC',
        'AVAX', 'LINK', 'LTC', 'BCH'
    ]

    CURRENCIES = ['usd', 'eur', 'rub']
    BINANCE_CURRENCIES = ['USDT', 'BUSD', 'BTC']

    DAYS_OPTIONS = ['30', '90', '180']
    TIMEFRAMES = ['1d', '1w']

    def on_start(self):
        """Инициализация пользователя при старте"""
        self.user_id = f"user_{random.randint(1000, 9999)}"
        print(f"[{self.user_id}] Начал тестирование")

    @task(3)  # CoinGecko тестируем в 3 раза чаще
    def test_coingecko_correlation(self):
        """Тестирование /correlation (CoinGecko)"""

        # 1. Получаем форму
        self.client.get("/correlation", name="GET /correlation (форма)")

        # 2. Готовим данные
        num_cryptos = random.randint(2, 5)
        selected_cryptos = random.sample(self.COINGECKO_CRYPTOS, num_cryptos)

        # 3. Отправляем POST запрос
        form_data = {
            'days': random.choice(self.DAYS_OPTIONS),
            'timeframe': random.choice(self.TIMEFRAMES),
            'currency': random.choice(self.CURRENCIES)
        }

        # Добавляем криптовалюты
        for crypto in selected_cryptos:
            form_data['cryptos'] = crypto

        response = self.client.post("/correlation",
                                    data=form_data,
                                    name="POST /correlation (расчет)")

        # Простая проверка ответа
        if response.status_code == 200:
            print(f"[{self.user_id}] CoinGecko OK: {num_cryptos} крипт")
        else:
            print(f"[{self.user_id}] CoinGecko ERROR: {response.status_code}")

    @task(2)  # Binance тестируем в 2 раза реже
    def test_binance_correlation(self):
        """Тестирование /correlation_binance"""

        # 1. Получаем форму
        self.client.get("/correlation_binance", name="GET /correlation_binance (форма)")

        # 2. Готовим данные
        num_cryptos = random.randint(2, 4)
        selected_cryptos = random.sample(self.BINANCE_CRYPTOS, num_cryptos)
        cryptos_string = ','.join(selected_cryptos)

        # 3. Отправляем POST запрос
        form_data = {
            "cryptos": cryptos_string,
            "days": random.choice(['7', '30', '90']),
            "currency": random.choice(self.BINANCE_CURRENCIES)
        }

        response = self.client.post("/correlation_binance",
                                    data=form_data,
                                    name="POST /correlation_binance (расчет)")

        # Простая проверка ответа
        if response.status_code == 200:
            print(f"[{self.user_id}] Binance OK: {num_cryptos} крипт")
        else:
            print(f"[{self.user_id}] Binance ERROR: {response.status_code}")

    @task(1)  # Проверка результатов - самая редкая задача
    def check_results(self):
        """Проверяем результаты с GET параметрами"""

        # Простые GET запросы с параметрами
        params = {
            'cryptos': 'bitcoin,ethereum,solana',
            'days': random.choice(self.DAYS_OPTIONS),
            'timeframe': random.choice(self.TIMEFRAMES),
            'currency': random.choice(self.CURRENCIES)
        }

        self.client.get("/correlation", params=params, name="GET /correlation (результаты)")

        # Binance вариант
        binance_params = {
            'cryptos': 'ETH,BNB,SOL',
            'days': random.choice(['7', '30', '90']),
            'currency': random.choice(self.BINANCE_CURRENCIES)
        }

        self.client.get("/correlation_binance", params=binance_params, name="GET /correlation_binance (результаты)")

        print(f"[{self.user_id}] Проверил результаты")


def main():
    """Простой запуск с выбором режима"""
    print("=" * 60)
    print("📊 ТЕСТИРОВАНИЕ КОРРЕЛЯЦИЙ КРИПТОВАЛЮТ")
    print("=" * 60)
    print("\nДоступные режимы:")
    print("  1. Веб-интерфейс (по умолчанию)")
    print("  2. Быстрый тест (10 users, 1m)")
    print("  3. Средний тест (20 users, 3m)")
    print("  4. Стресс-тест (50 users, 5m)")
    print("\nДля веб-интерфейса откройте: http://localhost:8089")
    print("=" * 60)

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("\nВыберите режим (1-4 или Enter для веб): ").strip()

    import subprocess

    # Настройки по умолчанию
    host = "http://hauptmann.su"  # Замените на ваш хост

    if mode == "2" or mode == "fast":
        cmd = f"locust -f {__file__} --host={host} --users=10 --spawn-rate=2 --run-time=1m --headless"
    elif mode == "3" or mode == "normal":
        cmd = f"locust -f {__file__} --host={host} --users=20 --spawn-rate=3 --run-time=3m --headless"
    elif mode == "4" or mode == "stress":
        cmd = f"locust -f {__file__} --host={host} --users=50 --spawn-rate=5 --run-time=5m --headless"
    else:
        # Веб-интерфейс по умолчанию
        cmd = f"locust -f {__file__} --host={host} --web-host=localhost --web-port=8089"

    print(f"\nЗапуск: {cmd}")
    print("-" * 60)

    try:
        subprocess.run(cmd.split(), check=True)
    except KeyboardInterrupt:
        print("\nТест прерван")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
# locust_correlation.py - Упрощенная версия
import random
import sys
import subprocess

# Проверка установки locust
try:
    from locust import HttpUser, task, between

    LOCUST_INSTALLED = True
except ImportError:
    LOCUST_INSTALLED = False
    print("❌ ОШИБКА: Locust не установлен!")
    print("\nУстановите Locust командой:")
    print("pip install locust")
    print("\nИли создайте виртуальное окружение и установите зависимости:")
    print("python -m venv .venv")
    print(".venv\\Scripts\\activate  # Windows")
    print("source .venv/bin/activate  # Linux/Mac")
    print("pip install locust")
    sys.exit(1)


class CorrelationUser(HttpUser):
    """
    Простой пользователь для тестирования эндпоинтов корреляции
    """

    # Базовые настройки
    wait_time = between(5, 15)

    # Тестовые данные
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


def check_locust_command():
    """Проверка доступности команды locust в системе"""
    try:
        # Проверяем, есть ли locust в PATH
        result = subprocess.run(['locust', '--version'],
                                capture_output=True,
                                text=True,
                                timeout=2)
        if result.returncode == 0:
            print(f"✅ Locust найден: {result.stdout.strip()}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("⚠️  Команда 'locust' не найдена в PATH")
    print("   Попробуйте активировать виртуальное окружение:")
    print("   Windows: .venv\\Scripts\\activate")
    print("   Linux/Mac: source .venv/bin/activate")
    return False


def run_locust_command(cmd_parts):
    """Запуск команды locust с обработкой ошибок"""
    try:
        print(f"\n🚀 Запуск: {' '.join(cmd_parts)}")
        print("-" * 60)

        # Запускаем процесс
        process = subprocess.Popen(cmd_parts,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   universal_newlines=True)

        # Выводим вывод в реальном времени
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode == 0:
            print("\n✅ Тест завершен успешно")
        else:
            print(f"\n❌ Тест завершен с кодом ошибки: {process.returncode}")

    except KeyboardInterrupt:
        print("\n\n🛑 Тест прерван пользователем")
    except FileNotFoundError:
        print("\n❌ Ошибка: Команда 'locust' не найдена")
        print("Убедитесь, что:")
        print("1. Locust установлен: pip install locust")
        print("2. Виртуальное окружение активировано")
        print("3. Команда 'locust' доступна в PATH")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")


def main():
    """Простой запуск с выбором режима"""
    print("=" * 60)
    print("📊 ТЕСТИРОВАНИЕ КОРРЕЛЯЦИЙ КРИПТОВАЛЮТ")
    print("=" * 60)

    # Проверяем доступность команды locust
    if not check_locust_command():
        response = input("\nПродолжить все равно? (y/N): ").lower().strip()
        if response != 'y':
            print("Отмена запуска")
            return

    print("\nДоступные режимы:")
    print("  1. Веб-интерфейс (по умолчанию)")
    print("  2. Быстрый тест (10 users, 1m)")
    print("  3. Средний тест (20 users, 3m)")
    print("  4. Стресс-тест (50 users, 5m)")
    print("  5. Настроить хост (текущий хост будет изменен)")
    print("\nДля веб-интерфейса откройте: http://localhost:8089")
    print("=" * 60)

    # Настройки по умолчанию
    host = "http://hauptmann.su"  # Исправлено: добавлен https

    # Получаем режим
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("\nВыберите режим (1-5 или Enter для веб): ").strip()

    # Обработка выбора хоста
    if mode == "5" or mode == "host":
        new_host = input(f"\nТекущий хост: {host}\nВведите новый хост: ").strip()
        if new_host:
            host = new_host
        print(f"Хост установлен: {host}")
        mode = input("\nТеперь выберите режим тестирования (1-4): ").strip()

    # Формируем команду
    cmd_parts = ['locust', '-f', __file__, '--host', host]

    if mode == "2" or mode == "fast":
        cmd_parts.extend(['--users', '10', '--spawn-rate', '2', '--run-time', '1m', '--headless'])
    elif mode == "3" or mode == "normal":
        cmd_parts.extend(['--users', '20', '--spawn-rate', '3', '--run-time', '3m', '--headless'])
    elif mode == "4" or mode == "stress":
        cmd_parts.extend(['--users', '50', '--spawn-rate', '5', '--run-time', '5m', '--headless'])
    else:
        # Веб-интерфейс по умолчанию
        cmd_parts.extend(['--web-host', 'localhost', '--web-port', '8089'])
        print(f"\n🌐 Веб-интерфейс будет доступен по адресу: http://localhost:8089")
        print("   После запуска откройте этот адрес в браузере")

    # Запускаем команду
    run_locust_command(cmd_parts)


if __name__ == "__main__":
    main()
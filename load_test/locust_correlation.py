# locust_correlation.py
import random
import time

from locust import HttpUser, task, between


class CorrelationUser(HttpUser):
    """
    Пользователь, который тестирует эндпоинты корреляции:
    - /correlation (CoinGecko API)
    - /correlation_binance (Binance API)
    """

    wait_time = between(5, 15)

    # Конфигурация теста
    COINGECKO_CRYPTOS = [
        'bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano',
        'solana', 'polkadot', 'dogecoin', 'matic-network', 'chainlink',
        'litecoin', 'bitcoin-cash', 'stellar', 'monero', 'ethereum-classic'
    ]

    BINANCE_CRYPTOS = [
        'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'MATIC',
        'AVAX', 'LINK', 'LTC', 'BCH', 'XLM', 'XMR', 'ETC'
    ]

    CURRENCIES = ['usd', 'eur', 'rub']
    BINANCE_CURRENCIES = ['USDT', 'BUSD', 'BTC']

    DAYS_OPTIONS = ['30', '90', '180', '365']
    TIMEFRAMES = ['1d', '1w', '1M']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binance_results = None
        self.coingecko_results = None
        self.test_start_time = None
        self.user_id = None

    def on_start(self):
        """Инициализация пользователя"""
        self.user_id = f"corr_user_{random.randint(1000, 9999)}"
        self.test_start_time = time.time()
        self.coingecko_results = []
        self.binance_results = []

        print(f"[{self.user_id}] Начал тестирование корреляций")

    @task(3)  # Чаще тестируем CoinGecko
    def test_coingecko_correlation(self):
        """Тестирование эндпоинта /correlation (CoinGecko)"""

        # 1. Сначала GET запрос для получения формы
        with self.client.get("/correlation",
                             name="GET /correlation (форма)",
                             catch_response=True) as get_response:

            if get_response.status_code != 200:
                get_response.failure(f"GET failed: {get_response.status_code}")
                return

            get_response.success()

        # 2. Подготавливаем данные для POST запроса
        num_cryptos = random.randint(3, 7)  # 3-7 криптовалют для анализа
        selected_cryptos = random.sample(self.COINGECKO_CRYPTOS, num_cryptos)

        correlation_data = {
            "cryptos": selected_cryptos,
            "days": random.choice(self.DAYS_OPTIONS),
            "timeframe": random.choice(self.TIMEFRAMES),
            "currency": random.choice(self.CURRENCIES)
        }

        # Для отправки массива в форме
        form_data = {}
        for i, crypto in enumerate(selected_cryptos):
            form_data[f'cryptos'] = crypto  # Все с одним ключом

        form_data['days'] = correlation_data['days']
        form_data['timeframe'] = correlation_data['timeframe']
        form_data['currency'] = correlation_data['currency']

        # 3. Отправляем POST запрос для расчета корреляций
        start_time = time.time()

        with self.client.post("/correlation",
                              data=form_data,
                              name="POST /correlation (расчет)",
                              catch_response=True) as post_response:

            response_time = int((time.time() - start_time) * 1000)

            if post_response.status_code == 200:
                # Проверяем содержимое ответа
                content = post_response.text

                success_indicators = [
                    'корреляция' in content.lower(),
                    'correlation' in content.lower(),
                    'график' in content.lower(),
                    'bitcoin' in content.lower(),
                    'результат' in content.lower()
                ]

                if any(success_indicators):
                    post_response.success()

                    # Сохраняем метрики
                    self.coingecko_results.append({
                        'cryptos_count': num_cryptos,
                        'days': correlation_data['days'],
                        'response_time_ms': response_time,
                        'timestamp': time.time()
                    })

                    print(f"[{self.user_id}] CoinGecko: {num_cryptos} cryptos, "
                          f"{correlation_data['days']} days, {response_time}ms")

                    # Дополнительная проверка - ищем JSON данные
                    if 'plot_json' in content or 'var graph' in content:
                        print(f"[{self.user_id}] График сгенерирован успешно")

                else:
                    post_response.failure("Страница не содержит ожидаемых данных корреляции")

            elif post_response.status_code == 429:
                # Rate limiting - ожидаем и повторяем
                post_response.failure("Rate limit exceeded (429)")
                time.sleep(5)

            elif post_response.status_code == 500:
                post_response.failure("Server error (500)")
                # Можно добавить логирование

            else:
                post_response.failure(f"Unexpected status: {post_response.status_code}")

    @task(2)  # Реже тестируем Binance
    def test_binance_correlation(self):
        """Тестирование эндпоинта /correlation_binance"""

        # 1. GET запрос для формы
        with self.client.get("/correlation_binance",
                             name="GET /correlation_binance (форма)",
                             catch_response=True) as get_response:

            if get_response.status_code != 200:
                get_response.failure(f"GET failed: {get_response.status_code}")
                return

            get_response.success()

        # 2. Подготовка данных для Binance
        num_cryptos = random.randint(2, 5)  # Binance может быть более тяжелым
        selected_cryptos = random.sample(self.BINANCE_CRYPTOS, num_cryptos)

        # Форматируем как ожидает форма (через запятую)
        cryptos_string = ','.join(selected_cryptos)

        binance_data = {
            "cryptos": cryptos_string,
            "days": random.choice(['7', '30', '90']),
            "currency": random.choice(self.BINANCE_CURRENCIES)
        }

        # 3. POST запрос к Binance эндпоинту
        start_time = time.time()

        with self.client.post("/correlation_binance",
                              data=binance_data,
                              name="POST /correlation_binance (расчет)",
                              catch_response=True) as post_response:

            response_time = int((time.time() - start_time) * 1000)

            if post_response.status_code == 200:
                content = post_response.text

                success_indicators = [
                    'binance' in content.lower(),
                    'корреляция' in content.lower(),
                    'btc' in content.lower(),
                    'usdt' in content.lower(),
                    'график' in content.lower()
                ]

                if any(success_indicators):
                    post_response.success()

                    self.binance_results.append({
                        'cryptos_count': num_cryptos,
                        'days': binance_data['days'],
                        'currency': binance_data['currency'],
                        'response_time_ms': response_time,
                        'timestamp': time.time()
                    })

                    print(f"[{self.user_id}] Binance: {num_cryptos} cryptos, "
                          f"{binance_data['days']} days, {response_time}ms")

                else:
                    post_response.failure("Страница Binance не содержит ожидаемых данных")

            elif post_response.status_code == 429:
                post_response.failure("Binance rate limit (429)")
                time.sleep(10)  # Binance требует больше времени

            elif post_response.status_code == 502 or post_response.status_code == 504:
                # Gateway timeout - Binance API может не отвечать
                post_response.failure(f"Gateway error: {post_response.status_code}")
                time.sleep(15)

            else:
                post_response.failure(f"Unexpected status: {post_response.status_code}")

    @task(1)  # Проверка результатов
    def check_correlation_results(self):
        """Проверка ранее рассчитанных корреляций через GET с параметрами"""

        # Для CoinGecko
        if self.coingecko_results:
            latest = self.coingecko_results[-1]

            # Эмулируем просмотр результатов с параметрами
            params = {
                'cryptos': 'bitcoin,ethereum,solana',  # Пример
                'days': latest['days'],
                'timeframe': random.choice(self.TIMEFRAMES),
                'currency': random.choice(self.CURRENCIES)
            }

            self.client.get("/correlation",
                            params=params,
                            name="GET /correlation (с параметрами)")

        # Для Binance
        if self.binance_results:
            latest = self.binance_results[-1]

            params = {
                'cryptos': 'ETH,BNB,SOL',
                'days': latest['days'],
                'currency': latest['currency']
            }

            self.client.get("/correlation_binance",
                            params=params,
                            name="GET /correlation_binance (с параметрами)")

    def on_stop(self):
        """Вызывается при завершении пользователя"""
        duration = time.time() - self.test_start_time

        print(f"\n[{self.user_id}] Итоги тестирования:")
        print(f"  CoinGecko запросов: {len(self.coingecko_results)}")
        print(f"  Binance запросов: {len(self.binance_results)}")

        if self.coingecko_results:
            avg_time = sum(r['response_time_ms'] for r in self.coingecko_results) / len(self.coingecko_results)
            print(f"  Среднее время CoinGecko: {avg_time:.0f}ms")

        if self.binance_results:
            avg_time = sum(r['response_time_ms'] for r in self.binance_results) / len(self.binance_results)
            print(f"  Среднее время Binance: {avg_time:.0f}ms")
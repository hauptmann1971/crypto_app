# locust_correlation_monitored.py
from locust import HttpUser, task, between
import random
import time
import requests


class MonitoredCorrelationUser(HttpUser):
    """Пользователь с мониторингом внешних API"""

    wait_time = between(10, 30)  # Долгие паузы, т.к. расчеты тяжелые

    def on_start(self):
        self.coingecko_api_status = self.check_coingecko_api()
        self.binance_api_status = self.check_binance_api()

        print(f"CoinGecko API: {'✓' if self.coingecko_api_status else '✗'}")
        print(f"Binance API: {'✓' if self.binance_api_status else '✗'}")

    def check_coingecko_api(self):
        """Проверка доступности CoinGecko API"""
        try:
            response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

    def check_binance_api(self):
        """Проверка доступности Binance API"""
        try:
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

    @task
    def test_coingecko_with_fallback(self):
        """Тест CoinGecko с fallback на кэш если API недоступно"""

        if not self.coingecko_api_status:
            print("CoinGecko API недоступно, используем упрощенный запрос")
            # Упрощенный запрос без тяжелых вычислений
            data = {
                "cryptos": ["bitcoin", "ethereum"],
                "days": "30",
                "timeframe": "1d",
                "currency": "usd"
            }
        else:
            # Полный запрос
            data = {
                "cryptos": random.sample(['bitcoin', 'ethereum', 'binancecoin', 'solana'], 3),
                "days": random.choice(["30", "90"]),
                "timeframe": random.choice(["1d", "1w"]),
                "currency": random.choice(["usd", "eur"])
            }

        with self.client.post("/correlation",
                              data=data,
                              name="POST /correlation (с проверкой API)",
                              catch_response=True) as response:

            if response.status_code == 200:
                # Проверяем, нет ли сообщений об ошибках API
                content = response.text.lower()

                if 'api limit' in content or 'rate limit' in content:
                    print("Обнаружено ограничение API")
                    self.coingecko_api_status = False
                    time.sleep(30)  # Ждем перед следующим запросом

                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task
    def test_binance_with_timeout(self):
        """Тест Binance с обработкой таймаутов"""

        if not self.binance_api_status:
            # Пропускаем запрос если Binance недоступен
            return

        # Binance запросы могут быть долгими
        data = {
            "cryptos": "ETH,BNB,SOL",
            "days": "30",
            "currency": "USDT"
        }

        with self.client.post("/correlation_binance",
                              data=data,
                              timeout=30,  # Увеличиваем таймаут для Binance
                              name="POST /correlation_binance (долгий)",
                              catch_response=True) as response:

            if response.status_code == 200:
                if 'таймаут' in response.text.lower() or 'timeout' in response.text.lower():
                    print("Binance запрос вызвал таймаут")
                response.success()
            elif response.status_code == 504:
                response.failure("Gateway Timeout - Binance API не отвечает")
                self.binance_api_status = False
            else:
                response.failure(f"Status: {response.status_code}")
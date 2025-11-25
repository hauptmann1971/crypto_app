import requests


def get_top_btc_pairs(limit=10):
    """
    Получить топ BTC пар по объему торгов
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/tickers"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            # Фильтруем только BTC пары и сортируем по объему
            btc_tickers = [
                ticker for ticker in data['tickers']
                if ticker.get('base', '').upper() == 'BTC'
            ]

            sorted_tickers = sorted(btc_tickers,
                                    key=lambda x: x.get('volume', 0),
                                    reverse=True)

            print(f"=== ТОП-{limit} BTC ПАР ПО ОБЪЕМУ ===")
            print("-" * 70)

            for i, ticker in enumerate(sorted_tickers[:limit], 1):
                base = ticker.get('base', '').upper()
                target = ticker.get('target', '').upper()
                volume = ticker.get('volume', 0)
                last_price = ticker.get('last', 0)
                exchange = ticker.get('market', {}).get('name', 'N/A')

                print(f"{i:2d}. {base}/{target:8} | "
                      f"Объем: ${volume:12,.2f} | "
                      f"Цена: ${last_price:10.2f} | "
                      f"Биржа: {exchange}")

            return sorted_tickers[:limit]

    except Exception as e:
        print(f"Ошибка: {e}")
        return None


# Получить топ пар
top_pairs = get_top_btc_pairs(15)
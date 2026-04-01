from modules.config import Config


def print_startup_banner() -> None:
    print("\n" + "=" * 60)
    print("CRYPTO CONVERTER APP")
    print("=" * 60)
    print(f"Debug mode: {'ON' if Config.DEBUG else 'OFF'}")
    print(f"Server: http://localhost:{Config.PORT}")
    print(f"Bot: {Config.BOT_USERNAME}")
    print(f"Popular cryptos: {len(Config.POPULAR_CRYPTOS)}")
    print("=" * 60 + "\n")

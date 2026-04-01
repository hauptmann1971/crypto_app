from modules.app_factory import create_app
from modules.app_logging import setup_logging
from modules.bootstrap import bootstrap_application
from modules.config import Config
from modules.runtime import print_startup_banner

setup_logging()

# Валидация конфигурации при старте
Config.validate()

app = create_app()
bootstrap_application()

if __name__ == '__main__':
    print_startup_banner()
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
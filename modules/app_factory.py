from pathlib import Path

from flask import Flask

from modules.config import Config
from modules.routes import register_routes


def create_app() -> Flask:
    """Create and configure Flask application instance."""
    project_root = Path(__file__).resolve().parent.parent
    template_folder = str(project_root / "templates")
    app = Flask(__name__, template_folder=template_folder)
    app.secret_key = Config.SECRET_KEY
    app.config['JSON_SORT_KEYS'] = False
    register_routes(app)
    return app

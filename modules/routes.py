from modules.blueprints import admin_bp, analytics_bp, auth_bp, market_bp


def register_routes(app):
    """Register all project blueprints."""

    app.register_blueprint(market_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(auth_bp)
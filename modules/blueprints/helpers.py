import json
from typing import Any, Dict

from flask import flash, redirect, url_for

from modules.database import get_db_connection_active


def ensure_db_connected():
    """
    Return redirect response when DB is disconnected, otherwise None.
    """
    if get_db_connection_active():
        return None

    flash("Соединение с базой данных отключено", 'error')
    return redirect(url_for('market.index'))


def safe_json_loads(payload: str) -> Dict[str, Any]:
    """Safely decode JSON payload into dict."""
    if not payload:
        return {}

    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}

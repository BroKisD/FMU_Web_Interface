from __future__ import annotations

import atexit
import signal
from typing import Optional

from flask import Flask

from .config import AppConfig, load_config
from .routes import api, startup_cleanup
from .storage import SessionStore


def create_app(config: Optional[AppConfig] = None) -> Flask:
    cfg = config or load_config()
    app = Flask(__name__, static_url_path="", static_folder=str(cfg.static_dir))
    app.config["APP_CONFIG"] = cfg
    app.extensions["session_store"] = SessionStore()
    app.register_blueprint(api)
    return app


def register_cleanup_handlers(app: Flask) -> None:
    cfg: AppConfig = app.config["APP_CONFIG"]
    store: SessionStore = app.extensions["session_store"]

    def cleanup_handler(signum=None, frame=None):
        print("\nCleaning up before shutdown...")
        store.clear(cfg.upload_dir)
        if signum:
            print(f"Received signal {signum}, exiting.")
        raise SystemExit(0)

    atexit.register(lambda: store.clear(cfg.upload_dir))
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    with app.app_context():
        print("Running startup cleanup...")
        startup_cleanup()

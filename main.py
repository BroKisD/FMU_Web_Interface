#!/usr/bin/env python3
from fmu_web import create_app, register_cleanup_handlers


app = create_app()


if __name__ == "__main__":
    register_cleanup_handlers(app)
    cfg = app.config["APP_CONFIG"]
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)

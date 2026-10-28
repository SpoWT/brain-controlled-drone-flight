"""HTTP layer. Loopback only; every state change needs the page token."""

from __future__ import annotations

import ipaddress
import logging
import secrets

from flask import Flask, abort, jsonify, render_template, request

from .session import SessionError
from .studio import Studio, StudioError

LOG = logging.getLogger("bci_game")


def _is_loopback(address: str | None) -> bool:
    try:
        return ipaddress.ip_address(address or "").is_loopback
    except ValueError:
        return False


def create_app(studio: Studio | None = None, *, port: int = 8790) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["TOKEN"] = secrets.token_urlsafe(32)
    app.config["STUDIO"] = studio or Studio()
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost"}
    studio_ = app.config["STUDIO"]

    @app.before_request
    def guard():
        if not _is_loopback(request.remote_addr):
            abort(403, description="Neural Ride only answers this computer.")
        if request.host not in allowed_hosts and not app.testing:
            abort(403, description="Unexpected host header.")
        if request.method != "GET":
            if not secrets.compare_digest(request.headers.get("X-Token", ""), app.config["TOKEN"]):
                abort(403, description="Missing or stale page token. Reload the page.")

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
            "media-src 'self' blob:; connect-src 'self'; font-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    def ok(payload: object = None, status: int = 200):
        return jsonify({"ok": True, "data": payload, "error": None}), status

    @app.errorhandler(StudioError)
    def studio_error(error):
        return jsonify({"ok": False, "data": None, "error": str(error)}), 409

    @app.errorhandler(ValueError)
    @app.errorhandler(SessionError)
    def bad_request(error):
        return jsonify({"ok": False, "data": None, "error": str(error)}), 400

    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({"ok": False, "data": None, "error": error.description}), 403

    @app.errorhandler(Exception)
    def unexpected(error):
        if hasattr(error, "code") and hasattr(error, "description"):
            return jsonify({"ok": False, "data": None, "error": error.description}), error.code
        LOG.exception("unhandled error")
        return jsonify({"ok": False, "data": None, "error": f"{type(error).__name__}: {error}"}), 500

    def body() -> dict:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("send a JSON object")
        return payload

    @app.get("/")
    def index():
        return render_template("index.html", token=app.config["TOKEN"])

    @app.get("/api/status")
    def status():
        return ok(studio_.status())

    @app.post("/api/connect")
    def connect():
        return ok(studio_.connect(str(body().get("mode", ""))))

    @app.post("/api/disconnect")
    def disconnect():
        return ok(studio_.disconnect())

    @app.get("/api/signal")
    def signal():
        return ok(studio_.signal())

    @app.post("/api/ride/start")
    def ride_start():
        payload = body()
        minutes = payload.get("minutes")
        if not isinstance(minutes, int):
            raise ValueError("minutes must be an integer")
        return ok(studio_.start_ride(str(payload.get("kind", "")), minutes))

    @app.post("/api/ride/events")
    def ride_events():
        return ok(studio_.add_events(body().get("events")))

    @app.post("/api/ride/finish")
    def ride_finish():
        return ok(studio_.finish_ride())

    @app.get("/api/live")
    def live():
        return ok(studio_.live_status())

    @app.get("/api/sessions")
    def sessions():
        return ok(studio_.sessions())

    @app.post("/api/sessions/<session_id>/exclude")
    def exclude(session_id: str):
        return ok(studio_.exclude(session_id, bool(body().get("excluded"))))

    @app.post("/api/train")
    def train():
        return ok(studio_.start_training(), 202)

    @app.get("/api/train")
    def training():
        return ok(studio_.training())

    @app.post("/api/stop")
    def stop():
        studio_.stop_everything()
        return ok(studio_.status())

    return app

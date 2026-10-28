"""Dashboard feeds: head and drone heading, turn rate and brain GO (read-only), plus head recentre."""
from __future__ import annotations


def create_dashboard_blueprint(transport, service, head_watch=None):
    from flask import Blueprint, jsonify, request

    blueprint = Blueprint("dashboard_feeds", __name__)

    @blueprint.get("/api/direction")
    def direction():
        state = dict(transport.direction_status())
        state["brain_go"] = bool(service.eeg_go_active())
        state["head_note"] = head_watch.note if head_watch is not None else ""
        return jsonify(state)

    @blueprint.post("/api/head/recenter")
    def recenter():
        """Where the goggles face now becomes straight ahead. Flight page token required."""
        try:
            service.recenter_head(request.headers.get("X-Flight-Token", ""))
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        return jsonify(status="ok")

    return blueprint

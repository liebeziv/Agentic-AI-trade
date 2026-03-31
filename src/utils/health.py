"""Health check HTTP server — exposes /health endpoint on port 8080."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class HealthStatus:
    """Snapshot of the system's current health."""

    status: str  # "ok" | "degraded" | "error"
    uptime_seconds: float
    checks: dict[str, bool] = field(default_factory=dict)


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP request handler — only GET /health and GET /metrics."""

    # Injected by HealthServer before the server is started
    _health_server: "HealthServer"

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (stdlib naming convention)
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self._send(404, "text/plain", b"Not found")

    # ------------------------------------------------------------------
    # /health — JSON response
    # ------------------------------------------------------------------

    def _handle_health(self) -> None:
        hs = self.__class__._health_server
        status: HealthStatus = hs._get_status()

        http_code = 200 if status.status == "ok" else 503
        body = json.dumps(
            {
                "status": status.status,
                "uptime_s": round(status.uptime_seconds, 1),
                "checks": status.checks,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ).encode()
        self._send(http_code, "application/json", body)

    # ------------------------------------------------------------------
    # /metrics — Prometheus-style plain text
    # ------------------------------------------------------------------

    def _handle_metrics(self) -> None:
        hs = self.__class__._health_server
        status: HealthStatus = hs._get_status()

        checks_total = len(status.checks)
        checks_passing = sum(1 for v in status.checks.values() if v)

        lines = [
            f"uptime_seconds {status.uptime_seconds:.1f}",
            f"checks_total {checks_total}",
            f"checks_passing {checks_passing}",
        ]
        body = "\n".join(lines).encode() + b"\n"
        self._send(200, "text/plain; version=0.0.4", body)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
        # Suppress default Apache-style access log; structlog handles it
        pass


class HealthServer:
    """
    Lightweight background HTTP server that exposes /health and /metrics.

    Usage::

        server = HealthServer(port=8080)
        server.register_check("db", lambda: store.is_connected())
        server.start()           # non-blocking
        ...
        server.stop()
    """

    def __init__(self, port: int = 8080) -> None:
        self._port = port
        self._checks: dict[str, Callable[[], bool]] = {}
        self._started_at: float = 0.0
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_check(self, name: str, fn: Callable[[], bool]) -> None:
        """Register a named health-check callable."""
        self._checks[name] = fn

    def start(self) -> None:
        """Start the health-check server in a daemon background thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("HealthServer already running", port=self._port)
            return

        self._started_at = time.monotonic()

        # Build a handler class with the server reference baked in so that
        # the stdlib HTTPServer can instantiate it without arguments.
        handler = type(
            "_BoundHandler",
            (_Handler,),
            {"_health_server": self},
        )

        self._httpd = HTTPServer(("0.0.0.0", self._port), handler)

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="health-server",
            daemon=True,
        )
        self._thread.start()
        log.info("HealthServer started", port=self._port)

    def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
            log.info("HealthServer stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_status(self) -> HealthStatus:
        """Evaluate all registered checks and build a HealthStatus."""
        results: dict[str, bool] = {}
        for name, fn in self._checks.items():
            try:
                results[name] = bool(fn())
            except Exception as exc:
                log.warning("Health check raised", check=name, error=str(exc))
                results[name] = False

        all_ok = all(results.values()) if results else True
        status_str = "ok" if all_ok else "degraded"
        uptime = time.monotonic() - self._started_at if self._started_at else 0.0

        return HealthStatus(
            status=status_str,
            uptime_seconds=uptime,
            checks=results,
        )

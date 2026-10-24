#!/usr/bin/env python3
"""OpenAI-compatible mock backend used by the LodeRouter end-to-end routing test.

One instance impersonates one routing level (easy / middle / hard). It answers
GET /v1/models so LodeRouter's startup probe succeeds, and POST
/v1/chat/completions with a JSON payload that names its level in three places:

  * the assistant content prefix, e.g. "[middle] ..."
  * the response "model" field (the model LodeRouter selected for that backend)
  * an extra "mock_backend" field (LodeRouter streams the backend body unchanged)

so a test can prove which mock backend actually served a routed request.

Standalone:
    python3 tools/router_e2e/mock_backend.py --level easy [--model mock-easy] [--port 0]

It prints one JSON line with the bound port and then serves until SIGINT/SIGTERM.

Programmatic use (what run_e2e.py does):
    backend = MockBackend("easy", model="mock-easy")
    backend.start()
    ... backend.base_url ...
    backend.stop()
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LEVELS = ("easy", "middle", "hard")


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockLodeRouterBackend/1.0"
    protocol_version = "HTTP/1.1"

    # Keep stdout/stderr clean: run_e2e.py is a test driver, not an access log.
    def log_message(self, fmt, *args):  # noqa: D401 - stdlib hook
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/models":
            self._send_json(
                404,
                {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}},
            )
            return
        serve = self.server
        self._send_json(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": serve.model,
                        "object": "model",
                        "created": 0,
                        "owned_by": f"mock-{serve.level}-backend",
                    }
                ],
            },
        )

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/chat/completions":
            self._send_json(
                404,
                {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}},
            )
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            request = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_json(
                400,
                {"error": {"message": f"invalid JSON body: {exc}", "type": "invalid_request_error"}},
            )
            return

        serve = self.server
        messages = request.get("messages")
        messages = messages if isinstance(messages, list) else []
        last_user = ""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    last_user = content
                    break

        record = {
            "received_at": time.time(),
            "path": self.path,
            "model": request.get("model"),
            "authorization": self.headers.get("Authorization"),
            "content_type": self.headers.get("Content-Type"),
            "messages_count": len(messages),
            "last_user_content": last_user,
        }
        with serve.lock:
            serve.requests.append(record)

        model = request.get("model") or serve.model
        content = (
            f"[{serve.level}] mock backend reply: served by the {serve.level} mock "
            f"backend (model={model})."
        )
        payload = {
            "id": f"chatcmpl-mock-{serve.level}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "mock_backend": serve.level,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self._send_json(200, payload)


class MockBackend:
    """A running OpenAI-compatible mock backend for one routing level."""

    def __init__(self, level: str, model: str | None = None, port: int = 0):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.level = level
        self.model = model or f"mock-{level}"
        self._requests: list[dict] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._server.daemon_threads = True
        self._server.level = level
        self._server.model = self.model
        self._server.lock = self._lock
        self._server.requests = self._requests
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "MockBackend":
        if self._thread is not None:
            raise RuntimeError(f"mock backend {self.level} is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name=f"mock-{self.level}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def snapshot_requests(self) -> list[dict]:
        with self._lock:
            return [dict(record) for record in self._requests]

    def request_count(self) -> int:
        with self._lock:
            return len(self._requests)

    def __enter__(self) -> "MockBackend":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible mock backend for LodeRouter e2e tests")
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--model", default=None, help="model id to advertise (default: mock-<level>)")
    parser.add_argument("--port", type=int, default=0, help="0 picks an ephemeral port")
    args = parser.parse_args(argv)

    backend = MockBackend(args.level, model=args.model, port=args.port)
    backend.start()
    print(
        json.dumps(
            {
                "level": backend.level,
                "model": backend.model,
                "port": backend.port,
                "url": backend.base_url,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    stop = threading.Event()

    def _handle_signal(signum, frame):  # noqa: ARG001 - signal handler signature
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        while not stop.is_set():
            stop.wait(0.25)
    finally:
        backend.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

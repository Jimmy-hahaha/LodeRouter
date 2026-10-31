#!/usr/bin/env python3
"""End-to-end routing test for LodeRouter + the laya C++ judge.

Starts three OpenAI-compatible mock backends (one per level), a temporary
config that routes easy/middle/hard to them, launches ./build/LodeRouter with
the real laya judge, and asserts that representative chitchat / math / code
requests are forwarded to the expected level.

Usage (from the repository root):
    python3 tools/router_e2e/run_e2e.py
    python3 tools/router_e2e/run_e2e.py --binary build/LodeRouter --json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from mock_backend import MockBackend  # noqa: E402

LEVELS = ("easy", "middle", "hard")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, {"_raw": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"_raw": body}


def wait_until_ready(base_url: str, proc: subprocess.Popen, deadline: float) -> dict:
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"LodeRouter exited with code {proc.returncode}")
        try:
            status, body = http_json(base_url + "/v1/models", timeout=2.0)
            if status == 200 and isinstance(body.get("data"), list):
                return body
            last_error = f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - readiness probe
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"LodeRouter did not become ready: {last_error}")


def user_turn(content: str) -> dict:
    return {"role": "user", "content": content}


CASES = [
    {
        "name": "chitchat",
        "expected": "easy",
        "messages": [user_turn("你好呀，今天心情不错，陪我聊聊天吧")],
    },
    {
        "name": "math",
        "expected": "middle",
        "messages": [user_turn("帮我算一下 17×23 等于多少，要有步骤")],
    },
    {
        "name": "code",
        "expected": "hard",
        "messages": [user_turn("用 Python 写一个快速排序，并给出单元测试")],
    },
    {
        "name": "code-debug-history",
        "expected": "hard",
        "messages": [
            user_turn("我的 React 组件重复渲染，怎么定位？"),
            {"role": "assistant", "content": "先检查依赖数组和 setState 的调用位置。"},
            user_turn("按你说的加了 useMemo 还是不行，报错 Maximum update depth exceeded"),
        ],
    },
    {
        "name": "chitchat-short",
        "expected": "easy",
        "messages": [user_turn("最近压力有点大，陪我聊会儿")],
    },
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO / "build" / "LodeRouter"))
    parser.add_argument("--model-dir", default=str(REPO / "laya"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument("--keep-logs", action="store_true", help="keep the router log file")
    args = parser.parse_args(argv)

    binary = Path(args.binary)
    model_dir = Path(args.model_dir)
    if not binary.is_file():
        print(f"run_e2e: missing router binary: {binary}", file=sys.stderr)
        return 2
    if not (model_dir / "model.onnx").is_file():
        print(f"run_e2e: missing model: {model_dir / 'model.onnx'}", file=sys.stderr)
        return 2

    backends = {level: MockBackend(level).start() for level in LEVELS}
    router_port = free_port()
    router_base = f"http://127.0.0.1:{router_port}"
    log_file = tempfile.NamedTemporaryFile(
        prefix="loderouter-e2e-", suffix=".log", delete=not args.keep_logs
    )
    config = tempfile.NamedTemporaryFile(
        prefix="loderouter-e2e-", suffix=".json", mode="w", encoding="utf-8", delete=True
    )
    proc = None
    report = {"cases": [], "passed": False}
    try:
        json.dump(
            {
                "host": "127.0.0.1",
                "port": router_port,
                "judge_backend": "laya",
                "judge_model_dir": str(model_dir.resolve()),
                "judge_auto_download": False,
                "backends": {
                    level: {"url": backends[level].base_url, "model": backends[level].model}
                    for level in LEVELS
                },
            },
            config,
            ensure_ascii=False,
        )
        config.flush()
        env = dict(os.environ)
        env["LODEROUTER_CONFIG"] = config.name
        proc = subprocess.Popen(
            [str(binary)],
            cwd=str(REPO),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        models = wait_until_ready(router_base, proc, time.time() + args.timeout)

        for case in CASES:
            status, body = http_json(
                router_base + "/v1/chat/completions",
                {
                    "model": "auto",
                    "messages": case["messages"],
                    "stream": False,
                },
            )
            served = body.get("mock_backend") if isinstance(body, dict) else None
            content = ""
            if isinstance(body, dict):
                try:
                    content = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    content = ""
            ok = status == 200 and served == case["expected"] and f"[{case['expected']}]" in content
            report["cases"].append(
                {
                    "name": case["name"],
                    "expected": case["expected"],
                    "served_by": served,
                    "status": status,
                    "ok": ok,
                }
            )
            print(
                f"{'PASS' if ok else 'FAIL'} {case['name']}: expected {case['expected']}, "
                f"served_by={served}, status={status}"
            )

        counts = {level: backends[level].request_count() for level in LEVELS}
        expected_counts = {}
        for case in CASES:
            expected_counts[case["expected"]] = expected_counts.get(case["expected"], 0) + 1
        counts_ok = all(counts[level] == expected_counts.get(level, 0) for level in LEVELS)
        report["backend_requests"] = counts
        report["advertised_models"] = [m.get("id") for m in models.get("data", [])]
        report["passed"] = counts_ok and all(c["ok"] for c in report["cases"])
        print(f"backend requests: {counts}")
        if not counts_ok:
            print(f"FAIL backend request counts: expected {expected_counts}", file=sys.stderr)
            report["passed"] = False
        return 0 if report["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - test driver reports any failure
        print(f"run_e2e: {type(exc).__name__}: {exc}", file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for backend in backends.values():
            backend.stop()
        if not args.keep_logs:
            try:
                report["router_log_tail"] = Path(log_file.name).read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:]
            except OSError:
                pass
        config.close()
        log_file.close()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

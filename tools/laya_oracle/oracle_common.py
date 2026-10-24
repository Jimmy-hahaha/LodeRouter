"""Shared constants and helpers for tools/laya_oracle.

Stdlib only on purpose: the C++ parity orchestrator (laya/cpp/tests/engine_parity.py)
runs under the system Python and must import this without numpy/onnxruntime/torch.
Model-specific code lives in dump_reference.py, which runs under the project venv.

Layout (all paths are absolute, independent of the current directory):

    tools/laya_oracle/                 this toolkit
      oracle_common.py                 shared constants/helpers (this file)
      dump_reference.py                Python oracle generator (venv python)
      compare_engine.py                C++ output vs oracle comparator
      reference/
        tokenizer_corpus.jsonl         HF tokenizer corpus
        eval_reference.jsonl           330-row reference (logits/probs/ids)
        baseline_report.json           accuracy/confusion/confidence report
        parity_cases.jsonl             generated CLI input cases (by engine_parity.py)
        engine_parity_report.json      generated e2e report (by engine_parity.py)

Run the generator from the repository root with the project venv:

    .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]
REFERENCE_DIR = TOOLS_DIR / "reference"
MODEL_DIR = REPO_ROOT / "laya"
EVAL_JSONL = REPO_ROOT / "finetune" / "data" / "eval.jsonl"
TRAIN_JSONL = REPO_ROOT / "finetune" / "data" / "train.jsonl"
QUESTIONS_JSON = MODEL_DIR / "questions.json"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
ONNX_PATH = MODEL_DIR / "model.onnx"
CONFIG_PATH = MODEL_DIR / "rl_agent_config.json"
VENV_PYTHON = REPO_ROOT / ".cache" / "laya-ft" / "venv" / "bin" / "python"

TOKENIZER_CORPUS_PATH = REFERENCE_DIR / "tokenizer_corpus.jsonl"
EVAL_REFERENCE_PATH = REFERENCE_DIR / "eval_reference.jsonl"
BASELINE_REPORT_PATH = REFERENCE_DIR / "baseline_report.json"
PARITY_CASES_PATH = REFERENCE_DIR / "parity_cases.jsonl"
ENGINE_OUTPUT_PATH = REFERENCE_DIR / "engine_output.jsonl"
PARITY_REPORT_PATH = REFERENCE_DIR / "engine_parity_report.json"

DUMP_SCRIPT = TOOLS_DIR / "dump_reference.py"
COMPARE_SCRIPT = TOOLS_DIR / "compare_engine.py"

# The judge's question, exactly as shipped in laya/questions.json and
# finetune/build_dataset.py.  Kept here only as a convenience; the reference
# generator reads laya/questions.json as the source of truth.
LABELS = ("chitchat", "math", "code")
QTYPE_SCORE = 1


def now_iso() -> str:
    """Local timezone timestamp, second resolution, e.g. 2026-10-24T21:30:00+08:00."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GiB"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    """Read a JSONL file and return a list of objects."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    """Write JSONL atomically (same-directory temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path, obj) -> None:
    """Write pretty JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def git_commit(repo=REPO_ROOT):
    """HEAD commit of the workspace, or None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load_questions(path=QUESTIONS_JSON):
    """Return the shipped judge question definition ({type, instructions, criteria})."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["category"]


def instruction_and_levels(path=QUESTIONS_JSON):
    q = load_questions(path)
    return q["instructions"], list(q["criteria"])


def load_reference(path=EVAL_REFERENCE_PATH):
    """Load eval_reference.jsonl keyed by the reference row id (e.g. 'eval:0001')."""
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str):
            raise ValueError(f"{path}: reference row without a string id: {row.get('id')!r}")
        if rid in out:
            raise ValueError(f"{path}: duplicate reference id {rid!r}")
        out[rid] = row
    return out


def file_info(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"path": str(path.relative_to(REPO_ROOT)), "exists": False}
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

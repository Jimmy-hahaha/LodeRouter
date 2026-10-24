#!/usr/bin/env python3
"""Run the tokenizer parity test (C++ vs Hugging Face reference corpus).

The corpus is produced by the project venv:

    .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py --tokenizer-only

Usage (from anywhere):
    python3 laya/cpp/tests/tokenizer_parity.py [--binary PATH] [--corpus PATH]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # laya/cpp/tests
REPO = HERE.parents[2]                          # repository root
DEFAULT_CORPUS = REPO / "tools" / "laya_oracle" / "reference" / "tokenizer_corpus.jsonl"
DEFAULT_BINARY_CANDIDATES = (
    REPO / "build-laya" / "laya-tokenizer-parity",
    REPO / "build" / "laya" / "laya-tokenizer-parity",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--tokenizer", default=str(REPO / "laya" / "tokenizer.json"))
    args = parser.parse_args(argv)

    binary = Path(args.binary) if args.binary else next(
        (path for path in DEFAULT_BINARY_CANDIDATES if path.is_file()), None
    )
    if binary is None or not binary.is_file():
        print(
            "tokenizer_parity: binary not found. Build it first:\n"
            "  cmake -S laya/cpp -B build-laya -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build-laya --target laya-tokenizer-parity",
            file=sys.stderr,
        )
        return 2
    corpus = Path(args.corpus)
    if not corpus.is_file():
        print(
            f"tokenizer_parity: missing corpus {corpus}. Generate it with:\n"
            "  .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py --tokenizer-only",
            file=sys.stderr,
        )
        return 2
    completed = subprocess.run(
        [str(binary), args.tokenizer, str(corpus)],
        text=True,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

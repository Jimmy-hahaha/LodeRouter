#!/usr/bin/env python3
"""End-to-end parity check: C++ laya engine vs the Python reference oracle.

Steps:
  1. read the 330-row reference (tools/laya_oracle/reference/eval_reference.jsonl);
  2. write laya-cli batch cases;
  3. run  laya-cli batch --model laya --input cases --output out --dump-sequence;
  4. verify token ids/sequences exactly, predictions/accuracy, and (in strict
     mode) probabilities and logits within 1e-4.

ORT note: this Int8 graph's optimized kernels changed numerically between
ONNX Runtime 1.29 and 1.31. Exact probability/logit parity requires the same
ORT runtime as the oracle (1.31.x). The default "auto" mode therefore checks
sequences and accuracy, and reports numeric drift; use --strict for exact
numeric parity.

The reference is produced (once) by:
    .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py

Usage (from the repository root):
    python3 laya/cpp/tests/engine_parity.py
    python3 laya/cpp/tests/engine_parity.py --strict
    python3 laya/cpp/tests/engine_parity.py --cli build-laya/laya-cli
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # laya/cpp/tests
REPO = HERE.parents[2]                          # repository root
DEFAULT_REFERENCE = REPO / "tools" / "laya_oracle" / "reference" / "eval_reference.jsonl"
DEFAULT_MODEL_DIR = REPO / "laya"
DEFAULT_CLI_CANDIDATES = (
    REPO / "build-laya" / "laya-cli",
    REPO / "build" / "laya-cli",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_cli(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    return next((path for path in DEFAULT_CLI_CANDIDATES if path.is_file()), None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default=None, help="path to laya-cli")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--strict", action="store_true",
                        help="require probabilities/logits within 1e-4 (same ORT version)")
    parser.add_argument("--prob-tol", type=float, default=1e-4)
    parser.add_argument("--logit-tol", type=float, default=1e-4)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    args = parser.parse_args(argv)

    reference_path = Path(args.reference)
    if not reference_path.is_file():
        print(
            f"engine_parity: missing reference {reference_path}. Generate it with:\n"
            "  .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py",
            file=sys.stderr,
        )
        return 2
    cli = find_cli(args.cli)
    if cli is None:
        print(
            "engine_parity: laya-cli not found. Build it first:\n"
            "  cmake -S laya/cpp -B build-laya -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build-laya --target laya-cli",
            file=sys.stderr,
        )
        return 2

    reference = read_jsonl(reference_path)
    if not reference:
        print("engine_parity: reference is empty", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="laya-parity-") as tmp:
        tmp_path = Path(tmp)
        cases_path = tmp_path / "cases.jsonl"
        with cases_path.open("w", encoding="utf-8") as handle:
            for row in reference:
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "state": row["state"],
                            "instruction": row["instruction"],
                            "levels": row["levels"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        output_path = tmp_path / "engine.jsonl"
        command = [
            str(cli),
            "batch",
            "--model",
            args.model,
            "--input",
            str(cases_path),
            "--output",
            str(output_path),
            "--dump-sequence",
        ]
        print("engine_parity: running", " ".join(command))
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print(f"engine_parity: laya-cli failed ({completed.returncode})", file=sys.stderr)
            return 1
        engine = read_jsonl(output_path)

    by_id = {row["id"]: row for row in engine}
    missing = [row["id"] for row in reference if row["id"] not in by_id]
    if missing:
        print(f"engine_parity: {len(missing)} missing rows, e.g. {missing[:5]}", file=sys.stderr)
        return 1
    extra = [row["id"] for row in engine if row["id"] not in {r["id"] for r in reference}]
    if extra:
        print(f"engine_parity: {len(extra)} unexpected rows, e.g. {extra[:5]}", file=sys.stderr)
        return 1

    sequence_bad = []
    label_bad = []
    accuracy = 0
    reference_accuracy = 0
    flips = []
    prob_diffs = []
    logit_diffs = []
    for row in reference:
        got = by_id[row["id"]]
        for field in ("state_ids", "head_ids", "sequence", "markers"):
            if row.get(field) != got.get(field):
                sequence_bad.append((row["id"], field))
                break
        if row["expected_index"] == got["index"]:
            accuracy += 1
        if got.get("label") != row.get("predicted_label"):
            label_bad.append(row["id"])
        if row["expected_index"] == row["predicted_index"]:
            reference_accuracy += 1
        if got["index"] != row["predicted_index"]:
            flips.append((row["id"], row["predicted_index"], got["index"]))
        if len(row["probabilities"]) == len(got["probabilities"]):
            prob_diffs.append(
                max(abs(a - b) for a, b in zip(row["probabilities"], got["probabilities"]))
            )
        if len(row["logits"]) == len(got["logits"]):
            logit_diffs.append(
                max(abs(a - b) for a, b in zip(row["logits"], got["logits"]))
            )

    total = len(reference)
    accuracy_rate = accuracy / total
    reference_rate = reference_accuracy / total
    max_prob = max(prob_diffs, default=0.0)
    max_logit = max(logit_diffs, default=0.0)

    print(f"rows:              {total}")
    print(f"sequence ids:      {total - len(sequence_bad)}/{total} exact")
    print(f"accuracy:          engine {accuracy}/{total} = {accuracy_rate:.4%}  "
          f"reference {reference_accuracy}/{total} = {reference_rate:.4%}")
    print(f"argmax flips:      {len(flips)}/{total} (reference -> engine)")
    print(f"label mismatches:  {len(label_bad)}/{total}")
    print(f"probabilities:     max|diff|={max_prob:.3e}")
    print(f"logits:            max|diff|={max_logit:.3e}")
    print(f"mean answer conf:  engine {statistics.fmean(g['confidence'] for g in engine):.4f}")

    if sequence_bad:
        for row_id, field in sequence_bad[:10]:
            print(f"  sequence mismatch {row_id}: {field}", file=sys.stderr)
        print("engine_parity: FAIL (input sequence differs from the reference)", file=sys.stderr)
        return 1

    if args.strict:
        if max_prob > args.prob_tol or max_logit > args.logit_tol or flips or label_bad:
            print(
                f"engine_parity: FAIL strict numeric parity "
                f"(prob {max_prob:.3e}/{args.prob_tol}, logit {max_logit:.3e}/{args.logit_tol}, "
                f"{len(flips)} argmax flips, {len(label_bad)} label mismatches). Use the same "
                f"ONNX Runtime version as the oracle (1.31.x) or drop --strict.",
                file=sys.stderr,
            )
            return 1
        print("engine_parity: PASS (strict numeric parity)")
        return 0

    if accuracy_rate + args.max_accuracy_drop < reference_rate:
        print(
            f"engine_parity: FAIL accuracy drop {reference_rate - accuracy_rate:.2%} "
            f"> {args.max_accuracy_drop:.2%}",
            file=sys.stderr,
        )
        return 1
    if max_prob > args.prob_tol or max_logit > args.logit_tol:
        print(
            "engine_parity: WARN numeric drift beyond the strict tolerance "
            f"(prob max {max_prob:.3e}, logit max {max_logit:.3e}); this is expected when the "
            "linked ONNX Runtime differs from the oracle's (1.29 vs 1.31). "
            "Sequences and accuracy still match; use --strict with ONNX Runtime 1.31 for exact parity."
        )
    print("engine_parity: PASS (sequences exact; accuracy within tolerance)")
    if flips:
        print("  argmax flips:")
        for row_id, ref, got in flips[:10]:
            print(f"    {row_id}: reference {ref}, engine {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

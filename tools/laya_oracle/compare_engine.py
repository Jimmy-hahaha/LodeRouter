#!/usr/bin/env python3
"""Compare a C++ laya-cli batch JSONL output against the Python reference oracle.

The gate (laya/cpp/SPEC.md sections 6 and 8) requires, for every eval row:

  * identical tokenizer / sequence ids: state_ids, head_ids, sequence, markers
  * identical argmax index (and therefore identical label)
  * temperature-scaled probabilities within 1e-4 absolute
  * raw logits within 1e-4 absolute
  * score and confidence within 1e-4 absolute

The reference file (tools/laya_oracle/reference/eval_reference.jsonl, produced by
dump_reference.py) is generated one row per ONNX session run.  Do NOT generate it
with a multi-row batch: this int8 graph is batch-composition dependent because
DynamicQuantizeLinear keeps one activation scale per tensor.

CLI contract (implemented by laya/cpp/cli.cpp, owned by the Lead):

    laya-cli batch --model laya --input cases.jsonl --output out.jsonl [--dump-sequence]
    input lines:  {"id","state","instruction","levels":[...]}
    output lines: {"id","index","label","confidence","score","probabilities":[...],
                   "logits":[...],"state_ids":[...],"head_ids":[...],
                   "sequence":[...],"markers":[...]}

Usage (stdlib only; system python works):

    python3 tools/laya_oracle/compare_engine.py \
        --engine-output tools/laya_oracle/reference/engine_output.jsonl
    python3 tools/laya_oracle/compare_engine.py --engine-output out.jsonl \
        --json-report /tmp/compare.json --prob-tol 1e-4 --logit-tol 1e-4

Exit codes: 0 = pass, 1 = comparison failure, 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oracle_common as oc  # noqa: E402

# C++ ScoreResult.confidence is max(p) (the calibrated answer confidence in
# laya.onnx_agent's answer["answer_confidence"]); entropy_confidence is the
# separate 1 - H(p)/log(k) field in laya.h.
C_FIELD_TO_REFERENCE = {
    "confidence": "answer_confidence",
    "score": "score",
}

SEQUENCE_FIELDS = ("state_ids", "head_ids", "sequence", "markers")


class CompareError(RuntimeError):
    """Bad input files (usage/IO class)."""


def _load_rows(path):
    if not Path(path).exists():
        raise CompareError(f"file not found: {path}")
    try:
        rows = oc.read_jsonl(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise CompareError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise CompareError(f"{path}: no JSONL rows")
    by_id = {}
    duplicates = []
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            raise CompareError(f"{path}: row without a non-empty string id: {row!r:.200}")
        if rid in by_id:
            duplicates.append(rid)
        by_id[rid] = row
    if duplicates:
        raise CompareError(f"{path}: duplicate ids: {sorted(set(duplicates))[:10]}")
    return rows, by_id


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int_list(value):
    if not isinstance(value, list):
        return None
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int):
            if _is_number(v) and float(v).is_integer():
                out.append(int(v))
            else:
                return None
        else:
            out.append(int(v))
    return out


def _as_float_list(value):
    if not isinstance(value, list):
        return None
    out = []
    for v in value:
        if not _is_number(v) or not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out


def _fmt_elem_list(values, limit=8):
    if values is None:
        return "None"
    shown = ", ".join(repr(v) for v in values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return "[" + shown + "]"


def _array_first_diff(expected, actual, tol):
    """Return (index, expected_i, actual_i, max_abs_diff) or None if within tol."""
    if len(expected) != len(actual):
        return {"length_mismatch": [len(expected), len(actual)]}
    best = -1
    best_diff = 0.0
    for i, (a, b) in enumerate(zip(expected, actual)):
        d = abs(a - b)
        if d > best_diff:
            best_diff = d
            best = i
        if d > tol:
            pass
    if best_diff <= tol:
        return None
    return {
        "index": best,
        "expected": expected[best],
        "actual": actual[best],
        "abs_diff": best_diff,
        "max_abs_diff": best_diff,
        "expected_window": expected[max(0, best - 2): best + 3],
        "actual_window": actual[max(0, best - 2): best + 3],
    }


def _sequence_mismatch(expected, actual):
    if len(expected) != len(actual):
        return {"kind": "length", "expected_len": len(expected), "actual_len": len(actual)}
    for i, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            return {
                "kind": "element",
                "index": i,
                "expected": a,
                "actual": b,
                "expected_window": expected[max(0, i - 2): i + 3],
                "actual_window": actual[max(0, i - 2): i + 3],
            }
    return None


def compare(reference_path, engine_output_path, prob_tol=1e-4, logit_tol=1e-4,
            value_tol=1e-4, require_sequence=True, allow_extra=False,
            max_details=20, verbose=False):
    """Compare engine output against the reference; return a report dict."""
    ref_rows, ref_by_id = _load_rows(reference_path)
    eng_rows, eng_by_id = _load_rows(engine_output_path)
    ref_ids = [r["id"] for r in ref_rows]

    missing = [rid for rid in ref_ids if rid not in eng_by_id]
    extra = [rid for rid in eng_by_id if rid not in ref_by_id]
    common = [rid for rid in ref_ids if rid in eng_by_id]

    checks = {
        "id": {"reference": len(ref_rows), "engine": len(eng_rows),
               "matched": len(common), "missing": missing, "extra": extra},
        "sequence": {"fields": list(SEQUENCE_FIELDS) if require_sequence else [],
                     "checked": 0, "mismatches": 0, "missing_fields": 0},
        "index": {"checked": 0, "mismatches": 0, "max_abs_diff": 0.0},
        "label": {"checked": 0, "mismatches": 0},
        "probabilities": {"checked": 0, "mismatches": 0, "max_abs_diff": 0.0},
        "logits": {"checked": 0, "mismatches": 0, "max_abs_diff": 0.0},
        "score": {"checked": 0, "mismatches": 0, "max_abs_diff": 0.0},
        "confidence": {"checked": 0, "mismatches": 0, "max_abs_diff": 0.0,
                       "reference_field": "answer_confidence (max p)",
                       "entropy_lookalike": 0},
        "index_vs_probability_argmax": {"checked": 0, "mismatches": 0},
    }
    details = []
    details_overflow = 0

    def add_detail(rid, field, expected, actual, **extra_fields):
        nonlocal details_overflow
        if len(details) < max_details:
            entry = {"id": rid, "field": field, "expected": expected, "actual": actual}
            entry.update(extra_fields)
            details.append(entry)
        else:
            details_overflow += 1

    engine_pred = {}
    for rid in common:
        ref = ref_by_id[rid]
        eng = eng_by_id[rid]
        failed_here = []

        # 1. sequence ids: exact, field by field
        if require_sequence:
            for field in SEQUENCE_FIELDS:
                if field not in eng or field not in ref:
                    checks["sequence"]["missing_fields"] += 1
                    failed_here.append(field)
                    add_detail(rid, f"sequence.{field}", "present", "missing")
                    continue
                exp = _as_int_list(ref[field])
                act = _as_int_list(eng[field])
                if exp is None or act is None:
                    checks["sequence"]["missing_fields"] += 1
                    failed_here.append(field)
                    add_detail(rid, f"sequence.{field}", "integer list", eng.get(field))
                    continue
                checks["sequence"]["checked"] += 1
                diff = _sequence_mismatch(exp, act)
                if diff is not None:
                    checks["sequence"]["mismatches"] += 1
                    failed_here.append(f"sequence.{field}")
                    add_detail(rid, f"sequence.{field}", exp, act, **diff)

        # 2. index (argmax)
        if "index" not in eng:
            checks["index"]["mismatches"] += 1
            failed_here.append("index")
            add_detail(rid, "index", ref["predicted_index"], "missing")
        else:
            act_index = eng.get("index")
            if not _is_number(act_index) or not float(act_index).is_integer():
                checks["index"]["mismatches"] += 1
                failed_here.append("index")
                add_detail(rid, "index", ref["predicted_index"], act_index, reason="not an integer")
            else:
                act_index = int(act_index)
                checks["index"]["checked"] += 1
                if act_index != int(ref["predicted_index"]):
                    checks["index"]["mismatches"] += 1
                    failed_here.append("index")
                    add_detail(rid, "index", int(ref["predicted_index"]), act_index)

        # 3. label
        if "label" in eng:
            checks["label"]["checked"] += 1
            if eng["label"] != ref["predicted_label"]:
                checks["label"]["mismatches"] += 1
                failed_here.append("label")
                add_detail(rid, "label", ref["predicted_label"], eng["label"])
        else:
            checks["label"]["mismatches"] += 1
            failed_here.append("label")
            add_detail(rid, "label", ref["predicted_label"], "missing")

        # 4. probabilities and logits
        for field, tol, check_key in (
            ("probabilities", prob_tol, "probabilities"),
            ("logits", logit_tol, "logits"),
        ):
            exp = _as_float_list(ref.get(field))
            if field not in eng:
                checks[check_key]["mismatches"] += 1
                failed_here.append(field)
                add_detail(rid, field, _fmt_elem_list(exp), "missing")
                continue
            act = _as_float_list(eng.get(field))
            if exp is None or act is None:
                checks[check_key]["mismatches"] += 1
                failed_here.append(field)
                add_detail(rid, field, _fmt_elem_list(exp), eng.get(field))
                continue
            checks[check_key]["checked"] += 1
            diff = _array_first_diff(exp, act, tol)
            if diff is not None:
                checks[check_key]["mismatches"] += 1
                failed_here.append(field)
                if "max_abs_diff" in diff:
                    checks[check_key]["max_abs_diff"] = max(
                        checks[check_key]["max_abs_diff"], diff["max_abs_diff"])
                add_detail(rid, field, _fmt_elem_list(exp), _fmt_elem_list(act),
                           tolerance=tol, **diff)
            else:
                # still track the observed max diff for reporting
                if exp:
                    observed = max(abs(a - b) for a, b in zip(exp, act))
                    checks[check_key]["max_abs_diff"] = max(
                        checks[check_key]["max_abs_diff"], observed)

        # 5. scalar fields: score and confidence
        for field in ("score", "confidence"):
            ref_field = C_FIELD_TO_REFERENCE[field]
            exp = float(ref[ref_field])
            if field not in eng:
                checks[field]["mismatches"] += 1
                failed_here.append(field)
                add_detail(rid, field, exp, "missing")
                continue
            act = eng.get(field)
            if not _is_number(act) or not math.isfinite(float(act)):
                checks[field]["mismatches"] += 1
                failed_here.append(field)
                add_detail(rid, field, exp, act, reason="not a finite number")
                continue
            act = float(act)
            checks[field]["checked"] += 1
            d = abs(act - exp)
            checks[field]["max_abs_diff"] = max(checks[field]["max_abs_diff"], d)
            if d > value_tol:
                checks[field]["mismatches"] += 1
                failed_here.append(field)
                hint = None
                if field == "confidence":
                    entropy = float(ref["confidence_entropy"])
                    if abs(act - entropy) <= value_tol:
                        hint = ("engine confidence equals the entropy confidence "
                                "(1 - H(p)/log k), not max(p); laya.h ScoreResult.confidence "
                                "is max(p) and entropy_confidence is the separate field")
                        checks["confidence"]["entropy_lookalike"] += 1
                add_detail(rid, field, exp, act, abs_diff=d, hint=hint)

        # 6. engine-internal consistency: index must be argmax(probabilities)
        probs = _as_float_list(eng.get("probabilities"))
        idx = eng.get("index")
        if probs is not None and _is_number(idx) and float(idx).is_integer() and probs:
            checks["index_vs_probability_argmax"]["checked"] += 1
            argmax = max(range(len(probs)), key=lambda i: probs[i])
            if int(argmax) != int(idx):
                checks["index_vs_probability_argmax"]["mismatches"] += 1
                failed_here.append("index != argmax(probabilities)")
                add_detail(rid, "index_vs_probability_argmax", int(idx), argmax)

        engine_pred[rid] = int(eng["index"]) if _is_number(eng.get("index")) else None

    # accuracy / confusion comparison
    labels = list(oc.LABELS)
    k = len(labels)

    def confusion_and_accuracy(predicted_by_id):
        matrix = [[0] * k for _ in range(k)]
        correct = 0
        counted = 0
        for rid in common:
            ref = ref_by_id[rid]
            expected = int(ref["expected_index"])
            pred = predicted_by_id.get(rid)
            if pred is None or not 0 <= pred < k:
                continue
            counted += 1
            matrix[expected][pred] += 1
            correct += int(pred == expected)
        return {
            "items": counted,
            "correct": correct,
            "accuracy": (correct / counted) if counted else None,
            "confusion_matrix": matrix,
        }

    ref_acc = confusion_and_accuracy({rid: int(ref_by_id[rid]["predicted_index"]) for rid in common})
    eng_acc = confusion_and_accuracy(engine_pred)
    accuracy_match = (
        ref_acc["items"] == eng_acc["items"]
        and ref_acc["correct"] == eng_acc["correct"]
        and ref_acc["confusion_matrix"] == eng_acc["confusion_matrix"]
    )

    # failure summary
    mismatch_keys = [
        "sequence", "index", "label", "probabilities", "logits", "score", "confidence",
        "index_vs_probability_argmax",
    ]
    failures = []
    if missing:
        failures.append(f"missing engine rows: {len(missing)}")
    if extra and not allow_extra:
        failures.append(f"unexpected extra engine rows: {len(extra)}")
    if checks["sequence"]["missing_fields"]:
        failures.append(f"sequence fields missing: {checks['sequence']['missing_fields']}")
    for key in mismatch_keys:
        if checks[key].get("mismatches"):
            failures.append(f"{key}: {checks[key]['mismatches']} mismatch(es)")
    if not accuracy_match:
        failures.append("accuracy/confusion differs from reference")

    report = {
        "schema": "laya-oracle-engine-compare/1",
        "generated_at": oc.now_iso(),
        "reference": str(reference_path),
        "engine_output": str(engine_output_path),
        "tolerances": {"probabilities": prob_tol, "logits": logit_tol,
                       "value": value_tol, "require_sequence": require_sequence},
        "rows": {"reference": len(ref_rows), "engine": len(eng_rows),
                 "compared": len(common), "missing": missing, "extra": extra},
        "checks": checks,
        "accuracy": {
            "reference": {k2: v2 for k2, v2 in ref_acc.items() if k2 != "confusion_matrix"},
            "engine": {k2: v2 for k2, v2 in eng_acc.items() if k2 != "confusion_matrix"},
            "match": accuracy_match,
            "confusion_matrix": {
                "labels": labels,
                "reference": ref_acc["confusion_matrix"],
                "engine": eng_acc["confusion_matrix"],
                "rows_expected_cols_predicted": True,
            },
        },
        "details": details,
        "details_overflow": details_overflow,
        "failures": failures,
        "pass": not failures,
    }
    return report


def print_report(report, verbose=False):
    rows = report["rows"]
    checks = report["checks"]
    tol = report["tolerances"]
    print(f"reference:    {report['reference']}  ({rows['reference']} rows)")
    print(f"engine:       {report['engine_output']}  ({rows['engine']} rows)")
    print(f"compared rows: {rows['compared']}  missing: {len(rows['missing'])}  "
          f"extra: {len(rows['extra'])}")
    print(f"tolerances:   probabilities {tol['probabilities']:g}, logits {tol['logits']:g}, "
          f"values {tol['value']:g}")

    seq = checks["sequence"]
    if tol["require_sequence"]:
        exact = seq["checked"] - seq["mismatches"]
        print(f"sequence ids: {exact}/{seq['checked']} exact"
              + (f" ({seq['missing_fields']} missing fields)" if seq['missing_fields'] else ""))
    idx = checks["index"]
    print(f"argmax index: {idx['checked'] - idx['mismatches']}/{idx['checked']} match")
    for key in ("probabilities", "logits"):
        c = checks[key]
        print(f"{key:13s} max|diff|={c['max_abs_diff']:.3e}  mismatches={c['mismatches']}/{c['checked']}")
    for key in ("score", "confidence"):
        c = checks[key]
        print(f"{key:13s} max|diff|={c['max_abs_diff']:.3e}  mismatches={c['mismatches']}/{c['checked']}")
    lab = checks["label"]
    print(f"labels:       {lab['checked'] - lab['mismatches']}/{lab['checked']} match")
    ipc = checks["index_vs_probability_argmax"]
    print(f"index==argmax(probabilities): {ipc['checked'] - ipc['mismatches']}/{ipc['checked']}")
    acc = report["accuracy"]
    ref_a = acc["reference"]["accuracy"]
    eng_a = acc["engine"]["accuracy"]
    print(f"accuracy:     reference {acc['reference']['correct']}/{acc['reference']['items']}"
          f" = {ref_a:.4%}   engine {acc['engine']['correct']}/{acc['engine']['items']}"
          f" = {eng_a:.4%}   {'MATCH' if acc['match'] else 'DIFFERS'}")
    cm = acc["confusion_matrix"]
    print(f"confusion:    reference {cm['reference']}")
    print(f"              engine    {cm['engine']}")

    if report["details"]:
        print(f"\nfirst {len(report['details'])} mismatches"
              + (f" (+{report['details_overflow']} more)" if report["details_overflow"] else "") + ":")
        for d in report["details"]:
            print(f"  {d['id']} {d['field']}: expected {str(d['expected'])[:160]}")
            print(f"                   actual   {str(d['actual'])[:160]}")
            for extra_key in ("index", "abs_diff", "tolerance", "hint", "expected_len",
                              "actual_len", "expected_window", "actual_window"):
                if extra_key in d and d[extra_key] not in (None, ""):
                    print(f"                   {extra_key}: {str(d[extra_key])[:160]}")
    if report["failures"]:
        print("\nFAILURES: " + "; ".join(report["failures"]))
    print("\nRESULT: " + ("PASS" if report["pass"] else "FAIL"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare laya-cli batch output with the Python reference oracle.")
    parser.add_argument("--reference", default=str(oc.EVAL_REFERENCE_PATH),
                        help="reference JSONL (default: tools/laya_oracle/reference/eval_reference.jsonl)")
    parser.add_argument("--engine-output", required=True,
                        help="JSONL produced by 'laya-cli batch'")
    parser.add_argument("--prob-tol", type=float, default=1e-4)
    parser.add_argument("--logit-tol", type=float, default=1e-4)
    parser.add_argument("--value-tol", type=float, default=1e-4,
                        help="tolerance for score/confidence scalars")
    parser.add_argument("--no-require-sequence", action="store_true",
                        help="skip exact state_ids/head_ids/sequence/markers checks")
    parser.add_argument("--allow-extra", action="store_true",
                        help="do not fail when the engine emits rows not in the reference")
    parser.add_argument("--json-report", default=None,
                        help="write the machine-readable report to this path")
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = compare(
            reference_path=args.reference,
            engine_output_path=args.engine_output,
            prob_tol=args.prob_tol,
            logit_tol=args.logit_tol,
            value_tol=args.value_tol,
            require_sequence=not args.no_require_sequence,
            allow_extra=args.allow_extra,
            max_details=args.max_details,
            verbose=args.verbose,
        )
    except CompareError as exc:
        print(f"compare_engine.py: {exc}", file=sys.stderr)
        return 2

    print_report(report, verbose=args.verbose)
    if args.json_report:
        oc.write_json(args.json_report, report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

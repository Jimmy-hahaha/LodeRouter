#!/usr/bin/env python3
"""Score an exported three-class judge on eval.jsonl, with the ONNX graph.

This is the closest Python check to what the C++ judge does: the same graph,
the same instruction and the same three score levels.

    python3 finetune/verify_onnx.py \
      --model laya \
      --onnx laya/model.onnx \
      --data finetune/data/eval.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict

from laya.onnx_agent import ONNXAgent

LABELS = ["chitchat", "math", "code"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint dir (tokenizer + config)")
    parser.add_argument("--onnx", required=True, help="exported model.onnx")
    parser.add_argument("--data", default="finetune/data/eval.jsonl")
    args = parser.parse_args()

    agent = ONNXAgent(args.model, onnx_path=args.onnx)
    total = correct = 0
    per_class = defaultdict(lambda: [0, 0])
    confusion = Counter()

    with open(args.data, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            expected = int(row["expected"]["category"])
            result = agent.predict(row["state"], row["questions"])
            answer = result["answers"]["category"]
            predicted = int(round(float(answer["score"])))
            total += 1
            correct += predicted == expected
            per_class[expected][0] += 1
            per_class[expected][1] += predicted == expected
            confusion[(expected, predicted)] += 1

    print(f"accuracy: {correct}/{total} = {correct / max(total, 1):.3f}")
    for idx, label in enumerate(LABELS):
        n, ok = per_class[idx]
        print(f"  {label:8s} {ok}/{n} = {ok / max(n, 1):.3f}")
    print("confusion (expected -> predicted):")
    for (expected, predicted), n in sorted(confusion.items()):
        if expected != predicted:
            print(f"  {LABELS[expected]} -> {LABELS[predicted]}: {n}")


if __name__ == "__main__":
    main()

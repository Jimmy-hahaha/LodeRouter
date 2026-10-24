# Verification report

Restored laya C++ runtime — tokenizer, ONNX session, engine, CLI, tests.
Everything below was run on 2026-10-24 in this workspace
(Linux, GCC 16.2.1, CMake 4.4.4, Python 3.14 venv with transformers /
onnxruntime 1.31.0, system ONNX Runtime 1.29.0).

## 1. Tokenizer parity (exact)

Reference: tools/laya_oracle/reference/tokenizer_corpus.jsonl, produced by
tools/laya_oracle/dump_reference.py with the Hugging Face fast tokenizer.
Corpus: 2,083 texts — all 1,680 dataset states, the real judge prompt texts,
all 249 added tokens alone/embedded, whitespace/newline/tab runs, CJK, emoji,
combining marks, private-use code points, truncation boundaries (48 tokens),
and a randomized Unicode fuzz set.

    ctest --test-dir build-laya -R tokenizer_parity
    # 1/1 Test #3: tokenizer_parity ... Passed 1.07 sec

    # or directly:
    ./build-laya/laya-tokenizer-parity laya/tokenizer.json \
        tools/laya_oracle/reference/tokenizer_corpus.jsonl
    # tokenizer parity: 2083 rows, ids mismatches=0, trunc48 mismatches=0,
    #                   special/trunc mismatches=0, decode mismatches=0

Checks include add_special_tokens=false, right truncation at 48 tokens, and
the HF special-token reservation rule ([bos] + content[:46] + [eos]).

## 2. Sequence + score parity (exact, ONNX Runtime 1.31)

The Python oracle ran all 330 rows of finetune/data/eval.jsonl through the
official pipeline (laya.common.build_sequence + ONNXAgent) with
max_len=512, head_max_len=256, temperature[1]=1.1558657884597778.

The C++ laya-cli ran the same rows one at a time (no padding) and produced:

    compared rows: 330  missing: 0  extra: 0
    sequence ids: 1320/1320 exact        (state_ids/head_ids/sequence/markers)
    argmax index: 330/330 match
    probabilities max|diff| = 1.192e-07  mismatches=0/330
    logits        max|diff| = 0.000e+00  mismatches=0/330
    accuracy:     reference 307/330 = 93.0303%   engine 307/330 = 93.0303%
    confusion:    [[97,5,4],[0,103,8],[1,5,107]] on both sides
    RESULT: PASS

Golden sample "你好，世界": state_ids [30485,235607,235365,9979], head_len 78,
markers [13,35,55], logits [5.777190685,-4.032849312,-1.220180392],
confidence 0.99745148, entropy confidence 0.98317859 — identical to the
Python reference.

Run with the same ONNX Runtime as the oracle (1.31):

    # build a CLI linked to the venv ORT 1.31 (the system package is 1.29)
    mkdir -p .scratch/ort131
    ln -sf $PWD/.cache/laya-ft/venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0 .scratch/ort131/libonnxruntime.so.1
    ln -sf $PWD/.cache/laya-ft/venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0 .scratch/ort131/libonnxruntime.so
    g++ -std=c++20 -O2 -I laya/cpp -I laya/cpp/third_party laya/cpp/cli.cpp \
        -Lbuild-laya -llaya -L.scratch/ort131 -lonnxruntime -pthread \
        -Wl,-rpath,$PWD/.scratch/ort131 -o .scratch/laya-cli131
    python3 laya/cpp/tests/engine_parity.py --cli .scratch/laya-cli131 --strict

## 3. ONNX Runtime 1.29 vs 1.31 (numeric drift, not a rewrite bug)

The model is dynamically int8-quantized. ONNX Runtime's optimized int8 kernels
for this graph changed between 1.29 (system package on this machine) and 1.31
(the oracle's runtime). With the same inputs and exact same token sequences:

    ORT 1.31: logits [5.777190685, -4.032849312, -1.220180392]  (matches Python)
    ORT 1.29: logits [5.952621937, -3.191021442, -2.724203587]

On the 330 eval rows, the system-ORT build gives 93.3333% accuracy
(308/330) versus the reference's 93.0303% (307/330); 7/330 argmax flips,
max probability diff 0.391. Token sequences are still 1320/1320 exact.
Accuracy and routing quality are therefore equivalent for the judge; exact
probability parity requires ONNX Runtime 1.31.x.

`laya/cpp/tests/engine_parity.py` defaults to the pragmatic check (sequences
exact + accuracy within 1 percentage point) and prints the drift warning.
`--strict` requires 1e-4 numeric parity and the same runtime as the oracle.

Note: finetune/train_report records 94.55% held-out accuracy, but the exported
checkpoint committed under laya/ scores 93.03% (Python reference) and 93.33%
(ORT 1.29) on the same 330 rows. The C++ engine reproduces the Python reference
exactly, so this gap predates the rewrite (checkpoint export vs train report).

## 4. C++ test suite

    ctest --test-dir build-laya --output-on-failure
    1/3 engine_smoke ......... Passed  2.00 sec  (loads laya/, scores sample)
    2/3 onnx_session_smoke ... Passed  1.17 sec  (real graph, shapes, errors)
    3/3 tokenizer_parity ..... Passed  1.07 sec  (exact HF ids)
    100% tests passed

`laya-engine-smoke` output:

    loaded laya: max_len=512 head_max_len=256 vocab=256000
    score: label=chitchat: ..., index=0 confidence=0.999085 ...
    state_ids=4 head_ids=78 sequence=83 markers=3

(confidence 0.9991 uses the system ORT 1.29; with 1.31 it is 0.99745 as above.)

## 5. Router end-to-end routing

Mock OpenAI-compatible backends + real LodeRouter process + real laya judge:

    python3 tools/router_e2e/run_e2e.py
    PASS chitchat: expected easy, served_by=easy, status=200
    PASS math: expected middle, served_by=middle, status=200
    PASS code: expected hard, served_by=hard, status=200
    PASS code-debug-history: expected hard, served_by=hard, status=200
    PASS chitchat-short: expected easy, served_by=easy, status=200
    backend requests: {'easy': 2, 'middle': 1, 'hard': 2}

## 6. Build

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    # laya: onnxruntime (package) -> found
    # laya: /home/.../LodeRouter/laya/cpp
    # [100%] Built target LodeRouter

No compiler warnings from the laya sources (-Wall -Wextra -Wpedantic).

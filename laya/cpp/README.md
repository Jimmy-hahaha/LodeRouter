# laya C++ runtime

Pure C++20 inference runtime for the **laya** decision model — a ModernBERT
encoder plus a small decision head exported to ONNX. It reproduces the
Hugging Face reference pipeline (`laya.common.build_sequence` and
`laya.onnx_agent.ONNXAgent`) token-for-token and score-for-score on CPU, with
no Python at runtime.

This project is the in-tree restoration of the lost sibling `laya/cpp`
project. LodeRouter's root `CMakeLists.txt` auto-detects it (a sibling
`../laya/cpp` checkout is still supported via `-DLAYA_DIR=`).

## Model directory

A model directory contains:

| file | required | description |
| --- | --- | --- |
| `model.onnx` | yes | exported laya decision graph |
| `tokenizer.json` | yes | Hugging Face fast-tokenizer file |
| `rl_agent_config.json` | optional | `max_len`, `head_max_len`, `temperature` |
| `tokenizer/` | optional | fallback location for `tokenizer.json` |

The repository's `laya/` directory already has the fine-tuned three-class
judge (chitchat / math / code) — it is about 925 MB.

## Build

Standalone:

    cmake -S laya/cpp -B build-laya -DCMAKE_BUILD_TYPE=Release
    cmake --build build-laya -j
    ./build-laya/laya-cli info --model laya

As part of LodeRouter (the default):

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

Requirements: CMake ≥ 3.20, a C++20 compiler, Threads, and ONNX Runtime
(headers + `libonnxruntime`). The build first tries `find_package(onnxruntime
CONFIG)`, then probes `/usr/include/onnxruntime` + `/usr/lib`. Override the
probe with `-DLAYA_ONNXRUNTIME_ROOT=<prefix>`.

## CLI

    # classify a request into the three judge levels
    ./build-laya/laya-cli score --model laya \
        --state "用 Python 写一个快速排序" \
        --instruction "Which category does this request belong to?" \
        --levels "chitchat: casual conversation, greetings, small talk, jokes, feelings, opinions;math: mathematics, arithmetic, algebra, geometry, calculus, probability, proofs;code: programming, debugging, refactoring, software design, APIs, databases, tooling"

    # inspect the model and tokenizer
    ./build-laya/laya-cli info --model laya

    # tokenize a string
    ./build-laya/laya-cli encode --model laya --text "你好，世界" --json

    # batch JSONL (fields: id, state, instruction, levels)
    ./build-laya/laya-cli batch --model laya --input cases.jsonl \
        --output predictions.jsonl --dump-sequence

`--json` prints machine-readable output. `batch --dump-sequence` also returns
`state_ids`, `head_ids`, `sequence`, `markers` and raw `logits`, which is what
the parity harness compares against the Python oracle.

## API

    #include "laya.h"

    laya::Engine engine("laya");              // throws std::runtime_error
    auto result = engine.score(
        "帮我算一下 17×23",
        "Which category does this request belong to?",
        { "chitchat: ...", "math: ...", "code: ..." });

    result.index;         // argmax level, 0..levels.size()-1
    result.label;         // levels[result.index]
    result.confidence;    // max(probabilities), temperature calibrated
    result.probabilities; // softmax over the levels
    result.score;         // expected level, sum(i * p_i)

`Engine` is thread-safe (inference is serialized internally). Model loading is
heavy: ~925 MB of weights, several seconds and roughly 1–2 GB of RAM. Reuse one
`Engine` per process rather than constructing it per request.

Additional question types:

    engine.choice(state, instruction, {"option a", "option b"});
    engine.noul(state, instruction);          // boolean decision

For debugging and parity tests, `score(..., &trace)` fills a `ScoreTrace` with
the exact token ids, marker positions and raw logits.

## Tokenizer

`tokenizer.cpp` implements the Hugging Face `tokenizer.json` pipeline used by
the laya checkpoints directly:

* added-token extraction (longest match, `lstrip`/`rstrip`),
* `Replace(" " -> "▁")` normalizer,
* `Metaspace(prepend_scheme=always, split=true)` pre-tokenizer,
* BPE with byte fallback (`<0xNN>`), 256 000-token vocabulary,
* `<bos>` / `<eos>` template processing when requested,
* Replace + ByteFallback + Fuse decoder, and right truncation.

It uses a streaming JSON parse of the 34 MB tokenizer file and keeps the
vocabulary/merge tables in flat hash maps.

## Verification

The runtime is verified against the Python reference (full evidence and
numbers: [VERIFICATION.md](VERIFICATION.md)):

* `tests/tokenizer_parity.py` / `tests/tokenizer_parity.cpp` — exact id
  equality on 2,083 texts: all 1,680 dataset states, the real prompts, all
  added tokens, whitespace/newline runs, abnormal Unicode and 48-token
  truncation boundaries. Decoder round-trips (including added tokens, byte
  fallback and skip_special_tokens) are checked as well.
* `tests/engine_parity.py` + `tools/laya_oracle/` — all 330 held-out eval rows,
  comparing token ids/sequences exactly, argmax/accuracy, probabilities and
  logits against the Python `ONNXAgent` baseline.
* `tests/engine_smoke.cpp`, `tests/onnx_session_smoke.cpp` — C++ smoke tests.
* `tools/router_e2e/run_e2e.py` — full LodeRouter process against mock
  OpenAI-compatible backends, asserting chitchat/math/code routing.

Build the standalone tests and run them from the repository root:

    cmake -S laya/cpp -B build-laya -DCMAKE_BUILD_TYPE=Release
    cmake --build build-laya -j
    ctest --test-dir build-laya --output-on-failure

    python3 laya/cpp/tests/tokenizer_parity.py
    python3 laya/cpp/tests/engine_parity.py            # sequence + accuracy
    python3 laya/cpp/tests/engine_parity.py --strict   # requires ORT 1.31

### ONNX Runtime version and numeric parity

The model is dynamically int8-quantized. ONNX Runtime's optimized int8 kernels
changed between 1.29 (common system package) and 1.31 (the version used to
build the Python oracle). The C++ engine reproduces the reference exactly when
linked against the same runtime (330/330 eval rows, max probability difference
1.2e-7); with 1.29, token sequences and accuracy still match, but individual
probabilities/logits differ (93.33% vs 93.03% eval accuracy, 7/330 argmax
flips). This is a runtime kernel difference, not a rewrite difference. Use
`engine_parity.py --strict` with ONNX Runtime 1.31 for bit-level verification.

## Dependencies

* ONNX Runtime (system installation, C++ API)
* Threads
* nlohmann/json (vendored at `third_party/nlohmann/json.hpp`, MIT)

No Python, no network access, and no other runtime dependency.

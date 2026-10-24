# laya C++ runtime — design and parity spec

This document describes what the C++ runtime reproduces, the reference
pipeline semantics, and the parity tests that gate completion. See
VERIFICATION.md for the measured results and README.md for usage.

## 1. What we are rebuilding

The LodeRouter judge calls a local laya model through a small C++ library:

    laya::Engine engine(model_dir);
    auto r = engine.score(state, instruction, levels);

The original sibling project (../laya/cpp: CMakeLists.txt, laya.h, laya.cpp) was
lost. The model artifacts still exist at ../laya (from this repo: laya/model.onnx,
laya/tokenizer.json, laya/rl_agent_config.json, ...). Because the sandbox cannot
write outside the repo, the rebuilt project lives **in-tree** at laya/cpp and the
root CMakeLists auto-detects it (a sibling path is still supported).

Pure C++ (C++20) at runtime: no Python, no shelling out. ONNX Runtime C++ API is
used for inference. The only vendored dependency is nlohmann/json (header-only,
third_party/nlohmann/json.hpp) for JSON parsing.

## 2. Environment

- Workspace: the LodeRouter repository root
- Model dir: laya/ (model.onnx 925 MB, tokenizer.json 34 MB, rl_agent_config.json)
- Python reference venv: .cache/laya-ft/venv/bin/python
  - has transformers, onnxruntime 1.31.0, and the laya Python package
    (.cache/laya-ft/venv/lib/python3.14/site-packages/laya)
- System ONNX Runtime: /usr/include/onnxruntime/onnxruntime_cxx_api.h,
  /usr/lib/libonnxruntime.so (1.29.0), CMake package onnxruntime::onnxruntime.
- Compiler: GCC 16, CMake 4.4. System has ~8 GB free RAM; model load is heavy.

Reference files to read:
- laya/rl_agent_config.json (max_len 512, head_max_len 256, temperature list)
- laya/questions.json (score question: 3 levels)
- .cache/laya-ft/venv/lib/python3.14/site-packages/laya/common.py
  (serialize_state, render_criterion, render_options, build_head,
   build_sequence, confidence_from_probs, answer_confidence, temp_bucket)
- .cache/laya-ft/venv/lib/python3.14/site-packages/laya/onnx_agent.py
  (_encode_state, _decode_answers, _collate_items_numpy)
- finetune/build_dataset.py (training rows / instruction text)
- finetune/data/{train,eval}.jsonl (1,350 / 330 rows)

## 3. Layout

- laya/cpp/laya.h            public API: Engine, ScoreResult/ChoiceResult/NoulResult
- laya/cpp/laya.cpp          Engine: config, prompt assembly, batching, decoding
- laya/cpp/tokenizer.h/.cpp  Hugging Face tokenizer.json pipeline in pure C++
- laya/cpp/onnx_session.h/.cpp  ONNX Runtime C++ session wrapper
- laya/cpp/cli.cpp           laya-cli (score / choice / noul / encode / batch / info)
- laya/cpp/CMakeLists.txt    standalone build, CLI, CTest targets
- laya/cpp/tests/            smoke tests + parity drivers
- laya/cpp/third_party/      vendored nlohmann/json (MIT)
- laya/cpp/README.md         usage
- laya/cpp/VERIFICATION.md   measured parity/accuracy/build results
- tools/laya_oracle/         Python reference oracle and comparator
- tools/router_e2e/          mock backends + LodeRouter end-to-end routing test
- CMakeLists.txt (root)      LodeRouter build; auto-detects laya/cpp

## 4. Tokenizer specification (must match HF exactly)

tokenizer.json pipeline:
- normalizer: Replace(pattern=" ", content="\u2581")  (U+2581)
- pre_tokenizer: Metaspace(replacement="\u2581", prepend_scheme="always", split=true)
- model: BPE, vocab_size=256000, 580,604 merges, byte_fallback=true,
  no continuing_subword_prefix, no end_of_word_suffix, ignore_merges=false
- post_processor: TemplateProcessing <bos> A <eos> (only when add_special_tokens)
- decoder: Replace("\u2581" -> " "), ByteFallback, Fuse
- added_tokens: 249 entries, special flag and lstrip/rstrip flags; they are
  matched in the raw input before normalization. e.g. "<mask>" -> id 4,
  "<bos>" -> id 2, "<pad>" -> id 0. Each non-added segment is normalized and
  metaspace-prefixed independently (verified: "a<bos>b" -> [476, 2, 518]).
- Special ids: pad=0, eos=1, bos=2, unk=3, mask=4, cls=2, sep=1.

Encoding algorithm (add_special_tokens=false):
1. Scan raw UTF-8 text for added-token matches (longest match; honour lstrip /
   rstrip); emit their ids directly. For each plain segment:
   a. normalize: replace every ASCII space 0x20 with U+2581.
   b. metaspace: prepend U+2581 when the segment is non-empty and does not
      already start with U+2581; split into words at U+2581 boundaries, each
      word keeping its leading U+2581 (e.g. "\u2581\u2581leading\u2581spaces"
      -> ["\u2581", "\u2581leading", "\u2581spaces"]).
   c. BPE each word: start from Unicode code points; characters absent from
      vocab fall back to <0xNN> byte tokens; repeatedly merge the adjacent pair
      with the lowest merge rank until none. (Match HF tokenizers exactly.)
2. If add_special_tokens: prepend bos id 2, append eos id 1.
3. If max_length > 0: right-truncate ids to max_length.

Python oracle commands (run from repo root):
    .cache/laya-ft/venv/bin/python - <<'PY'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('laya/tokenizer', local_files_only=True)
    print(tok('你好，世界', add_special_tokens=False)['input_ids'])
    print(tok('a<bos>b', add_special_tokens=False)['input_ids'])
    print(tok('  leading spaces', add_special_tokens=False)['input_ids'])
    print(tok('<mask>hi', add_special_tokens=False)['input_ids'])
    PY

Known golden values (add_special_tokens=false):
- "hello world" -> [25612, 2134]
- "你好，世界" -> [30485, 235607, 235365, 9979]
- "a<bos>b" -> [476, 2, 518]
- "<mask>hi" -> [4, 5827]
- "  leading spaces" -> [235248, 8133, 14130]
- "a\tb\nc" -> [476, 226, 235268, 108, 498]
- "x\ue000y" -> [1141, 240539, 235267]
- "\x00" -> [235248, 217]

Parity requirement for the tokenizer: exact id equality on
(a) all 1,680 states from finetune/data/*.jsonl,
(b) all three score instructions/levels (prompt texts),
(c) a generated adversarial corpus: empty strings, whitespace runs, tabs/newlines,
    CJK, emoji, combining marks, surrogate-adjacent/rare code points, added tokens
    in the middle of words, long strings, and truncation at 48 tokens.
The script laya/cpp/tests/tokenizer_parity.py must run and report 0 mismatches.

## 5. Sequence construction (Lead, in laya.cpp)

This is laya.common.build_sequence for the internal question dict:
q = {"t": "score", "ins": instruction, "crit": [levels...]}.

render_options(score) = ["level 0: <level0>", "level 1: <level1>", ...]
render_criterion(string) = the string unchanged.

build_head(tok, q, head_max_len):
  mask = "<mask>"
  head_ids = encode("%s question: %s" % (q.t, instruction.replace(mask, " ")),
                    add_special_tokens=false)
  for each option i:
      opt_ids[i] = [mask_id] + encode(" " + option_i.replace(mask, " "),
                                      add_special_tokens=false, max_length=48)
  opt_budget = head_max_len - sum(len(o) for o in opt_ids)
  if opt_budget < 16:
      per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
      opt_ids = [o[:per] for o in opt_ids]
      opt_budget = head_max_len - sum(len(o) for o in opt_ids)
  head_ids = head_ids[:max(8, opt_budget)]
  ids = [cls] + head_ids + [sep]
  markers[i] = len(ids); ids.extend(opt_ids[i])
  ids.append(sep)

build_sequence(tok, state, q, max_len, head_max_len):
  ids, markers = build_head(...)
  room = max(0, max_len - len(ids) - 1)
  state_ids = encode(state.replace(mask, " "), add_special_tokens=false)
  ids = ids + state_ids[:room] + [sep]
  markers = [m for m in markers if m < max_len]
  ids = ids[:max_len]

For the fine-tuned judge: max_len=512, head_max_len=256,
instruction="Which category does this request belong to?",
levels = ["chitchat: casual conversation, greetings, small talk, jokes, feelings, opinions",
          "math: mathematics, arithmetic, algebra, geometry, calculus, probability, proofs",
          "code: programming, debugging, refactoring, software design, APIs, databases, tooling"],
state = "你好，世界" gives:
  head len 78, markers [13, 35, 55], state_ids [30485, 235607, 235365, 9979],
  total seq len 83.

## 6. ONNX graph and decoding

Inputs (int64 unless noted): input_ids [B,L], attention_mask [B,L],
marker_pos [B,K], marker_mask [B,K] bool, qtype [B].
Outputs: logits float [B,K], act_logits float [B,2] (act unused for our API).

qtype: choice=0, score=1, noul=2.
temperature: array from rl_agent_config.json, clamped to [0.5, 5.0]; index by
qtype. For the judge score question use temperature[1] = 1.1558657884597778.

Decode a score/choice row:
  z = logits[row, :k] / temperature[qtype]
  p = softmax(z)
  index = argmax(p); confidence = max(p); score = sum(i*p[i])
  entropy_confidence = 1 - H(p)/log(k) for k>=2 else 1.0
  label = levels/options[index]

Golden sample "你好，世界" (score, 3 options):
  logits = [5.7771907, -4.0328493, -1.2201804]
  p      = [0.99745148, 0.00020557, 0.00234295]
  index  = 0, score = 0.0049, confidence = 0.9974515,
  entropy_confidence = 0.9831786, label = levels[0].
Python ONNXAgent on the same input returns the same probabilities with
answer_confidence 0.9975 and confidence 0.9832.

## 7. Batching

Engine::score_batch(states, instruction, levels) collates one row per state
into a single ONNX run: pad sequences on the right with pad_id 0,
attention_mask 1 for real tokens and 0 for padding, marker_pos/marker_mask
identical for every row, qtype=1. Results are index-aligned with states.
Single-state score() does not pad (one row).

Because the exported model is dynamically int8-quantized (one activation scale
per tensor), a batched run can differ numerically from scoring the same states
individually. score() is the canonical path; score_batch() is a throughput
path and is documented as such in laya.h.

## 8. Verification gates (all met; see VERIFICATION.md)

1. Tokenizer: 2,083/2,083 texts exact against Hugging Face (ids, 48-token
   truncation, special-token reservation).
2. Engine: 330/330 eval rows with exact token sequences; with ONNX Runtime
   1.31 the argmax is 330/330, logits exact and probabilities within 1.2e-7.
3. Accuracy: 307/330 = 93.03% on both sides with ORT 1.31; 308/330 = 93.33%
   with the system ORT 1.29 (int8 kernel drift; sequences still exact).
4. LodeRouter builds warning-free, loads the judge, and routes chitchat / math
   / code / code-with-history / short-chitchat requests to the right backend.
5. laya-cli works for score/choice/noul/encode/batch; all three CTest tests
   pass (engine smoke, ONNX session smoke, tokenizer parity).

## 9. Build commands

Standalone:
    cmake -S laya/cpp -B build-laya -DCMAKE_BUILD_TYPE=Release
    cmake --build build-laya -j
    ctest --test-dir build-laya --output-on-failure
    ./build-laya/laya-cli score --model laya \
        --state "你好" --instruction "Which category does this request belong to?" \
        --levels "a;b;c"

Router integration:
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    ./build/LodeRouter

Runtime cost: loading the 925 MB model takes several seconds and ~1-2 GB RAM;
allow generous timeouts when running parity tests. Run one model process at a
time on this machine.

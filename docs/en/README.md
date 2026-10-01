[English](./README.md) | [简体中文](../zh-CN/README.md)
# LodeRouter

Judges question difficulty locally with Laya and routes each request to the matching model, keeping answer quality high while cutting inference cost.

## Core Features

- Driven by the local Laya judge: difficulty is decided on-device by the ONNX model from the sibling laya project, with no extra model API call
- Decisions are based on a summary compressed out of the whole conversation rather than the last user message
- Routes simple questions to lightweight models (`model_easy`)
- Routes medium-difficulty questions to balanced models (`model_middle`)
- Routes complex questions to high-performance models (`model_hard`)

## Quick Start

### 1. Build

```bash
cd path_to_router
mkdir build && cd build
cmake ..
cmake --build .
```

Build dependencies: CMake ≥ 3.20, a C++20 compiler, OpenSSL and Threads, and onnxruntime; optional dependencies are `fmt` (enabled automatically when present) and the `mold` linker (auto-detected, or set explicitly via `-DLODEROUTER_LINKER=`).

The judge model comes from the **sibling `laya` project** (local ONNX inference), expected at `../laya/cpp` by default and overridable with `-DLAYA_DIR=/path/to/laya/cpp`. The build compiles laya and its tokenizer dependency as well, which requires a system onnxruntime (`libonnxruntime.so` and `onnxruntime/onnxruntime_cxx_api.h`).

### 2. The judge model (laya)

Difficulty is graded by a local ONNX laya model — no API key involved. Three ways to get it:

**1) Automatic download (default)**: on the first run, if no model is found locally it is **downloaded in the background** while the router serves immediately (requests fall back to the `easy` level until it is ready):

```
[info] laya judge: model not found locally, downloading techtheist/laya-onnx/resolve/main/multilingual in the background (about 873 MB, one time only)
[warn] judge model is not ready yet (see the log line above)
[info] easy backend is found: http://127.0.0.1:8088 -> Qwen3.5-0.8B
[info] router listening on 127.0.0.1:8080
[info] model.onnx: 32 MB ...
...
[info] laya judge loaded from /home/you/.cache/LodeRouter/laya-onnx-multilingual (downloaded in background)
```

- Downloads come from `hf-mirror.com` by default and fall back to `huggingface.co` when unreachable; set `HF_ENDPOINT` to pin a single site, e.g. `HF_ENDPOINT=https://huggingface.co`
- It fetches `techtheist/laya-onnx`'s `multilingual` export (= an int8 quantization of `convaiinnovations/laya-multilingual`, 873 MB). The upstream `convaiinnovations/laya` repo only ships PyTorch weights — **no ONNX** — so a community export is used; its I/O names (`logits` + `act_logits`) match what the laya C++ runtime expects
- Downloads are split into 32 MB segments, one request each; a failed segment is retried and the transfer continues from where it stopped (the mirrors throttle long connections and drop them after a few hundred MB — throughput measured at 1-2 MB/s, so the whole file takes about 10 minutes). An interrupted run also resumes from the existing `.part` instead of starting over
- Files land in the cache directory: `~/.cache/LodeRouter/laya-onnx-multilingual` on Linux, `~/Library/Caches/LodeRouter/laya-onnx-multilingual` on macOS, `%LOCALAPPDATA%\LodeRouter\Cache\laya-onnx-multilingual` on Windows
- To opt out: set `"judge_auto_download": false` in the config, or `LODE_JUDGE_AUTO_DOWNLOAD=0` in the environment

**2) Manual download**: only two files are needed, both in the same directory:

```bash
mkdir -p ~/laya-onnx && cd ~/laya-onnx
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual   # or https://huggingface.co/...
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"    # 873 MB; -C - resumes an interrupted transfer
```

Then point the config at it (the `LODE_JUDGE_MODEL_DIR` environment variable works too):

```json
{ "judge_model_dir": "/home/you/laya-onnx" }
```

**3) Bring your own model**: any laya-format ONNX directory works (`model.onnx` + `tokenizer.json`, with `<bos>` / `<eos>` / `<mask>` in the tokenizer). For instance, export your own from any checkpoint with the upstream `edgejev` tool:

```bash
pip install 'edgejev[build]'
edgejev build --backend laya --model convaiinnovations/laya-multilingual --out ./my-laya-onnx
```

### 3. Configuration

Use the compiled `config_tool` (interactive: it asks for `host` / `port` / `backend_url` / judge backend / `judge_url` / `judge_model_dir` / `api_key` / `judge_api_key` / the three model names, then per level `backends.<level>.{url,model,api_key}` — leave an answer empty to keep the default), or configure manually:

| Platform | Configuration Directory |
| :--- | :--- |
| **Linux** | `~/.config/LodeRouter` (or `$XDG_CONFIG_HOME/LodeRouter`) |
| **macOS** | `~/Library/Application Support/LodeRouter` |
| **Windows** | `%APPDATA%\LodeRouter` (i.e., `C:\Users\Username\AppData\Roaming\LodeRouter`) |

Create a `config.json` file in the directory. All available fields are listed below (every one of them may stay empty, and unused blocks can be dropped):

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend_url": "",
  "judge_backend": "laya",
  "judge_url": "",
  "judge_model_dir": "",
  "judge_auto_download": true,
  "api_key": "",
  "judge_api_key": "",
  "model_easy": "",
  "model_middle": "",
  "model_hard": "",
  "backends": {
    "easy":   { "url": "", "model": "", "api_key": "" },
    "middle": { "url": "", "model": "", "api_key": "" },
    "hard":   { "url": "", "model": "", "api_key": "" }
  }
}
```

**Configuration Details**:
- `port` / `host`: The port and address on which the service listens; defaults to `127.0.0.1:8080`. `port` must be a number (an invalid value falls back to the default 8080); an empty `host` does not fall back, so write it explicitly. It must not be the same port as the service behind `backend_url` / `judge_url` / any `backends.*.url`, otherwise startup fails with an error
- `backend_url`: The address of the backend model service, including the port (e.g. `http://127.0.0.1:8088` for a llama.cpp server); a URL without a port falls back to port 80; a listen address such as `0.0.0.0` / `[::]` is rewritten to the loopback address (`127.0.0.1` / `[::1]`, with a log notice). This is the default address for all three levels
- `judge_backend`: Judge backend, either `laya` (the local ONNX model from the sibling laya project, the default) or `http` (the OpenAI-compatible service at `judge_url`)
- `judge_url`: The address of the difficulty judgment service (used when `judge_backend` is `http`)
- `judge_model_dir`: The model directory for the `laya` backend (must contain `model.onnx` and `tokenizer.json`). When empty, candidates are probed in order: the laya model path recorded at build time → the executable's directory → the current working directory → the auto-download cache directory; the `LODE_JUDGE_MODEL_DIR` environment variable overrides it. An explicitly configured but invalid directory is reported as an error instead of falling back to another directory
- `judge_auto_download`: whether to download the laya model automatically when it is not found locally (default `true`, see the previous section); `LODE_JUDGE_AUTO_DOWNLOAD=0` disables it. Set `HF_ENDPOINT` to pick the download site
- `judge_api_key`: The API Key required to call the difficulty judgment service (only used by the `http` backend)
- `api_key`: The API Key required to call the backend model service (the default key for all three levels)
- `model_easy` / `model_middle` / `model_hard`: The model names corresponding to the three difficulty levels (the default models)
- `backends.easy` / `backends.middle` / `backends.hard`: **per-level service definitions**, each with `url` / `model` / `api_key`. Every field can be omitted, in which case it inherits `backend_url` / `model_<level>` / `api_key`

**Which style to use**: with a single backend, fill in only `backend_url` + `model_easy` / `model_middle` / `model_hard` (`api_key` as needed) and skip `backends` entirely. When the three levels point at different services, **write all three into `backends.*`**, each with its own `url` / `model` / `api_key` — one uniform shape that shows at a glance where each level goes:

```json
{
  "backends": {
    "easy":   { "url": "http://127.0.0.1:8088",    "model": "Qwen3.5-0.8B",      "api_key": "" },
    "middle": { "url": "http://127.0.0.1:8088",    "model": "MiniCPM5-1B",       "api_key": "" },
    "hard":   { "url": "https://api.deepseek.com", "model": "deepseek-reasoner", "api_key": "sk-xxxxxxxx" }
  }
}
```

Here `easy` / `middle` use the local llama.cpp (no key needed) while `hard` goes to DeepSeek in the cloud (an `https://...` URL means HTTPS). You may also write only the level that differs (e.g. just `backends.hard`) and let the others inherit the defaults above — that merely splits one setting across two places, so it is not recommended.

For a given level, `backends.<level>.model` **wins**: when non-empty it overrides `model_<level>`, and the latter has no effect if both are set.

The `laya` backend uses laya's `score` head to grade the request into three difficulty buckets (chitchat / most everyday questions / complex reasoning and hard programming) and maps the result to a level. On failure it falls back to `easy`. The judge does not see the last message alone but a structured summary compressed from the whole conversation (see below).

**Config file lookup order**: the path given by the `MYAPP_CONFIG` environment variable → the user config directory → the executable's directory → the current working directory (the in-repo `config/config.json` is a sample template). Runtime logs go to `router.log` in the current directory and are echoed to the terminal (set `LODE_LOG_CONSOLE=0` to disable console output).

**Startup checks**: backends are probed per level (`GET /v1/models`) with duplicate URLs probed once, logging `easy/middle/hard backend is found: <url> -> <model>`. A level missing `url` or `model` is reported in the log only, and a request actually landing on it returns `502`. A judge that fails to load or an unreachable backend is a warning, not a startup failure; only a backend or judge address colliding with the listen address aborts startup.

### 4. Run

In the terminal, navigate to the directory containing `LodeRouter` and execute:

```bash
./LodeRouter
```

Once started, the router listens on the address configured in `config.json` and forwards each request to the service of the level the judge picked.

### 5. Usage

The target service of every level (`backend_url` / `backends.*.url`) must expose an **OpenAI-compatible API** (`/v1/chat/completions`); the judge defaults to the local ONNX model from the sibling laya project (`judge_backend` = `laya`) and can also be an OpenAI-compatible remote service (`judge_backend` = `http`).

The router serves `/v1/chat/completions` and `/v1/models`, so any client that speaks the **OpenAI interface format** works. Point `base_url` at the router's listen address, put anything in `api_key` (it is not validated) and use `auto` as the model:

```json
{
  "base_url": "http://127.0.0.1:8080",
  "api_key": "anything",
  "model": "auto"
}
```

- The `model` field in the request is overwritten with the model of the level the judge picked, so whatever the client sends makes no difference (a request without `messages` falls back to the `easy` level and keeps the client's `model` value)
- `GET /v1/models` returns `auto` plus the model name of each level (`owned_by` marks which level it belongs to), which is handy for clients that fetch the model list
- Upstream responses are always relayed to the client as SSE (`text/event-stream`), so enable streaming on the client
- Every decision is logged, e.g. `middle -> cloud-mid @ http://127.0.0.1:8095 (ctx 59 tok / 3 turn / 3 err)`

### 6. What the judge receives

The judge (`laya` or `http`) does not see the last user message alone but a **structured summary** compressed from the whole conversation (the section labels below are Chinese, exactly as the router emits them):

```
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
[起点] 帮我重构 utils 模块
[上文] 明白了，我换个思路。
```

- Patches are kept in priority order `[任务] > [报错] > [约束] > [失败] > [起点] > [上文]` within the budget, and the same sentence never occupies two slots
- The budget is 256 tokens (laya's `state` slot has 445 tokens available at `max_len=512`, leaving ample headroom); a 20-turn conversation measured 207 tokens in practice
- So the same `hi` is graded `easy` in a single-turn chat but reaches `middle` when the context carries "retried 3 times + error"

## Project Structure

```
.
├── CMakeLists.txt
├── config
│   └── config.json
├── docs
│   ├── en
│   │   └── README.md
│   └── zh-CN
│       └── README.md
├── include
│   ├── cpp-httplib
│   ├── json
│   └── spdlog
├── LICENSE
└── src
    ├── compress.h
    ├── config.h
    ├── config_tool.cpp
    ├── judge.h
    ├── judge_laya.h
    ├── log.h
    ├── main.cpp
    └── router.h

```

## Open Source License

MIT License

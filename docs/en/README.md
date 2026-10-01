[English](./README.md) | [简体中文](../zh-CN/README.md)
# LodeRouter

Judges question difficulty locally with Laya and routes each request to the matching model, keeping answer quality high while cutting inference cost.

## Core Features

- Local Laya judge: difficulty is decided on-device by an ONNX model, with no model API call
- Decisions are based on a summary compressed out of the whole conversation, not the last user message
- Routes to `model_easy` / `model_middle` / `model_hard` by difficulty
- The three levels may point at different services, mixing local llama.cpp with cloud APIs

## Quick Start

### 1. Build

```bash
mkdir build && cd build
cmake .. && cmake --build .
```

Requires CMake ≥ 3.20, a C++20 compiler, OpenSSL, Threads and onnxruntime; optional `fmt` (enabled when present) and the `mold` linker (auto-detected, or `-DLODEROUTER_LINKER=`). The judge model comes from the sibling laya project (default `../laya/cpp`, override with `-DLAYA_DIR=`), which is built along with the router.

### 2. The judge model (laya)

No API key needed. Pick one of three ways:

**1) Automatic download (default)**: on the first run, if no model is found locally it is **downloaded in the background** while the service serves immediately (requests fall back to the `easy` level until it is ready):

```
[info] laya judge: model not found locally, downloading techtheist/laya-onnx/resolve/main/multilingual in the background (about 873 MB, one time only)
[info] router listening on 127.0.0.1:8080
[info] model.onnx: 64 MB ...
[info] laya judge loaded from /home/you/.cache/LodeRouter/laya-onnx-multilingual (downloaded in background)
```

- It fetches the `multilingual` int8 quantization from `techtheist/laya-onnx` (873 MB): upstream `convaiinnovations/laya` only ships PyTorch weights and **no ONNX**, while this community export uses the I/O names (`logits` + `act_logits`) the laya C++ runtime expects
- Downloads come from `hf-mirror.com` by default and fall back to `huggingface.co`; set `HF_ENDPOINT` to pin a single site
- Transfers are split into 32 MB segments, retried on failure and resumed from the existing `.part` (mirrors throttle or drop long connections; measured 1-2 MB/s, so the whole file takes about 10 minutes)
- Cache directory: `~/.cache/LodeRouter/laya-onnx-multilingual` on Linux, `~/Library/Caches/LodeRouter/…` on macOS, `%LOCALAPPDATA%\LodeRouter\Cache\…` on Windows
- To opt out: `"judge_auto_download": false` in the config, or `LODE_JUDGE_AUTO_DOWNLOAD=0`

**2) Manual download**: put `model.onnx` and `tokenizer.json` in the same directory:

```bash
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"   # 873 MB; -C - resumes
```

Then set `"judge_model_dir": "/home/you/laya-onnx"` (or the `LODE_JUDGE_MODEL_DIR` environment variable).

**3) Bring your own model**: any laya-format ONNX directory works (`model.onnx` + `tokenizer.json`, with `<bos>` / `<eos>` / `<mask>` in the tokenizer). You can also export one from another checkpoint with `edgejev build --backend laya --model <HF id or local path> --out ./my-laya-onnx`.

### 3. Configuration

Generate it with `config_tool`, or write `~/.config/LodeRouter/config.json` by hand (macOS `~/Library/Application Support/LodeRouter`, Windows `%APPDATA%\LodeRouter`). All available fields:

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

**Configuration details**:

- `port` / `host`: listen address, default `127.0.0.1:8080`. `port` must be a number (invalid values fall back to 8080); an empty `host` does not fall back, so write it explicitly. It must not collide with any backend / judge port, otherwise startup fails
- `backend_url`: backend model service address (e.g. `http://127.0.0.1:8088`), the default address for all three levels; a `0.0.0.0` / `[::]` listen address is rewritten to loopback with a log notice
- `judge_backend`: `laya` (local ONNX model, the default) or `http` (the OpenAI-compatible service at `judge_url`)
- `judge_url`: judge service address when `judge_backend` is `http`
- `judge_model_dir`: laya model directory (needs `model.onnx` + `tokenizer.json`). When empty, candidates are probed in order: the path recorded at build time → the executable's directory → the current working directory → the download cache; an explicitly configured but invalid directory is an error, with no fallback
- `judge_auto_download`: whether to download the laya model when it is missing locally (default `true`)
- `judge_api_key`: API key for the judge service (only used by the `http` backend)
- `api_key`: default API key for the backend services
- `model_easy` / `model_middle` / `model_hard`: default model names for the three levels
- `backends.<level>`: per-level `url` / `model` / `api_key`; omitted fields inherit the three defaults above

With a single backend, `backend_url` + the three `model_*` entries are enough. When the levels point at different services, write all three into `backends.*`:

```json
{
  "backends": {
    "easy":   { "url": "http://127.0.0.1:8088",    "model": "Qwen3.5-0.8B",      "api_key": "" },
    "middle": { "url": "http://127.0.0.1:8088",    "model": "MiniCPM5-1B",       "api_key": "" },
    "hard":   { "url": "https://api.deepseek.com", "model": "deepseek-reasoner", "api_key": "sk-xxxxxxxx" }
  }
}
```

Above, `easy` / `middle` use the local llama.cpp (no key needed) while `hard` goes to DeepSeek in the cloud (an `https://...` URL means HTTPS). For a given level `backends.<level>.model` wins: when non-empty it overrides `model_<level>`.

**Other behaviour**: config lookup order is `MYAPP_CONFIG` → user config directory → the executable's directory → the current working directory (the in-repo `config/config.json` is a template); logs go to `router.log` in the current directory and to the terminal (`LODE_LOG_CONSOLE=0` writes the file only). At startup every level's backend is probed and logged as `easy/middle/hard backend is found: <url> -> <model>`; a level missing `url` / `model` is only logged, and a request actually landing on it returns `502`. A judge that fails to load or an unreachable backend is a warning — only an address colliding with the listen address aborts startup.

### 4. Run

```bash
./LodeRouter
```

### 5. Usage

Every level's service must speak an **OpenAI-compatible API** (`/v1/chat/completions`). Point the client's `base_url` at the router, put anything in `api_key` (not validated) and use `auto` as `model`:

```json
{ "base_url": "http://127.0.0.1:8080", "api_key": "anything", "model": "auto" }
```

- The request's `model` is replaced with the model of the chosen level (a request without `messages` falls back to `easy` and keeps the client's `model`)
- `GET /v1/models` returns `auto` plus each level's model name (`owned_by` marks the level)
- Upstream responses are always relayed as SSE (`text/event-stream`), so enable streaming on the client
- Every decision is logged: `middle -> cloud-mid @ http://127.0.0.1:8095 (ctx 59 tok / 3 turn / 3 err)`

### 6. What the judge receives

The judge does not see the last user message alone but a **structured summary** compressed from the whole conversation:

```
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
```

Patches are kept in priority order `[任务] > [报错] > [约束] > [失败] > [起点] > [上文]` within a 256-token budget (laya's `state` slot has 445 tokens available; a 20-turn conversation measured 207 in practice). So the same `hi` is `easy` in a single-turn chat but `middle` when the context carries "retried 3 times + error".

## Project Structure

```
├── CMakeLists.txt
├── config/config.json
├── docs/{en,zh-CN}/README.md
├── include/{cpp-httplib,json,spdlog}
└── src/   # compress config config_tool download judge judge_laya log main router
```

## Open Source License

MIT License

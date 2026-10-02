<div align="center">

# LodeRouter — Full Documentation

Judges question difficulty locally with Laya and routes each request to the matching model, keeping answer quality high while cutting inference cost.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()

**English** · [简体中文](../zh-CN/README.md)

</div>

## Contents

- [Quick start](#quick-start)
- [The judge model](#the-judge-model)
- [Configuration](#configuration)
- [Running and usage](#running-and-usage)
- [What the judge receives](#what-the-judge-receives)
- [Other behaviour](#other-behaviour)
- [Project structure](#project-structure)

## Quick start

```bash
mkdir build && cd build
cmake .. && cmake --build .
./config_tool    # generate the config interactively
./LodeRouter     # listens on 127.0.0.1:8080 by default
```

Requires CMake ≥ 3.20, a C++20 compiler, OpenSSL, Threads and onnxruntime; optional `fmt` (enabled when present) and the `mold` linker (auto-detected, or `-DLODEROUTER_LINKER=`). The judge model comes from the sibling laya project (default `../laya/cpp`, override with `-DLAYA_DIR=`), which is built along with the router.

## The judge model

No API key is needed. Pick one of three ways.

### Automatic download (default)

On the first run, if no model is found locally it is downloaded in the background while the service serves immediately (requests fall back to the `easy` level until it is ready):

```text
[info] laya judge: model not found locally, downloading techtheist/laya-onnx/resolve/main/multilingual in the background (about 873 MB, one time only)
[info] router listening on 127.0.0.1:8080
[info] model.onnx: 64 MB ...
[info] laya judge loaded from /home/you/.cache/LodeRouter/laya-onnx-multilingual (downloaded in background)
```

| Item | Details |
| --- | --- |
| Model | The `multilingual` int8 quantization from `techtheist/laya-onnx` (873 MB). Upstream `convaiinnovations/laya` ships PyTorch weights only and no ONNX; this community export uses the I/O names (`logits` + `act_logits`) the laya C++ runtime expects |
| Mirrors | `hf-mirror.com` by default, falling back to `huggingface.co`; set `HF_ENDPOINT` to pin a single site |
| Resume | Transfers are split into 32 MB segments, retried on failure and resumed from the existing `.part` (mirrors throttle or drop long connections; measured 1–2 MB/s, so the whole file takes about 10 minutes) |
| Cache | Linux `~/.cache/LodeRouter/laya-onnx-multilingual`, macOS `~/Library/Caches/LodeRouter/…`, Windows `%LOCALAPPDATA%\LodeRouter\Cache\…` |
| Opt out | `"judge_auto_download": false` in the config, or `LODE_JUDGE_AUTO_DOWNLOAD=0` |

### Manual download

Put `model.onnx` and `tokenizer.json` in the same directory:

```bash
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"   # 873 MB; -C - resumes
```

Then set `"judge_model_dir": "/home/you/laya-onnx"` (or the `LODE_JUDGE_MODEL_DIR` environment variable).

### Bring your own model

Any laya-format ONNX directory works (`model.onnx` + `tokenizer.json`, with `<bos>` / `<eos>` / `<mask>` in the tokenizer). You can also export one from another checkpoint:

```bash
edgejev build --backend laya --model <HF id or local path> --out ./my-laya-onnx
```

## Configuration

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

| Field | Default | Description |
| --- | --- | --- |
| `host` | `127.0.0.1` | Listen address. An empty value does not fall back, so write it explicitly |
| `port` | `8080` | Listen port, must be a number (invalid values fall back to 8080). It must not collide with any backend / judge port, otherwise startup fails |
| `backend_url` | empty | Backend model service address (e.g. `http://127.0.0.1:8088`), the default address for all three levels. A `0.0.0.0` / `[::]` listen address is rewritten to loopback with a log notice |
| `api_key` | empty | Default API key for the backend services |
| `model_easy` / `model_middle` / `model_hard` | empty | Default model names for the three levels |
| `judge_backend` | `laya` | `laya` (local ONNX model) or `http` (the OpenAI-compatible service at `judge_url`) |
| `judge_url` | empty | Judge service address when `judge_backend` is `http` |
| `judge_api_key` | empty | API key for the judge service (only used by the `http` backend) |
| `judge_model_dir` | empty | laya model directory (needs `model.onnx` + `tokenizer.json`). When empty, candidates are probed in order: the path recorded at build time → the executable's directory → the current working directory → the download cache; an explicitly configured but invalid directory is an error, with no fallback |
| `judge_auto_download` | `true` | Whether to download the laya model when it is missing locally |
| `backends.<level>` | empty | Per-level `url` / `model` / `api_key`; omitted fields inherit the defaults above |

### Per-level backends

Fields omitted inside `backends.<level>` inherit `backend_url` / `model_<level>` / `api_key`; for a given level, `backends.<level>.model` wins when non-empty.

```json
{
  "backends": {
    "easy":   { "url": "http://127.0.0.1:8088", "model": "<small model>",  "api_key": "" },
    "middle": { "url": "http://127.0.0.1:8088", "model": "<medium model>", "api_key": "" },
    "hard":   { "url": "https://<cloud host>",   "model": "<large model>", "api_key": "<key>" }
  }
}
```

Above, `easy` / `middle` use the local llama.cpp (no key needed) while `hard` goes to a cloud API (an `https://` URL means HTTPS).

## Running and usage

```bash
./LodeRouter
```

Every level's service must speak an **OpenAI-compatible API** (`/v1/chat/completions`). Point the client's `base_url` at the router, put anything in `api_key` (not validated) and use `auto` as `model`:

```json
{ "base_url": "http://127.0.0.1:8080", "api_key": "anything", "model": "auto" }
```

| Behaviour | Details |
| --- | --- |
| Model replacement | The request's `model` is replaced with the model of the chosen level; a request without `messages` falls back to `easy` and keeps the client's `model` |
| `GET /v1/models` | Returns `auto` plus each level's model name (`owned_by` marks the level) |
| Response style | Upstream responses are always relayed as SSE (`text/event-stream`), so enable streaming on the client |
| Logging | Every decision is logged: `middle -> cloud-mid @ http://127.0.0.1:8095 (ctx 59 tok / 3 turn / 3 err)` |

## What the judge receives

The judge does not see the last user message alone but a **structured summary** compressed from the whole conversation:

```text
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
```

Patches are kept in priority order `[任务] > [报错] > [约束] > [失败] > [起点] > [上文]` within a 256-token budget (laya's `state` slot has 445 tokens available; a 20-turn conversation measured 207 in practice). So the same `hi` is `easy` in a single-turn chat but `middle` when the context carries "retried 3 times + error".

## Other behaviour

| Item | Details |
| --- | --- |
| Config lookup order | `MYAPP_CONFIG` → user config directory → the executable's directory → the current working directory (the in-repo `config/config.json` is a template) |
| Logging | Writes `router.log` in the current directory and prints to the terminal; `LODE_LOG_CONSOLE=0` writes the file only |
| Startup probe | Every level's backend is probed by url and logged as `easy/middle/hard backend is found: <url> -> <model>` |
| Missing config | A level missing `url` / `model` is only logged; a request actually landing on it returns `502` |
| Failure handling | A judge that fails to load or an unreachable backend is a warning — only an address colliding with the listen address aborts startup |

## Project structure

```text
├── CMakeLists.txt
├── config/config.json              # config template
├── docs/{en,zh-CN}/README.md       # full documentation
├── include/{cpp-httplib,json,spdlog}
└── src/                            # compress config config_tool download judge judge_laya log main router
```

## Open source license

[MIT License](../../LICENSE)

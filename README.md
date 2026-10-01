[English](./docs/en/README.md) | [简体中文](./docs/zh-CN/README.md)

# LodeRouter

以本地 Laya 裁判判断问题难度，自动路由到对应模型，在保证回答质量的同时降低推理成本。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()

## 核心功能

- Laya 本地裁判：难度判断在本地 ONNX 模型上完成，不调用任何模型 API
- 判断依据是整段对话压缩出的概要，而不是最后一句用户消息
- 按难度路由到 `model_easy` / `model_middle` / `model_hard`
- 三个挡位可指向不同服务，本地 llama.cpp 与云端 API 混用

## 快速开始

### 1. 编译

```bash
mkdir build && cd build
cmake .. && cmake --build .
```

依赖 CMake ≥ 3.20、C++20、OpenSSL、Threads、onnxruntime。裁判模型来自同目录的 laya 项目（默认 `../laya/cpp`，可用 `-DLAYA_DIR=` 指定）。

### 2. 获取裁判模型（laya）

不需要 API Key，三种方式任选：

**自动下载（默认）**：首次启动时本地找不到模型，会**在后台下载**，服务立即可用（期间请求按 `easy` 档处理，下完自动接管）：

```
[info] laya judge: model not found locally, downloading techtheist/laya-onnx/resolve/main/multilingual in the background (about 873 MB, one time only)
[info] router listening on 127.0.0.1:8080
[info] model.onnx: 64 MB ...
[info] laya judge loaded from /home/you/.cache/LodeRouter/laya-onnx-multilingual (downloaded in background)
```

下载的是 `techtheist/laya-onnx` 的 `multilingual` int8 量化版（873 MB，I/O 名与 laya C++ 端一致；官方 `convaiinnovations/laya` 只有 PyTorch 权重、没有 ONNX）。默认走 `hf-mirror.com`，不通回退 `huggingface.co`，可用 `HF_ENDPOINT` 指定唯一站点。下载按 32 MB 切段、失败自动重试并从断点续传。缓存目录：Linux `~/.cache/LodeRouter`、macOS `~/Library/Caches/LodeRouter`、Windows `%LOCALAPPDATA%\LodeRouter\Cache`。关闭自动下载：配置写 `"judge_auto_download": false`，或设 `LODE_JUDGE_AUTO_DOWNLOAD=0`。

**手动下载**：两个文件放进同一目录即可：

```bash
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"   # 873 MB，-C - 断了可续传
```

再在配置里写 `"judge_model_dir": "/home/you/laya-onnx"`。

**自己的模型**：任何 laya 格式的 ONNX 目录（`model.onnx` + `tokenizer.json`，tokenizer 里有 `<bos>` / `<eos>` / `<mask>`）都行，也可以用 `edgejev build --backend laya --model <HF id 或本地路径> --out ./my-laya-onnx` 从别的 checkpoint 导出。

### 3. 配置

用编译好的 `config_tool` 问答式生成，或手动写到 `~/.config/LodeRouter/config.json`（macOS / Windows 路径见 `config_tool` 提示）：

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend_url": "",
  "judge_backend": "laya",
  "api_key": "",
  "model_easy": "",
  "model_middle": "",
  "model_hard": ""
}
```

只有一个后端时填 `backend_url` + 三个 `model_*` 就够；三个挡位指向不同服务时，统一写进 `backends.*`：

```json
{
  "backends": {
    "easy":   { "url": "http://127.0.0.1:8088",    "model": "Qwen3.5-0.8B",      "api_key": "" },
    "middle": { "url": "http://127.0.0.1:8088",    "model": "MiniCPM5-1B",       "api_key": "" },
    "hard":   { "url": "https://api.deepseek.com", "model": "deepseek-reasoner", "api_key": "sk-xxxxxxxx" }
  }
}
```

`backends.<level>` 里省掉的字段继承 `backend_url` / `model_<level>` / `api_key`；同一挡位的模型名以 `backends.<level>.model` 为准。其余字段（`judge_url`、`judge_model_dir`、`judge_auto_download`、`judge_api_key`）和各字段的完整说明见 [docs/zh-CN/README.md](./docs/zh-CN/README.md)。

### 4. 运行与使用

```bash
./LodeRouter
```

每个挡位的服务都要支持 **OpenAI 兼容 API**。客户端把 `base_url` 指向路由器地址、`api_key` 随便填（不校验）、`model` 填 `auto`：

```json
{ "base_url": "http://127.0.0.1:8080", "api_key": "anything", "model": "auto" }
```

请求里的 `model` 会被换成判定挡位的模型名；`GET /v1/models` 返回 `auto` 和三档模型名；响应始终以 SSE 透传给客户端。日志里能看到每次判定，例如 `middle -> deepseek-reasoner @ https://api.deepseek.com (ctx 59 tok / 3 turn / 3 err)`。

## 难度判断的输入

裁判看到的不是最后一句用户消息，而是整段对话压缩出的结构化概要（预算 256 token）：

```
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
```

同一句「hi」，单轮会话判 `easy`，带着「失败重试 3 次 + 报错」上下文的会话判 `middle`。

## 项目结构

```
├── CMakeLists.txt
├── config/config.json          # 配置模板
├── docs/{en,zh-CN}/README.md   # 完整文档
├── include/{cpp-httplib,json,spdlog}
└── src/                        # compress config config_tool download judge judge_laya log main router
```

## 开源协议

MIT License

<div align="center">

# LodeRouter 完整文档

以本地 Laya 裁判判断问题难度，自动路由到对应模型，在保证回答质量的同时降低推理成本。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()

[English](../en/README.md) · **简体中文**

</div>

## 目录

- [快速开始](#快速开始)
- [裁判模型](#裁判模型)
- [配置项](#配置项)
- [运行与使用](#运行与使用)
- [难度判断的输入](#难度判断的输入)
- [其他行为](#其他行为)
- [项目结构](#项目结构)

## 快速开始

```bash
mkdir build && cd build
cmake .. && cmake --build .
./config_tool    # 问答式生成配置
./LodeRouter     # 默认监听 127.0.0.1:8080
```

依赖 CMake ≥ 3.20、支持 C++20 的编译器、OpenSSL、Threads、onnxruntime；可选 `fmt`（存在时自动启用）与 `mold` 链接器（自动探测，也可用 `-DLODEROUTER_LINKER=` 指定）。裁判模型来自同目录的 laya 项目（默认 `../laya/cpp`，可用 `-DLAYA_DIR=` 指定），构建时会一并编译。

## 裁判模型

不需要任何 API Key，三种方式任选。

### 自动下载（默认）

首次启动时本地找不到模型，会在后台下载，服务立即可用（期间请求按 `easy` 档处理，下完自动接管）：

```text
[info] laya judge: model not found locally, downloading techtheist/laya-onnx/resolve/main/multilingual in the background (about 873 MB, one time only)
[info] router listening on 127.0.0.1:8080
[info] model.onnx: 64 MB ...
[info] laya judge loaded from /home/you/.cache/LodeRouter/laya-onnx-multilingual (downloaded in background)
```

| 项目 | 说明 |
| --- | --- |
| 模型 | `techtheist/laya-onnx` 的 `multilingual` int8 量化版（873 MB）。官方 `convaiinnovations/laya` 只有 PyTorch 权重、没有 ONNX，这份社区导出的输入输出名（`logits` + `act_logits`）与 laya C++ 端一致 |
| 下载站 | 默认 `hf-mirror.com`（国内可直连），不通回退 `huggingface.co`；`HF_ENDPOINT` 可指定唯一站点 |
| 断点续传 | 按 32 MB 切段，失败自动重试并从 `.part` 继续（镜像会限速甚至掐断，实测 1~2 MB/s，整份约 10 分钟） |
| 缓存位置 | Linux `~/.cache/LodeRouter/laya-onnx-multilingual`、macOS `~/Library/Caches/LodeRouter/…`、Windows `%LOCALAPPDATA%\LodeRouter\Cache\…` |
| 关闭下载 | 配置写 `"judge_auto_download": false`，或设 `LODE_JUDGE_AUTO_DOWNLOAD=0` |

### 手动下载

把 `model.onnx` 与 `tokenizer.json` 放进同一目录即可：

```bash
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"   # 873 MB，-C - 断了可续传
```

再写 `"judge_model_dir": "/home/you/laya-onnx"`（也可用环境变量 `LODE_JUDGE_MODEL_DIR`）。

### 自己的模型

任何 laya 格式的 ONNX 目录都能用（`model.onnx` + `tokenizer.json`，tokenizer 里有 `<bos>` / `<eos>` / `<mask>`）。也可以从别的 checkpoint 导出：

```bash
edgejev build --backend laya --model <HF id 或本地路径> --out ./my-laya-onnx
```

## 配置项

用 `config_tool` 问答式生成，或手写 `~/.config/LodeRouter/config.json`（macOS `~/Library/Application Support/LodeRouter`、Windows `%APPDATA%\LodeRouter`）。全部可用字段：

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

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `host` | `127.0.0.1` | 监听地址。留空不会退回默认值，请显式填写 |
| `port` | `8080` | 监听端口，必须是数字（非法值退回 8080）。不能与任何后端 / 裁判地址的端口重合，否则启动即报错退出 |
| `backend_url` | 空 | 后端模型服务地址（如 `http://127.0.0.1:8088`），三个挡位的默认地址。写成 `0.0.0.0` / `[::]` 会自动改成回环地址并打日志 |
| `api_key` | 空 | 后端模型服务的默认 API Key |
| `model_easy` / `model_middle` / `model_hard` | 空 | 三个挡位的默认模型名 |
| `judge_backend` | `laya` | `laya`（本地 ONNX 模型）或 `http`（`judge_url` 的 OpenAI 兼容接口） |
| `judge_url` | 空 | `judge_backend` 为 `http` 时的裁判服务地址 |
| `judge_api_key` | 空 | 裁判服务的 API Key（仅 `http` 后端使用） |
| `judge_model_dir` | 空 | laya 模型目录（需含 `model.onnx` 与 `tokenizer.json`）。留空时按「构建时记录的路径 → 可执行文件所在目录 → 当前工作目录 → 下载缓存」探测；显式配置但无效会直接报错，不回退 |
| `judge_auto_download` | `true` | 本地没有 laya 模型时是否自动下载 |
| `backends.<level>` | 空 | 按挡位指定 `url` / `model` / `api_key`，省掉的字段继承上面的默认值 |

### 三档指向不同服务

`backends.<level>` 里省略的字段继承 `backend_url` / `model_<level>` / `api_key`；同一挡位的模型名以 `backends.<level>.model` 为准（非空时覆盖 `model_<level>`）。

```json
{
  "backends": {
    "easy":   { "url": "http://127.0.0.1:8088", "model": "<小模型名>",   "api_key": "" },
    "middle": { "url": "http://127.0.0.1:8088", "model": "<中等模型名>", "api_key": "" },
    "hard":   { "url": "https://<云端地址>",     "model": "<大模型名>",   "api_key": "<key>" }
  }
}
```

上例里 `easy` / `middle` 走本地 llama.cpp（不需要 key），`hard` 走云端 API（`url` 写 `https://` 即走 HTTPS）。

## 运行与使用

```bash
./LodeRouter
```

每个挡位的服务都要支持 **OpenAI 兼容 API**（`/v1/chat/completions`）。客户端把 `base_url` 指向路由器地址、`api_key` 随便填（不校验）、`model` 填 `auto`：

```json
{ "base_url": "http://127.0.0.1:8080", "api_key": "anything", "model": "auto" }
```

| 行为 | 说明 |
| --- | --- |
| 模型替换 | 请求里的 `model` 会被换成判定挡位的模型名；没有 `messages` 时按 `easy` 档处理，并保留客户端给的 `model` |
| `GET /v1/models` | 返回 `auto` 与三个挡位各自的模型名（`owned_by` 标出挡位） |
| 响应方式 | 上游响应始终以 SSE（`text/event-stream`）透传，客户端需要开启流式接收 |
| 日志 | 每次判定都记录：`middle -> cloud-mid @ http://127.0.0.1:8095 (ctx 59 tok / 3 turn / 3 err)` |

## 难度判断的输入

裁判看到的不是最后一句用户消息，而是整段对话压缩出来的**结构化概要**：

```text
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
```

按 `[任务] > [报错] > [约束] > [失败] > [起点] > [上文]` 的优先级在预算内裁剪，同一句话不会重复占位；预算 256 token（laya 的 `state` 槽可用 445 token，实测 20 轮长会话也只压到 207）。所以同一句「hi」，单轮判 `easy`，带「失败重试 3 次 + 报错」上下文则判 `middle`。

## 其他行为

| 项目 | 说明 |
| --- | --- |
| 配置查找顺序 | `MYAPP_CONFIG` → 用户配置目录 → 可执行文件所在目录 → 当前工作目录（仓库内 `config/config.json` 是模板） |
| 日志 | 写当前目录的 `router.log` 并打印到终端；`LODE_LOG_CONSOLE=0` 只写文件 |
| 启动探测 | 按 url 逐档探测后端，打印 `easy/middle/hard backend is found: <url> -> <model>` |
| 缺配置 | 某档缺 `url` / `model` 只打日志，请求真落到该档才返回 `502` |
| 失败处理 | 裁判或后端探测失败都只告警，只有地址与监听地址重合时才退出 |

## 项目结构

```text
├── CMakeLists.txt
├── config/config.json              # 配置模板
├── docs/{en,zh-CN}/README.md       # 完整文档
├── include/{cpp-httplib,json,spdlog}
└── src/                            # compress config config_tool download judge judge_laya log main router
```

## 开源协议

[MIT License](../../LICENSE)

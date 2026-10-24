<div align="center">

# LodeRouter 完整文档

以本地微调的 Laya 裁判判断请求类别（闲聊 / 数学 / 代码），自动路由到对应模型，在保证回答质量的同时降低推理成本。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()

[English](../en/README.md) · **简体中文**

</div>

## 目录

- [快速开始](#快速开始)
- [裁判模型](#裁判模型)
- [配置项](#配置项)
- [运行与使用](#运行与使用)
- [类别判断的输入](#类别判断的输入)
- [其他行为](#其他行为)
- [项目结构](#项目结构)

## 快速开始

```bash
mkdir build && cd build
cmake .. && cmake --build .
./config_tool    # 问答式生成配置
./LodeRouter     # 默认监听 127.0.0.1:8080
```

依赖 CMake ≥ 3.20、支持 C++20 的编译器、OpenSSL、Threads、onnxruntime；可选 `fmt`（存在时自动启用）与 `mold` 链接器（自动探测，也可用 `-DLODEROUTER_LINKER=` 指定）。裁判的纯 C++ 运行时在仓库内 `laya/cpp`（与 `laya/` 里的模型配套），构建时会一并编译；也可用 `-DLAYA_DIR=` 指定外部 laya 项目。

## 裁判模型

不需要任何 API Key。裁判必须先按本任务的“闲聊 / 数学 / 代码”三分类微调一次；训练好的 **int8** 模型随仓库提交在 `laya/`（约 925 MB，单文件）。

### 微调并导出（默认做法）

`finetune/` 里有可复现的数据与命令：下载 `convaiinnovations/laya-multilingual` 基础权重，在本机 CPU 上用 1350 条三分类样本微调，再导出 ONNX：

```bash
python3 finetune/build_dataset.py
# 详见 finetune/README.md：下载权重 → laya-train → export_onnx.py
```

导出目录包含 `model.onnx` 与 `tokenizer.json`，默认放在仓库根 `laya/`（构建时自动探测），也可以写：

```json
{ "judge_model_dir": "/home/you/laya" }
```

（环境变量 `LODE_JUDGE_MODEL_DIR` 优先级更高。）找不到微调模型时请求按 `easy`（闲聊）处理并写日志。

### 原版 laya（不推荐）

`techtheist/laya-onnx` 的 `multilingual` 是原版难度模型（873 MB），只分 trivial/easy/moderate/hard，不区分闲聊和代码。自动下载默认关闭；需要临时调试时手动取一份：

```bash
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"   # 873 MB，-C - 断了可续传
```

再设 `"judge_model_dir"` 指向该目录；或把 `"judge_auto_download"` 设为 `true`，让它下载到缓存目录（Linux `~/.cache/LodeRouter/laya-onnx-multilingual`）。原版会把数学和代码误判成难度挡位，仅用于调试。

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
| `judge_auto_download` | `false` | 本地没有微调模型时是否自动下载**原版** laya（只分难度，不分工况；仅用于调试） |
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
| 模型替换 | 请求里的 `model` 会被换成判定类别的模型名；没有 `messages` 时按 `easy`（闲聊）处理，并保留客户端给的 `model` |
| `GET /v1/models` | 返回 `auto` 与三个挡位各自的模型名（`owned_by` 标出挡位：`闲聊-backend` / `数学-backend` / `代码-backend`） |
| 响应方式 | 上游响应始终以 SSE（`text/event-stream`）透传，客户端需要开启流式接收 |
| 日志 | 每次判定都记录：`数学(middle) -> cloud-mid @ http://127.0.0.1:8095 (ctx 59 tok / 3 turn / 3 err)` |

## 类别判断的输入

裁判看到的不是最后一句用户消息，而是整段对话压缩出来的**结构化概要**：

```text
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
```

按 `[任务] > [报错] > [约束] > [失败] > [起点] > [上文]` 的优先级在预算内裁剪，同一句话不会重复占位；预算 256 token（laya 的 `state` 槽可用 445 token，实测 20 轮长会话也只压到 207）。所以同一句「hi」，单轮判「闲聊」（`easy`），带「失败重试 3 次 + 报错」上下文则判「代码」（`hard`）。

## 其他行为

| 项目 | 说明 |
| --- | --- |
| 配置查找顺序 | `MYAPP_CONFIG` → 用户配置目录 → 可执行文件所在目录 → 当前工作目录（仓库内 `config/config.json` 是模板） |
| 日志 | 写当前目录的 `router.log` 并打印到终端；`LODE_LOG_CONSOLE=0` 只写文件 |
| 启动探测 | 按 url 逐档探测后端，打印 `闲聊(easy)/数学(middle)/代码(hard) backend is found: <url> -> <model>` |
| 缺配置 | 某档缺 `url` / `model` 只打日志，请求真落到该档才返回 `502` |
| 失败处理 | 裁判或后端探测失败都只告警，只有地址与监听地址重合时才退出 |

## 项目结构

```text
├── CMakeLists.txt
├── config/config.json              # 配置模板
├── finetune/                       # 三分类数据、微调与 ONNX 导出
├── docs/{en,zh-CN}/README.md       # 完整文档
├── include/{cpp-httplib,json,spdlog}
├── laya/                           # 微调后的 ONNX 模型 + laya/cpp 纯 C++ 运行时
└── src/                            # compress config config_tool download judge judge_laya log main router
```

## 开源协议

[MIT License](../../LICENSE)

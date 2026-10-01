[English](../en/README.md) | [简体中文](./README.md)
# LodeRouter

以本地 Laya 裁判判断问题难度，自动路由到对应模型，在保证回答质量的同时降低推理成本。

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

依赖 CMake ≥ 3.20、支持 C++20 的编译器、OpenSSL、Threads、onnxruntime；可选 `fmt`（存在时自动启用）与 `mold` 链接器（自动探测，也可用 `-DLODEROUTER_LINKER=` 指定）。裁判模型来自同目录的 laya 项目（默认 `../laya/cpp`，可用 `-DLAYA_DIR=` 指定），构建时会一并编译。

### 2. 裁判模型（laya）

不需要任何 API Key，三种方式任选：

**① 自动下载（默认）**：首次启动时本地找不到模型，会**在后台下载**，服务立即可用（期间请求按 `easy` 档处理，下完自动接管）：

```
[info] laya judge: model not found locally, downloading techtheist/laya-onnx/resolve/main/multilingual in the background (about 873 MB, one time only)
[info] router listening on 127.0.0.1:8080
[info] model.onnx: 64 MB ...
[info] laya judge loaded from /home/you/.cache/LodeRouter/laya-onnx-multilingual (downloaded in background)
```

- 下载的是 `techtheist/laya-onnx` 的 `multilingual` int8 量化版（873 MB）：官方 `convaiinnovations/laya` 只有 PyTorch 权重、没有 ONNX，这份社区导出的输入输出名（`logits` + `act_logits`）与 laya C++ 端一致
- 默认走 `hf-mirror.com`（国内可直连），不通回退 `huggingface.co`；可用 `HF_ENDPOINT` 指定唯一站点
- 按 32 MB 切段下载，失败自动重试并从 `.part` 续传（镜像对长连接会限速甚至掐断，实测 1~2 MB/s，整份约 10 分钟）
- 缓存目录：Linux `~/.cache/LodeRouter/laya-onnx-multilingual`、macOS `~/Library/Caches/LodeRouter/…`、Windows `%LOCALAPPDATA%\LodeRouter\Cache\…`
- 关闭自动下载：配置写 `"judge_auto_download": false`，或设 `LODE_JUDGE_AUTO_DOWNLOAD=0`

**② 手动下载**：`model.onnx` 与 `tokenizer.json` 放进同一目录即可：

```bash
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"   # 873 MB，-C - 断了可续传
```

再写 `"judge_model_dir": "/home/you/laya-onnx"`（也可用环境变量 `LODE_JUDGE_MODEL_DIR`）。

**③ 自己的模型**：任何 laya 格式的 ONNX 目录都能用（`model.onnx` + `tokenizer.json`，tokenizer 里有 `<bos>` / `<eos>` / `<mask>`），也可以用官方 `edgejev build --backend laya --model <HF id 或本地路径> --out ./my-laya-onnx` 从别的 checkpoint 导出。

### 3. 配置

`config_tool` 问答式生成，或手动写到 `~/.config/LodeRouter/config.json`（macOS `~/Library/Application Support/LodeRouter`、Windows `%APPDATA%\LodeRouter`）。全部可用字段：

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

**配置项说明**：

- `port` / `host`：监听地址，默认 `127.0.0.1:8080`。`port` 必须是数字（非法值退回 8080）；`host` 留空不会退回默认值，请显式填写。不能与任何后端 / 裁判地址的端口重合，否则启动即报错退出
- `backend_url`：后端模型服务地址（如 `http://127.0.0.1:8088`），三个挡位的默认地址；写成 `0.0.0.0` / `[::]` 会自动改成回环地址并打日志
- `judge_backend`：`laya`（本地 ONNX 模型，默认）或 `http`（`judge_url` 的 OpenAI 兼容接口）
- `judge_url`：`judge_backend` 为 `http` 时的裁判服务地址
- `judge_model_dir`：laya 模型目录（需含 `model.onnx` 与 `tokenizer.json`）。留空时按「构建时记录的路径 → 可执行文件所在目录 → 当前工作目录 → 下载缓存」探测；显式配置但无效会直接报错，不回退
- `judge_auto_download`：本地没有 laya 模型时是否自动下载（默认 `true`）
- `judge_api_key`：裁判服务的 API Key（仅 `http` 后端使用）
- `api_key`：后端模型服务的默认 API Key
- `model_easy` / `model_middle` / `model_hard`：三个挡位的默认模型名
- `backends.<level>`：按挡位指定 `url` / `model` / `api_key`，省掉的字段继承上面三个默认值

只用同一个后端时，填 `backend_url` + 三个 `model_*` 就够；三档分别指向不同服务时，统一写进 `backends.*`：

```json
{
  "backends": {
    "easy":   { "url": "http://127.0.0.1:8088",    "model": "Qwen3.5-0.8B",      "api_key": "" },
    "middle": { "url": "http://127.0.0.1:8088",    "model": "MiniCPM5-1B",       "api_key": "" },
    "hard":   { "url": "https://api.deepseek.com", "model": "deepseek-reasoner", "api_key": "sk-xxxxxxxx" }
  }
}
```

上例里 `easy` / `middle` 走本地 llama.cpp（不需要 key），`hard` 走云端 DeepSeek（`url` 写 `https://...` 就是 HTTPS）。同一挡位的模型名以 `backends.<level>.model` 为准（非空时覆盖 `model_<level>`）。

**其他**：配置文件查找顺序为 `MYAPP_CONFIG` → 用户配置目录 → 可执行文件所在目录 → 当前工作目录（仓库内 `config/config.json` 是模板）；日志写当前目录的 `router.log` 并打印到终端（`LODE_LOG_CONSOLE=0` 只写文件）。启动时按 url 逐档探测后端并打印 `easy/middle/hard backend is found: <url> -> <model>`，某档缺 `url` / `model` 只打日志、请求真落到该档才返回 `502`；裁判或后端探测失败都只告警，只有地址与监听地址重合时才退出。

### 4. 运行

```bash
./LodeRouter
```

### 5. 使用

每个挡位的服务都要支持 **OpenAI 兼容 API**（`/v1/chat/completions`）。客户端把 `base_url` 指向路由器地址、`api_key` 随便填（不校验）、`model` 填 `auto`：

```json
{ "base_url": "http://127.0.0.1:8080", "api_key": "anything", "model": "auto" }
```

- 请求里的 `model` 会被换成判定挡位的模型名（没有 `messages` 时按 `easy` 档处理，并保留客户端给的 `model`）
- `GET /v1/models` 返回 `auto` 与三个挡位各自的模型名（`owned_by` 标出挡位）
- 上游响应始终以 SSE（`text/event-stream`）透传，客户端需要开启流式接收
- 日志里能看到每次判定：`middle -> cloud-mid @ http://127.0.0.1:8095 (ctx 59 tok / 3 turn / 3 err)`

### 6. 难度判断的输入

裁判看到的不是最后一句用户消息，而是整段对话压缩出来的**结构化概要**：

```
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
```

按 `[任务] > [报错] > [约束] > [失败] > [起点] > [上文]` 的优先级在预算内裁剪，同一句话不会重复占位；预算 256 token（laya 的 `state` 槽可用 445 token，实测 20 轮长会话也只压到 207）。所以同一句「hi」，单轮判 `easy`，带「失败重试 3 次 + 报错」上下文则判 `middle`。

## 项目结构

```
├── CMakeLists.txt
├── config/config.json
├── docs/{en,zh-CN}/README.md
├── include/{cpp-httplib,json,spdlog}
└── src/   # compress config config_tool download judge judge_laya log main router
```

## 开源协议

MIT License

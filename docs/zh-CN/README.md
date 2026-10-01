[English](../en/README.md) | [简体中文](./README.md)
# LodeRouter

以本地 Laya 裁判判断问题难度，自动路由到对应模型，在保证回答质量的同时降低推理成本。

## 核心功能

- Laya 本地裁判驱动：难度判断由同目录 laya 项目的 ONNX 模型在本地完成，不产生额外的模型 API 调用
- 判断依据是整段对话压缩出的概要，而不是最后一句用户消息
- 简单问题路由至轻量模型（`model_easy`）
- 中等难度路由至均衡模型（`model_middle`）
- 复杂问题路由至高性能模型（`model_hard`）

## 快速开始

### 1. 编译

```bash
cd path_to_router
mkdir build && cd build
cmake ..
cmake --build .
```

构建依赖：CMake ≥ 3.20、支持 C++20 的编译器、OpenSSL 与 Threads、onnxruntime；可选依赖 `fmt`（存在时自动启用）与 `mold` 链接器（自动探测，也可用 `-DLODEROUTER_LINKER=` 指定）。

裁判模型使用与 LodeRouter **同目录的 laya 项目**（本地 ONNX 推理），默认路径为 `../laya/cpp`，可用 `-DLAYA_DIR=/path/to/laya/cpp` 指定；构建时会一并编译 laya 及其 tokenizer 依赖（需要系统安装 onnxruntime：`libonnxruntime.so` 与 `onnxruntime/onnxruntime_cxx_api.h`）。

### 2. 裁判模型（laya）

难度判断用的是本地 ONNX 推理的 laya 模型，不需要任何 API Key。三种获得方式：

**① 自动下载（默认）**：第一次启动时如果本地找不到模型，会**在后台自动下载**，路由服务立刻可用（这期间请求都按 `easy` 档处理，下载完成后自动接管）：

```
[info] laya judge: model not found locally, downloading techtheist/laya-onnx/resolve/main/multilingual in the background (about 873 MB, one time only)
[warn] judge model is not ready yet (see the log line above)
[info] easy backend is found: http://127.0.0.1:8088 -> Qwen3.5-0.8B
[info] router listening on 127.0.0.1:8080
[info] model.onnx: 32 MB ...
...
[info] laya judge loaded from /home/you/.cache/LodeRouter/laya-onnx-multilingual (downloaded in background)
```

- 下载站默认 `hf-mirror.com`（国内可直连），不通会自动回退 `huggingface.co`；也可以用 `HF_ENDPOINT` 指定唯一站点，例如 `HF_ENDPOINT=https://huggingface.co`
- 下载的是 `techtheist/laya-onnx` 的 `multilingual` 导出（= `convaiinnovations/laya-multilingual` 的 int8 量化，873 MB）。官方 `convaiinnovations/laya` 仓库只有 PyTorch 权重、**没有 ONNX**，所以这里用社区导出——它的输入输出名（`logits` + `act_logits`）与 laya C++ 端要求的一致
- 下载按 32 MB 切段，每段一个请求，某段失败了会自动重试并从断点继续（国内镜像对长连接会限速、下到几百 MB 后还会直接掐断，实测 1~2 MB/s，整份约 10 分钟）；中断后重新启动同样从 `.part` 接着下，不会重头来
- 文件落在缓存目录：Linux `~/.cache/LodeRouter/laya-onnx-multilingual`、macOS `~/Library/Caches/LodeRouter/laya-onnx-multilingual`、Windows `%LOCALAPPDATA%\LodeRouter\Cache\laya-onnx-multilingual`
- 不想要自动下载：配置里写 `"judge_auto_download": false`，或设环境变量 `LODE_JUDGE_AUTO_DOWNLOAD=0`

**② 手动下载**：只需 `model.onnx` 与 `tokenizer.json` 两个文件放进同一目录：

```bash
mkdir -p ~/laya-onnx && cd ~/laya-onnx
BASE=https://hf-mirror.com/techtheist/laya-onnx/resolve/main/multilingual   # 或 https://huggingface.co/...
curl -L -o tokenizer.json "$BASE/tokenizer.json"
curl -L -C - -o model.onnx "$BASE/model_int8.onnx"    # 873 MB；-C - 断了可续传
```

然后在 `config.json` 里指过去（也可以用环境变量 `LODE_JUDGE_MODEL_DIR`）：

```json
{ "judge_model_dir": "/home/you/laya-onnx" }
```

**③ 换成自己的模型**：任何 laya 格式的 ONNX 目录都能用（需含 `model.onnx` + `tokenizer.json`，且 tokenizer 里有 `<bos>` / `<eos>` / `<mask>`）。例如用官方 `edgejev` 从任意 checkpoint 自己导出：

```bash
pip install 'edgejev[build]'
edgejev build --backend laya --model convaiinnovations/laya-multilingual --out ./my-laya-onnx
```

### 3. 配置

用编译好的 `config_tool`（问答式：依次问 `host` / `port` / `backend_url` / 裁判后端 / `judge_url` / `judge_model_dir` / `api_key` / `judge_api_key` / 三个模型名，最后逐档问 `backends.<level>.{url,model,api_key}`，留空即沿用默认值），或手动配置：

| 平台 | 配置目录 |
| :--- | :--- |
| **Linux** | `~/.config/LodeRouter` (或 `$XDG_CONFIG_HOME/LodeRouter`) |
| **macOS** | `~/Library/Application Support/LodeRouter` |
| **Windows** | `%APPDATA%\LodeRouter` (即 `C:\Users\用户名\AppData\Roaming\LodeRouter`) |

目录下创建 `config.json` 文件。全部可用字段如下（都可以留空，用不到的整块删掉也行）：

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
- `port` / `host`：服务监听的端口和地址，默认 `127.0.0.1:8080`。`port` 必须是数字，非法值会退回默认 8080；`host` 写空字符串不会退回默认值，请显式填写。不能与 `backend_url` / `judge_url` / 各 `backends.*.url` 所指向服务的端口相同，否则启动时直接报错退出
- `backend_url`：后端模型服务的地址，需带上端口（例如 llama.cpp server 的 `http://127.0.0.1:8088`）；省略端口时按 URL 默认端口 80 处理；写成 `0.0.0.0` / `[::]` 这类监听地址时会自动改写成回环地址（`127.0.0.1` / `[::1]`，日志中会提示）。三个挡位的默认地址
- `judge_backend`：裁判后端，`laya`（同目录 laya 项目的本地 ONNX 模型，默认）或 `http`（`judge_url` 的 OpenAI 兼容接口）
- `judge_url`：`judge_backend` 为 `http` 时的难度判断服务地址
- `judge_model_dir`：`judge_backend` 为 `laya` 时的模型目录（需包含 `model.onnx` 与 `tokenizer.json`）。留空时按「构建时记录的 laya 模型路径 → 可执行文件所在目录 → 当前工作目录 → 自动下载的缓存目录」依次探测；也可用环境变量 `LODE_JUDGE_MODEL_DIR` 指定。显式配置但目录无效时会直接报错，不会回退到其它目录
- `judge_auto_download`：本地找不到 laya 模型时是否自动下载（默认 `true`，见上一节）；也可用环境变量 `LODE_JUDGE_AUTO_DOWNLOAD=0` 关闭。下载站可用 `HF_ENDPOINT` 指定
- `judge_api_key`：调用难度判断服务所需的 API Key（仅 `http` 后端使用）
- `api_key`：调用后端模型服务所需的 API Key（三个挡位的默认密钥）
- `model_easy` / `model_middle` / `model_hard`：分别对应三种难度使用的模型名称（三个挡位的默认模型）
- `backends.easy` / `backends.middle` / `backends.hard`：**按挡位指定各自的服务**，每项含 `url` / `model` / `api_key`。每一项都可以省，省掉的字段继承 `backend_url` / `model_<level>` / `api_key`

**写法建议**：只有一个后端时，只填 `backend_url` + `model_easy` / `model_middle` / `model_hard`（`api_key` 按需），不用写 `backends`；三个挡位指向不同服务时，**建议三档都写进 `backends.*`**，每档都写全 `url` / `model` / `api_key`，格式统一、一眼能看出哪档走到哪里：

```json
{
  "backends": {
    "easy":   { "url": "http://127.0.0.1:8088",    "model": "Qwen3.5-0.8B",      "api_key": "" },
    "middle": { "url": "http://127.0.0.1:8088",    "model": "MiniCPM5-1B",       "api_key": "" },
    "hard":   { "url": "https://api.deepseek.com", "model": "deepseek-reasoner", "api_key": "sk-xxxxxxxx" }
  }
}
```

上例里 `easy` / `middle` 走本地 llama.cpp（不需要 key），`hard` 走云端 DeepSeek（`url` 写 `https://...` 就是 HTTPS）。也可以只写有差异的那一档（比如只写 `backends.hard`），其余挡位会继承上面的默认值——只是那样 `easy` / `middle` / `hard` 的配置分散在两处，不推荐。

同一档位的模型名**以 `backends.<level>.model` 为准**：它非空时会覆盖 `model_<level>`，两个都写时后者不生效。

`laya` 后端使用其 `score` 头把用户输入分成三档难度（闲聊/大多数日常问题/极复杂推理与高难度编程），再把结果映射到对应挡位；判定失败时回退到 `easy` 挡位。裁判的输入不是最后一句话，而是整段对话压缩出的结构化概要（见下）。

**配置文件查找顺序**：`MYAPP_CONFIG` 环境变量指定的路径 → 用户配置目录 → 可执行文件所在目录 → 当前工作目录（仓库内 `config/config.json` 为示例模板）。运行日志写入当前目录下的 `router.log`，同时打印到终端（设置 `LODE_LOG_CONSOLE=0` 可关闭终端输出）。

**启动检查**：启动时按 `url` 去重后分别探测三个挡位的后端（`GET /v1/models`），日志里逐档打印 `easy/middle/hard backend is found: <url> -> <model>`；某档的 `url` 或 `model` 没配只在日志里报错，请求真的落到该档时才返回 `502`。裁判加载失败、后端探测失败都只告警，不会阻止启动；只有后端 / 裁判地址与监听地址重合时才会直接退出。

### 4. 运行

在终端中进入 `LodeRouter` 所在目录，执行：

```bash
./LodeRouter
```

服务启动后监听 `config.json` 里配置的地址，并按问题难度把请求转发到对应挡位的服务。

### 5. 使用

每个挡位的目标服务（`backend_url` / `backends.*.url`）都需支持 **OpenAI 兼容 API**（`/v1/chat/completions`）；裁判默认使用同目录 laya 项目的本地 ONNX 模型（`judge_backend` 为 `laya`），也可换成 OpenAI 兼容的远程服务（`judge_backend` 为 `http`）。

路由对外提供 `/v1/chat/completions` 与 `/v1/models`，用支持 **OpenAI 接口格式**的客户端即可。把 `base_url` 指向路由器监听的地址，`api_key` 随便填（不校验），`model` 填 `auto`：

```json
{
  "base_url": "http://127.0.0.1:8080",
  "api_key": "anything",
  "model": "auto"
}
```

- 请求里的 `model` 会被覆盖成判定挡位的模型名，客户端填什么都一样（请求里没有 `messages` 时按 `easy` 挡位处理，并保留客户端给的 `model`）
- `GET /v1/models` 返回 `auto` 以及三个挡位各自的模型名（`owned_by` 标出所属挡位），便于客户端拉取列表
- 上游响应一律以 SSE（`text/event-stream`）透传给客户端，请让客户端开启流式接收
- 日志里能看到每次判定，例如：`middle -> cloud-mid @ http://127.0.0.1:8095 (ctx 59 tok / 3 turn / 3 err)`

### 6. 难度判断的输入

裁判（`laya` 或 `http`）看到的不是最后一句用户消息，而是整段对话压缩出来的**结构化概要**：

```
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
[起点] 帮我重构 utils 模块
[上文] 明白了，我换个思路。
```

- 概要按 `[任务] > [报错] > [约束] > [失败] > [起点] > [上文]` 的优先级在预算内裁剪，同一句话不会重复占位
- 预算 256 token（laya 的 `state` 槽在 `max_len=512` 下可用 445 token，留足余量），实测 20 轮长会话也只压到 207 token
- 因此同一句「hi」，单轮会话判 `easy`，而带着「失败重试 3 次 + 报错」上下文的会话会判到 `middle`

## 项目结构

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

## 开源协议

MIT License

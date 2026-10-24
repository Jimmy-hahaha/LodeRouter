<div align="center">

# LodeRouter

**先判断类别，再挑模型。**

闲聊交给小模型，数学和代码交给更合适的模型，回答质量不变、推理成本更低。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

[English](./docs/en/README.md) · [简体中文](./docs/zh-CN/README.md)

</div>

## 工作方式

```mermaid
flowchart LR
    C[客户端] --> R[LodeRouter]
    R --> J{本地 Laya 裁判}
    J -->|easy| A[小模型 / 闲聊]
    J -->|middle| M[中等模型 / 数学]
    J -->|hard| H[大模型 / 代码]
```

| 挡位 | 对应类别 | 后端示例 |
| --- | --- | --- |
| `easy` | 闲聊：问候、寒暄、情绪、观点 | 本地小模型 |
| `middle` | 数学：算术、代数、几何、概率、证明 | 中等模型 |
| `hard` | 代码：编程、调试、重构、架构、工具链 | 大模型 / 云端 API |

三个要点：

- **类别在本地判断** —— Laya 裁判是一个微调后的 ONNX 模型，跑在本机，不调用任何模型 API
- **看整段对话，不看最后一句** —— 裁判的输入是整段会话压缩出的概要（256 token 预算）
- **三档可以各走各的** —— 本地 llama.cpp 与云端 API 混用，每一档用自己的 key

## 快速开始

### 1. 编译

```bash
mkdir build && cd build
cmake .. && cmake --build .
```

需要 CMake ≥ 3.20、C++20 编译器、OpenSSL、Threads、onnxruntime。裁判的纯 C++ 运行时在仓库内 `laya/cpp`（与 `laya/` 里的模型配套），构建时一并编译；也可用 `-DLAYA_DIR=` 指向外部的 laya 项目。

### 2. 裁判模型（需要先准备）

裁判是针对本任务微调过的 laya 多语言模型：把请求分成闲聊、数学、代码三类，模型目录就叫 `laya`。训练好的 **int8** 模型（约 925 MB，单文件）随仓库一起提交；也可以把目录放到别处并写 `"judge_model_dir": "/path/to/laya"`。不需要任何 API Key。

找不到微调模型时请求按 `easy`（闲聊）处理并写日志。原版 laya 的自动下载默认关闭：原版只分难度、不分工况，加载它会把数学和代码误判；需要临时用原版调试时设 `"judge_auto_download": true`。

### 3. 配置

```bash
./config_tool    # 问答式生成 ~/.config/LodeRouter/config.json
```

只有一个后端时，手写这几行就够：

```json
{
  "backend_url": "http://127.0.0.1:8088",
  "model_easy": "<小模型名>",
  "model_middle": "<中等模型名>",
  "model_hard": "<大模型名>"
}
```

三档指向不同服务时写 `backends.*`，全部字段见[配置项](./docs/zh-CN/README.md#配置项)。

### 4. 启动，接上客户端

```bash
./LodeRouter    # 默认监听 127.0.0.1:8080
```

把客户端的 `base_url` 指向它、`model` 填 `auto`，`api_key` 随便填（不校验）：

```json
{ "base_url": "http://127.0.0.1:8080", "api_key": "anything", "model": "auto" }
```

各档服务都需要支持 OpenAI 兼容 API。响应以 SSE 透传，客户端记得开启流式接收；每次判定到哪一档、依据了多少上下文，都会写进日志。

## 裁判看到什么

不是最后一句用户消息，而是整段对话压缩出的概要：

```text
[轮次] 第 4 轮
[任务] 再试一次
[报错] 编译报错了：undefined reference to `parse_config'
[约束] 必须保持 ABI 兼容，不能升级依赖
```

所以同一句「hi」，单轮会话判「闲聊」（`easy`），带着「重试 3 次 + 报错」的上下文就判「代码」（`hard`）。

## 文档

| 文档 | 内容 |
| --- | --- |
| [完整文档（简体中文）](./docs/zh-CN/README.md) | 全部配置项、模型获取方式、日志与启动行为 |
| [Full documentation (English)](./docs/en/README.md) | Everything above, in English |

## 开源协议

[MIT License](LICENSE)

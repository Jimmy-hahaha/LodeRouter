[English](../en/README.md) | [简体中文](./README.md)
# LodeRouter

根据问题难度自动路由模型。

## 核心功能

- 自动识别用户问题的复杂度（简单/中等/困难）
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
### 2. 配置

使用编译完了的 `config_tool` 或使用如下方法手动配置

| 平台 | 配置目录 |
| :--- | :--- |
| **Linux** | `~/.config/LodeRouter` (或 `$XDG_CONFIG_HOME/LodeRouter`) |
| **macOS** | `~/Library/Application Support/LodeRouter` |
| **Windows** | `%APPDATA%\LodeRouter` (即 `C:\Users\用户名\AppData\Roaming\LodeRouter`) |
目录下创建 `config.json` 文件，内容如下：

```json
{
  "port": "",
  "host": "",
  "backend_url": "",
  "judge_url": "",
  "judge_api_key": "",
  "api_key": "",
  "model_easy": "",
  "model_middle": "",
  "model_hard": ""
}
```

**配置项说明**：
- `port` / `host`：服务监听的端口和地址
- `backend_url`：后端模型服务的地址
- `judge_url`：难度判断服务的地址
- `judge_api_key`：调用难度判断服务所需的 API Key
- `api_key`：调用后端模型服务所需的 API Key
- `model_easy` / `model_middle` / `model_hard`：分别对应三种难度使用的模型名称

### 3. 运行

在终端中进入 `router` 所在目录，执行：

```bash
./router
```

服务启动后，将监听config.json中配置的的地址，自动根据问题难度路由至对应模型。

### 4.使用

目前仅支持**OpenAI兼容API**的模型(包括评判模型和各个难度的输出模型)

路由对外提供 `/v1/chat/completions` 端点，格式与 OpenAI 一致
使用时请用支持 **OpenAI 的接口格式** 的客户端：

```json
"url":"config里写的url"
"key":"随便写一个(未校验)"
"model":"auto"
```
## 项目结构

```
.
├── CMakeLists.txt
├── docs
│   ├── en
│   │   └── README.md
│   └── zh-CN
│       └── README.md
├── include
│   ├── cfgpath
│   │   ├── cfgpath.h
│   │   └── LICENSE
│   ├── httplib.h
│   └── passwordPrompt_cpp
│       ├── LICENSE
│       ├── passwordPrompt.cpp
│       └── passwordPrompt.h
├── LICENSE
└── src
    ├── choose_model.cpp
    ├── choose_model.hpp
    ├── edit_config.cpp
    └── main.cpp

```

## 开源协议

MIT License

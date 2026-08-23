[English](./README.md) | [简体中文](../zh-CN/README.md)
# LodeRouter

Automatically routes models based on question difficulty, ensuring answer quality while reducing inference costs.

## Core Features

- Automatically identifies the complexity of user questions (simple/medium/hard)
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
### 2. Configuration

Use the compiled `config_tool` or manually configure as follows:

| Platform | Configuration Directory |
| :--- | :--- |
| **Linux** | `~/.config/LodeRouter` (or `$XDG_CONFIG_HOME/LodeRouter`) |
| **macOS** | `~/Library/Application Support/LodeRouter` |
| **Windows** | `%APPDATA%\LodeRouter` (i.e., `C:\Users\Username\AppData\Roaming\LodeRouter`) |

Create a `config.json` file in the directory with the following content:

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

**Configuration Details**:
- `port` / `host`: The port and address on which the service listens
- `backend_url`: The address of the backend model service
- `judge_url`: The address of the difficulty judgment service
- `judge_api_key`: The API Key required to call the difficulty judgment service
- `api_key`: The API Key required to call the backend model service
- `model_easy` / `model_middle` / `model_hard`: The model names corresponding to the three difficulty levels

### 3. Run

In the terminal, navigate to the directory containing `router` and execute:

```bash
./router
```

After the service starts, it will listen on the address configured in `config.json` and automatically route questions to the corresponding model based on difficulty.

### 4. Usage

Currently, only models that are **OpenAI-compatible APIs** are supported (including the judgment model and the output models for each difficulty level).

The router exposes the `/v1/chat/completions` endpoint externally, with a format consistent with OpenAI.

When using it, please use a client that supports the **OpenAI interface format**:

```json
"url": "the URL written in the config",
"key": "anything (not validated)",
"model": "auto"
```
## Project Structure

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

## Open Source License

MIT License
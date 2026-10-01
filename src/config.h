#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include "../include/json/json.hpp"
#include "spdlog/spdlog.h"
#include "log.h"

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#include <conio.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <climits>
#include <unistd.h>
#include <termios.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#include <termios.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace cfg {

inline fs::path getExecutablePath() {
#if defined(_WIN32)
    wchar_t buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len == 0 || len == MAX_PATH) return {};
    return fs::path(buf);
#elif defined(__APPLE__)
    char buf[PATH_MAX];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};
    return fs::weakly_canonical(fs::path(buf));
#elif defined(__linux__)
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return {};
    buf[len] = '\0';
    return fs::path(buf);
#else
    return {};
#endif
}

inline fs::path getHomeDir() {
#if defined(_WIN32)
    const wchar_t* home = _wgetenv(L"USERPROFILE");
    if (home) return fs::path(home);
    const wchar_t* drive = _wgetenv(L"HOMEDRIVE");
    const wchar_t* path = _wgetenv(L"HOMEPATH");
    if (drive && path) return fs::path(std::wstring(drive) + path);
    return {};
#else
    const char* home = std::getenv("HOME");
    return home ? fs::path(home) : fs::path();
#endif
}

inline fs::path getUserConfigDir(const std::string& appName) {
#if defined(_WIN32)
    const wchar_t* appdata = _wgetenv(L"APPDATA");
    if (appdata) return fs::path(appdata) / appName;
    return {};
#elif defined(__APPLE__)
    return getHomeDir() / "Library" / "Application Support" / appName;
#elif defined(__linux__)
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    if (xdg && *xdg) return fs::path(xdg) / appName;
    return getHomeDir() / ".config" / appName;
#else
    return getHomeDir() / ("." + appName);
#endif
}

// 缓存目录：自动下载的模型放这里，既不污染配置目录也不污染安装目录。
// Linux 走 XDG_CACHE_HOME/~/.cache，macOS 走 ~/Library/Caches，Windows 走 %LOCALAPPDATA%。
inline fs::path getUserCacheDir(const std::string& appName) {
#if defined(_WIN32)
    const wchar_t* local = _wgetenv(L"LOCALAPPDATA");
    if (local) return fs::path(local) / appName / "Cache";
    return {};
#elif defined(__APPLE__)
    return getHomeDir() / "Library" / "Caches" / appName;
#elif defined(__linux__)
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && *xdg) return fs::path(xdg) / appName;
    return getHomeDir() / ".cache" / appName;
#else
    return getHomeDir() / ("." + appName) / "cache";
#endif
}

inline fs::path locateConfigFile(const std::string& appName,
                                 const std::string& fileName = "config.json") {
    if (const char* env = std::getenv("MYAPP_CONFIG")) {
        if (*env) return fs::path(env);
    }

    fs::path userConfig = getUserConfigDir(appName) / fileName;
    if (!userConfig.empty() && fs::exists(userConfig)) return userConfig;

    fs::path exeDir = getExecutablePath().parent_path();
    if (!exeDir.empty()) {
        fs::path exeConfig = exeDir / fileName;
        if (fs::exists(exeConfig)) return exeConfig;
    }

    fs::path cwdConfig = fs::current_path() / fileName;
    if (fs::exists(cwdConfig)) return cwdConfig;

    return userConfig;
}

// 后端地址常直接从 llama.cpp server 的 --host 抄成 0.0.0.0 / ::，但这两个地址只在 bind
// 时有意义，作为客户端目的地址连不上。这里把 URL 里的监听地址改写成回环地址。
inline std::string normalize_client_host(const std::string& url) {
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return url;

    const size_t host_begin = scheme_end + 3;
    size_t host_end = url.find_first_of("/?#", host_begin);
    if (host_end == std::string::npos) host_end = url.size();

    std::string authority = url.substr(host_begin, host_end - host_begin);
    if (authority.empty()) return url;

    std::string userinfo;  // user:pass@
    if (size_t at = authority.rfind('@'); at != std::string::npos) {
        userinfo = authority.substr(0, at + 1);
        authority = authority.substr(at + 1);
    }

    std::string host = authority, suffix;  // suffix 保留 ":port"，原样拼回
    if (host.empty()) return url;
    if (host.front() == '[') {  // [::1]:8088
        const size_t close = host.find(']');
        if (close == std::string::npos) return url;
        host = host.substr(0, close + 1);
        suffix = authority.substr(close + 1);
    } else if (size_t colon = host.rfind(':'); colon != std::string::npos) {
        host = host.substr(0, colon);
        suffix = authority.substr(colon);
    }

    std::string replacement;
    if (host == "0.0.0.0") replacement = "127.0.0.1";
    else if (host == "::" || host == "[::]") replacement = "[::1]";
    else return url;

    return url.substr(0, host_begin) + userinfo + replacement + suffix + url.substr(host_end);
}

inline json loadJson(const fs::path& file) {
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
        throw std::runtime_error("unable to open config file: " + file.string());
    }
    return json::parse(ifs);
}

inline void saveJson(const fs::path& file, const json& j) {
    if (!file.parent_path().empty()) {
        fs::create_directories(file.parent_path());
    }
    std::ofstream ofs(file);
    if (!ofs.is_open()) {
        throw std::runtime_error("unable to write config file: " + file.string());
    }
    ofs << j.dump(4);
}

inline json defaultConfig() {
    return json{
        {"host", "127.0.0.1"},
        {"port", 8080},
        {"backend_url", ""},
        {"judge_backend", "laya"},
        {"judge_url", ""},
        {"judge_model_dir", ""},
        // 本地找不到 laya 模型时，是否自动从 HuggingFace 下载（约 873 MB）
        {"judge_auto_download", true},
        {"api_key", ""},
        {"judge_api_key", ""},
        {"model_easy", ""},
        {"model_middle", ""},
        {"model_hard", ""},
        // 三个档位可以各自指向不同的服务（本地 llama.cpp 或云端 API）：
        // 这里留空，启动时按上面的默认值补齐，需要区分再逐档填写。
        {"backends", {
            {"easy",   {{"url", ""}, {"model", ""}, {"api_key", ""}}},
            {"middle", {{"url", ""}, {"model", ""}, {"api_key", ""}}},
            {"hard",   {{"url", ""}, {"model", ""}, {"api_key", ""}}}
        }}
    };
}

}

namespace {

std::string get_str(const json& j, const std::string& name) {
    if (j.is_object() && j.contains(name) && j[name].is_string()) {
        return j[name].get<std::string>();
    }
    return {};
}

std::string read_masked() {
    std::string s;
#if defined(_WIN32)
    std::cout << std::flush;
    int ch;
    while ((ch = _getch()) != '\r' && ch != '\n') {
        if (ch == '\b' || ch == 0x7f) {
            if (!s.empty()) {
                s.pop_back();
                std::cout << "\b \b" << std::flush;
            }
        } else {
            s.push_back((char)ch);
            std::cout << '*' << std::flush;
        }
    }
    std::cout << '\n';
#else
    termios oldt, newt;
    if (tcgetattr(STDIN_FILENO, &oldt) != 0) {
        std::getline(std::cin, s);
        return s;
    }
    newt = oldt;
    newt.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    char c;
    while (std::cin.get(c)) {
        if (c == '\n' || c == '\r') break;
        if (c == '\b' || c == 127) {
            if (!s.empty()) {
                s.pop_back();
                std::cout << "\b \b" << std::flush;
            }
            continue;
        }
        s.push_back(c);
        std::cout << '*' << std::flush;
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    std::cout << '\n';
#endif
    return s;
}

}

inline std::string host = "127.0.0.1";
inline int port = 8080;
inline std::string backend_url = {};
// 裁判后端：laya（同目录 laya 项目的本地 ONNX 模型）或 http（judge_url 的 OpenAI 兼容接口）
inline std::string judge_backend = "laya";
inline std::string judge_url = {};
// laya 模型目录（含 model.onnx 与 tokenizer.json）；为空时自动探测
inline std::string judge_model_dir = {};
// 本地找不到 laya 模型时是否自动从 HuggingFace 下载；LODE_JUDGE_AUTO_DOWNLOAD=0 可临时关闭
inline bool judge_auto_download = true;
inline std::string api_key = {};
inline std::string judge_api_key = {};
inline std::string model_easy = {};
inline std::string model_middle = {};
inline std::string model_hard = {};

// 难度挡位。裁判只需要回答"该用哪一档"，具体用哪个服务由挡位决定，
// 这样三个档位可以分别指向本地 llama.cpp 或云端 API。
enum class Level { Easy = 0, Middle = 1, Hard = 2 };

// 一个挡位的目标服务：url 为空表示没配（不可用），model 为空表示没配。
struct Backend {
    std::string url;
    std::string model;
    std::string api_key;
    bool ready() const { return !url.empty() && !model.empty(); }
};

// 三个挡位最终解析出来的目标；缺省值来自 backend_url / api_key / model_*
inline Backend backend_easy{};
inline Backend backend_middle{};
inline Backend backend_hard{};

inline const char* level_name(Level lv) {
    switch (lv) {
        case Level::Middle: return "middle";
        case Level::Hard:   return "hard";
        default:            return "easy";
    }
}

inline const Backend& backend_for(Level lv) {
    switch (lv) {
        case Level::Middle: return backend_middle;
        case Level::Hard:   return backend_hard;
        default:            return backend_easy;
    }
}

inline void init_config() {
    fs::path file = cfg::locateConfigFile("LodeRouter");
    if (file.empty()) {
        log_error("unable to determine config path");
        return;
    }

    if (!fs::exists(file)) {
        try {
            cfg::saveJson(file, cfg::defaultConfig());
            log_warn("config file created, edit it or run config_tool: " + file.string());
        } catch (const std::exception& e) {
            log_error(e.what());
        }
        return;
    }

    try {
        json j = cfg::loadJson(file);
        if (!j.is_object()) {
            log_error("config is not a JSON object: " + file.string());
            return;
        }
        host = get_str(j, "host");
        if (j.contains("port") && j["port"].is_number()) port = j["port"].get<int>();
        backend_url = get_str(j, "backend_url");
        if (j.contains("judge_backend") && j["judge_backend"].is_string()) {
            judge_backend = j["judge_backend"].get<std::string>();
        }
        judge_url = get_str(j, "judge_url");
        judge_model_dir = get_str(j, "judge_model_dir");
        if (j.contains("judge_auto_download") && j["judge_auto_download"].is_boolean()) {
            judge_auto_download = j["judge_auto_download"].get<bool>();
        }
        api_key = get_str(j, "api_key");
        judge_api_key = get_str(j, "judge_api_key");
        model_easy = get_str(j, "model_easy");
        model_middle = get_str(j, "model_middle");
        model_hard = get_str(j, "model_hard");

        // 后端地址抄了 --host 0.0.0.0 时也能连上
        auto fix_listen_addr = [](std::string& url, const char* name) {
            const std::string fixed = cfg::normalize_client_host(url);
            if (fixed != url) {
                log_warn(std::string(name) + " " + url +
                         " uses a listen address, using " + fixed + " instead");
                url = fixed;
            }
        };
        fix_listen_addr(backend_url, "backend_url");
        fix_listen_addr(judge_url, "judge_url");

        // 按挡位解析目标服务：先从 backend_url / api_key / model_* 填缺省值，
        // 再用 backends.<level>.{url,model,api_key} 里非空的字段覆盖。
        // 于是旧的 config.json（没有 backends）行为完全不变。
        auto load_backend = [&](const char* name, Backend& b, const std::string& fallback_model) {
            b.url = backend_url;
            b.model = fallback_model;
            b.api_key = api_key;

            if (!j.contains("backends") || !j["backends"].is_object()) return;
            const json& bj = j["backends"];
            if (!bj.contains(name) || !bj[name].is_object()) return;
            const json& o = bj[name];

            std::string v;
            if (!(v = get_str(o, "url")).empty()) b.url = v;
            if (!(v = get_str(o, "model")).empty()) b.model = v;
            if (!(v = get_str(o, "api_key")).empty()) b.api_key = v;
        };
        load_backend("easy", backend_easy, model_easy);
        load_backend("middle", backend_middle, model_middle);
        load_backend("hard", backend_hard, model_hard);

        fix_listen_addr(backend_easy.url, "backends.easy.url");
        fix_listen_addr(backend_middle.url, "backends.middle.url");
        fix_listen_addr(backend_hard.url, "backends.hard.url");

        if (judge_backend != "laya" && judge_backend != "http") {
            log_warn("unknown judge_backend '" + judge_backend + "', falling back to 'laya'");
            judge_backend = "laya";
        }
    } catch (const std::exception& e) {
        log_error(e.what());
    }
}

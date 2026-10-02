#pragma once

#include <cstdlib>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>
#include "laya.h"
#include "config.h"
#include "download.h"
#include "log.h"

// ───────── laya 裁判：用本地 ONNX 模型打分，把输入判成 easy / middle / hard ─────────
namespace lode_judge_laya {

// ───────── 引擎与互斥量：引擎串行推理，后台热加载与请求共用一把锁 ─────────
inline std::unique_ptr<laya::Engine> engine;
inline std::mutex engine_mutex;

// ───────── 难度问法：laya 官方 router preset 的 4 档问法，映射到 3 档 ─────────
inline const std::vector<std::string>& levels() {
    static const std::vector<std::string> lv = {
        "trivial: a lookup or one-liner",
        "easy: short answer, no reasoning",
        "moderate: several steps",
        "hard: long multi-step reasoning or specialist knowledge",
    };
    return lv;
}

inline constexpr const char* instruction = "How hard is `request` for a language model?";

// ───────── 目录探测：构建时路径 → 可执行文件旁 → 工作目录，认 model.onnx + tokenizer.json ─────────
inline bool looks_like_model_dir(const fs::path& dir) {
    std::error_code ec;
    return fs::is_regular_file(dir / "model.onnx", ec) &&
           fs::is_regular_file(dir / "tokenizer.json", ec);
}

inline fs::path normalize(const fs::path& p) {
    std::error_code ec;
    fs::path abs = fs::weakly_canonical(p, ec);
    return ec ? p : abs;
}

inline fs::path auto_detect_model_dir() {
    std::vector<fs::path> candidates;

#ifdef LODEROUTER_LAYA_MODEL_DIR
    candidates.emplace_back(LODEROUTER_LAYA_MODEL_DIR);
#endif
    fs::path exe_dir = cfg::getExecutablePath().parent_path();
    if (!exe_dir.empty()) {
        candidates.push_back(exe_dir / "laya-zh-onnx-int8");
        candidates.push_back(exe_dir / ".." / "laya" / "laya-zh-onnx-int8");
    }
    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
        candidates.push_back(cwd / "laya-zh-onnx-int8");
        candidates.push_back(cwd / ".." / "laya" / "laya-zh-onnx-int8");
    }

    for (const auto& c : candidates) {
        fs::path p = normalize(c);
        if (looks_like_model_dir(p)) return p;
    }
    return {};
}

// ───────── 自动下载：本地无模型时从 HuggingFace 镜像取一份（约 873 MB，仅首次） ─────────
inline constexpr const char* kHfRepo = "techtheist/laya-onnx/resolve/main/multilingual";

inline fs::path downloaded_model_dir() {
    return cfg::getUserCacheDir("LodeRouter") / "laya-onnx-multilingual";
}

inline bool auto_download_allowed() {
    const char* env = std::getenv("LODE_JUDGE_AUTO_DOWNLOAD");
    if (env && *env) {
        std::string v = env;
        if (v == "0" || v == "off" || v == "false" || v == "no") return false;
    }
    return judge_auto_download;
}

inline bool download_laya_model() {
    const std::string prefix = kHfRepo;
    const fs::path dir = downloaded_model_dir();
    const std::vector<std::pair<std::string, fs::path>> files = {
        {prefix + "/model_int8.onnx", dir / "model.onnx"},
        {prefix + "/tokenizer.json", dir / "tokenizer.json"},
    };

    std::string chosen;
    for (const auto& ep : lode_dl::candidates()) {
        if (lode_dl::probe(ep, prefix + "/rl_agent_config.json")) {
            chosen = ep;
            break;
        }
        log_warn("laya judge: cannot reach " + ep + ", trying the next download mirror");
    }
    if (chosen.empty()) {
        log_error("laya judge: no download mirror is reachable (tried hf-mirror.com and "
                  "huggingface.co); set HF_ENDPOINT to a working mirror, or download manually "
                  "(see README)");
        return false;
    }

    log_info("laya judge: downloading from " + chosen + " (about 873 MB, one time only)");
    std::string err;
    if (!lode_dl::download_files(chosen, files, err)) {
        log_error("laya judge: download failed: " + err);
        return false;
    }
    return looks_like_model_dir(dir);
}

// ───────── 模型解析：显式配置优先，其次自动探测与已下载的缓存 ─────────
inline fs::path resolve_model_dir() {
    std::string explicit_dir = judge_model_dir;
    if (const char* env = std::getenv("LODE_JUDGE_MODEL_DIR"); env && *env) {
        explicit_dir = env;
    }

    if (!explicit_dir.empty()) {
        fs::path p = normalize(explicit_dir);
        if (!looks_like_model_dir(p)) {
            log_error("laya judge: '" + p.string() + "' is not a laya model dir (need model.onnx + tokenizer.json)");
            return {};
        }
        return p;
    }

    fs::path p = auto_detect_model_dir();
    if (!p.empty()) return p;

    fs::path cache = downloaded_model_dir();
    if (looks_like_model_dir(cache)) return cache;
    return {};
}

// ───────── 启动初始化：后台下载并在就绪后热加载，缺模型时才请求下载 ─────────
inline void start_background_download() {
    std::thread([]() {
        if (!download_laya_model()) {
            log_error("laya judge: automatic download failed; see README for manual download, "
                      "or set judge_model_dir (judgement keeps falling back to easy)");
            return;
        }
        fs::path dir = downloaded_model_dir();
        try {
            auto eng = std::make_unique<laya::Engine>(dir.string());
            {
                std::lock_guard<std::mutex> lock(engine_mutex);
                engine = std::move(eng);
            }
            log_info("laya judge loaded from " + dir.string() + " (downloaded in background)");
        } catch (const std::exception& e) {
            log_error("laya judge: engine init failed after download: " + std::string(e.what()));
        }
    }).detach();
}

inline bool init_laya_judge() {
    fs::path dir = resolve_model_dir();
    if (dir.empty()) {
        if (!auto_download_allowed()) {
            log_error("laya judge: model dir not found; set judge_model_dir in config or "
                      "LODE_JUDGE_MODEL_DIR (auto download is disabled)");
            return false;
        }
        log_info(std::string("laya judge: model not found locally, downloading ") + kHfRepo +
                 " in the background (about 873 MB, one time only)");
        start_background_download();
        return false;
    }

    try {
        engine = std::make_unique<laya::Engine>(dir.string());
    } catch (const std::exception& e) {
        log_error("laya judge: engine init failed: " + std::string(e.what()));
        engine.reset();
        return false;
    }
    log_info("laya judge loaded from " + dir.string());
    return true;
}

// ───────── 判定入口：调 score 头取挡位，失败一律回退 easy ─────────
inline Level ask_judge_laya(const std::string& input) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    if (!engine) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            log_warn("laya judge: engine not ready yet (model still downloading?) - "
                     "falling back to easy until it is");
        }
        return Level::Easy;
    }
    try {
        auto ans = engine->score(input, instruction, levels());
        Level lv = Level::Easy;
        switch (ans.index) {
            case 2: lv = Level::Middle; break;
            case 3: lv = Level::Hard; break;
            default: lv = Level::Easy; break;
        }
        log_info(ans.label + " (" + std::to_string(ans.confidence) + ") -> " + level_name(lv));
        return lv;
    } catch (const std::exception& e) {
        log_error("laya judge: " + std::string(e.what()));
    }
    return Level::Easy;
}

}

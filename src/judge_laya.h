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

// 裁判后端：同目录 laya 项目（../laya）的本地 ONNX 模型。
// 用 laya 的 score 头把用户输入分成 easy / middle / hard 三档；
// 每档具体指向哪个服务/模型由 config.json 里的 backends 决定。

namespace lode_judge_laya {

inline std::unique_ptr<laya::Engine> engine;
// laya 引擎串行推理：一次 Run 内部已用满 onnxruntime 线程池，
// 并发 Run 只会互相抢 CPU，这里直接串行以保证可预测的行为。
inline std::mutex engine_mutex;

// 难度档次：laya 官方 router preset 的 difficulty 问法（4 档），
// 映射到路由器的 3 档：trivial/easy -> easy，moderate -> middle，hard -> hard。
// 4 档比 3 档更容易让模型把"极端简单"和"需要几步推理"区分开。
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

// 依次尝试：构建时记录的 laya 模型路径 → 可执行文件所在目录 → 当前工作目录
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

// ---- 首次运行自动下载 ----
// 本地找不到 laya 模型时，从 HuggingFace 拉一份到缓存目录。
// 用 techtheist/laya-onnx 的 multilingual 导出（= convaiinnovations/laya-multilingual
// 的 int8 量化）：输入输出名与 laya.cpp 期望的完全一致（input_ids/attention_mask/
// marker_pos/marker_mask/qtype → logits + act_logits），tokenizer 也用 <bos>/<eos>/<mask>。
// 官方 convaiinnovations/laya 仓库只有 PyTorch 权重、没有 ONNX，所以这里用社区导出。
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

    // 先挑一个通的站再下大文件；默认顺序是 hf-mirror.com → huggingface.co
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

// 返回本地已有的 laya 模型目录：显式配置（config 或环境变量）优先，且无效即报错；
// 未显式配置时依次走「构建时记录的路径 / 可执行文件旁 / 工作目录 → 之前下载的缓存」。
// 都没有就返回空，要不要下载由 init_laya_judge 决定。
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
    if (looks_like_model_dir(cache)) return cache;  // 之前已经下载过
    return {};
}

// 后台下载 + 完成后热加载：873 MB 在国内镜像上要几分钟，
// 同步下载会让路由器"启动后一直没反应"，所以先把服务跑起来，
// 这期间判定回退 easy，模型就绪后自动接管。
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

// 返回这次请求该用的挡位（与 http 后端语义一致：判定失败则回退 easy）
inline Level ask_judge_laya(const std::string& input) {
    // engine 可能正被后台下载线程替换，读和推理都放在同一把锁下
    std::lock_guard<std::mutex> lock(engine_mutex);
    if (!engine) {
        // 后台下载期间每个请求都会走到这里，只提示一次，别刷屏
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
            default: lv = Level::Easy; break;  // trivial / easy
        }
        log_info(ans.label + " (" + std::to_string(ans.confidence) + ") -> " + level_name(lv));
        return lv;
    } catch (const std::exception& e) {
        log_error("laya judge: " + std::string(e.what()));
    }
    return Level::Easy;
}

} // namespace lode_judge_laya

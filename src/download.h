#pragma once

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../include/cpp-httplib/httplib.h"
#include "log.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ───────── 模型下载器：首次运行时从 HuggingFace 镜像把 laya 模型拉到本地缓存 ─────────
namespace lode_dl {

// ───────── 分段参数：每段 32 MB、最多 400 段，镜像掐断后接着下一段续传 ─────────
constexpr uint64_t kSegmentBytes = 32ull * 1024 * 1024;
constexpr int kMaxSegments = 400;

// ───────── 下载源：默认 hf-mirror.com → huggingface.co，HF_ENDPOINT 可指定唯一点 ─────────
inline std::string trim_slashes(std::string s) {
    while (!s.empty() && s.back() == '/') s.pop_back();
    return s;
}

inline std::vector<std::string> candidates() {
    if (const char* env = std::getenv("HF_ENDPOINT"); env && *env) {
        return {trim_slashes(std::string(env))};
    }
    return {"https://hf-mirror.com", "https://huggingface.co"};
}

inline httplib::Client make_client(const std::string& endpoint) {
    httplib::Client cli(endpoint);
    cli.set_connection_timeout(15);
    cli.set_read_timeout(60);
    cli.set_follow_location(true);
    return cli;
}

// ───────── 能力探测：先用小文件探活，再确认支持 Range 并取回文件总大小 ─────────
inline bool probe(const std::string& endpoint, const std::string& probe_repo_path) {
    if (probe_repo_path.empty()) return true;
    auto cli = make_client(endpoint);
    if (!cli.is_valid()) return false;
    cli.set_connection_timeout(10);
    cli.set_read_timeout(20);
    auto res = cli.Get("/" + probe_repo_path);
    return res && res->status == 200 && res->body.size() > 10;
}

inline bool probe_range(const std::string& endpoint, const std::string& repo_path,
                        uint64_t have, uint64_t& total_size) {
    auto cli = make_client(endpoint);
    if (!cli.is_valid()) return false;
    cli.set_read_timeout(30);

    httplib::Headers headers = {
        {"Range", "bytes=" + std::to_string(have) + "-" + std::to_string(have + 1023)}
    };
    auto res = cli.Get("/" + repo_path, headers);
    if (!res || res->status != 206) return false;

    const std::string cr = res->get_header_value("Content-Range");
    const size_t slash = cr.rfind('/');
    if (slash == std::string::npos) return false;
    try {
        total_size = std::stoull(cr.substr(slash + 1));
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

// ───────── 取数方式：按 Range 追加一段，不支持 Range 时整文件重下 ─────────
inline bool fetch_segment(const std::string& endpoint, const std::string& repo_path,
                          uint64_t from, uint64_t to, const fs::path& part, std::string& err) {
    auto cli = make_client(endpoint);
    if (!cli.is_valid()) {
        err = "invalid endpoint '" + endpoint + "'";
        return false;
    }

    std::ofstream ofs(part, std::ios::binary | std::ios::app);
    if (!ofs) {
        err = "cannot write " + part.string();
        return false;
    }

    uint64_t got = 0;
    auto receiver = [&](const char* data, size_t len) -> bool {
        ofs.write(data, static_cast<std::streamsize>(len));
        got += len;
        return static_cast<bool>(ofs);
    };

    httplib::Headers headers = {
        {"Range", "bytes=" + std::to_string(from) + "-" + std::to_string(to)}
    };
    auto res = cli.Get("/" + repo_path, headers, receiver);
    ofs.close();

    if (!res) {
        err = "cannot reach " + endpoint + " (" + httplib::to_string(res.error()) + ")";
        return false;
    }
    if (res->status != 206) {
        std::error_code tec;
        fs::resize_file(part, from, tec);
        err = "HTTP " + std::to_string(res->status) + " (expected 206 for a ranged request)";
        return false;
    }
    if (got == 0) {
        err = "empty response";
        return false;
    }
    return true;
}

inline bool download_whole(const std::string& endpoint, const std::string& repo_path,
                           const fs::path& part, const std::string& what, std::string& err) {
    auto cli = make_client(endpoint);
    if (!cli.is_valid()) {
        err = "invalid endpoint '" + endpoint + "'";
        return false;
    }

    std::ofstream ofs(part, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        err = "cannot write " + part.string();
        return false;
    }

    uint64_t total = 0, last_report = 0;
    auto receiver = [&](const char* data, size_t len) -> bool {
        ofs.write(data, static_cast<std::streamsize>(len));
        if (!ofs) return false;
        total += len;
        if (total - last_report >= kSegmentBytes) {
            last_report = total;
            log_info(what + ": " + std::to_string(total / (1024 * 1024)) + " MB ...");
        }
        return true;
    };

    auto res = cli.Get("/" + repo_path, httplib::Headers{}, receiver);
    ofs.close();

    auto drop_if_empty = [&]() {
        std::error_code tec;
        if (fs::file_size(part, tec) == 0 && !tec) fs::remove(part, tec);
    };

    if (!res) {
        err = "cannot reach " + endpoint + " (" + httplib::to_string(res.error()) + ")";
        drop_if_empty();
        return false;
    }
    if (res->status != 200 || total == 0) {
        err = "HTTP " + std::to_string(res->status) + " for an unranged request";
        drop_if_empty();
        return false;
    }
    return true;
}

// ───────── 单文件下载：写 .part，下完才改名，失败保留断点供下次续传 ─────────
inline bool download_file(const std::string& endpoint, const std::string& repo_path,
                          const fs::path& out, const std::string& what, std::string& err) {
    std::error_code ec;
    if (!out.parent_path().empty()) fs::create_directories(out.parent_path(), ec);

    fs::path part = out;
    part += ".part";

    uint64_t have = 0;
    if (fs::exists(part, ec)) {
        uint64_t size = fs::file_size(part, ec);
        if (!ec) have = size;
    }

    uint64_t total = 0;
    if (!probe_range(endpoint, repo_path, have, total)) {
        log_info(what + ": server does not support ranged download, fetching in one go");
        if (!download_whole(endpoint, repo_path, part, what, err)) return false;
        fs::rename(part, out, ec);
        if (ec) {
            err = "cannot rename " + part.string() + ": " + ec.message();
            return false;
        }
        log_info(what + ": saved to " + out.string());
        return true;
    }

    if (have >= total) {
        log_info(what + ": already complete (" + std::to_string(total / (1024 * 1024)) + " MB)");
    } else if (have > 0) {
        log_info(what + ": resuming from " + std::to_string(have / (1024 * 1024)) + " MB / " +
                 std::to_string(total / (1024 * 1024)) + " MB");
    }

    for (int segment = 0; have < total && segment < kMaxSegments; ++segment) {
        const uint64_t to = std::min(have + kSegmentBytes, total) - 1;
        if (!fetch_segment(endpoint, repo_path, have, to, part, err)) {
            log_warn(what + ": segment at " + std::to_string(have / (1024 * 1024)) +
                     " MB failed (" + err + "), retrying");
            uint64_t size = fs::file_size(part, ec);
            if (ec) {
                err = "cannot stat " + part.string();
                return false;
            }
            if (size == have) continue;
            have = size;
            continue;
        }
        uint64_t size = fs::file_size(part, ec);
        if (ec) {
            err = "cannot stat " + part.string();
            return false;
        }
        if (size == have) {
            err = "server returned no data for the requested range";
            return false;
        }
        have = size;
        if (have % (2 * kSegmentBytes) < kSegmentBytes) {
            log_info(what + ": " + std::to_string(have / (1024 * 1024)) + " MB / " +
                     std::to_string(total / (1024 * 1024)) + " MB ...");
        }
    }

    if (have != total) {
        err = "incomplete after retries (" + std::to_string(have) + "/" + std::to_string(total) +
              " bytes); the partial file is kept, so the next run resumes from here";
        return false;
    }

    fs::rename(part, out, ec);
    if (ec) {
        err = "cannot rename " + part.string() + ": " + ec.message();
        return false;
    }
    log_info(what + ": " + std::to_string(total / (1024 * 1024)) + " MB -> " + out.string());
    return true;
}

// ───────── 批量下载：按顺序拉一组文件，任一失败即中止 ─────────
inline bool download_files(const std::string& endpoint,
                           const std::vector<std::pair<std::string, fs::path>>& files,
                           std::string& err) {
    for (const auto& [repo_path, out] : files) {
        if (!download_file(endpoint, repo_path, out, out.filename().string(), err)) return false;
    }
    return true;
}

}

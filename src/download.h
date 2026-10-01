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

// HuggingFace 文件下载器：第一次运行、本地又找不到 laya 模型时用它把模型拉下来。
// 复用项目已有的 cpp-httplib + OpenSSL（已开 HTTPS），不引入额外依赖。
//
// 默认先走 hf-mirror.com（国内可直连），不通再回退 huggingface.co；
// 也可以用 HF_ENDPOINT 指定唯一下载站（例如 HF_ENDPOINT=https://huggingface.co）。
namespace lode_dl {

// 每一段的大小。国内镜像对单个长连接会限速、下到几百 MB 后还会直接掐断
// （实测 873 MB 的文件在 256 MB / 533 MB 两处断掉），所以按固定大小切段，
// 每段一个请求，断了就接着下一段继续。
constexpr uint64_t kSegmentBytes = 32ull * 1024 * 1024;
// 一段最多重试这么多轮，避免镜像长时间抽风时无限循环
constexpr int kMaxSegments = 400;

inline std::string trim_slashes(std::string s) {
    while (!s.empty() && s.back() == '/') s.pop_back();
    return s;
}

// 候选下载站，按顺序尝试。显式设置了 HF_ENDPOINT 就只用它，不再偷偷换站。
inline std::vector<std::string> candidates() {
    if (const char* env = std::getenv("HF_ENDPOINT"); env && *env) {
        return {trim_slashes(std::string(env))};
    }
    return {"https://hf-mirror.com", "https://huggingface.co"};
}

inline httplib::Client make_client(const std::string& endpoint) {
    httplib::Client cli(endpoint);
    cli.set_connection_timeout(15);
    cli.set_read_timeout(60);       // 单个 socket 读的超时，不是整个下载的总时长
    cli.set_follow_location(true);  // HF / 镜像都会 302 到 CDN，httplib 默认不跟随
    return cli;
}

// 用几百字节的小文件探活：站点通不通要在下 873 MB 之前就知道。
inline bool probe(const std::string& endpoint, const std::string& probe_repo_path) {
    if (probe_repo_path.empty()) return true;
    auto cli = make_client(endpoint);
    if (!cli.is_valid()) return false;
    cli.set_connection_timeout(10);
    cli.set_read_timeout(20);
    auto res = cli.Get("/" + probe_repo_path);
    return res && res->status == 200 && res->body.size() > 10;
}

// 探测能否按 Range 分段下载：请求 1 KB，返回 206 才算支持，同时解析出文件总大小。
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

    // Content-Range: bytes 335544320-335545343/915648029
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

// 取一段 [from, to] 追加到 part 末尾（调用方保证 from == part 当前大小）。
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
        // 网络断了：已经落盘的部分是合法的前缀，留着下次继续
        err = "cannot reach " + endpoint + " (" + httplib::to_string(res.error()) + ")";
        return false;
    }
    if (res->status != 206) {
        // 服务端没按 Range 回：body 可能是整个文件，已经追加进去的是垃圾，必须回滚
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

// 不支持 Range 时的退路：一次性下载整个文件（先截断 .part）。
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

    // 一个字节都没下到就别留下空的 .part 文件
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

// 下载 {endpoint}/{repo_path} 到 out。
// 先写 out.part，只有下完整了才改名；失败时**保留** .part，下次启动从这里续。
// repo_path 形如 "owner/repo/resolve/main/path/to/file"。
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
        // 服务端不给 Range：只能一次性下完（网络一抖就得重来，所以是退路）
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
            // 这一段没拿到（镜像掐断、限速超时……）：把已有的字节留着，接着重试
            log_warn(what + ": segment at " + std::to_string(have / (1024 * 1024)) +
                     " MB failed (" + err + "), retrying");
            uint64_t size = fs::file_size(part, ec);
            if (ec) {
                err = "cannot stat " + part.string();
                return false;
            }
            if (size == have) continue;  // 一个字节都没进来，直接重试同一段
            have = size;
            continue;
        }
        uint64_t size = fs::file_size(part, ec);
        if (ec) {
            err = "cannot stat " + part.string();
            return false;
        }
        if (size == have) {  // 服务端说 206 却什么都没给，避免死循环
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

// 按顺序下载一组文件；任意一个失败就返回 false（err 里是原因）。
inline bool download_files(const std::string& endpoint,
                           const std::vector<std::pair<std::string, fs::path>>& files,
                           std::string& err) {
    for (const auto& [repo_path, out] : files) {
        if (!download_file(endpoint, repo_path, out, out.filename().string(), err)) return false;
    }
    return true;
}

} // namespace lode_dl

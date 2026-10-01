#pragma once
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../include/cpp-httplib/httplib.h"
#include "../include/json/json.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "compress.h"
#include "config.h"
#include "judge.h"
#include "log.h"
using json = nlohmann::json;

// 每个挡位用它自己的 api_key 调它自己的服务（云端 API 各有各的 key），
// 客户端带来的 Authorization 一概不透传：那是客户端自己的凭据。
inline void add_backend_auth(httplib::Headers& headers, const std::string& key) {
    if (!key.empty()) headers.emplace("Authorization", "Bearer " + key);
}

// 探测/检查结果：失败时带上可读原因，避免只剩下一句光秃秃的 "not found"。
struct Probe {
    bool ok = false;
    std::string detail;
};

// httplib 会把 URL 里省略的端口补成 80/443，这里返回实际连接的目标，
// 于是漏写端口的 "http://0.0.0.0" 会显示成 "0.0.0.0:80"。
inline std::string endpoint_of(const httplib::Client& cli) {
    return cli.host() + ":" + std::to_string(cli.port());
}

// 探测 {base_url}/v1/models：能连通且返回 200 才算后端可用。
inline Probe probe_models(const std::string& base_url) {
    if (base_url.empty()) return {false, "url is empty (backend_url/judge_url not set?)"};

    httplib::Client cli(base_url);
    cli.set_connection_timeout(3);  // 连不上时尽快出结果，而不是干等默认超时
    if (!cli.is_valid()) return {false, "invalid url '" + base_url + "'"};

    auto res = cli.Get("/v1/models");
    if (!res) {
        return {false, "cannot reach " + endpoint_of(cli) + " (" +
                       httplib::to_string(res.error()) + ")"};
    }
    if (res->status != 200) {
        return {false, endpoint_of(cli) + " answered HTTP " +
                       std::to_string(res->status) + " for /v1/models"};
    }
    return {true, {}};
}

inline bool is_local_host(const std::string& h) {
    return h.empty() || h == "0.0.0.0" || h == "127.0.0.1" || h == "localhost" ||
           h == "::" || h == "::1" || h == "[::1]";
}

// 后端与本机监听地址重合时两者不可能同时监听，属于配置错误：
// 直接退出，免得把 bind 失败误读成"后端没起来"。
inline Probe check_router_port_conflict(const std::string& base_url) {
    if (base_url.empty() || !is_local_host(host)) return {true, {}};
    httplib::Client cli(base_url);
    if (!cli.is_valid() || !is_local_host(cli.host()) || cli.port() != port) return {true, {}};
    return {false, endpoint_of(cli) + " is the same address this router listens on (" +
                   host + ":" + std::to_string(port) +
                   "); change port or point it at the real backend"};
}

// 三个挡位常常指向同一个服务（比如都指本地 llama.cpp），探测/冲突检查按 url 去重，
// 免得同一个地址被反复刷日志。
inline std::vector<std::pair<Level, const Backend*>> distinct_backends() {
    std::vector<std::pair<Level, const Backend*>> out;
    for (Level lv : {Level::Easy, Level::Middle, Level::Hard}) {
        const Backend& b = backend_for(lv);
        bool seen = false;
        for (const auto& [_, prev] : out) {
            if (prev->url == b.url) { seen = true; break; }
        }
        if (!seen) out.push_back({lv, &b});
    }
    return out;
}

inline int run_router() {
    for (const auto& [lv, b] : distinct_backends()) {
        const std::string tag = std::string("backends.") + level_name(lv);
        if (b->url.empty()) {
            log_error(tag + ".url is not configured (set backend_url, or backends.*.url per level)");
            continue;
        }
        if (b->model.empty()) {
            log_error(tag + ".model is not configured (set model_" + level_name(lv) +
                      ", or backends." + level_name(lv) + ".model)");
        }
        if (auto c = check_router_port_conflict(b->url); !c.ok) {
            log_error(tag + ".url " + c.detail);
            return 1;
        }
    }
    if (judge_backend == "http") {
        if (auto c = check_router_port_conflict(judge_url); !c.ok) {
            log_error("judge_url " + c.detail);
            return 1;
        }
    }

    for (const auto& [lv, b] : distinct_backends()) {
        if (b->url.empty()) continue;
        if (auto p = probe_models(b->url); !p.ok)
            log_error(std::string(level_name(lv)) + " backend not found: " + p.detail +
                      " (check backend_url / backends.*.url in config.json)");
        else
            log_info(std::string(level_name(lv)) + " backend is found: " + b->url +
                     " -> " + (b->model.empty() ? "(no model)" : b->model));
    }
    if (judge_backend == "http") {
        if (auto p = probe_models(judge_url); !p.ok)
            log_error("judge model not found: " + p.detail + " (check judge_url in config.json)");
        else
            log_info("judge model is found");
    } else {
        // 首次运行会在后台下载模型（见 judge_laya.h）：此时尚未就绪是预期行为，不是错误
        if (!lode_judge_laya::init_laya_judge())
            log_warn("judge model is not ready yet (see the log line above)");
        else
            log_info("judge model is found");
    }

    httplib::Server svr;

    // httplib 默认优先设置 SO_REUSEPORT：端口已被占用时 bind 依然会"成功"，
    // 内核把请求随机分给两个进程，故障极难定位。这里只设 SO_REUSEADDR，
    // 让端口冲突干脆地表现为 bind 失败。
    svr.set_socket_options([](socket_t sock) {
        httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
    });

    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"}
    });
    svr.Options("/.*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json resp = {{"object", "list"}, {"data", json::array()}};
        resp["data"].push_back({{"id", "auto"}, {"object", "model"}, {"created", 0}, {"owned_by", "router"}});
        // 再把三个挡位各自的模型列出来，客户端 /models 时能看到真实可用的名字
        for (Level lv : {Level::Easy, Level::Middle, Level::Hard}) {
            const Backend& b = backend_for(lv);
            if (b.model.empty()) continue;
            bool dup = false;
            for (const auto& item : resp["data"]) {
                if (item.value("id", "") == b.model) { dup = true; break; }
            }
            if (!dup) {
                resp["data"].push_back({{"id", b.model}, {"object", "model"}, {"created", 0},
                                        {"owned_by", std::string(level_name(lv)) + "-backend"}});
            }
        }
        res.set_content(resp.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        std::string body = req.body;
        // 没有判定结果时（比如没带 messages）回退到 easy 挡位，与旧行为一致
        const Backend* target = &backend_easy;
        try {
            json reqj = json::parse(body);
            if (reqj.contains("messages")) {
                // 裁判的输入是压缩后的整段上下文：多轮里的约束、报错、重试次数、
                // 会话起点意图都是难度信号，只喂最后一条 user 消息会全丢掉。
                lode_ctx::Summary ctx = lode_ctx::summarize(reqj["messages"]);
                if (!ctx.text.empty()) {
                    Level lv = ask_judge(ctx.text);
                    target = &backend_for(lv);
                    reqj["model"] = target->model;
                    log_info(std::string(level_name(lv)) + " -> " +
                             (target->model.empty() ? "(no model)" : target->model) +
                             " @ " + target->url + " (ctx " +
                             std::to_string(ctx.tokens) + " tok / " +
                             std::to_string(ctx.turns) + " turn / " +
                             std::to_string(ctx.error_patches) + " err)");
                }
            }
            body = reqj.dump();
        } catch (const std::exception& e) {
            log_error("ERROR: " + std::string(e.what()));
        } catch (...) {
            log_error("Unknown ERROR");
        }
        if (!target->ready()) {
            res.status = 502;
            res.set_content("{\"error\":{\"message\":\"backend url/model is not configured for this level\",\"type\":\"api_error\"}}", "application/json");
            return;
        }
        // 目标挡位的 url / key 在判定之后才确定，所以转发头在这里拼
        const std::string target_url = target->url;
        httplib::Headers fwd_headers;
        add_backend_auth(fwd_headers, target->api_key);
        std::string fwd_ct = req.get_header_value("Content-Type");
        if (!fwd_ct.empty()) { fwd_headers.emplace("Content-Type", fwd_ct); }
        res.set_chunked_content_provider("text/event-stream", [body, fwd_headers, target_url](size_t offset, httplib::DataSink& sink) {
            auto forward = [&](const char* data, size_t len) -> bool {
                sink.write(data, len);
                return true;
            };
            // 每个请求用独立的 Client：httplib 的 Server 是多线程的，
            // 共享同一个 Client 并发发送会产生数据竞争。
            httplib::Client cli(target_url);
            auto res = cli.Post("/v1/chat/completions", fwd_headers, body, "application/json", forward, nullptr);
            if (!res) {
                log_error("backend unreachable: " + target_url + " (" +
                          httplib::to_string(res.error()) + ")");
            } else if (res->status != 200) {
                // 响应头此时已经作为 200 发出去了，客户端只会看到空回复；
                // 至少把真实状态码和错误内容留在日志里（云端大多是 key / 配额问题）。
                log_error("backend " + target_url + " answered HTTP " +
                          std::to_string(res->status) + " (check api_key / model for this level)");
            }
            sink.done();
            return true;
        });
    });

    // 先 bind 再宣布 "listening"：否则端口被占时日志会谎报启动成功
    if (!svr.bind_to_port(host, port)) {
        log_error("failed to listen on " + host + ":" + std::to_string(port) +
                  " (port already in use?)");
        return 1;
    }
    log_info("router listening on " + host + ":" + std::to_string(port));
    svr.listen_after_bind();
    return 0;
}

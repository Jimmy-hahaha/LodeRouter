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

// ───────── 路由服务：按难度挡位把 /v1/chat/completions 转发到对应后端 ─────────

// ───────── 后端鉴权：每个挡位用自己的 api_key，客户端凭据一律不透传 ─────────
inline void add_backend_auth(httplib::Headers& headers, const std::string& key) {
    if (!key.empty()) headers.emplace("Authorization", "Bearer " + key);
}

// ───────── 可用性探测：GET {base_url}/v1/models 返回 200 才算后端可用 ─────────
struct Probe {
    bool ok = false;
    std::string detail;
};

inline std::string endpoint_of(const httplib::Client& cli) {
    return cli.host() + ":" + std::to_string(cli.port());
}

inline Probe probe_models(const std::string& base_url) {
    if (base_url.empty()) return {false, "url is empty (backend_url/judge_url not set?)"};

    httplib::Client cli(base_url);
    cli.set_connection_timeout(3);
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

// ───────── 端口冲突检查：后端与本机监听地址重合属于配置错误，直接退出 ─────────
inline bool is_local_host(const std::string& h) {
    return h.empty() || h == "0.0.0.0" || h == "127.0.0.1" || h == "localhost" ||
           h == "::" || h == "::1" || h == "[::1]";
}

inline Probe check_router_port_conflict(const std::string& base_url) {
    if (base_url.empty() || !is_local_host(host)) return {true, {}};
    httplib::Client cli(base_url);
    if (!cli.is_valid() || !is_local_host(cli.host()) || cli.port() != port) return {true, {}};
    return {false, endpoint_of(cli) + " is the same address this router listens on (" +
                   host + ":" + std::to_string(port) +
                   "); change port or point it at the real backend"};
}

// ───────── 启动自检：按 url 去重后检查地址冲突并探测各挡位与裁判 ─────────
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
        if (!lode_judge_laya::init_laya_judge())
            log_warn("judge model is not ready yet (see the log line above)");
        else
            log_info("judge model is found");
    }

    // ───────── HTTP 服务：监听选项、CORS 预检与模型列表 ─────────
    httplib::Server svr;

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

    // ───────── 转发接口：判定挡位后把请求转给对应后端并流式回传 ─────────
    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        std::string body = req.body;
        const Backend* target = &backend_easy;
        try {
            json reqj = json::parse(body);
            if (reqj.contains("messages")) {
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
            httplib::Client cli(target_url);
            auto res = cli.Post("/v1/chat/completions", fwd_headers, body, "application/json", forward, nullptr);
            if (!res) {
                log_error("backend unreachable: " + target_url + " (" +
                          httplib::to_string(res.error()) + ")");
            } else if (res->status != 200) {
                log_error("backend " + target_url + " answered HTTP " +
                          std::to_string(res->status) + " (check api_key / model for this level)");
            }
            sink.done();
            return true;
        });
    });

    // ───────── 启动：bind 成功后才宣布 listening 并进入监听 ─────────
    if (!svr.bind_to_port(host, port)) {
        log_error("failed to listen on " + host + ":" + std::to_string(port) +
                  " (port already in use?)");
        return 1;
    }
    log_info("router listening on " + host + ":" + std::to_string(port));
    svr.listen_after_bind();
    return 0;
}

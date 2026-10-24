#pragma once
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../include/cpp-httplib/httplib.h"
#include "../include/json/json.hpp"
#include <algorithm>
#include <cctype>
#include "config.h"
#include "judge_laya.h"
#include "log.h"
using json = nlohmann::json;

// ───────── http 裁判：调 judge_url 上的 OpenAI 兼容接口，按回复里的词判类别 ─────────
inline Level ask_judge_http(const std::string& input) {
    json req = {
        {"messages", {
            {{"role", "user"}, {"content", "给下面的请求分类，只回答一个词：chitchat(闲聊)、math(数学)、code(代码)。\n请求：" + input}}
        }},
        {"max_tokens", 5}
    };
    httplib::Headers headers;
    if (!judge_api_key.empty()) { headers.emplace("Authorization", "Bearer " + judge_api_key); }
    httplib::Client cli(judge_url);
    auto res = cli.Post("/v1/chat/completions", headers, req.dump(), "application/json");
    if (!res) {
        log_error("judge unreachable: " + judge_url);
        return Level::Easy;
    }
    try {
        auto resp = json::parse(res->body);
        std::string result = resp["choices"][0]["message"]["content"];
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (result.find("code") != std::string::npos) return Level::Hard;
        if (result.find("math") != std::string::npos) return Level::Middle;
    } catch (const std::exception& e) {
        log_error(e.what());
    }
    return Level::Easy;
}

// ───────── 裁判分派：按 judge_backend 选本地 laya 模型或 http 接口，只回挡位 ─────────
inline Level ask_judge(const std::string& input) {
    if (judge_backend == "http") return ask_judge_http(input);
    return lode_judge_laya::ask_judge_laya(input);
}

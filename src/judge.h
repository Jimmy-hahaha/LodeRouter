#pragma once
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../include/cpp-httplib/httplib.h"
#include "../include/json/json.hpp"
#include "config.h"
#include "judge_laya.h"
#include "log.h"
using json = nlohmann::json;

// 裁判后端之一：judge_url 上的 OpenAI 兼容接口
inline Level ask_judge_http(const std::string& input) {
    json req = {
        {"messages", {
            {{"role", "user"}, {"content", "给下面的问题做难度评级，只回答一个词：easy(闲聊)、middle(大多数问题)、hard(极复杂推理，高难度编程)。\n问题：" + input}}
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
        if (result.find("hard") != std::string::npos) return Level::Hard;
        if (result.find("middle") != std::string::npos) return Level::Middle;
    } catch (const std::exception& e) {
        log_error(e.what());
    }
    return Level::Easy;
}

// 按 judge_backend 分派：laya（本地 ONNX 裁判模型）或 http（远程接口）。
// 只回挡位，具体用哪个服务/模型由调用方按挡位去 backends 里查。
inline Level ask_judge(const std::string& input) {
    if (judge_backend == "http") return ask_judge_http(input);
    return lode_judge_laya::ask_judge_laya(input);
}

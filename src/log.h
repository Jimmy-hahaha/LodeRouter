#pragma once
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include "spdlog/spdlog.h"
#include "../include/spdlog/sinks/basic_file_sink.h"
#include "../include/spdlog/sinks/stdout_color_sinks.h"

// ───────── 日志：同时写 router.log 与终端，LODE_LOG_CONSOLE=0 可只写文件 ─────────
void init_logger() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("router.log"));

    const char* console_env = std::getenv("LODE_LOG_CONSOLE");
    bool console = !(console_env && (std::string(console_env) == "0" ||
                                     std::string(console_env) == "off"));
    if (console) {
        sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    }

    auto logger = std::make_shared<spdlog::logger>("router", sinks.begin(), sinks.end());
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->flush_on(spdlog::level::warn);
    spdlog::register_logger(logger);
}

// ───────── 输出封装：info / warn / error 三个级别，统一走 "router" logger ─────────
inline void log_info(const std::string& msg) {
    auto logger = spdlog::get("router");
    if (logger) logger->info("{}", msg);
    else spdlog::info("{}", msg);
}

inline void log_warn(const std::string& msg) {
    auto logger = spdlog::get("router");
    if (logger) logger->warn("{}", msg);
    else spdlog::warn("{}", msg);
}

inline void log_error(const std::string& msg) {
    auto logger = spdlog::get("router");
    if (logger) logger->error("{}", msg);
    else spdlog::error("{}", msg);
}

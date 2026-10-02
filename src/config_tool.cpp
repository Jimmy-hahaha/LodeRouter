#include <iostream>
#include <string>
#include "log.h"
#include "config.h"

namespace {

std::string ask(const std::string& name) {
    std::cout << name << ": " << std::flush;
    std::string v;
    std::getline(std::cin, v);
    return v;
}

std::string ask_hidden(const std::string& name) {
    std::cout << name << " (input is hidden): " << std::flush;
    return read_masked();
}

}

int main() {
    init_logger();

    fs::path file = cfg::locateConfigFile("LodeRouter");
    if (file.empty()) {
        log_error("unable to determine config path");
        return 1;
    }

    json config = cfg::defaultConfig();

    std::string host_in = ask("host");
    if (!host_in.empty()) config["host"] = host_in;

    std::string port_in = ask("port");
    if (!port_in.empty()) {
        try {
            config["port"] = std::stoi(port_in);
        } catch (const std::exception&) {
            std::cout << "invalid port '" << port_in << "', keeping default" << std::endl;
        }
    }

    config["backend_url"] = ask("backend_url");

    std::string judge_backend_in = ask("judge_backend (laya|http)");
    if (!judge_backend_in.empty()) {
        if (judge_backend_in == "laya" || judge_backend_in == "http") {
            config["judge_backend"] = judge_backend_in;
        } else {
            std::cout << "invalid judge_backend '" << judge_backend_in
                      << "', keeping default 'laya'" << std::endl;
        }
    }

    config["judge_url"] = ask("judge_url");
    config["judge_model_dir"] = ask("judge_model_dir (empty = auto-detect)");
    config["api_key"] = ask_hidden("api_key");
    config["judge_api_key"] = ask_hidden("judge_api_key");
    config["model_easy"] = ask("model_easy");
    config["model_middle"] = ask("model_middle");
    config["model_hard"] = ask("model_hard");

    // ───────── 挡位配置：三个挡位可各自指定 url / model / api_key，留空即沿用上面的默认值 ─────────
    std::cout << "\n每个挡位可以单独指定 url/model/api_key，留空即沿用上面的默认值。\n";
    for (const char* lv : {"easy", "middle", "hard"}) {
        std::cout << "-- " << lv << " --\n";
        std::string url = ask(std::string("backends.") + lv + ".url");
        std::string model = ask(std::string("backends.") + lv + ".model");
        std::string key = ask_hidden(std::string("backends.") + lv + ".api_key");
        if (!url.empty() || !model.empty() || !key.empty()) {
            config["backends"][lv]["url"] = url;
            config["backends"][lv]["model"] = model;
            config["backends"][lv]["api_key"] = key;
        }
    }

    try {
        cfg::saveJson(file, config);
        log_info("config saved to " + file.string());
        std::cout << "Configuration saved to " << file.string() << std::endl;
    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
    return 0;
}

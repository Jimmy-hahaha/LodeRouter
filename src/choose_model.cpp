#include "../include/httplib.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <filesystem>
#include "../include/cfgpath/cfgpath.h"
using json = nlohmann::json;
namespace fs = std::filesystem;

#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RESET   "\033[0m"
#define LOG_ERROR(msg_e) do { \
    auto now = std::chrono::system_clock::now(); \
    auto time = std::chrono::system_clock::to_time_t(now); \
    std::string time_str = std::ctime(&time); \
    time_str.pop_back(); \
    std::cerr << '[' << COLOR_RED << "ERROR" << COLOR_RESET << "] " << time_str << " - " << msg_e << std::endl; \
    } while(0)

static std::string get_config_path(){
    char cfgdir[MAX_PATH];
    try{
        get_user_config_file(cfgdir, sizeof(cfgdir), "LodeRouter");
        fs::create_directories(cfgdir);
    }
    catch(...){
        LOG_ERROR("Can't save the config");
    }
    return std::string(cfgdir) + "/config.json";
}
static std::string config_get(const std::string& key, const std::string& def = "") {
    try {
        std::ifstream file(get_config_path());
        json data = json::parse(file);
        if (data.contains(key) && data[key].is_string()) {
            std::string v = data[key].get<std::string>();
            if (!v.empty()) return v;
        }
    } catch (...) {
    }
    return def;
}

static std::string get_judge_url()     { return config_get("judge_url"); }
static std::string get_judge_api_key() { return config_get("judge_api_key"); }

std::string ask_judge(const std::string& input) {
    static const std::string judge_url = get_judge_url();
    static const std::string judge_api_key = get_judge_api_key();
    static const std::string model_easy   = config_get("model_easy");
    static const std::string model_middle = config_get("model_middle");
    static const std::string model_hard   = config_get("model_hard");

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
    if (!res) return model_easy;
    auto resp = json::parse(res->body);
    std::string result = resp["choices"][0]["message"]["content"];
    if (result.find("hard") != std::string::npos) return model_hard;
    if (result.find("middle") != std::string::npos) return model_middle;
    return model_easy;
}

#include "../include/passwordPrompt_cpp/passwordPrompt.h"
#include <nlohmann/json.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include "../include/cfgpath/cfgpath.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RESET   "\033[0m"

#define LOG_INFO(msg) do { \
    auto now = std::chrono::system_clock::now(); \
    auto time = std::chrono::system_clock::to_time_t(now); \
    std::string time_str = std::ctime(&time); \
    time_str.pop_back(); \
    std::cerr << '[' << COLOR_GREEN << "INFO" << COLOR_RESET << "] " << time_str << " - " << msg << std::endl; \
    } while(0)

#define LOG_ERROR(msg_e) do { \
    auto now = std::chrono::system_clock::now(); \
    auto time = std::chrono::system_clock::to_time_t(now); \
    std::string time_str = std::ctime(&time); \
    time_str.pop_back(); \
    std::cerr << '[' << COLOR_RED << "ERROR" << COLOR_RESET << "] " << time_str << " - " << msg_e << std::endl; \
    } while(0)

std::string port, host, backend_url, judge_url_t, router_api_key;
std::string model_easy, model_middle, model_hard;

std::string getapikey(std::string prompt){
    std::cout << prompt;
    std::string pwd = PasswordPrompt::getPassword("");
    return pwd;
}

void create_config_interactively(std::string config_path) {
    std::cout << "port: ";          std::getline(std::cin, port);
    std::cout << "host: ";          std::getline(std::cin, host);
    std::cout << "backend_url: ";   std::getline(std::cin, backend_url);
    std::cout << "judge_url: ";     std::getline(std::cin, judge_url_t);
    router_api_key = getapikey("api_key: ");
    std::cout << "model_easy: ";    std::getline(std::cin, model_easy);
    std::cout << "model_middle: ";  std::getline(std::cin, model_middle);
    std::cout << "model_hard: ";    std::getline(std::cin, model_hard);

    json config;
    config["port"] = port;
    config["host"] = host;
    config["backend_url"] = backend_url;
    config["judge_url"] = judge_url_t;
    config["api_key"] = router_api_key;
    config["model_easy"] = model_easy;
    config["model_middle"] = model_middle;
    config["model_hard"] = model_hard;

    std::ofstream file(config_path);
    if (file.is_open()) {
        file << config.dump(4);
        file.close();
        std::string saved_info = "Config saved to" + config_path; 
        LOG_INFO("Config saved to ../config/config.json");
    } else {
        LOG_ERROR("failed to create config file");
    }
}

std::string get_config_path(){
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

int main(){
    
    create_config_interactively(get_config_path());
    return 0;
}
#include "../include/httplib.h"
#include <iostream>
#include "choose_model.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <filesystem>
#include "../include/cfgpath/cfgpath.h"
using json = nlohmann::json;
namespace fs = std::filesystem;

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

//---------------------------------------------------------------------------------------------------

json get_default_config() {
    return json{
        {"host", "127.0.0.1"},
        {"port", 8080},
        {"judge_url", ""},
        {"backend_url", ""},
        {"api_key", ""},
        {"model_easy", ""},
        {"model_middle", ""},
        {"model_hard", ""}
    };
}

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

int port;
std::string backend_url;
std::string host;
std::string judge_url_t;
std::string router_api_key;

httplib::Client* backend_client = nullptr;

bool check_auth(const httplib::Request& req) {
    if (router_api_key.empty()) return true;
    return req.get_header_value("Authorization") == "Bearer " + router_api_key;
}

void get_config(){
    std::string config_path = get_config_path();
    if (fs::exists(config_path)) {
        try{
            std::ifstream file(config_path);
            json data = json::parse(file);
            port = data["port"];
            host = data["host"];
            backend_url = data["backend_url"];
            judge_url_t = data["judge_url"];
            router_api_key = data.value("api_key", "");
      }
        catch(const std::exception& e){
            std::string err_msg_config = "ERROR: " + std::string(e.what());
            LOG_ERROR(err_msg_config);
        }
      }
    else{
      LOG_ERROR("Config file missing. Creating default config");
      fs::path dir = fs::path(config_path).parent_path();
        if (!dir.empty() && !fs::exists(dir)) {
            fs::create_directories(dir);
        }
        json config = get_default_config();
        std::ofstream out(config_path);
        if (out.is_open()) {
            out << config.dump(4) << std::endl;
            out.close();
            std::ifstream file(config_path);
            json data = json::parse(file);
            port = data["port"];
            host = data["host"];
            backend_url = data["backend_url"];
            judge_url_t = data["judge_url"];
            router_api_key = data.value("api_key", "");
        }
        else{
          LOG_ERROR("can't write config");
        }
    }
}


//--------------------------------------------------------------------------------------------------

bool is_alive(const std::string& base_url) {
    httplib::Client cli(base_url);
    auto res = cli.Get("/v1/models");
    return res && res->status == 200;
}

int main() {
    LOG_INFO("Program is Start");

    get_config();

    backend_client = new httplib::Client(backend_url);
    
    httplib::Server svr;
//外部支持访问------------------------------------------------------------------------------------------
    svr.set_default_headers({
           {"Access-Control-Allow-Origin", "*"},
           {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
           {"Access-Control-Allow-Headers", "Content-Type, Authorization"}
       });
    svr.Options("/.*", [](const httplib::Request&, httplib::Response& res) {
           res.status = 204;
    });
//------------------------------------------------------------------------------------------------------
    if (!is_alive(backend_url)) {
      LOG_ERROR("Backend not found");
    }
    else {
      LOG_INFO("Backend is found");
    }
    if (!is_alive(judge_url_t)) {
      LOG_ERROR("Judge model not found");
    }
    else{
      LOG_INFO("Judge model is found");
    }
//改写model字段-----------------------------------------------------------------------------------------
    svr.Get("/v1/models", [](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"Invalid API key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}", "application/json");
            return;
        }
        json resp = {{"object", "list"}, {"data", json::array()}};
        resp["data"].push_back({{"id", "auto"}, {"object", "model"}, {"created", 0}, {"owned_by", "router"}});
        res.set_content(resp.dump(), "application/json");
    });
//-------------------------------------------------------------------------------------------------------
    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"Invalid API key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}", "application/json");
            return;
        }
        std::string body = req.body;
        httplib::Headers fwd_headers;
        std::string auth = req.get_header_value("Authorization");
        if (!auth.empty()) { fwd_headers.emplace("Authorization", auth); }
        std::string fwd_ct = req.get_header_value("Content-Type");
        if (!fwd_ct.empty()) { fwd_headers.emplace("Content-Type", fwd_ct); }
        try {
            json reqj = json::parse(body);
            if (reqj.contains("messages")){
            auto& msgs = reqj["messages"];
              for(auto it = msgs.rbegin(); it != msgs.rend(); ++it){
                  if ((*it).value("role","")=="user" && (*it)["content"].is_string()){
                      reqj["model"] = ask_judge((*it)["content"]);
                      std::string model_name = reqj["model"];
                      std::string messages_info=model_name+" is start";
                      LOG_INFO(messages_info);
                      
                      break;
                   }
              }
          }
          body = reqj.dump();
        }
        catch(const std::exception& e){
          std::string err_msg = "ERROR: " + std::string(e.what());
          LOG_ERROR(err_msg);
        }
        catch(...){
          LOG_ERROR("Unknown ERROR");
        }
        res.set_chunked_content_provider("text/event-stream", [body, fwd_headers](size_t offset, httplib::DataSink& sink) {
                auto forward = [&](const char* data, size_t len) -> bool {
                sink.write(data, len);
                return true;
            };
            if (backend_client) {
                auto res = backend_client->Post("/v1/chat/completions", fwd_headers, body, "application/json", forward,nullptr);
                if (!res) {
                    std::string err = "backend unreachable:" + backend_url;
                    LOG_ERROR(err);
                }
            }
            sink.done();
            return true;
        });
    });
    svr.listen(host, port);
    return 0;
}
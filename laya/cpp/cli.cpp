// laya-cli: command line front end for the laya C++ runtime.
//
//   laya-cli info   --model DIR
//   laya-cli score  --model DIR --state TEXT --instruction TEXT --levels "a;b;c" [--json]
//   laya-cli choice --model DIR --state TEXT --instruction TEXT --options "a;b" [--json]
//   laya-cli noul   --model DIR --state TEXT --instruction TEXT [--false-criterion TEXT]
//                    [--true-criterion TEXT] [--json]
//   laya-cli encode --model DIR --text TEXT [--special] [--json]
//   laya-cli batch  --model DIR --input cases.jsonl [--output out.jsonl] [--dump-sequence]
//
// batch input lines: {"id": "...", "state": "...", "instruction": "...",
//                     "levels": ["...", "..."]}
// batch output lines: {"id","index","label","confidence","score","probabilities",
//                      "logits","state_ids","head_ids","sequence","markers"}
// (the last five only with --dump-sequence).

#include "laya.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

struct Args {
    std::string command;
    std::map<std::string, std::string> values;
    bool json = false;
    bool dump_sequence = false;
    bool special = false;
    bool help = false;
};

const std::vector<std::string> kFlagNames = {
    "json", "dump-sequence", "special", "help",
};

Args parse_args(int argc, char** argv) {
    Args args;
    if (argc < 2) {
        args.help = true;
        return args;
    }
    const std::string first = argv[1];
    if (first == "--help" || first == "-h" || first == "help") {
        args.help = true;
        return args;
    }
    args.command = first;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            throw std::invalid_argument("unexpected argument: '" + arg + "'");
        }
        std::string key = arg.substr(2);
        if (std::find(kFlagNames.begin(), kFlagNames.end(), key) != kFlagNames.end()) {
            if (key == "json") args.json = true;
            else if (key == "dump-sequence") args.dump_sequence = true;
            else if (key == "special") args.special = true;
            else if (key == "help") args.help = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for --" + key);
        }
        args.values[key] = argv[++i];
    }
    return args;
}

std::string require(const Args& args, const std::string& key) {
    auto it = args.values.find(key);
    if (it == args.values.end() || it->second.empty()) {
        throw std::invalid_argument("missing required option --" + key);
    }
    return it->second;
}

std::string optional(const Args& args, const std::string& key,
                     const std::string& fallback = {}) {
    auto it = args.values.find(key);
    return it == args.values.end() ? fallback : it->second;
}

std::vector<std::string> split_semicolons(const std::string& text) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t pos = text.find(';', start);
        if (pos == std::string::npos) {
            if (start < text.size() || !out.empty()) out.push_back(text.substr(start));
            break;
        }
        out.push_back(text.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

std::string read_stdin() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

std::string resolve_text(const std::string& text) {
    return text == "-" ? read_stdin() : text;
}

void print_json(const json& value) {
    std::cout << value.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
}

void print_probabilities(const std::vector<float>& probabilities,
                         const std::vector<std::string>* labels = nullptr) {
    std::cout << "probabilities:";
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        std::cout << "  [" << i << "] " << probabilities[i];
        if (labels && i < labels->size()) std::cout << "  " << (*labels)[i];
        std::cout << '\n';
    }
}

std::vector<std::string> levels_from_json(const json& line) {
    if (!line.contains("levels") || !line["levels"].is_array()) return {};
    std::vector<std::string> levels;
    for (const auto& level : line["levels"]) {
        if (!level.is_string()) {
            throw std::invalid_argument("batch 'levels' entries must be strings");
        }
        levels.push_back(level.get<std::string>());
    }
    return levels;
}

void print_usage() {
    std::cout << R"(laya-cli - pure C++ runtime for the laya decision model

usage:
  laya-cli --version
  laya-cli info   --model DIR
  laya-cli score  --model DIR --state TEXT --instruction TEXT --levels "a;b;c" [--json]
  laya-cli choice --model DIR --state TEXT --instruction TEXT --options "a;b" [--json]
  laya-cli noul   --model DIR --state TEXT --instruction TEXT [--false-criterion TEXT]
                   [--true-criterion TEXT] [--json]
  laya-cli encode --model DIR --text TEXT [--special] [--json]
  laya-cli batch  --model DIR --input cases.jsonl [--output out.jsonl] [--dump-sequence]

Use '-' as TEXT/state to read it from stdin. Batch entries are JSONL records:
  {"id": "row-1", "state": "...", "instruction": "...", "levels": ["...", "..."]}
)";
}

json score_json(const laya::ScoreResult& result) {
    json out;
    out["index"] = result.index;
    out["label"] = result.label;
    out["confidence"] = result.confidence;
    out["entropy_confidence"] = result.entropy_confidence;
    out["score"] = result.score;
    out["probabilities"] = result.probabilities;
    return out;
}

json trace_json(const laya::ScoreTrace& trace) {
    json out;
    out["state_ids"] = trace.state_ids;
    out["head_ids"] = trace.head_ids;
    out["sequence"] = trace.sequence;
    out["markers"] = trace.markers;
    out["logits"] = trace.logits;
    return out;
}

int run_info(const Args& args) {
    laya::Engine engine(require(args, "model"));
    json out;
    out["model_dir"] = engine.model_dir();
    out["max_len"] = engine.max_len();
    out["head_max_len"] = engine.head_max_len();
    out["vocab_size"] = engine.vocab_size();
    out["pad_token_id"] = engine.tokenizer().pad_id();
    out["bos_token_id"] = engine.tokenizer().bos_id();
    out["eos_token_id"] = engine.tokenizer().eos_id();
    out["mask_token_id"] = engine.tokenizer().mask_id();
    out["cls_token_id"] = engine.tokenizer().cls_id();
    out["sep_token_id"] = engine.tokenizer().sep_id();
    print_json(out);
    return 0;
}

int run_score(const Args& args) {
    laya::Engine engine(require(args, "model"));
    const std::string state = resolve_text(require(args, "state"));
    const std::string instruction = require(args, "instruction");
    const std::vector<std::string> levels = split_semicolons(require(args, "levels"));
    laya::ScoreTrace trace;
    laya::ScoreResult result = engine.score(state, instruction, levels, &trace);
    if (args.json) {
        json out = score_json(result);
        out["logits"] = trace.logits;
        print_json(out);
    } else {
        std::cout << "label:      " << result.label << '\n'
                  << "index:      " << result.index << '\n'
                  << "confidence: " << result.confidence << '\n'
                  << "entropy:    " << result.entropy_confidence << '\n'
                  << "score:      " << result.score << '\n';
        print_probabilities(result.probabilities, &levels);
    }
    return 0;
}

int run_choice(const Args& args) {
    laya::Engine engine(require(args, "model"));
    const std::string state = resolve_text(require(args, "state"));
    const std::string instruction = require(args, "instruction");
    const std::vector<std::string> options = split_semicolons(require(args, "options"));
    laya::ChoiceResult result = engine.choice(state, instruction, options);
    if (args.json) {
        json out;
        out["index"] = result.index;
        out["label"] = result.label;
        out["confidence"] = result.confidence;
        out["probabilities"] = result.probabilities;
        print_json(out);
    } else {
        std::cout << "label:      " << result.label << '\n'
                  << "index:      " << result.index << '\n'
                  << "confidence: " << result.confidence << '\n';
        print_probabilities(result.probabilities, &options);
    }
    return 0;
}

int run_noul(const Args& args) {
    laya::Engine engine(require(args, "model"));
    const std::string state = resolve_text(require(args, "state"));
    const std::string instruction = require(args, "instruction");
    laya::NoulResult result = engine.noul(state, instruction,
                                          optional(args, "false-criterion"),
                                          optional(args, "true-criterion"));
    if (args.json) {
        json out;
        out["value"] = result.value;
        out["label"] = result.label;
        out["probability"] = result.probability;
        out["confidence"] = result.confidence;
        print_json(out);
    } else {
        std::cout << "label:       " << result.label << '\n'
                  << "P(true):     " << result.probability << '\n'
                  << "confidence:  " << result.confidence << '\n';
    }
    return 0;
}

int run_encode(const Args& args) {
    laya::Engine engine(require(args, "model"));
    const std::string text = resolve_text(require(args, "text"));
    const std::vector<std::int32_t> ids = engine.encode(text, args.special);
    if (args.json) {
        json out;
        out["ids"] = ids;
        out["tokens"] = json::array();
        for (std::int32_t id : ids) out["tokens"].push_back(engine.tokenizer().token(id));
        print_json(out);
    } else {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i] << ' ' << engine.tokenizer().token(ids[i]) << '\n';
        }
    }
    return 0;
}

int run_batch(const Args& args) {
    laya::Engine engine(require(args, "model"));
    const std::string input = require(args, "input");
    const std::string output = optional(args, "output");
    const std::string default_instruction = optional(args, "instruction");

    std::ifstream in(input);
    if (!in) throw std::runtime_error("cannot open input file '" + input + "'");
    std::ofstream out_file;
    std::ostream* out = &std::cout;
    if (!output.empty()) {
        out_file.open(output, std::ios::trunc);
        if (!out_file) throw std::runtime_error("cannot open output file '" + output + "'");
        out = &out_file;
    }

    std::string line;
    std::size_t index = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json record;
        try {
            record = json::parse(line);
        } catch (const std::exception& e) {
            throw std::runtime_error("input line " + std::to_string(index + 1) +
                                     ": invalid JSON: " + e.what());
        }
        if (!record.is_object()) {
            throw std::runtime_error("input line " + std::to_string(index + 1) +
                                     ": expected a JSON object");
        }
        const std::string id = record.contains("id") && record["id"].is_string()
            ? record["id"].get<std::string>()
            : std::to_string(index);
        const std::string state = record.value("state", std::string{});
        const std::string instruction = record.contains("instruction") &&
                                                record["instruction"].is_string()
            ? record["instruction"].get<std::string>()
            : default_instruction;
        std::vector<std::string> levels = levels_from_json(record);
        if (levels.empty()) levels = split_semicolons(optional(args, "levels"));
        if (levels.empty()) {
            throw std::runtime_error("input line " + std::to_string(index + 1) +
                                     ": missing 'levels'");
        }
        if (instruction.empty()) {
            throw std::runtime_error("input line " + std::to_string(index + 1) +
                                     ": missing 'instruction'");
        }

        laya::ScoreTrace trace;
        laya::ScoreResult result = engine.score(state, instruction, levels, &trace);
        json row = score_json(result);
        row["id"] = id;
        row["logits"] = trace.logits;
        if (args.dump_sequence) {
            json t = trace_json(trace);
            for (auto it = t.begin(); it != t.end(); ++it) row[it.key()] = it.value();
        }
        (*out) << row.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
        ++index;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.help || args.command.empty() || args.command == "help") {
            print_usage();
            return 0;
        }
        if (args.command == "--version" || args.command == "version") {
            std::cout << "laya-cli 1.0.0" << '\n';
            return 0;
        }
        if (args.command == "info") return run_info(args);
        if (args.command == "score") return run_score(args);
        if (args.command == "choice") return run_choice(args);
        if (args.command == "noul") return run_noul(args);
        if (args.command == "encode") return run_encode(args);
        if (args.command == "batch") return run_batch(args);
        throw std::invalid_argument("unknown command '" + args.command + "'");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "laya-cli: %s\n", e.what());
        return 1;
    }
}

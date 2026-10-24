// Tokenizer parity test: compares the C++ tokenizer against the Hugging Face
// reference corpus written by tools/laya_oracle/dump_reference.py.
//
//   laya-tokenizer-parity <tokenizer.json> <tokenizer_corpus.jsonl>
//
// Each corpus row has "text", "ids" (add_special_tokens=false) and
// "ids_trunc48" (truncation=True, max_length=48). The test also checks that
// add_special_tokens=true at max_length=48 reserves the two template tokens
// exactly like HF: [bos] + content[:46] + [eos].

#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

bool same(const std::vector<std::int32_t>& got, const std::vector<int>& want) {
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) return false;
    }
    return true;
}

void print_mismatch(const char* label, const json& rec, const std::vector<int>& want,
                    const std::vector<std::int32_t>& got) {
    std::fprintf(stderr, "MISMATCH %s id=%s text=%s\n", label,
                 rec.value("id", "?").c_str(), json(rec.value("text", "")).dump().c_str());
    std::fprintf(stderr, "  hf :");
    for (int v : want) std::fprintf(stderr, " %d", v);
    std::fprintf(stderr, "\n  cpp:");
    for (auto v : got) std::fprintf(stderr, " %d", v);
    std::fprintf(stderr, "\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <tokenizer.json> <tokenizer_corpus.jsonl>\n", argv[0]);
        return 2;
    }
    try {
        const auto tokenizer = laya::Tokenizer::from_file(argv[1]);
        std::ifstream in(argv[2]);
        if (!in) {
            std::fprintf(stderr, "cannot open corpus '%s'\n", argv[2]);
            return 2;
        }

        std::size_t checked = 0;
        std::size_t id_bad = 0;
        std::size_t trunc_bad = 0;
        std::size_t special_bad = 0;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const json rec = json::parse(line);
            const std::string text = rec.at("text").get<std::string>();
            const std::vector<int> expected = rec.at("ids").get<std::vector<int>>();
            const std::vector<std::int32_t> got = tokenizer->encode(text, false);
            ++checked;
            if (!same(got, expected)) {
                if (++id_bad <= 10) print_mismatch("ids", rec, expected, got);
            }
            if (rec.contains("ids_trunc48")) {
                const std::vector<int> expected_trunc =
                    rec.at("ids_trunc48").get<std::vector<int>>();
                const std::vector<std::int32_t> got_trunc =
                    tokenizer->encode(text, false, 48);
                if (!same(got_trunc, expected_trunc)) {
                    if (++trunc_bad <= 5) {
                        print_mismatch("trunc48", rec, expected_trunc, got_trunc);
                    }
                }
            }
            const std::vector<std::int32_t> full = tokenizer->encode(text, false);
            const std::vector<std::int32_t> special =
                tokenizer->encode(text, true, 48);
            std::vector<std::int32_t> wanted;
            wanted.push_back(tokenizer->bos_id());
            const std::size_t keep = std::min<std::size_t>(full.size(), 46);
            wanted.insert(wanted.end(), full.begin(),
                          full.begin() + static_cast<std::ptrdiff_t>(keep));
            wanted.push_back(tokenizer->eos_id());
            if (special != wanted) {
                if (++special_bad <= 5) {
                    std::fprintf(stderr, "MISMATCH special/trunc id=%s text=%s\n",
                                 rec.value("id", "?").c_str(),
                                 json(text).dump().c_str());
                }
            }
        }

        // A few decode checks mirror the HF decoder: Replace(▁ -> " "),
        // ByteFallback and Fuse, plus skip_special_tokens.
        struct DecodeCase {
            std::string text;
            bool skip_special;
            std::string expected;
        };
        const std::vector<DecodeCase> decode_cases = {
            {"hello world", false, " hello world"},
            {"你好，世界", false, " 你好，世界"},
            {"a<bos>b", false, " a<bos> b"},
            {"a<bos>b", true, " a b"},
            {"<mask>hi", false, "<mask> hi"},
            {"<mask>hi", true, " hi"},
            {"  leading spaces", false, "  leading spaces"},
            {"a\tb\nc", false, " a\tb\n c"},
            {std::string(1, '\0'), false, std::string(" \0", 2)},
        };
        std::size_t decode_bad = 0;
        for (const auto& test : decode_cases) {
            const std::vector<std::int32_t> ids = tokenizer->encode(test.text, false);
            const std::string got = tokenizer->decode(ids, test.skip_special);
            if (got != test.expected) {
                ++decode_bad;
                std::printf("decode mismatch text=%s skip_special=%d: got '%s', expected '%s'\n",
                            json(test.text).dump().c_str(), static_cast<int>(test.skip_special),
                            json(got).dump().c_str(), json(test.expected).dump().c_str());
            }
        }
        const std::string replacement = "\xEF\xBF\xBD";
        const std::int32_t byte_80 = tokenizer->token_id("<0x80>");
        if (byte_80 >= 0) {
            const std::string got = tokenizer->decode({byte_80}, false);
            if (got != replacement) {
                ++decode_bad;
                std::printf("decode mismatch <0x80>: got '%s', expected U+FFFD\n",
                            json(got).dump().c_str());
            }
        }

        std::printf("tokenizer parity: %zu rows, ids mismatches=%zu, trunc48 mismatches=%zu, "
                    "special/trunc mismatches=%zu, decode mismatches=%zu\n",
                    checked, id_bad, trunc_bad, special_bad, decode_bad);
        return (id_bad == 0 && trunc_bad == 0 && special_bad == 0 && decode_bad == 0) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tokenizer parity: ERROR: %s\n", e.what());
        return 1;
    }
}

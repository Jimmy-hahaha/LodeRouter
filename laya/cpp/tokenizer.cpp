#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace laya {
namespace {

constexpr std::uint32_t kMetaspace = 0x2581;  // U+2581 LOWER ONE EIGHTH BLOCK
constexpr std::uint32_t kReplacement = 0xFFFD;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("laya tokenizer: cannot open '" + path + "'");
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("laya tokenizer: cannot read '" + path + "'");
    }
    return data;
}

void append_utf8(std::uint32_t cp, std::string& out) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        append_utf8(kReplacement, out);
    }
}

// Decode one UTF-8 code point. Invalid bytes are consumed one at a time and
// surfaced as U+FFFD, matching Python's 'replace' error handler closely enough
// for the laya pipeline (all real inputs are valid UTF-8).
std::uint32_t decode_utf8(std::string_view text, std::size_t& index) {
    const auto byte = [&](std::size_t i) {
        return static_cast<unsigned char>(text[i]);
    };
    const unsigned char c0 = byte(index);
    if (c0 < 0x80) {
        ++index;
        return c0;
    }
    std::size_t len = 0;
    std::uint32_t cp = 0;
    if ((c0 & 0xE0) == 0xC0) {
        len = 2;
        cp = c0 & 0x1F;
    } else if ((c0 & 0xF0) == 0xE0) {
        len = 3;
        cp = c0 & 0x0F;
    } else if ((c0 & 0xF8) == 0xF0) {
        len = 4;
        cp = c0 & 0x07;
    } else {
        ++index;
        return kReplacement;
    }
    if (index + len > text.size()) {
        ++index;
        return kReplacement;
    }
    for (std::size_t k = 1; k < len; ++k) {
        const unsigned char cc = byte(index + k);
        if ((cc & 0xC0) != 0x80) {
            ++index;
            return kReplacement;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    // Reject overlong forms, surrogates and out-of-range values.
    if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
        (len == 4 && cp < 0x10000) || cp > 0x10FFFF ||
        (cp >= 0xD800 && cp <= 0xDFFF)) {
        ++index;
        return kReplacement;
    }
    index += len;
    return cp;
}

bool is_whitespace(std::uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

struct CodePoint {
    std::uint32_t value = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
};

std::vector<CodePoint> decode_codepoints(std::string_view text) {
    std::vector<CodePoint> cps;
    cps.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t begin = i;
        const std::uint32_t cp = decode_utf8(text, i);
        cps.push_back({cp, begin, i});
    }
    return cps;
}

std::string normalize_spaces(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        if (c == ' ') {
            append_utf8(kMetaspace, out);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

bool valid_utf8(std::string_view bytes) {
    std::size_t i = 0;
    while (i < bytes.size()) {
        const unsigned char c0 = static_cast<unsigned char>(bytes[i]);
        if (c0 < 0x80) {
            ++i;
            continue;
        }
        std::size_t len = 0;
        std::uint32_t cp = 0;
        if ((c0 & 0xE0) == 0xC0) {
            len = 2;
            cp = c0 & 0x1F;
        } else if ((c0 & 0xF0) == 0xE0) {
            len = 3;
            cp = c0 & 0x0F;
        } else if ((c0 & 0xF8) == 0xF0) {
            len = 4;
            cp = c0 & 0x07;
        } else {
            return false;
        }
        if (i + len > bytes.size()) return false;
        for (std::size_t k = 1; k < len; ++k) {
            const unsigned char cc = static_cast<unsigned char>(bytes[i + k]);
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
            (len == 4 && cp < 0x10000) || cp > 0x10FFFF ||
            (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += len;
    }
    return true;
}

// ───────── SAX handler: extracts exactly the fields the pipeline needs ─────────
struct AddedToken {
    std::int32_t id = 0;
    std::string content;
    bool lstrip = false;
    bool rstrip = false;
    bool special = false;
};

struct MergeInfo {
    std::uint32_t rank = 0;
    std::int32_t result = -1;
};

class TokenizerJson final : public nlohmann::json_sax<nlohmann::json> {
public:
    using number_integer_t = nlohmann::json::number_integer_t;
    using number_unsigned_t = nlohmann::json::number_unsigned_t;
    using number_float_t = nlohmann::json::number_float_t;
    using string_t = nlohmann::json::string_t;
    using binary_t = nlohmann::json::binary_t;

    std::string error;
    std::string model_type;
    bool byte_fallback = false;
    std::string normalizer_type;
    std::string pre_tokenizer_type;
    std::string post_processor_type;
    std::string decoder_type;
    std::unordered_map<std::string, std::int32_t> vocab;
    std::unordered_map<std::uint64_t, MergeInfo> merges;
    std::vector<AddedToken> added_tokens;

    bool null() override { return true; }

    bool boolean(bool value) override {
        if (in_added_token_) {
            if (last_key_ == "lstrip") current_added_.lstrip = value;
            else if (last_key_ == "rstrip") current_added_.rstrip = value;
            else if (last_key_ == "special") current_added_.special = value;
        } else if (model_depth_ > 0 && depth_ == model_depth_ && last_key_ == "byte_fallback") {
            byte_fallback = value;
        }
        return true;
    }

    bool number_integer(number_integer_t value) override {
        return number_impl(value, value >= 0);
    }

    bool number_unsigned(number_unsigned_t value) override {
        return number_impl(static_cast<std::int64_t>(value), true);
    }

    bool number_float(number_float_t, const string_t&) override { return true; }

    bool string(string_t& value) override {
        if (in_added_token_) {
            if (last_key_ == "content") current_added_.content = value;
            return true;
        }
        if (pair_depth_ > 0 && depth_ == pair_depth_) {
            merge_pair_.push_back(value);
            return true;
        }
        if (merges_depth_ > 0 && depth_ == merges_depth_ + 1) {
            // Newer tokenizers serialise a merge as "left right".
            const auto space = value.find(' ');
            if (space == std::string::npos) {
                return fail("malformed merge entry: '" + value + "'");
            }
            merge_pair_.clear();
            merge_pair_.push_back(value.substr(0, space));
            merge_pair_.push_back(value.substr(space + 1));
            return consume_merge_pair();
        }
        if (model_depth_ > 0 && depth_ == model_depth_ && last_key_ == "type") {
            model_type = value;
            return true;
        }
        if (normalizer_depth_ > 0 && depth_ == normalizer_depth_ && last_key_ == "type") {
            normalizer_type = value;
            return true;
        }
        if (pre_tokenizer_depth_ > 0 && depth_ == pre_tokenizer_depth_ &&
            last_key_ == "type") {
            pre_tokenizer_type = value;
            return true;
        }
        if (post_processor_depth_ > 0 && depth_ == post_processor_depth_ &&
            last_key_ == "type") {
            post_processor_type = value;
            return true;
        }
        if (decoder_depth_ > 0 && depth_ == decoder_depth_ && last_key_ == "type") {
            decoder_type = value;
            return true;
        }
        return true;
    }

    bool binary(binary_t&) override { return true; }

    bool start_object(std::size_t) override {
        ++depth_;
        if (added_array_depth_ > 0 && depth_ == added_array_depth_ + 1) {
            in_added_token_ = true;
            current_added_ = AddedToken{};
        } else if (depth_ == 2 && last_key_ == "model") {
            model_depth_ = 2;
        } else if (model_depth_ > 0 && depth_ == model_depth_ + 1 &&
                   last_key_ == "vocab") {
            vocab_depth_ = depth_;
        } else if (depth_ == 2 && last_key_ == "normalizer") {
            normalizer_depth_ = 2;
        } else if (depth_ == 2 && last_key_ == "pre_tokenizer") {
            pre_tokenizer_depth_ = 2;
        } else if (depth_ == 2 && last_key_ == "post_processor") {
            post_processor_depth_ = 2;
        } else if (depth_ == 2 && last_key_ == "decoder") {
            decoder_depth_ = 2;
        }
        return true;
    }

    bool key(string_t& value) override {
        last_key_ = value;
        if (vocab_depth_ > 0 && depth_ == vocab_depth_) {
            pending_vocab_key_ = value;
            have_pending_key_ = true;
        }
        return true;
    }

    bool end_object() override {
        if (in_added_token_ && added_array_depth_ > 0 &&
            depth_ == added_array_depth_ + 1) {
            added_tokens.push_back(std::move(current_added_));
            in_added_token_ = false;
        } else if (vocab_depth_ > 0 && depth_ == vocab_depth_) {
            vocab_depth_ = 0;
        } else if (model_depth_ > 0 && depth_ == model_depth_) {
            model_depth_ = 0;
        } else if (normalizer_depth_ > 0 && depth_ == normalizer_depth_) {
            normalizer_depth_ = 0;
        } else if (pre_tokenizer_depth_ > 0 && depth_ == pre_tokenizer_depth_) {
            pre_tokenizer_depth_ = 0;
        } else if (post_processor_depth_ > 0 && depth_ == post_processor_depth_) {
            post_processor_depth_ = 0;
        } else if (decoder_depth_ > 0 && depth_ == decoder_depth_) {
            decoder_depth_ = 0;
        }
        --depth_;
        return true;
    }

    bool start_array(std::size_t) override {
        ++depth_;
        if (depth_ == 2 && last_key_ == "added_tokens") {
            added_array_depth_ = 2;
        } else if (model_depth_ > 0 && depth_ == model_depth_ + 1 &&
                   last_key_ == "merges") {
            merges_depth_ = depth_;
        } else if (merges_depth_ > 0 && depth_ == merges_depth_ + 1) {
            pair_depth_ = depth_;
            merge_pair_.clear();
        }
        return true;
    }

    bool end_array() override {
        if (pair_depth_ > 0 && depth_ == pair_depth_) {
            if (!consume_merge_pair()) return false;
            pair_depth_ = 0;
        } else if (merges_depth_ > 0 && depth_ == merges_depth_) {
            merges_depth_ = 0;
        } else if (added_array_depth_ > 0 && depth_ == added_array_depth_) {
            added_array_depth_ = 0;
        }
        --depth_;
        return true;
    }

    bool parse_error(std::size_t position, const std::string& last_token,
                     const nlohmann::detail::exception& ex) override {
        error = "parse error at byte " + std::to_string(position) + " near '" + last_token +
                "': " + ex.what();
        return false;
    }

private:
    bool fail(std::string message) {
        if (error.empty()) error = std::move(message);
        return false;
    }

    bool number_impl(std::int64_t value, bool non_negative) {
        if (in_added_token_) {
            if (last_key_ == "id") {
                if (!non_negative || value > std::numeric_limits<std::int32_t>::max()) {
                    return fail("added token id is out of range");
                }
                current_added_.id = static_cast<std::int32_t>(value);
            }
            return true;
        }
        if (vocab_depth_ > 0 && depth_ == vocab_depth_ && have_pending_key_) {
            if (!non_negative || value > std::numeric_limits<std::int32_t>::max()) {
                return fail("vocab id is out of range");
            }
            vocab.emplace(std::move(pending_vocab_key_), static_cast<std::int32_t>(value));
            have_pending_key_ = false;
            return true;
        }
        return true;
    }

    bool consume_merge_pair() {
        if (merge_pair_.size() != 2) {
            return fail("merge entry does not have two tokens");
        }
        const auto left = vocab.find(merge_pair_[0]);
        const auto right = vocab.find(merge_pair_[1]);
        if (left == vocab.end() || right == vocab.end()) {
            return fail("merge references an unknown token: '" + merge_pair_[0] + "' + '" +
                        merge_pair_[1] + "'");
        }
        const auto merged = vocab.find(merge_pair_[0] + merge_pair_[1]);
        if (merged == vocab.end()) {
            return fail("merge result is missing from vocab: '" + merge_pair_[0] +
                        merge_pair_[1] + "'");
        }
        const std::uint64_t key = (static_cast<std::uint64_t>(
                                       static_cast<std::uint32_t>(left->second))
                                   << 32) |
                                  static_cast<std::uint32_t>(right->second);
        merges.emplace(key, MergeInfo{next_rank_++, merged->second});
        merge_pair_.clear();
        return true;
    }

    std::size_t depth_ = 0;
    std::string last_key_;
    std::size_t added_array_depth_ = 0;
    std::size_t model_depth_ = 0;
    std::size_t vocab_depth_ = 0;
    std::size_t merges_depth_ = 0;
    std::size_t pair_depth_ = 0;
    std::size_t normalizer_depth_ = 0;
    std::size_t pre_tokenizer_depth_ = 0;
    std::size_t post_processor_depth_ = 0;
    std::size_t decoder_depth_ = 0;
    bool in_added_token_ = false;
    AddedToken current_added_;
    std::string pending_vocab_key_;
    bool have_pending_key_ = false;
    std::vector<std::string> merge_pair_;
    std::uint32_t next_rank_ = 0;
};

std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
}

}  // namespace

struct Tokenizer::Impl {
    std::unordered_map<std::string, std::int32_t> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::int32_t, std::size_t> added_index;
    std::vector<AddedToken> added;
    std::unordered_map<std::uint64_t, MergeInfo> merges;
    std::unordered_map<std::uint32_t, std::int32_t> char_token;
    std::array<std::int32_t, 256> byte_token{};
    std::int32_t pad_id = -1;
    std::int32_t bos_id = -1;
    std::int32_t eos_id = -1;
    std::int32_t unk_id = -1;
    std::int32_t mask_id = -1;

    struct TrieNode {
        std::unordered_map<std::uint32_t, std::int32_t> next;
        std::int32_t added = -1;
    };
    std::vector<TrieNode> trie;

    struct Segment {
        std::int32_t added_id = -1;
        std::size_t begin = 0;
        std::size_t end = 0;
    };
    struct Match {
        std::size_t begin = 0;
        std::size_t end = 0;
        std::size_t added = 0;
    };

    void load(const std::string& path);
    void build_indexes();
    void build_trie();

    std::vector<Match> find_matches(const std::vector<CodePoint>& cps) const;
    std::vector<Segment> extract(const std::vector<CodePoint>& cps) const;
    std::vector<std::string> metaspace_pieces(std::string_view normalized) const;
    std::vector<std::int32_t> bpe_word(std::string_view word) const;
    std::vector<std::int32_t> encode_raw(std::string_view text) const;
    std::string token_of(std::int32_t id) const;
};

void Tokenizer::Impl::load(const std::string& path) {
    const std::string data = read_file(path);
    TokenizerJson sax;
    if (!nlohmann::json::sax_parse(data, &sax)) {
        throw std::runtime_error("laya tokenizer: invalid '" + path + "': " +
                                 (sax.error.empty() ? "JSON parse failed" : sax.error));
    }
    if (sax.model_type != "BPE") {
        throw std::runtime_error("laya tokenizer: unsupported model type '" + sax.model_type +
                                 "' (expected BPE)");
    }
    if (sax.normalizer_type != "Replace") {
        throw std::runtime_error("laya tokenizer: unsupported normalizer '" +
                                 sax.normalizer_type + "' (expected Replace)");
    }
    if (sax.pre_tokenizer_type != "Metaspace") {
        throw std::runtime_error("laya tokenizer: unsupported pre_tokenizer '" +
                                 sax.pre_tokenizer_type + "' (expected Metaspace)");
    }
    vocab = std::move(sax.vocab);
    merges = std::move(sax.merges);
    added = std::move(sax.added_tokens);
    build_indexes();
    build_trie();
}

void Tokenizer::Impl::build_indexes() {
    if (vocab.empty()) {
        throw std::runtime_error("laya tokenizer: empty vocabulary");
    }
    std::int32_t max_id = -1;
    for (const auto& [token, id] : vocab) {
        (void)token;
        if (id < 0) throw std::runtime_error("laya tokenizer: negative vocab id");
        max_id = std::max(max_id, id);
    }
    id_to_token.assign(static_cast<std::size_t>(max_id) + 1, std::string{});
    for (const auto& [token, id] : vocab) {
        id_to_token[static_cast<std::size_t>(id)] = token;
    }

    added_index.clear();
    for (std::size_t i = 0; i < added.size(); ++i) {
        added_index[added[i].id] = i;
    }

    const auto require = [this](const char* token) {
        const auto it = vocab.find(token);
        if (it == vocab.end()) {
            throw std::runtime_error(std::string("laya tokenizer: missing special token '") +
                                     token + "'");
        }
        return it->second;
    };
    pad_id = require("<pad>");
    bos_id = require("<bos>");
    eos_id = require("<eos>");
    unk_id = require("<unk>");
    mask_id = require("<mask>");

    char_token.clear();
    char_token.reserve(vocab.size());
    for (const auto& [token, id] : vocab) {
        std::size_t index = 0;
        const std::uint32_t cp = decode_utf8(token, index);
        if (index == token.size()) {
            char_token.emplace(cp, id);
        }
    }

    byte_token.fill(-1);
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        const auto it = vocab.find(name);
        if (it != vocab.end()) byte_token[static_cast<std::size_t>(b)] = it->second;
    }
}

void Tokenizer::Impl::build_trie() {
    trie.clear();
    trie.emplace_back();
    for (std::size_t i = 0; i < added.size(); ++i) {
        const std::string& content = added[i].content;
        if (content.empty()) continue;
        std::int32_t node = 0;
        std::size_t index = 0;
        while (index < content.size()) {
            const std::uint32_t cp = decode_utf8(content, index);
            const auto it = trie[static_cast<std::size_t>(node)].next.find(cp);
            if (it == trie[static_cast<std::size_t>(node)].next.end()) {
                const std::int32_t next = static_cast<std::int32_t>(trie.size());
                trie.emplace_back();
                trie[static_cast<std::size_t>(node)].next.emplace(cp, next);
                node = next;
            } else {
                node = it->second;
            }
        }
        trie[static_cast<std::size_t>(node)].added = static_cast<std::int32_t>(i);
    }
}

std::vector<Tokenizer::Impl::Match> Tokenizer::Impl::find_matches(
    const std::vector<CodePoint>& cps) const {
    std::vector<Match> matches;
    std::size_t p = 0;
    while (p < cps.size()) {
        std::int32_t node = 0;
        std::int32_t best = -1;
        std::size_t best_end = p;
        std::size_t i = p;
        while (i < cps.size()) {
            const auto it = trie[static_cast<std::size_t>(node)].next.find(cps[i].value);
            if (it == trie[static_cast<std::size_t>(node)].next.end()) break;
            node = it->second;
            ++i;
            if (trie[static_cast<std::size_t>(node)].added >= 0) {
                best = trie[static_cast<std::size_t>(node)].added;
                best_end = i;
            }
        }
        if (best < 0) {
            ++p;
            continue;
        }
        matches.push_back({p, best_end, static_cast<std::size_t>(best)});
        p = best_end;
    }
    return matches;
}

std::vector<Tokenizer::Impl::Segment> Tokenizer::Impl::extract(
    const std::vector<CodePoint>& cps) const {
    std::vector<Segment> segments;
    if (cps.empty()) return segments;

    std::vector<Match> matches = find_matches(cps);
    std::size_t cursor = 0;
    for (const Match& match : matches) {
        std::size_t begin = match.begin;
        std::size_t end = match.end;
        const AddedToken& token = added[match.added];
        if (token.lstrip) {
            std::size_t j = begin;
            while (j > cursor && is_whitespace(cps[j - 1].value)) --j;
            begin = std::max(j, cursor);
        }
        if (token.rstrip) {
            std::size_t j = end;
            while (j < cps.size() && is_whitespace(cps[j].value)) ++j;
            end = j;
        }
        if (cursor < begin) {
            segments.push_back({-1, cursor, begin});
        }
        segments.push_back({token.id, begin, end});
        cursor = end;
    }
    if (cursor < cps.size()) {
        segments.push_back({-1, cursor, cps.size()});
    }
    return segments;
}

std::vector<std::string> Tokenizer::Impl::metaspace_pieces(
    std::string_view normalized) const {
    std::vector<std::string> pieces;
    if (normalized.empty()) return pieces;

    std::vector<std::size_t> offsets;
    std::size_t i = 0;
    while (i < normalized.size()) offsets.push_back(i), decode_utf8(normalized, i);
    if (offsets.empty()) return pieces;

    // prepend U+2581 unless the text already starts with it
    bool starts_with_metaspace = false;
    {
        std::size_t j = 0;
        const std::uint32_t first = decode_utf8(normalized, j);
        starts_with_metaspace = first == kMetaspace;
    }

    std::vector<std::size_t> cuts;
    for (std::size_t off : offsets) {
        std::size_t j = off;
        if (decode_utf8(normalized, j) == kMetaspace) cuts.push_back(off);
    }
    if (!starts_with_metaspace) {
        std::string prepended;
        prepended.reserve(normalized.size() + 3);
        append_utf8(kMetaspace, prepended);
        prepended.append(normalized);
        return metaspace_pieces(prepended);
    }
    for (std::size_t k = 0; k < cuts.size(); ++k) {
        const std::size_t begin = cuts[k];
        const std::size_t end = (k + 1 < cuts.size()) ? cuts[k + 1] : normalized.size();
        pieces.emplace_back(normalized.substr(begin, end - begin));
    }
    return pieces;
}

std::vector<std::int32_t> Tokenizer::Impl::bpe_word(std::string_view word) const {
    std::vector<std::int32_t> symbols;
    symbols.reserve(word.size());
    std::size_t index = 0;
    while (index < word.size()) {
        const std::size_t begin = index;
        const std::uint32_t cp = decode_utf8(word, index);
        const auto it = char_token.find(cp);
        if (it != char_token.end()) {
            symbols.push_back(it->second);
            continue;
        }
        const std::string_view bytes = word.substr(begin, index - begin);
        bool fallback_ok = true;
        for (const unsigned char b : bytes) {
            if (byte_token[b] < 0) {
                fallback_ok = false;
                break;
            }
        }
        if (fallback_ok) {
            for (const unsigned char b : bytes) symbols.push_back(byte_token[b]);
        } else {
            symbols.push_back(unk_id);
        }
    }

    while (symbols.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best = 0;
        bool found = false;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merges.find(pair_key(symbols[i], symbols[i + 1]));
            if (it != merges.end() && it->second.rank < best_rank) {
                best_rank = it->second.rank;
                best = i;
                found = true;
            }
        }
        if (!found) break;
        const auto it = merges.find(pair_key(symbols[best], symbols[best + 1]));
        symbols[best] = it->second.result;
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    return symbols;
}

std::vector<std::int32_t> Tokenizer::Impl::encode_raw(std::string_view text) const {
    std::vector<std::int32_t> ids;
    const std::vector<CodePoint> cps = decode_codepoints(text);
    const std::vector<Segment> segments = extract(cps);
    for (const Segment& segment : segments) {
        if (segment.added_id >= 0) {
            ids.push_back(segment.added_id);
            continue;
        }
        if (segment.begin >= segment.end) continue;
        const std::size_t byte_begin = cps[segment.begin].begin;
        const std::size_t byte_end = cps[segment.end - 1].end;
        const std::string normalized =
            normalize_spaces(std::string_view(text).substr(byte_begin, byte_end - byte_begin));
        for (const std::string& piece : metaspace_pieces(normalized)) {
            const std::vector<std::int32_t> word_ids = bpe_word(piece);
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }
    }
    return ids;
}

std::string Tokenizer::Impl::token_of(std::int32_t id) const {
    const auto added_it = added_index.find(id);
    if (added_it != added_index.end()) return added[added_it->second].content;
    if (id >= 0 && static_cast<std::size_t>(id) < id_to_token.size()) {
        return id_to_token[static_cast<std::size_t>(id)];
    }
    return {};
}

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::unique_ptr<Tokenizer> Tokenizer::from_file(const std::string& tokenizer_json_path) {
    auto tokenizer = std::unique_ptr<Tokenizer>(new Tokenizer());
    tokenizer->impl_->load(tokenizer_json_path);
    return tokenizer;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text,
                                            bool add_special_tokens) const {
    return encode(text, add_special_tokens, 0);
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text,
                                            bool add_special_tokens,
                                            std::size_t max_length) const {
    std::vector<std::int32_t> ids = impl_->encode_raw(text);
    if (add_special_tokens) {
        constexpr std::size_t kSpecialTokens = 2;  // <bos> + <eos>
        if (max_length > 0 && max_length >= kSpecialTokens &&
            ids.size() > max_length - kSpecialTokens) {
            ids.resize(max_length - kSpecialTokens);
        }
        ids.insert(ids.begin(), impl_->bos_id);
        ids.push_back(impl_->eos_id);
    } else if (max_length > 0 && ids.size() > max_length) {
        ids.resize(max_length);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids,
                              bool skip_special_tokens) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (const std::int32_t id : ids) {
        const auto added_it = impl_->added_index.find(id);
        if (added_it != impl_->added_index.end()) {
            const AddedToken& token = impl_->added[added_it->second];
            if (skip_special_tokens && token.special) continue;
            tokens.push_back(token.content);
            continue;
        }
        if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_token.size()) continue;
        const std::string& token = impl_->id_to_token[static_cast<std::size_t>(id)];
        if (token.empty()) continue;
        tokens.push_back(token);
    }

    for (std::string& token : tokens) {
        std::string replaced;
        replaced.reserve(token.size());
        std::size_t index = 0;
        while (index < token.size()) {
            const std::size_t begin = index;
            const std::uint32_t cp = decode_utf8(token, index);
            if (cp == kMetaspace) {
                replaced.push_back(' ');
            } else {
                replaced.append(token, begin, index - begin);
            }
        }
        token = std::move(replaced);
    }

    std::string output;
    std::string pending;
    const auto flush = [&]() {
        if (pending.empty()) return;
        if (valid_utf8(pending)) {
            output.append(pending);
        } else {
            for (std::size_t i = 0; i < pending.size(); ++i) append_utf8(kReplacement, output);
        }
        pending.clear();
    };

    for (const std::string& token : tokens) {
        if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' &&
            token[5] == '>') {
            const auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            const int hi = hex(token[3]);
            const int lo = hex(token[4]);
            if (hi >= 0 && lo >= 0) {
                pending.push_back(static_cast<char>((hi << 4) | lo));
                continue;
            }
        }
        flush();
        output.append(token);
    }
    flush();
    return output;
}

std::string Tokenizer::token(std::int32_t id) const { return impl_->token_of(id); }

std::int32_t Tokenizer::token_id(std::string_view token) const {
    const auto it = impl_->vocab.find(std::string(token));
    return it == impl_->vocab.end() ? -1 : it->second;
}

bool Tokenizer::has_token(std::string_view token) const { return token_id(token) >= 0; }

bool Tokenizer::is_special(std::int32_t id) const {
    const auto it = impl_->added_index.find(id);
    return it != impl_->added_index.end() && impl_->added[it->second].special;
}

std::size_t Tokenizer::vocab_size() const { return impl_->id_to_token.size(); }

std::int32_t Tokenizer::pad_id() const { return impl_->pad_id; }
std::int32_t Tokenizer::bos_id() const { return impl_->bos_id; }
std::int32_t Tokenizer::eos_id() const { return impl_->eos_id; }
std::int32_t Tokenizer::unk_id() const { return impl_->unk_id; }
std::int32_t Tokenizer::mask_id() const { return impl_->mask_id; }
std::int32_t Tokenizer::cls_id() const { return impl_->bos_id; }
std::int32_t Tokenizer::sep_id() const { return impl_->eos_id; }
std::string Tokenizer::mask_token() const { return token(impl_->mask_id); }
std::string Tokenizer::sep_token() const { return token(impl_->eos_id); }

std::string Tokenizer::normalize(std::string_view text) const {
    return normalize_spaces(text);
}

std::vector<std::string> Tokenizer::pre_tokenize(std::string_view text) const {
    std::vector<std::string> pieces;
    const std::vector<CodePoint> cps = decode_codepoints(text);
    for (const Impl::Segment& segment : impl_->extract(cps)) {
        if (segment.added_id >= 0) {
            pieces.push_back(impl_->token_of(segment.added_id));
            continue;
        }
        if (segment.begin >= segment.end) continue;
        const std::size_t byte_begin = cps[segment.begin].begin;
        const std::size_t byte_end = cps[segment.end - 1].end;
        const std::string normalized =
            normalize_spaces(text.substr(byte_begin, byte_end - byte_begin));
        for (std::string& piece : impl_->metaspace_pieces(normalized)) {
            pieces.push_back(std::move(piece));
        }
    }
    return pieces;
}

}  // namespace laya

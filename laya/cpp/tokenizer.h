#pragma once

// laya tokenizer: a dependency-free C++ implementation of the Hugging Face
// "tokenizer.json" pipeline used by the laya checkpoints.
//
// Supported pipeline (exactly what the shipped laya models use):
//   * AddedVocabulary extraction for "added_tokens" (special and non-special),
//     including the lstrip/rstrip flags.
//   * Normalizer: Replace(" " -> "\u2581") (U+2581, LOWER ONE EIGHTH BLOCK).
//   * PreTokenizer: Metaspace(replacement="\u2581", prepend_scheme="always",
//     split=true).
//   * Model: BPE with byte fallback (unknown characters become "<0xNN>" tokens).
//   * PostProcessor: TemplateProcessing (<bos> A <eos>) when
//     add_special_tokens is requested.
//   * Decoder: Replace("\u2581" -> " ") + ByteFallback + Fuse.
//
// The class is immutable after construction, so encode()/decode() are safe to
// call concurrently.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace laya {

class Tokenizer {
public:
    // Throws std::runtime_error on I/O errors, malformed JSON, or an
    // unsupported tokenizer pipeline.
    static std::unique_ptr<Tokenizer> from_file(const std::string& tokenizer_json_path);

    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Encode UTF-8 text. add_special_tokens=true wraps the sequence with the
    // post-processor template (<bos> ... <eos>). max_length>0 right-truncates
    // the final id sequence (matching HF truncation=True, max_length=N).
    std::vector<std::int32_t> encode(std::string_view text,
                                     bool add_special_tokens = false) const;
    std::vector<std::int32_t> encode(std::string_view text,
                                     bool add_special_tokens,
                                     std::size_t max_length) const;

    // Decode ids back to text. skip_special_tokens drops added/special tokens.
    std::string decode(const std::vector<std::int32_t>& ids,
                       bool skip_special_tokens = true) const;

    // Vocabulary lookup.
    std::string token(std::int32_t id) const;
    std::int32_t token_id(std::string_view token) const;   // -1 when absent
    bool has_token(std::string_view token) const;
    bool is_special(std::int32_t id) const;
    std::size_t vocab_size() const;

    // Special ids/tokens from the model configuration.
    std::int32_t pad_id() const;
    std::int32_t bos_id() const;
    std::int32_t eos_id() const;
    std::int32_t unk_id() const;
    std::int32_t mask_id() const;
    std::int32_t cls_id() const;
    std::int32_t sep_id() const;
    std::string mask_token() const;
    std::string sep_token() const;

    // Introspection helpers, used by the parity tests. normalize() applies the
    // normalizer to a whole string; pre_tokenize() applies added-token
    // extraction, normalization and metaspace splitting, in that order.
    std::string normalize(std::string_view text) const;
    std::vector<std::string> pre_tokenize(std::string_view text) const;

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace laya

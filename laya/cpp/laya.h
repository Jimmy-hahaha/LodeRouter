#pragma once

// Pure C++ runtime for the laya decision model (ModernBERT encoder + decision
// head exported to ONNX). It reproduces the Hugging Face reference pipeline
// (laya.common.build_sequence + laya.onnx_agent.ONNXAgent) token-for-token and
// score-for-score on CPU, with no Python at runtime.
//
// Model directory layout:
//   model.onnx             exported decision graph
//   tokenizer.json         Hugging Face fast-tokenizer file
//   rl_agent_config.json   {"max_len": 512, "head_max_len": 256,
//                           "temperature": [t_choice, t_score, t_noul], ...}
//
// Example:
//   laya::Engine engine("path/to/laya");
//   auto r = engine.score("帮我写个快排", "Which category does this request belong to?",
//                         {"chitchat: ...", "math: ...", "code: ..."});
//   // r.index in [0, levels.size()), r.label == levels[r.index],
//   // r.confidence == max(probabilities)

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer.h"

namespace laya {

// Optional debug trace for a score call, used by laya-cli and the parity
// harness. It exposes the exact token sequence and raw logits.
struct ScoreTrace {
    std::vector<std::int32_t> state_ids;
    std::vector<std::int32_t> head_ids;
    std::vector<std::int32_t> sequence;
    std::vector<std::int32_t> markers;
    std::vector<float> logits;
};

// Result of a score question: levels are ordered from low to high.
struct ScoreResult {
    std::int32_t index = 0;                 // argmax level
    float score = 0.0f;                     // expected level: sum(i * p_i)
    float confidence = 0.0f;                // max(p_i), the calibrated answer confidence
    float entropy_confidence = 0.0f;        // 1 - H(p) / log(k)
    std::string label;                      // levels[index]
    std::vector<float> probabilities;       // softmax over levels, temperature-scaled
};

// Result of a choice question: exactly one option is selected.
struct ChoiceResult {
    std::int32_t index = 0;
    float confidence = 0.0f;                // max(p_i)
    std::string label;                      // options[index]
    std::vector<float> probabilities;
};

// Result of a noul (boolean) question.
struct NoulResult {
    bool value = false;                     // probability(true) >= probability(false)
    float probability = 0.0f;               // p(true)
    float confidence = 0.0f;                // max(p(true), p(false))
    std::string label;                      // "false" or "true"
};

struct EngineOptions {
    // 0 keeps the ONNX Runtime default. Use 1 for latency-sensitive single
    // requests, or the core count for throughput.
    int intra_op_threads = 0;
    int inter_op_threads = 0;
    bool enable_all_optimizations = true;
    bool verbose = false;

    // Overrides for the files inside the model directory.
    std::string onnx_filename = "model.onnx";
    std::string tokenizer_filename = "tokenizer.json";
    std::string config_filename = "rl_agent_config.json";

    // Overrides for rl_agent_config.json values.
    std::optional<int> max_len;
    std::optional<int> head_max_len;
};

class Engine {
public:
    // Throws std::runtime_error if the directory, model, tokenizer or config is
    // missing/invalid, or if ONNX Runtime cannot create the session.
    explicit Engine(const std::string& model_dir);
    Engine(const std::string& model_dir, const EngineOptions& options);
    ~Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Score questions: the model reads the state + instruction + one rendered
    // level per entry and returns a categorical distribution over levels.
    // levels must not be empty.
    ScoreResult score(std::string_view state,
                      std::string_view instruction,
                      const std::vector<std::string>& levels,
                      ScoreTrace* trace = nullptr);
    // Batched scoring, one ONNX run for all states. NOTE: this graph is
    // dynamically int8-quantized, and ONNX Runtime computes one activation
    // scale per tensor, so a batched run can differ numerically from scoring
    // the same states one at a time. Use score() for canonical per-request
    // values; score_batch() is a throughput path.
    std::vector<ScoreResult> score_batch(const std::vector<std::string>& states,
                                         std::string_view instruction,
                                         const std::vector<std::string>& levels);

    // Choice questions: pick exactly one option.
    ChoiceResult choice(std::string_view state,
                        std::string_view instruction,
                        const std::vector<std::string>& options);

    // Noul questions: decide whether the statement holds. Optional criteria
    // describe the "false" and "true" answers (matching the Python noul
    // question's criteria dict); empty criteria use the laya defaults.
    NoulResult noul(std::string_view state,
                    std::string_view instruction,
                    std::string_view false_criterion = {},
                    std::string_view true_criterion = {});

    // Tokenizer access (exposed for tests/CLI, immutable and thread-safe).
    const Tokenizer& tokenizer() const;
    std::vector<std::int32_t> encode(std::string_view text,
                                     bool add_special_tokens = false) const;
    std::string decode(const std::vector<std::int32_t>& ids) const;

    // Effective configuration.
    const std::string& model_dir() const noexcept;
    int max_len() const noexcept;
    int head_max_len() const noexcept;
    std::size_t vocab_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace laya

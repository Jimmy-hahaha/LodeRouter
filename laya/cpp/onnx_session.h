#pragma once

// Thin, owning wrapper around an ONNX Runtime CPU session for the laya
// decision graph. The laya graph takes five tensors:
//
//   input_ids      int64   [batch, seq]
//   attention_mask int64   [batch, seq]
//   marker_pos     int64   [batch, markers]
//   marker_mask    bool    [batch, markers]
//   qtype          int64   [batch]
//
// and returns:
//
//   logits         float32 [batch, markers]
//   act_logits     float32 [batch, act_classes]
//
// The wrapper validates buffer sizes, reports ONNX Runtime errors as
// std::runtime_error, and serializes concurrent run() calls on the same
// session (ORT sessions are not re-entrant).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace laya {

struct ModelInputs {
    std::vector<std::int64_t> input_ids;       // batch * seq_len
    std::vector<std::int64_t> attention_mask;  // batch * seq_len
    std::vector<std::int64_t> marker_pos;      // batch * num_markers
    std::vector<std::uint8_t> marker_mask;     // batch * num_markers, 0/1
    std::vector<std::int64_t> qtype;           // batch
    std::int64_t batch = 0;
    std::int64_t seq_len = 0;
    std::int64_t num_markers = 0;
};

struct ModelOutputs {
    std::vector<float> logits;       // batch * num_markers
    std::vector<float> act_logits;   // batch * act_classes
    std::int64_t batch = 0;
    std::int64_t num_markers = 0;
    std::int64_t act_classes = 0;

    float logit(std::int64_t row, std::int64_t marker) const {
        return logits[static_cast<std::size_t>(row * num_markers + marker)];
    }
};

class OnnxSession {
public:
    struct Options {
        int intra_op_threads = 0;   // 0 -> ONNX Runtime default
        int inter_op_threads = 0;   // 0 -> ONNX Runtime default
        bool enable_all_optimizations = true;
        bool verbose = false;
    };

    explicit OnnxSession(const std::string& model_path);
    OnnxSession(const std::string& model_path, const Options& options);
    ~OnnxSession();
    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;
    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;

    ModelOutputs run(const ModelInputs& inputs);

    const std::string& model_path() const;
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace laya

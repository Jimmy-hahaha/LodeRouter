// Smoke test for laya::OnnxSession: runs the real laya ONNX decision graph once
// and exercises the wrapper's validation and error reporting.
//
// Usage:
//   onnx_session_smoke <model-directory | path/to/model.onnx>
//
// Set LAYA_ONNX_VERBOSE=1 to enable ONNX Runtime session logging on stderr.

#include "onnx_session.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace {

std::string resolve_model_file(const std::string& argument) {
    std::filesystem::path path(argument);
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) {
        path /= "model.onnx";
    }
    return path.string();
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void print_values(const char* label, const std::vector<float>& values) {
    std::cout << label << " [" << values.size() << "]: [";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << values[i];
    }
    std::cout << "]\n";
}

bool all_finite(const std::vector<float>& values) {
    for (const float value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

laya::ModelInputs make_valid_inputs() {
    laya::ModelInputs inputs;
    inputs.batch = 1;
    inputs.seq_len = 8;
    inputs.num_markers = 3;
    inputs.input_ids = {2, 476, 518, 30485, 235607, 9979, 1, 0};
    inputs.attention_mask = {1, 1, 1, 1, 1, 1, 1, 1};
    inputs.marker_pos = {2, 4, 6};
    inputs.marker_mask = {1, 1, 1};
    inputs.qtype = {1};
    return inputs;
}

// Runs an invalid input and requires a std::runtime_error whose message
// mentions the problematic tensor (or field).
void expect_rejected(laya::OnnxSession& session,
                     const laya::ModelInputs& inputs,
                     const std::string& expected_fragment,
                     const char* description) {
    try {
        const laya::ModelOutputs outputs = session.run(inputs);
        (void)outputs;
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        std::cout << "  " << description << "\n    -> std::runtime_error: " << message << "\n";
        require(message.find(expected_fragment) != std::string::npos,
                std::string("error message for '") + description + "' does not mention '" +
                    expected_fragment + "'");
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("'") + description +
                                 "' threw a non-runtime_error exception (" +
                                 typeid(error).name() + "): " + error.what());
    }
    throw std::runtime_error(std::string("invalid input '") + description +
                             "' was accepted without throwing");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <model-directory | path/to/model.onnx>\n";
        return 2;
    }

    try {
        const std::string model_file = resolve_model_file(argv[1]);
        std::cout << "onnx_session_smoke: model '" << model_file << "'\n";

        laya::OnnxSession::Options options;
        options.verbose = std::getenv("LAYA_ONNX_VERBOSE") != nullptr;

        laya::OnnxSession session(model_file, options);
        require(session.model_path() == model_file, "model_path() does not round-trip");

        const std::vector<std::string> input_names = session.input_names();
        const std::vector<std::string> output_names = session.output_names();
        const std::vector<std::string> expected_inputs{
            "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
        const std::vector<std::string> expected_outputs{"logits", "act_logits"};
        require(input_names == expected_inputs, "unexpected input_names()");
        require(output_names == expected_outputs, "unexpected output_names()");
        std::cout << "inputs:";
        for (const std::string& name : input_names) std::cout << ' ' << name;
        std::cout << "\noutputs:";
        for (const std::string& name : output_names) std::cout << ' ' << name;
        std::cout << '\n';

        const laya::ModelInputs inputs = make_valid_inputs();
        std::cout << "running batch=1 seq_len=8 markers=3 qtype=1 ...\n" << std::flush;
        const laya::ModelOutputs outputs = session.run(inputs);

        std::cout << "output shapes: logits [" << outputs.batch << ", " << outputs.num_markers
                  << "], act_logits [" << outputs.batch << ", " << outputs.act_classes << "]\n";
        require(outputs.batch == 1, "unexpected output batch");
        require(outputs.num_markers == 3, "unexpected output num_markers");
        require(outputs.act_classes == 2, "unexpected act_logits classes (expected 2)");
        require(outputs.logits.size() == 3, "logits size mismatch");
        require(outputs.act_logits.size() == 2, "act_logits size mismatch");
        require(all_finite(outputs.logits), "logits contain non-finite values");
        require(all_finite(outputs.act_logits), "act_logits contain non-finite values");
        print_values("logits", outputs.logits);
        print_values("act_logits", outputs.act_logits);
        std::cout << "logit(0,0) = " << outputs.logit(0, 0) << '\n';

        // The session must stay usable after a move.
        laya::OnnxSession moved(std::move(session));
        require(moved.model_path() == model_file, "moved session lost the model path");
        const laya::ModelOutputs moved_outputs = moved.run(inputs);
        require(moved_outputs.logits.size() == outputs.logits.size(),
                "moved session produced a different logits size");
        for (std::size_t i = 0; i < outputs.logits.size(); ++i) {
            require(std::fabs(moved_outputs.logits[i] - outputs.logits[i]) <= 1e-4f,
                    "moved session produced different logits");
        }
        std::cout << "re-run after move: identical logits\n";

        std::cout << "invalid input checks:\n";
        {
            laya::ModelInputs bad = inputs;
            bad.input_ids.pop_back();
            expect_rejected(moved, bad, "input_ids", "input_ids is one element short");
        }
        {
            laya::ModelInputs bad = inputs;
            bad.marker_mask = {1, 1};
            expect_rejected(moved, bad, "marker_mask", "marker_mask is one element short");
        }
        {
            laya::ModelInputs bad = inputs;
            bad.batch = 0;
            expect_rejected(moved, bad, "batch", "batch is zero");
        }
        {
            laya::ModelInputs bad = inputs;
            bad.marker_pos = {2, 4, 99};
            expect_rejected(moved, bad, "marker_pos", "marker_pos points past the sequence");
        }
        {
            laya::ModelInputs bad = inputs;
            bad.qtype.clear();
            expect_rejected(moved, bad, "qtype", "qtype is empty");
        }

        std::cout << "PASS: onnx_session_smoke\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

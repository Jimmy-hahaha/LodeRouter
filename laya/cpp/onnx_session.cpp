#include "onnx_session.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace laya {
namespace {

using int64 = std::int64_t;

// Tensor names of the laya decision graph. They are part of the exported model
// contract, so they are validated against the session metadata at construction.
constexpr const char* kInputIds = "input_ids";
constexpr const char* kAttentionMask = "attention_mask";
constexpr const char* kMarkerPos = "marker_pos";
constexpr const char* kMarkerMask = "marker_mask";
constexpr const char* kQtype = "qtype";
constexpr const char* kLogits = "logits";
constexpr const char* kActLogits = "act_logits";

// The order used to feed the graph and to expose input_names()/output_names().
constexpr std::array<const char*, 5> kInputNames{
    kInputIds, kAttentionMask, kMarkerPos, kMarkerMask, kQtype};
constexpr std::array<const char*, 2> kOutputNames{kLogits, kActLogits};

struct TensorSpec {
    const char* name;
    ONNXTensorElementDataType element_type;
    std::size_t rank;
};

constexpr std::array<TensorSpec, 5> kInputSpecs{{
    {kInputIds, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 2},
    {kAttentionMask, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 2},
    {kMarkerPos, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 2},
    {kMarkerMask, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 2},
    {kQtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 1},
}};

constexpr std::array<TensorSpec, 2> kOutputSpecs{{
    {kLogits, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 2},
    {kActLogits, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 2},
}};

// Metadata of one graph tensor as declared by the model (dynamic dimensions are -1).
struct TensorMeta {
    std::string name;
    ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64> shape;
};

const char* element_type_name(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        default:
            return "an unsupported element type";
    }
}

std::string shape_to_string(const std::vector<int64>& shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) out << ", ";
        out << shape[i];
    }
    out << ']';
    return out.str();
}

std::string names_to_string(const std::vector<std::string>& names) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i != 0) out << ", ";
        out << '\'' << names[i] << '\'';
    }
    out << ']';
    return out.str();
}

[[noreturn]] void fail(const std::string& context, const std::string& detail) {
    throw std::runtime_error(context + ": " + detail);
}

// ONNX Runtime requires a single Env per process. The C++ wrappers construct it
// lazily; a function-local static keeps it alive for the whole process. It is
// deliberately never destroyed so that sessions owned by static objects cannot
// outlive the runtime environment during shutdown.
Ort::Env& global_env() {
    static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "laya");
    return *env;
}

// Map every ONNX Runtime (and generic) exception onto std::runtime_error while
// keeping the original ORT message. std::runtime_error is rethrown untouched so
// validation errors keep their full context.
template <typename Fn>
decltype(auto) with_ort_errors(const std::string& context, Fn&& fn) {
    try {
        return fn();
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(context + ": ONNX Runtime error: " + e.what());
    } catch (const std::runtime_error&) {
        throw;
    } catch (const std::exception& e) {
        throw std::runtime_error(context + ": " + e.what());
    }
}

std::basic_string<ORTCHAR_T> native_model_path(const std::string& model_path) {
#if defined(_WIN32)
    return std::filesystem::path(model_path).wstring();
#else
    return model_path;
#endif
}

std::vector<std::string> read_tensor_names(const Ort::Session& session, bool inputs) {
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Ort::AllocatedStringPtr name =
            inputs ? session.GetInputNameAllocated(i, allocator)
                   : session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

TensorMeta read_tensor_meta(const Ort::TypeInfo& type_info,
                            const TensorSpec& spec,
                            const std::string& context,
                            const char* kind) {
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        fail(context, std::string(kind) + " '" + spec.name + "' is not a dense tensor");
    }
    const Ort::ConstTensorTypeAndShapeInfo info = type_info.GetTensorTypeAndShapeInfo();
    TensorMeta meta;
    meta.name = spec.name;
    meta.element_type = info.GetElementType();
    meta.shape = info.GetShape();
    if (meta.element_type != spec.element_type) {
        fail(context, std::string(kind) + " '" + spec.name + "' has element type " +
                          element_type_name(meta.element_type) + ", expected " +
                          element_type_name(spec.element_type));
    }
    if (meta.shape.size() != spec.rank) {
        fail(context, std::string(kind) + " '" + spec.name + "' has shape " +
                          shape_to_string(meta.shape) + " (rank " +
                          std::to_string(meta.shape.size()) + "), expected rank " +
                          std::to_string(spec.rank));
    }
    return meta;
}

std::size_t checked_product(std::size_t lhs,
                            std::size_t rhs,
                            const std::string& context,
                            const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        fail(context, std::string(what) + " does not fit in a buffer size");
    }
    return lhs * rhs;
}

// Validate an actual tensor shape against the shape declared in the model
// metadata. Negative metadata dimensions are dynamic and accept any value.
void check_static_shape(const TensorMeta& meta,
                        const std::vector<int64>& actual,
                        const std::string& context,
                        const char* kind) {
    if (actual.size() != meta.shape.size()) {
        fail(context, std::string(kind) + " '" + meta.name + "' has rank " +
                          std::to_string(actual.size()) + " (shape " + shape_to_string(actual) +
                          "), but the model metadata declares shape " +
                          shape_to_string(meta.shape));
    }
    for (std::size_t dim = 0; dim < actual.size(); ++dim) {
        if (meta.shape[dim] >= 0 && meta.shape[dim] != actual[dim]) {
            fail(context, std::string(kind) + " '" + meta.name + "' has shape " +
                              shape_to_string(actual) + ", but the model metadata declares shape " +
                              shape_to_string(meta.shape));
        }
    }
}

std::vector<int64> float_output_shape(const Ort::Value& value,
                                      const TensorMeta& meta,
                                      const std::string& context) {
    if (!value.IsTensor()) {
        fail(context, "output '" + meta.name + "' is not a dense tensor");
    }
    const Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
    const std::vector<int64> shape = info.GetShape();
    if (info.GetElementType() != meta.element_type) {
        fail(context, "output '" + meta.name + "' has element type " +
                          element_type_name(info.GetElementType()) + ", expected " +
                          element_type_name(meta.element_type));
    }
    if (shape.size() != 2) {
        fail(context, "output '" + meta.name + "' has shape " + shape_to_string(shape) +
                          " (rank " + std::to_string(shape.size()) + "), expected rank 2");
    }
    return shape;
}

std::vector<float> copy_float_output(const Ort::Value& value,
                                     const std::vector<int64>& shape,
                                     const std::string& context,
                                     const char* name) {
    std::size_t count = 1;
    for (const int64 dim : shape) {
        if (dim < 0) {
            fail(context, std::string("output '") + name +
                              "' has an unresolved dimension in shape " + shape_to_string(shape));
        }
        count = checked_product(count, static_cast<std::size_t>(dim), context,
                                "output element count");
    }
    const float* data = value.GetTensorData<float>();
    if (data == nullptr) {
        fail(context, std::string("output '") + name + "' has no data");
    }
    return std::vector<float>(data, data + count);
}

}  // namespace

struct OnnxSession::Impl {
    Impl(std::string model_path, const OnnxSession::Options& options);

    ModelOutputs run(const ModelInputs& inputs);
    std::string describe_request(const ModelInputs& inputs) const;
    void validate_inputs(const ModelInputs& inputs) const;

    std::string model_path_;
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::Session session_{nullptr};
    Ort::RunOptions run_options_{nullptr};
    std::array<TensorMeta, 5> inputs_;
    std::array<TensorMeta, 2> outputs_;
    std::mutex mutex_;
};

OnnxSession::Impl::Impl(std::string model_path, const OnnxSession::Options& options)
    : model_path_(std::move(model_path)) {
    const std::string context =
        "laya ONNX session: cannot initialize model '" + model_path_ + "'";

    with_ort_errors(context, [&]() {
        if (model_path_.empty()) {
            fail(context, "the model path is empty");
        }
        std::error_code ec;
        if (!std::filesystem::exists(model_path_, ec) || ec) {
            fail(context, "the model file does not exist");
        }
        if (!std::filesystem::is_regular_file(model_path_, ec) || ec) {
            fail(context, "the model path is not a regular file");
        }

        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        run_options_ = Ort::RunOptions();

        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(options.enable_all_optimizations
                                                       ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                                       : GraphOptimizationLevel::ORT_DISABLE_ALL);
        if (options.intra_op_threads > 0) {
            session_options.SetIntraOpNumThreads(options.intra_op_threads);
        }
        if (options.inter_op_threads > 0) {
            session_options.SetInterOpNumThreads(options.inter_op_threads);
        }
        // ONNX Runtime logs to stderr. Keep it at errors-only unless the caller
        // asked for verbose logging of the session.
        session_options.SetLogSeverityLevel(options.verbose
                                                ? static_cast<int>(ORT_LOGGING_LEVEL_VERBOSE)
                                                : static_cast<int>(ORT_LOGGING_LEVEL_ERROR));

        const std::basic_string<ORTCHAR_T> native_path = native_model_path(model_path_);
        session_ = Ort::Session(global_env(), native_path.c_str(), session_options);

        const std::vector<std::string> graph_inputs = read_tensor_names(session_, true);
        const std::vector<std::string> graph_outputs = read_tensor_names(session_, false);

        for (std::size_t i = 0; i < kInputSpecs.size(); ++i) {
            const TensorSpec& spec = kInputSpecs[i];
            const auto it = std::find(graph_inputs.begin(), graph_inputs.end(), spec.name);
            if (it == graph_inputs.end()) {
                fail(context, std::string("required input tensor '") + spec.name +
                                  "' is not exposed by the graph; graph inputs are " +
                                  names_to_string(graph_inputs));
            }
            const std::size_t index =
                static_cast<std::size_t>(std::distance(graph_inputs.begin(), it));
            inputs_[i] = read_tensor_meta(session_.GetInputTypeInfo(index), spec, context, "input");
        }

        for (std::size_t i = 0; i < kOutputSpecs.size(); ++i) {
            const TensorSpec& spec = kOutputSpecs[i];
            const auto it = std::find(graph_outputs.begin(), graph_outputs.end(), spec.name);
            if (it == graph_outputs.end()) {
                fail(context, std::string("required output tensor '") + spec.name +
                                  "' is not exposed by the graph; graph outputs are " +
                                  names_to_string(graph_outputs));
            }
            const std::size_t index =
                static_cast<std::size_t>(std::distance(graph_outputs.begin(), it));
            outputs_[i] =
                read_tensor_meta(session_.GetOutputTypeInfo(index), spec, context, "output");
        }
    });
}

std::string OnnxSession::Impl::describe_request(const ModelInputs& inputs) const {
    return "laya ONNX session for model '" + model_path_ + "' (batch=" +
           std::to_string(inputs.batch) + ", seq_len=" + std::to_string(inputs.seq_len) +
           ", num_markers=" + std::to_string(inputs.num_markers) + ")";
}

void OnnxSession::Impl::validate_inputs(const ModelInputs& inputs) const {
    const std::string context = describe_request(inputs);
    if (inputs.batch < 1) {
        fail(context, "ModelInputs.batch must be at least 1, got " +
                          std::to_string(inputs.batch));
    }
    if (inputs.seq_len < 1) {
        fail(context, "ModelInputs.seq_len must be at least 1, got " +
                          std::to_string(inputs.seq_len));
    }
    if (inputs.num_markers < 1) {
        fail(context, "ModelInputs.num_markers must be at least 1, got " +
                          std::to_string(inputs.num_markers));
    }

    const std::size_t batch = static_cast<std::size_t>(inputs.batch);
    const std::size_t seq_len = static_cast<std::size_t>(inputs.seq_len);
    const std::size_t markers = static_cast<std::size_t>(inputs.num_markers);
    const std::size_t seq_elements = checked_product(batch, seq_len, context, "batch * seq_len");
    const std::size_t marker_elements =
        checked_product(batch, markers, context, "batch * num_markers");

    const auto require_size = [&](const char* name,
                                  std::size_t actual,
                                  std::size_t expected,
                                  const std::string& expected_desc) {
        if (actual != expected) {
            fail(context, "tensor '" + std::string(name) + "' expects " + expected_desc +
                              " = " + std::to_string(expected) + " elements, but the buffer has " +
                              std::to_string(actual));
        }
    };

    const std::string seq_desc = "batch * seq_len (" + std::to_string(inputs.batch) + " * " +
                                 std::to_string(inputs.seq_len) + ")";
    const std::string marker_desc = "batch * num_markers (" + std::to_string(inputs.batch) +
                                    " * " + std::to_string(inputs.num_markers) + ")";
    require_size(kInputIds, inputs.input_ids.size(), seq_elements, seq_desc);
    require_size(kAttentionMask, inputs.attention_mask.size(), seq_elements, seq_desc);
    require_size(kMarkerPos, inputs.marker_pos.size(), marker_elements, marker_desc);
    require_size(kMarkerMask, inputs.marker_mask.size(), marker_elements, marker_desc);
    require_size(kQtype, inputs.qtype.size(), batch, "batch (" + std::to_string(inputs.batch) + ")");

    for (std::size_t i = 0; i < inputs.marker_mask.size(); ++i) {
        if (inputs.marker_mask[i] > 1) {
            fail(context, "tensor 'marker_mask' must contain only 0/1 values, but element " +
                              std::to_string(i) + " is " +
                              std::to_string(static_cast<unsigned>(inputs.marker_mask[i])));
        }
    }

    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t marker = 0; marker < markers; ++marker) {
            const std::size_t index = row * markers + marker;
            if (inputs.marker_mask[index] == 0) continue;
            const int64 position = inputs.marker_pos[index];
            if (position < 0 || position >= inputs.seq_len) {
                fail(context, "tensor 'marker_pos' entry [" + std::to_string(row) + ", " +
                                  std::to_string(marker) + "] = " + std::to_string(position) +
                                  " is outside the sequence (seq_len = " +
                                  std::to_string(inputs.seq_len) + ")");
            }
        }
    }

    const std::array<std::vector<int64>, 5> requested_shapes{{
        {inputs.batch, inputs.seq_len},
        {inputs.batch, inputs.seq_len},
        {inputs.batch, inputs.num_markers},
        {inputs.batch, inputs.num_markers},
        {inputs.batch},
    }};
    for (std::size_t i = 0; i < kInputSpecs.size(); ++i) {
        check_static_shape(inputs_[i], requested_shapes[i], context, "input");
    }
}

ModelOutputs OnnxSession::Impl::run(const ModelInputs& inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    validate_inputs(inputs);

    const std::string request_context = describe_request(inputs);
    const std::string context = request_context + ": inference failed";
    const auto batch = static_cast<std::size_t>(inputs.batch);
    const auto seq_len = static_cast<std::size_t>(inputs.seq_len);
    const auto markers = static_cast<std::size_t>(inputs.num_markers);
    const std::size_t seq_elements = batch * seq_len;
    const std::size_t marker_elements = batch * markers;

    const std::array<int64, 2> seq_shape{inputs.batch, inputs.seq_len};
    const std::array<int64, 2> marker_shape{inputs.batch, inputs.num_markers};
    const std::array<int64, 1> qtype_shape{inputs.batch};

    // The input tensors point directly at the caller's buffers, so run() does
    // not copy the token ids. ONNX Runtime only reads these buffers during Run().
    std::vector<Ort::Value> values;
    values.reserve(kInputNames.size());
    values.push_back(Ort::Value::CreateTensor<int64>(
        memory_info_, const_cast<int64*>(inputs.input_ids.data()), seq_elements, seq_shape.data(),
        seq_shape.size()));
    values.push_back(Ort::Value::CreateTensor<int64>(
        memory_info_, const_cast<int64*>(inputs.attention_mask.data()), seq_elements,
        seq_shape.data(), seq_shape.size()));
    values.push_back(Ort::Value::CreateTensor<int64>(
        memory_info_, const_cast<int64*>(inputs.marker_pos.data()), marker_elements,
        marker_shape.data(), marker_shape.size()));
    values.push_back(Ort::Value::CreateTensor(
        memory_info_,
        static_cast<void*>(const_cast<std::uint8_t*>(inputs.marker_mask.data())),
        marker_elements * sizeof(std::uint8_t), marker_shape.data(), marker_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
    values.push_back(Ort::Value::CreateTensor<int64>(
        memory_info_, const_cast<int64*>(inputs.qtype.data()), batch, qtype_shape.data(),
        qtype_shape.size()));

    std::vector<Ort::Value> results = with_ort_errors(context, [&]() {
        return session_.Run(run_options_, kInputNames.data(), values.data(), values.size(),
                            kOutputNames.data(), kOutputNames.size());
    });

    if (results.size() != kOutputNames.size()) {
        fail(request_context, "the model returned " + std::to_string(results.size()) +
                                  " outputs, but 2 were requested");
    }

    ModelOutputs outputs;
    outputs.batch = inputs.batch;
    outputs.num_markers = inputs.num_markers;

    {
        const std::vector<int64> shape = float_output_shape(results[0], outputs_[0], request_context);
        check_static_shape(outputs_[0], shape, request_context, "output");
        const std::vector<int64> expected{inputs.batch, inputs.num_markers};
        if (shape != expected) {
            fail(request_context, "output '" + outputs_[0].name + "' has shape " +
                                      shape_to_string(shape) +
                                      ", but [batch, num_markers] = " +
                                      shape_to_string(expected) + " was expected");
        }
        outputs.logits = copy_float_output(results[0], shape, request_context, kLogits);
    }

    {
        const std::vector<int64> shape = float_output_shape(results[1], outputs_[1], request_context);
        check_static_shape(outputs_[1], shape, request_context, "output");
        if (shape[0] != inputs.batch) {
            fail(request_context, "output '" + outputs_[1].name + "' has shape " +
                                      shape_to_string(shape) + ", but batch = " +
                                      std::to_string(inputs.batch) + " was expected");
        }
        if (shape[1] < 1) {
            fail(request_context, "output '" + outputs_[1].name + "' has no action classes: " +
                                      shape_to_string(shape));
        }
        outputs.act_classes = shape[1];
        outputs.act_logits = copy_float_output(results[1], shape, request_context, kActLogits);
    }

    return outputs;
}

OnnxSession::OnnxSession(const std::string& model_path) : OnnxSession(model_path, Options{}) {}

OnnxSession::OnnxSession(const std::string& model_path, const Options& options)
    : impl_(std::make_unique<Impl>(model_path, options)) {}

OnnxSession::~OnnxSession() = default;

OnnxSession::OnnxSession(OnnxSession&&) noexcept = default;

OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

ModelOutputs OnnxSession::run(const ModelInputs& inputs) {
    if (!impl_) {
        throw std::runtime_error("laya ONNX session: run() called on a moved-from OnnxSession");
    }
    return impl_->run(inputs);
}

const std::string& OnnxSession::model_path() const {
    if (!impl_) {
        throw std::runtime_error(
            "laya ONNX session: model_path() called on a moved-from OnnxSession");
    }
    return impl_->model_path_;
}

std::vector<std::string> OnnxSession::input_names() const {
    if (!impl_) {
        throw std::runtime_error(
            "laya ONNX session: input_names() called on a moved-from OnnxSession");
    }
    std::vector<std::string> names;
    names.reserve(kInputNames.size());
    for (const char* name : kInputNames) {
        names.emplace_back(name);
    }
    return names;
}

std::vector<std::string> OnnxSession::output_names() const {
    if (!impl_) {
        throw std::runtime_error(
            "laya ONNX session: output_names() called on a moved-from OnnxSession");
    }
    std::vector<std::string> names;
    names.reserve(kOutputNames.size());
    for (const char* name : kOutputNames) {
        names.emplace_back(name);
    }
    return names;
}

}  // namespace laya

#include "laya.h"

#include "onnx_session.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace laya {
namespace {

constexpr int kQtypeChoice = 0;
constexpr int kQtypeScore = 1;
constexpr int kQtypeNoul = 2;

constexpr double kTempMin = 0.5;
constexpr double kTempMax = 5.0;

constexpr const char* kMaskFallback = "<mask>";

std::string replace_all(std::string text, std::string_view from, std::string_view to) {
    if (from.empty()) return text;
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("laya: cannot open '" + path.string() + "'");
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

double clamp_temperature(double t) {
    if (!std::isfinite(t)) return 1.0;
    if (t < kTempMin) return kTempMin;
    if (t > kTempMax) return kTempMax;
    return t;
}

const char* qtype_name(int qtype) {
    switch (qtype) {
        case kQtypeChoice: return "choice";
        case kQtypeScore: return "score";
        case kQtypeNoul: return "noul";
        default: return "score";
    }
}

struct Question {
    int qtype = kQtypeScore;
    std::string instruction;
    std::vector<std::string> options;  // rendered option texts, in label order
    std::vector<std::string> labels;   // caller-facing labels, in label order
};

struct Head {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> markers;
};

struct Row {
    std::vector<std::int32_t> head_ids;
    std::vector<std::int32_t> state_ids;
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> markers;
};

struct Decoded {
    std::vector<float> logits;          // raw model logits, k values
    std::vector<float> probabilities;   // temperature-scaled softmax, k values
    int index = 0;
    float score = 0.0f;
    float confidence = 0.0f;
    float entropy_confidence = 0.0f;
};

// laya.common.render_options: score levels are rendered as "level N: <text>",
// choice options are passed through, noul has fixed true/false legends.
Question make_score_question(std::string_view instruction,
                             const std::vector<std::string>& levels) {
    if (levels.empty()) {
        throw std::invalid_argument("laya::score: levels must not be empty");
    }
    Question q;
    q.qtype = kQtypeScore;
    q.instruction = std::string(instruction);
    q.options.reserve(levels.size());
    q.labels = levels;
    for (std::size_t i = 0; i < levels.size(); ++i) {
        q.options.push_back("level " + std::to_string(i) + ": " + levels[i]);
    }
    return q;
}

Question make_choice_question(std::string_view instruction,
                              const std::vector<std::string>& options) {
    if (options.empty()) {
        throw std::invalid_argument("laya::choice: options must not be empty");
    }
    Question q;
    q.qtype = kQtypeChoice;
    q.instruction = std::string(instruction);
    q.options = options;
    q.labels = options;
    return q;
}

Question make_noul_question(std::string_view instruction,
                            std::string_view false_criterion,
                            std::string_view true_criterion) {
    Question q;
    q.qtype = kQtypeNoul;
    q.instruction = std::string(instruction);
    const std::string false_text = false_criterion.empty()
        ? std::string("no, the statement does not hold")
        : std::string(false_criterion);
    const std::string true_text = true_criterion.empty()
        ? std::string("yes, the statement holds")
        : std::string(true_criterion);
    q.options = {"false: " + false_text, "true: " + true_text};
    q.labels = {"false", "true"};
    return q;
}

// laya.common.build_head: [CLS] <type> question: <instructions> [SEP]
//                           [MASK] opt0 [MASK] opt1 ... [SEP]
Head build_head(const Tokenizer& tok,
                const Question& q,
                int head_max_len,
                const std::string& mask_token) {
    Head head;
    const std::string ins = replace_all(q.instruction, mask_token, " ");
    const std::string head_text =
        std::string(qtype_name(q.qtype)) + " question: " + ins;
    std::vector<std::int32_t> prompt = tok.encode(head_text, false);

    std::vector<std::vector<std::int32_t>> options;
    options.reserve(q.options.size());
    std::int64_t option_tokens = 0;
    for (const auto& opt : q.options) {
        const std::string text = " " + replace_all(opt, mask_token, " ");
        std::vector<std::int32_t> ids = tok.encode(text, false, 48);
        ids.insert(ids.begin(), tok.mask_id());
        option_tokens += static_cast<std::int64_t>(ids.size());
        options.push_back(std::move(ids));
    }

    std::int64_t opt_budget = static_cast<std::int64_t>(head_max_len) - option_tokens;
    if (opt_budget < 16) {
        const auto n = static_cast<std::int64_t>(std::max<std::size_t>(1, options.size()));
        const std::int64_t per = std::max<std::int64_t>(4, (static_cast<std::int64_t>(head_max_len) - 16) / n);
        option_tokens = 0;
        for (auto& ids : options) {
            if (static_cast<std::int64_t>(ids.size()) > per) {
                ids.resize(static_cast<std::size_t>(per));
            }
            option_tokens += static_cast<std::int64_t>(ids.size());
        }
        opt_budget = static_cast<std::int64_t>(head_max_len) - option_tokens;
    }

    const std::int64_t keep = std::max<std::int64_t>(8, opt_budget);
    if (static_cast<std::int64_t>(prompt.size()) > keep) {
        prompt.resize(static_cast<std::size_t>(keep));
    }

    head.ids.push_back(tok.cls_id());
    head.ids.insert(head.ids.end(), prompt.begin(), prompt.end());
    head.ids.push_back(tok.sep_id());
    for (const auto& ids : options) {
        head.markers.push_back(static_cast<std::int32_t>(head.ids.size()));
        head.ids.insert(head.ids.end(), ids.begin(), ids.end());
    }
    head.ids.push_back(tok.sep_id());
    return head;
}

// laya.common.build_sequence: head + truncated state + closing [SEP].
Row build_row(const Tokenizer& tok,
              std::string_view state,
              const Question& q,
              int max_len,
              int head_max_len,
              const std::string& mask_token) {
    Head head = build_head(tok, q, head_max_len, mask_token);
    if (max_len <= 0) {
        throw std::invalid_argument("laya: max_len must be positive");
    }

    Row row;
    row.head_ids = head.ids;
    row.ids = std::move(head.ids);
    row.markers = std::move(head.markers);

    const std::string state_text = replace_all(std::string(state), mask_token, " ");
    row.state_ids = tok.encode(state_text, false);

    const std::int64_t room =
        std::max<std::int64_t>(0, static_cast<std::int64_t>(max_len) -
                                     static_cast<std::int64_t>(row.ids.size()) - 1);
    const std::size_t take =
        std::min<std::size_t>(row.state_ids.size(), static_cast<std::size_t>(room));
    row.ids.insert(row.ids.end(), row.state_ids.begin(),
                   row.state_ids.begin() + static_cast<std::ptrdiff_t>(take));
    row.ids.push_back(tok.sep_id());

    if (static_cast<std::int64_t>(row.ids.size()) > max_len) {
        row.ids.resize(static_cast<std::size_t>(max_len));
    }
    row.markers.erase(
        std::remove_if(row.markers.begin(), row.markers.end(),
                       [max_len](std::int32_t marker) { return marker >= max_len; }),
        row.markers.end());
    if (row.markers.size() != q.options.size()) {
        throw std::runtime_error(
            "laya: only " + std::to_string(row.markers.size()) + " of " +
            std::to_string(q.options.size()) +
            " option markers fit in max_len=" + std::to_string(max_len) +
            " with head_max_len=" + std::to_string(head_max_len) +
            "; lower head_max_len or raise max_len");
    }
    return row;
}

Decoded decode_row(const ModelOutputs& out,
               std::int64_t row,
               std::size_t k,
               double temperature) {
    if (k == 0) throw std::runtime_error("laya: model returned no option columns");
    Decoded decoded;
    decoded.logits.resize(k);
    decoded.probabilities.resize(k);

    double best = -std::numeric_limits<double>::infinity();
    std::vector<double> z(k);
    for (std::size_t i = 0; i < k; ++i) {
        const double logit = static_cast<double>(out.logit(row, static_cast<std::int64_t>(i)));
        decoded.logits[i] = static_cast<float>(logit);
        z[i] = logit / temperature;
        if (z[i] > best) best = z[i];
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        z[i] = std::exp(z[i] - best);
        sum += z[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("laya: non-finite softmax denominator");
    }

    double expected = 0.0;
    std::size_t argmax = 0;
    for (std::size_t i = 0; i < k; ++i) {
        const double p = z[i] / sum;
        decoded.probabilities[i] = static_cast<float>(p);
        expected += static_cast<double>(i) * p;
        if (p > static_cast<double>(decoded.probabilities[argmax])) argmax = i;
    }
    decoded.index = static_cast<int>(argmax);
    decoded.score = static_cast<float>(expected);
    decoded.confidence = decoded.probabilities[argmax];

    if (k < 2) {
        decoded.entropy_confidence = 1.0f;
    } else {
        double entropy = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double p = static_cast<double>(decoded.probabilities[i]);
            entropy -= p * std::log(std::max(p, 1e-12));
        }
        double conf = 1.0 - entropy / std::log(static_cast<double>(k));
        conf = std::clamp(conf, 0.0, 1.0);
        decoded.entropy_confidence = static_cast<float>(conf);
    }
    return decoded;
}

}  // namespace

struct Engine::Impl {
    std::string model_dir;
    EngineOptions options;
    int max_len = 512;
    int head_max_len = 192;
    std::array<double, 3> temperature{1.0, 1.0, 1.0};
    std::map<std::string, double> temperature_by_options;
    std::string mask_token = kMaskFallback;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<OnnxSession> session;
    std::mutex predict_mutex;

    Impl(std::string dir, EngineOptions opts);
    void load_config(const fs::path& path);
    double temperature_for(int qtype, std::size_t k) const;
    std::vector<Decoded> run_rows(const std::vector<Row>& rows, int qtype);
    Decoded predict_one(const Row& row,
                        int qtype,
                        ScoreTrace* trace = nullptr);
};

Engine::Impl::Impl(std::string dir, EngineOptions opts)
    : model_dir(std::move(dir)), options(std::move(opts)) {
    std::error_code ec;
    const fs::path root(model_dir);
    if (!fs::is_directory(root, ec)) {
        throw std::runtime_error("laya: model directory not found: '" + model_dir + "'");
    }

    const fs::path model_path = root / options.onnx_filename;
    if (!fs::is_regular_file(model_path, ec)) {
        throw std::runtime_error("laya: ONNX model not found: '" + model_path.string() + "'");
    }

    fs::path tokenizer_path = root / options.tokenizer_filename;
    if (!fs::is_regular_file(tokenizer_path, ec)) {
        const fs::path nested = root / "tokenizer" / "tokenizer.json";
        if (fs::is_regular_file(nested, ec)) {
            tokenizer_path = nested;
        } else {
            throw std::runtime_error("laya: tokenizer.json not found: '" +
                                     (root / options.tokenizer_filename).string() + "'");
        }
    }

    const fs::path config_path = root / options.config_filename;
    if (fs::is_regular_file(config_path, ec)) {
        load_config(config_path);
    }
    if (options.max_len) max_len = *options.max_len;
    if (options.head_max_len) head_max_len = *options.head_max_len;
    if (max_len <= 0) {
        throw std::runtime_error("laya: invalid max_len (" + std::to_string(max_len) + ")");
    }
    if (head_max_len < 0) {
        throw std::runtime_error("laya: invalid head_max_len (" +
                                 std::to_string(head_max_len) + ")");
    }

    tokenizer = Tokenizer::from_file(tokenizer_path.string());
    if (!tokenizer) {
        throw std::runtime_error("laya: failed to load tokenizer");
    }
    if (tokenizer->mask_id() < 0 || tokenizer->cls_id() < 0 || tokenizer->sep_id() < 0) {
        throw std::runtime_error(
            "laya: tokenizer.json is missing <mask>/<bos>/<eos> special tokens");
    }
    mask_token = tokenizer->mask_token();
    if (mask_token.empty()) mask_token = kMaskFallback;

    OnnxSession::Options session_options;
    session_options.intra_op_threads = options.intra_op_threads;
    session_options.inter_op_threads = options.inter_op_threads;
    session_options.enable_all_optimizations = options.enable_all_optimizations;
    session_options.verbose = options.verbose;
    session = std::make_unique<OnnxSession>(model_path.string(), session_options);
}

void Engine::Impl::load_config(const fs::path& path) {
    json cfg;
    try {
        cfg = json::parse(read_file(path));
    } catch (const std::exception& e) {
        throw std::runtime_error("laya: cannot parse '" + path.string() + "': " + e.what());
    }
    if (!cfg.is_object()) {
        throw std::runtime_error("laya: '" + path.string() + "' is not a JSON object");
    }
    if (cfg.contains("max_len") && cfg["max_len"].is_number_integer()) {
        max_len = cfg["max_len"].get<int>();
    }
    if (cfg.contains("head_max_len") && cfg["head_max_len"].is_number_integer()) {
        head_max_len = cfg["head_max_len"].get<int>();
    }
    if (cfg.contains("temperature") && cfg["temperature"].is_array()) {
        const auto& raw = cfg["temperature"];
        for (std::size_t i = 0; i < temperature.size() && i < raw.size(); ++i) {
            if (raw[i].is_number()) {
                temperature[i] = clamp_temperature(raw[i].get<double>());
            }
        }
    }
    if (cfg.contains("temperature_by_options") && cfg["temperature_by_options"].is_object()) {
        for (const auto& [bucket, value] : cfg["temperature_by_options"].items()) {
            if (value.is_number()) {
                temperature_by_options[bucket] = clamp_temperature(value.get<double>());
            }
        }
    }
}

double Engine::Impl::temperature_for(int qtype, std::size_t k) const {
    const char* size = k <= 2 ? "2" : (k <= 5 ? "3-5" : (k <= 10 ? "6-10" : "11+"));
    const std::string bucket = std::string(qtype_name(qtype)) + ":" + size;
    const auto it = temperature_by_options.find(bucket);
    if (it != temperature_by_options.end()) return it->second;
    return temperature[static_cast<std::size_t>(qtype)];
}

std::vector<Decoded> Engine::Impl::run_rows(const std::vector<Row>& rows, int qtype) {
    if (rows.empty()) return {};

    std::size_t longest = 0;
    std::size_t columns = 2;  // the exported graph needs at least two marker columns
    for (const auto& row : rows) {
        longest = std::max(longest, row.ids.size());
        columns = std::max(columns, row.markers.size());
    }

    ModelInputs inputs;
    inputs.batch = static_cast<std::int64_t>(rows.size());
    inputs.seq_len = static_cast<std::int64_t>(longest);
    inputs.num_markers = static_cast<std::int64_t>(columns);
    const auto total = rows.size() * longest;
    inputs.input_ids.assign(total, tokenizer->pad_id());
    inputs.attention_mask.assign(total, 0);
    inputs.marker_pos.assign(rows.size() * columns, 0);
    inputs.marker_mask.assign(rows.size() * columns, 0);
    inputs.qtype.assign(rows.size(), qtype);

    for (std::size_t r = 0; r < rows.size(); ++r) {
        const Row& row = rows[r];
        for (std::size_t c = 0; c < row.ids.size(); ++c) {
            inputs.input_ids[r * longest + c] = row.ids[c];
            inputs.attention_mask[r * longest + c] = 1;
        }
        for (std::size_t c = 0; c < row.markers.size(); ++c) {
            inputs.marker_pos[r * columns + c] = row.markers[c];
            inputs.marker_mask[r * columns + c] = 1;
        }
    }

    ModelOutputs outputs = session->run(inputs);
    if (outputs.batch != inputs.batch || outputs.num_markers < 1) {
        throw std::runtime_error("laya: unexpected model output shape");
    }

    std::vector<Decoded> decoded;
    decoded.reserve(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        decoded.push_back(decode_row(outputs, static_cast<std::int64_t>(r),
                                 rows[r].markers.size(),
                                 temperature_for(qtype, rows[r].markers.size())));
    }
    return decoded;
}

Decoded Engine::Impl::predict_one(const Row& row, int qtype, ScoreTrace* trace) {
    std::vector<Decoded> decoded = run_rows({row}, qtype);
    if (decoded.size() != 1) {
        throw std::runtime_error("laya: internal error: expected one decoded row");
    }
    if (trace) {
        trace->state_ids = row.state_ids;
        trace->head_ids = row.head_ids;
        trace->sequence = row.ids;
        trace->markers = row.markers;
        trace->logits = decoded.front().logits;
    }
    return decoded.front();
}

Engine::Engine(const std::string& model_dir)
    : impl_(std::make_unique<Impl>(model_dir, EngineOptions{})) {}

Engine::Engine(const std::string& model_dir, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(model_dir, options)) {}

Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

ScoreResult Engine::score(std::string_view state,
                          std::string_view instruction,
                          const std::vector<std::string>& levels,
                          ScoreTrace* trace) {
    Question q = make_score_question(instruction, levels);
    std::lock_guard<std::mutex> lock(impl_->predict_mutex);
    Row row = build_row(*impl_->tokenizer, state, q, impl_->max_len, impl_->head_max_len,
                        impl_->mask_token);
    Decoded decoded = impl_->predict_one(row, kQtypeScore, trace);

    ScoreResult result;
    result.index = decoded.index;
    result.score = decoded.score;
    result.confidence = decoded.confidence;
    result.entropy_confidence = decoded.entropy_confidence;
    result.label = q.labels[static_cast<std::size_t>(decoded.index)];
    result.probabilities = std::move(decoded.probabilities);
    return result;
}

std::vector<ScoreResult> Engine::score_batch(const std::vector<std::string>& states,
                                             std::string_view instruction,
                                             const std::vector<std::string>& levels) {
    if (states.empty()) return {};
    Question q = make_score_question(instruction, levels);
    std::vector<Row> rows;
    rows.reserve(states.size());
    for (const auto& state : states) {
        rows.push_back(build_row(*impl_->tokenizer, state, q, impl_->max_len,
                                 impl_->head_max_len, impl_->mask_token));
    }

    std::lock_guard<std::mutex> lock(impl_->predict_mutex);
    std::vector<Decoded> decoded = impl_->run_rows(rows, kQtypeScore);
    std::vector<ScoreResult> results;
    results.reserve(decoded.size());
    for (auto& d : decoded) {
        ScoreResult result;
        result.index = d.index;
        result.score = d.score;
        result.confidence = d.confidence;
        result.entropy_confidence = d.entropy_confidence;
        result.label = q.labels[static_cast<std::size_t>(d.index)];
        result.probabilities = std::move(d.probabilities);
        results.push_back(std::move(result));
    }
    return results;
}

ChoiceResult Engine::choice(std::string_view state,
                            std::string_view instruction,
                            const std::vector<std::string>& options) {
    Question q = make_choice_question(instruction, options);
    std::lock_guard<std::mutex> lock(impl_->predict_mutex);
    Row row = build_row(*impl_->tokenizer, state, q, impl_->max_len, impl_->head_max_len,
                        impl_->mask_token);
    Decoded decoded = impl_->predict_one(row, kQtypeChoice);

    ChoiceResult result;
    result.index = decoded.index;
    result.confidence = decoded.confidence;
    result.label = q.labels[static_cast<std::size_t>(decoded.index)];
    result.probabilities = std::move(decoded.probabilities);
    return result;
}

NoulResult Engine::noul(std::string_view state,
                        std::string_view instruction,
                        std::string_view false_criterion,
                        std::string_view true_criterion) {
    Question q = make_noul_question(instruction, false_criterion, true_criterion);
    std::lock_guard<std::mutex> lock(impl_->predict_mutex);
    Row row = build_row(*impl_->tokenizer, state, q, impl_->max_len, impl_->head_max_len,
                        impl_->mask_token);
    Decoded decoded = impl_->predict_one(row, kQtypeNoul);

    NoulResult result;
    result.value = decoded.index == 1;
    result.probability = decoded.probabilities[1];
    result.confidence = decoded.confidence;
    result.label = result.value ? "true" : "false";
    return result;
}

const Tokenizer& Engine::tokenizer() const { return *impl_->tokenizer; }

std::vector<std::int32_t> Engine::encode(std::string_view text,
                                         bool add_special_tokens) const {
    return impl_->tokenizer->encode(text, add_special_tokens);
}

std::string Engine::decode(const std::vector<std::int32_t>& ids) const {
    return impl_->tokenizer->decode(ids);
}

const std::string& Engine::model_dir() const noexcept { return impl_->model_dir; }
int Engine::max_len() const noexcept { return impl_->max_len; }
int Engine::head_max_len() const noexcept { return impl_->head_max_len; }
std::size_t Engine::vocab_size() const noexcept { return impl_->tokenizer->vocab_size(); }

}  // namespace laya

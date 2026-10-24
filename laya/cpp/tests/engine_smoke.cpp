// Smoke test: load a laya model directory and score one request.
#include "laya.h"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model-dir>\n", argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::vector<std::string> levels = {
        "chitchat: casual conversation, greetings, small talk, jokes, feelings, opinions",
        "math: mathematics, arithmetic, algebra, geometry, calculus, probability, proofs",
        "code: programming, debugging, refactoring, software design, APIs, databases, tooling",
    };
    try {
        laya::Engine engine(model_dir);
        std::printf("loaded %s: max_len=%d head_max_len=%d vocab=%zu\n",
                    engine.model_dir().c_str(), engine.max_len(),
                    engine.head_max_len(), engine.vocab_size());

        laya::ScoreTrace trace;
        laya::ScoreResult result = engine.score(
            "你好，世界", "Which category does this request belong to?", levels, &trace);
        std::printf("score: label=%s index=%d confidence=%.6f entropy=%.6f\n",
                    result.label.c_str(), result.index, result.confidence,
                    result.entropy_confidence);
        std::printf("state_ids=%zu head_ids=%zu sequence=%zu markers=%zu\n",
                    trace.state_ids.size(), trace.head_ids.size(),
                    trace.sequence.size(), trace.markers.size());
        if (result.probabilities.size() != levels.size()) {
            std::fprintf(stderr, "bad probability count\n");
            return 1;
        }
        const float total = std::accumulate(result.probabilities.begin(),
                                            result.probabilities.end(), 0.0f);
        if (std::fabs(total - 1.0f) > 1e-3f) {
            std::fprintf(stderr, "probabilities do not sum to one: %f\n", total);
            return 1;
        }
        if (!(result.confidence >= 0.0f && result.confidence <= 1.0f)) {
            std::fprintf(stderr, "invalid confidence\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "smoke test failed: %s\n", e.what());
        return 1;
    }
}

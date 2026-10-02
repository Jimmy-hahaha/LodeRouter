#pragma once

#include <string>
#include <vector>
#include <set>
#include "../include/json/json.hpp"

using json = nlohmann::json;

// ───────── 上下文压缩：把整段对话压成结构化概要，作为 laya 裁判的 state ─────────
namespace lode_ctx {

// ───────── token 预算：按 256 token 渲染，给 laya 的 512 上限留足余量 ─────────
constexpr int kTokenBudget = 256;
constexpr int kRenderBudget = kTokenBudget - 8;

// ───────── 文本工具：token 估算、UTF-8 截断、content 取值与单行化 ─────────
inline int est_tokens(const std::string& s) {
    int wide = 0, ascii = 0;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++ascii;
            ++i;
        } else {
            ++wide;
            i += (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        }
    }
    return wide + (ascii + 3) / 4;
}

inline std::string clip_utf8(const std::string& s, size_t max_chars) {
    size_t chars = 0, i = 0;
    for (; i < s.size(); ++chars) {
        if (chars == max_chars) break;
        unsigned char c = static_cast<unsigned char>(s[i]);
        i += (c < 0x80) ? 1 : (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
    }
    if (i >= s.size()) return s;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
    return s.substr(0, i) + "...";
}

inline std::string content_text(const json& content) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_array()) {
        std::string out;
        for (const auto& part : content) {
            if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                if (!out.empty()) out += " ";
                out += part["text"].get<std::string>();
            }
        }
        return out;
    }
    return {};
}

inline std::string squeeze_ascii_runs(const std::string& s) {
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 0x80) {
            size_t len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
            out += s.substr(i, len);
            i += len;
            continue;
        }
        size_t j = i;
        while (j < s.size() && s[j] == s[i]) ++j;
        if (j - i > 3) { out += s[i]; out += s[i]; out += s[i]; out += "..."; }
        else out += s.substr(i, j - i);
        i = j;
    }
    return out;
}

inline std::string one_line(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = false;
    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);
        bool space = (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ');
        if (space) {
            if (!prev_space && !out.empty()) out += ' ';
            prev_space = true;
        } else if (c >= 0x20 || c >= 0x80) {
            out += ch;
            prev_space = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return squeeze_ascii_runs(out);
}

// ───────── 信号抽取：按句切分并命中关键词表，找出约束与报错 ─────────
inline bool contains_any(const std::string& s, const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (s.find(k) != std::string::npos) return true;
    }
    return false;
}

inline std::vector<std::string> split_sentences(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = (c < 0x80) ? 1 : (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        std::string ch = s.substr(i, len);
        i += len;
        if (ch == "\n") {
            out.push_back(cur);
            cur.clear();
            continue;
        }
        cur += ch;
        if (ch == "。" || ch == "！" || ch == "？" || ch == "；" ||
            ch == "!" || ch == "?" || ch == ";" || ch == ".") {
            out.push_back(cur);
            cur.clear();
        }
    }
    out.push_back(cur);

    std::vector<std::string> cleaned;
    for (auto& line : out) {
        std::string t = one_line(line);
        if (!t.empty()) cleaned.push_back(t);
    }
    return cleaned;
}

inline const std::vector<std::string>& constraint_keys() {
    static const std::vector<std::string> k = {
        "必须", "不能", "不要", "禁止", "至少", "最多", "务必", "保持", "兼容",
        "只能", "不允许", "确保", "避免", "注意", "严格", "约束",
        "must", "cannot", "should not", "don't", "do not", "at least", "at most",
        "only", "without", "ensure", "avoid", "require", "compatible", "keep"
    };
    return k;
}

inline const std::vector<std::string>& error_keys() {
    static const std::vector<std::string> k = {
        "报错", "错误", "失败", "异常", "超时", "崩溃", "中断", "无法", "挂了", "宕机",
        "重试", "回滚", "traceback", "Traceback", "Exception", "error", "Error", "ERROR",
        "failed", "Failed", "FAILED", "timeout", "timed out", "refused", "undefined",
        "not found", "segmentation", "panic", "Uncaught", "ERR_", "5xx", "502", "503", "504"
    };
    return k;
}

// ───────── 渲染上限：单条补丁的字符数、任务描述长度与每类补丁的条数 ─────────
constexpr size_t kPatchChars = 90;
constexpr size_t kGoalChars = 300;
constexpr size_t kMaxPatchesPerKind = 3;

struct Summary {
    std::string text;
    int tokens = 0;
    int turns = 0;
    int error_patches = 0;
    int repeat_errors = 0;
};

// ───────── 压缩主流程：按 [任务] > [报错] > [约束] > [失败] > [起点] > [上文] 填预算 ─────────
inline Summary summarize(const json& messages) {
    Summary sum;
    if (!messages.is_array() || messages.empty()) return sum;

    std::vector<std::string> user_msgs, assistant_msgs;
    for (const auto& m : messages) {
        if (!m.is_object()) continue;
        std::string role = m.value("role", "");
        std::string text = m.contains("content") ? content_text(m["content"]) : std::string{};
        if (text.empty()) continue;
        if (role == "user") user_msgs.push_back(text);
        else if (role == "assistant") assistant_msgs.push_back(text);
    }
    sum.turns = static_cast<int>(user_msgs.size());

    // ───────── 语义单元抽取：约束与报错扫全量消息，去重后越晚出现越重要 ─────────
    std::vector<std::string> constraints, errors;
    std::set<std::string> seen_c, seen_e;
    int error_hits = 0;
    for (const auto& m : messages) {
        if (!m.is_object()) continue;
        std::string text = m.contains("content") ? content_text(m["content"]) : std::string{};
        if (text.empty()) continue;
        for (auto& sentence : split_sentences(text)) {
            std::string clipped = clip_utf8(sentence, kPatchChars);
            if (contains_any(sentence, error_keys())) {
                ++error_hits;
                if (seen_e.insert(clipped).second) errors.push_back(clipped);
            }
            if (contains_any(sentence, constraint_keys())) {
                if (seen_c.insert(clipped).second) constraints.push_back(clipped);
            }
        }
    }
    sum.error_patches = error_hits;
    sum.repeat_errors = sum.error_patches > static_cast<int>(errors.size())
                            ? sum.error_patches - static_cast<int>(errors.size())
                            : 0;

    std::string origin = user_msgs.empty() ? std::string{} : clip_utf8(one_line(user_msgs.front()), kPatchChars);
    std::string goal = user_msgs.empty() ? std::string{} : clip_utf8(one_line(user_msgs.back()), kGoalChars);
    std::string progress = assistant_msgs.empty() ? std::string{}
                                                  : clip_utf8(one_line(assistant_msgs.back()), kPatchChars);

    // ───────── 预算分配：骨架之外按优先级贪心填充，塞不下就截断本块 ─────────
    std::string text;
    auto already = [&](const std::string& body) {
        return !body.empty() && text.find(body) != std::string::npos;
    };
    auto append_block = [&](const std::string& head, const std::vector<std::string>& items) {
        if (items.empty()) return;
        std::string block = head;
        for (size_t i = 0; i < items.size(); ++i) {
            if (already(items[i])) continue;
            std::string part = (i ? " | " : "") + items[i];
            if (est_tokens(text + block + part) > kRenderBudget) break;
            block += part;
        }
        if (block != head) text += block + "\n";
    };
    auto append_line = [&](const std::string& head, const std::string& body) {
        if (body.empty() || already(body)) return;
        std::string line = head + body + "\n";
        if (est_tokens(text + line) > kRenderBudget) return;
        text += line;
    };

    if (sum.turns > 1) append_line("[轮次] ", "第 " + std::to_string(sum.turns) + " 轮");
    if (!goal.empty()) {
        std::string head = "[任务] ";
        size_t room = goal.size();
        while (room > 0 && est_tokens(text + head + clip_utf8(goal, room)) > kRenderBudget) {
            room = room * 3 / 4;
        }
        append_line(head, room ? clip_utf8(goal, room) : std::string{});
    }
    {
        std::vector<std::string> tail(errors.rbegin(), errors.rend());
        if (tail.size() > kMaxPatchesPerKind) tail.resize(kMaxPatchesPerKind);
        append_block("[报错] ", tail);
    }
    {
        std::vector<std::string> tail(constraints.rbegin(), constraints.rend());
        if (tail.size() > kMaxPatchesPerKind) tail.resize(kMaxPatchesPerKind);
        append_block("[约束] ", tail);
    }
    if (sum.repeat_errors > 0) {
        append_line("[失败] ", "同一报错重复 " + std::to_string(sum.repeat_errors + 1) +
                                   " 次，说明前几次尝试都没解决");
    }
    if (origin != goal) append_line("[起点] ", origin);
    append_line("[上文] ", progress);

    sum.text = text;
    sum.tokens = est_tokens(text);
    return sum;
}

}

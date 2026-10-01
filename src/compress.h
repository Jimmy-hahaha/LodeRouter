#pragma once

#include <string>
#include <vector>
#include <set>
#include "../include/json/json.hpp"

using json = nlohmann::json;

// 上下文压缩：把整段对话（messages）压成一段结构化概要，作为 laya 裁判的 state。
//
// 为什么需要：laya 现在只看得到最后一条 user 消息，多轮会话里的约束、报错、重试、
// 会话起点意图全丢了，难度自然判不准。
//
// 预算：laya 的 per_question_row 布局在 max_len=512 里给 state 留了 445 token
// （question 14 + 4 个档位选项 49 + 分隔符 3 = 66 被头部占掉）。
// 这里按 256 token 渲染，留足余量，保证 laya 端不会发生截断而丢掉判断依据。
namespace lode_ctx {

constexpr int kTokenBudget = 256;
// 渲染时按更小的额度裁剪：估算在极短文本上可能低估 1~2 token（实测误差 ±1），
// 留出余量后，"laya 实际收到的 ≤ 256" 才是站得住的承诺。
constexpr int kRenderBudget = kTokenBudget - 8;

// 估算 token 数：非 ASCII（中文）按 1 字 1 token，ASCII 按 4 字符 1 token。
// 实测该 tokenizer 中文约 0.6 token/字、英文约 0.22 token/字符，这个换算偏保守，
// 所以"估算 ≤ 256"能保证真实 token 数也在 256 以内。
inline int est_tokens(const std::string& s) {
    int wide = 0, ascii = 0;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++ascii;
            ++i;
        } else {
            ++wide;  // 一个多字节字符算 1 token，且不切在字节中间
            i += (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        }
    }
    return wide + (ascii + 3) / 4;
}

// 按 UTF-8 字符边界截断（不切出半个汉字），超长时补省略号
inline std::string clip_utf8(const std::string& s, size_t max_chars) {
    size_t chars = 0, i = 0;
    for (; i < s.size(); ++chars) {
        if (chars == max_chars) break;
        unsigned char c = static_cast<unsigned char>(s[i]);
        i += (c < 0x80) ? 1 : (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
    }
    if (i >= s.size()) return s;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;  // 回退到合法边界
    return s.substr(0, i) + "...";
}

// content 可能是 string，也可能是 OpenAI 的 [{type:"text", text:"..."}] 数组
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

// 折叠连续重复字符："xxxxxxxx" 这类填充会白吃预算，连续超过 3 个就压成 "..."
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

// 去掉换行/制表，压掉连续空白，让每个补丁都是单行，渲染时好排布
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
        } else if (c >= 0x20 || c >= 0x80) {  // 保留可打印字符与多字节序列
            out += ch;
            prev_space = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return squeeze_ascii_runs(out);
}

inline bool contains_any(const std::string& s, const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (s.find(k) != std::string::npos) return true;
    }
    return false;
}

// 按句末标点与换行切句（保留句子本身），便于按句抽取与去重
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

// 四类补丁的关键词。难度判断真正吃得上的信号都在这里：
// 约束（限制越多越难）、报错/失败（要调试就更难）、任务目标（由调用方单独取）。
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

// 单个补丁的渲染上限，避免一句话吃掉整个预算
constexpr size_t kPatchChars = 90;
constexpr size_t kGoalChars = 300;
constexpr size_t kMaxPatchesPerKind = 3;

struct Summary {
    std::string text;
    int tokens = 0;
    int turns = 0;
    int error_patches = 0;
    int repeat_errors = 0;  // 同一报错重复出现，等价于"失败次数"
};

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
        // system / tool 的内容并进错误与约束抽取，但不单独成行
    }
    sum.turns = static_cast<int>(user_msgs.size());

    // ---- 语义单元抽取 ----
    // 约束、报错都按句扫全量消息，去重后按"越晚出现越重要"排序
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

    // 会话起点意图（"难度继承"：一开始就很难的任务，后续追问通常也难）
    std::string origin = user_msgs.empty() ? std::string{} : clip_utf8(one_line(user_msgs.front()), kPatchChars);
    // 当前诉求：最后一条用户消息，是难度判断的主依据，优先保留
    std::string goal = user_msgs.empty() ? std::string{} : clip_utf8(one_line(user_msgs.back()), kGoalChars);
    // 上文进展：最后一条 assistant 回复的结尾
    std::string progress = assistant_msgs.empty() ? std::string{}
                                                  : clip_utf8(one_line(assistant_msgs.back()), kPatchChars);

    // ---- 预算分配 + 渲染 ----
    // 骨架固定开销；剩余额度按 [任务] > [报错] > [约束] > [失败] > [起点] > [上文] 贪心填充。
    std::string text;
    // 同一个补丁（比如 [任务] 那句同时命中约束关键词）只该占一份预算
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
    // 任务目标：预算不够时也只截断它自己，不整块丢掉
    if (!goal.empty()) {
        std::string head = "[任务] ";
        size_t room = goal.size();
        while (room > 0 && est_tokens(text + head + clip_utf8(goal, room)) > kRenderBudget) {
            room = room * 3 / 4;  // 逐步收缩，直到塞得下
        }
        append_line(head, room ? clip_utf8(goal, room) : std::string{});
    }
    {
        std::vector<std::string> tail(errors.rbegin(), errors.rend());  // 最近的报错优先
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

} // namespace lode_ctx

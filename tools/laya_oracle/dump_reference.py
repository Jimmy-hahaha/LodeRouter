#!/usr/bin/env python3
"""Independent Python reference oracle for the pure-C++ laya rewrite.

This script is the *oracle* side of the parity gate: it writes down what the
official Python pipeline (HF tokenizer + laya.common.build_sequence +
laya.onnx_agent.ONNXAgent inference + laya.common decoding) produces for the
judge, so the C++ engine can be compared against it without trusting either
implementation's tokenizer/sequence logic.

Outputs (under tools/laya_oracle/reference/ by default):

  tokenizer_corpus.jsonl
      Every text the C++ tokenizer must reproduce, with the Hugging Face ids:
        * all 1,680 dataset "state" strings (train 1350 + eval 330)
        * the real judge instruction / levels / rendered options / head text
        * all 249 added-token contents (runs of newlines, tabs, U+2581, tags)
        * an adversarial corpus: empty/whitespace, control chars, CJK, emoji,
          combining marks, private-use/rare code points, added tokens inside
          words, byte-fallback scripts, long strings and 46..51-token texts
          around the option-truncation boundary (max_length=48)
      Each record: {"id","kind","text",("note"),"n_tokens","ids","ids_trunc48"}
      ids are HF input_ids with add_special_tokens=False; ids_trunc48 adds
      truncation=True, max_length=48 (exactly what build_head uses for options).

  eval_reference.jsonl
      For all 330 rows of finetune/data/eval.jsonl, the exact pipeline with
      max_len=512, head_max_len=256 and the score temperature
      temperature[1] = 1.1558657884597778:
        state/instruction/levels, state_ids, head_ids, head_len, markers,
        full sequence, raw logits, temperature-scaled probabilities (full
        float32 and ONNXAgent-style rounded4), predicted index/label, score,
        confidence (max p), entropy_confidence, answer_confidence,
        expected index/label and correctness.

  baseline_report.json
      accuracy, per-class accuracy, confusion matrix, mean confidence, ECE,
      NLL, Brier scores, item count, timestamp, commands, versions, and
      sanity checks (SPEC golden sample, agreement with ONNXAgent.predict,
      batch invariance).

Usage, from the repository root, with the project venv (one model load):

    .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py
    .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py --tokenizer-only
    .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py --eval-only
    .cache/laya-ft/venv/bin/python tools/laya_oracle/dump_reference.py \
        --eval-only --limit 4 --out-dir /tmp/oracle-smoke

--limit N writes *.partial.jsonl / *.partial.json so a smoke run can never
clobber the full reference.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import oracle_common as oc  # noqa: E402

MASK_TOKEN = "<mask>"

# Golden tokenizer values from laya/cpp/SPEC.md section 4 (add_special_tokens=false).
SPEC_TOKENIZER_GOLDENS = {
    "hello world": [25612, 2134],
    "你好，世界": [30485, 235607, 235365, 9979],
    "a<bos>b": [476, 2, 518],
    "<mask>hi": [4, 5827],
    "  leading spaces": [235248, 8133, 14130],
    "a\tb\nc": [476, 226, 235268, 108, 498],
    "x\ue000y": [1141, 240539, 235267],
    "\x00": [235248, 217],
}

# Golden end-to-end sample from laya/cpp/SPEC.md section 6.
SPEC_SAMPLE_STATE = "你好，世界"
SPEC_SAMPLE_HEAD_LEN = 78
SPEC_SAMPLE_MARKERS = [13, 35, 55]
SPEC_SAMPLE_STATE_IDS = [30485, 235607, 235365, 9979]
SPEC_SAMPLE_SEQ_LEN = 83
SPEC_SAMPLE_LOGITS = [5.7771907, -4.0328493, -1.2201804]
SPEC_SAMPLE_PROBS = [0.99745148, 0.00020557, 0.00234295]


# ---------------------------------------------------------------------------
# tokenizer helpers / corpus
# ---------------------------------------------------------------------------
def encode_ids(tok, text, max_length=None):
    """HF tokenizer ids for text, add_special_tokens=False (+ optional truncation)."""
    kwargs = {"add_special_tokens": False}
    if max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = int(max_length)
    enc = tok(text, **kwargs)
    return [int(i) for i in enc["input_ids"]]


def load_tokenizer():
    from transformers import AutoTokenizer

    tok_dir = oc.TOKENIZER_DIR
    if not tok_dir.exists():
        # Same fallback as ONNXAgent: the config's encoder reference.
        with open(oc.CONFIG_PATH, "r", encoding="utf-8") as f:
            tok_dir = Path(json.load(f)["encoder"])
    tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    return tok


def iter_dataset_rows():
    """Yield (split, 1-based line number, parsed row) for train then eval."""
    for split, path in (("train", oc.TRAIN_JSONL), ("eval", oc.EVAL_JSONL)):
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                yield split, lineno, json.loads(line)


def corpus_record(cid, kind, text, ids, ids48=None, note=None):
    rec = {
        "id": cid,
        "kind": kind,
        "text": text,
        "n_tokens": len(ids),
        "ids": ids,
        "ids_trunc48": ids48 if ids48 is not None else encode_ids_dummy(ids),
    }
    if note:
        rec["note"] = note
    return rec


def encode_ids_dummy(ids):
    """ids_trunc48 for a record generated from already-encoded ids."""
    return list(ids[:48])


def adversarial_cases():
    """Deterministic adversarial corpus for tokenizer parity.

    Every entry is (id, kind, text, note).  It mirrors SPEC.md section 4(c) and
    adds the traps found in this checkpoint's added-token list: the tokenizer
    matches runs of newlines/tabs/U+2581 as single added tokens before
    normalization, so raw whitespace runs are not byte-fallback sequences.
    """
    cases = []

    def add(cid, kind, text, note=""):
        cases.append({"id": cid, "kind": kind, "text": text, "note": note})

    # Empty and space handling (normalizer: " " -> U+2581, then Metaspace).
    add("empty", "adv_empty", "")
    add("space1", "adv_whitespace", " ")
    add("spaces2", "adv_whitespace", "  ")
    add("spaces8", "adv_whitespace", " " * 8)
    add("spaces32", "adv_whitespace", " " * 32)
    add("leading_spaces", "adv_whitespace", "  leading spaces", "SPEC golden")
    add("trailing_spaces", "adv_whitespace", "trailing spaces  ")
    add("internal_spaces", "adv_whitespace", "a  b   c")
    add("many_words", "adv_whitespace", " ".join(["word"] * 50))
    add("space_word_space", "adv_whitespace", " a b c ")
    # Control whitespace; newline/tab runs are added tokens (ids 108..138,
    # 255969..255999, 139..168).  The longest runs are 30 newlines / 31 tabs /
    # 31 U+2581, so 31/32/35 are boundary cases.
    add("tab1", "adv_control", "\t")
    add("newline1", "adv_control", "\n")
    add("crlf", "adv_control", "\r\n")
    add("mixed_controls", "adv_control", "a\tb\nc")
    add("vertical_tab_formfeed", "adv_control", "\v\f")
    add("newlines3", "adv_control", "\n" * 3)
    add("newlines8", "adv_control", "\n" * 8)
    add("newlines30", "adv_control", "\n" * 30, "longest newline added token is 30")
    add("newlines31", "adv_control", "\n" * 31)
    add("newlines32", "adv_control", "\n" * 32)
    add("newlines35", "adv_control", "\n" * 35)
    add("tabs3", "adv_control", "\t" * 3)
    add("tabs31", "adv_control", "\t" * 31, "longest tab added token is 31")
    add("tabs32", "adv_control", "\t" * 32)
    add("metaspace1", "adv_whitespace", "\u2581", "literal U+2581 input")
    add("metaspace5", "adv_whitespace", "\u2581" * 5)
    add("metaspace31", "adv_whitespace", "\u2581" * 31)
    add("metaspace_mixed", "adv_whitespace", "a\u2581b\u2581\u2581c")
    add("whitespace_soup", "adv_control", " \t \n \r\n  \t")
    add("space_newline_runs", "adv_control", "word\n\n  \t\tend")
    # CJK / multilingual.
    add("cjk_short", "adv_cjk", "你好，世界", "SPEC golden")
    add("cjk_math", "adv_cjk", "数学求助：抛两次硬币，至少一次正面的概率是多少")
    add("cjk_code", "adv_cjk", "写个快速排序算法")
    add("japanese", "adv_cjk", "日本語のテストです")
    add("korean", "adv_cjk", "한국어 테스트")
    add("cjk_ext_b", "adv_cjk", "𠀀𠀁𠀂", "supplementary plane CJK")
    add("cjk_ext_b2", "adv_cjk", "𠮷野家")
    add("traditional", "adv_cjk", "繁體中文測試")
    add("fullwidth", "adv_cjk", "ＡＢＣ１２３，。")
    add("halfwidth_katakana", "adv_cjk", "ｱｲｳｴｵ")
    add("vietnamese", "adv_cjk", "Tiếng Việt có dấu")
    add("arabic", "adv_cjk", "مرحبا بالعالم")
    add("hebrew", "adv_cjk", "שלום עולם")
    add("hindi", "adv_cjk", "नमस्ते दुनिया")
    add("thai", "adv_cjk", "สวัสดีชาวโลก")
    add("cyrillic", "adv_cjk", "Привет, мир")
    add("greek", "adv_cjk", "Γειά σου Κόσμε")
    add("mixed_scripts", "adv_cjk", "hello 你好 123 😀")
    # Emoji (ZWJ sequences, flags, skin tones, variation selectors, tags).
    add("emoji_grin", "adv_emoji", "😀")
    add("emoji_party", "adv_emoji", "😀🎉🚀")
    add("emoji_family", "adv_emoji", "👨‍👩‍👧‍👦", "ZWJ sequence")
    add("emoji_flags", "adv_emoji", "🇨🇳🇺🇸")
    add("emoji_skin", "adv_emoji", "👍🏽")
    add("emoji_robot", "adv_emoji", "🤖💻")
    add("emoji_heart_vs16", "adv_emoji", "❤️")
    add("emoji_rainbow", "adv_emoji", "🏳️‍🌈")
    add("emoji_keycap", "adv_emoji", "1️⃣")
    add("emoji_tag_flag", "adv_emoji", "🏴󠁧󠁢󠁳󠁣󠁴󠁿")
    add("emoji_mixed", "adv_emoji", "代码 review 😅 please")
    # Combining marks (input is *not* NFC-normalized by this tokenizer).
    add("combining_acute", "adv_combining", "e\u0301")
    add("combining_multi", "adv_combining", "a\u0300\u0301\u0302")
    add("combining_dot", "adv_combining", "Z\u0301\u0323")
    add("combining_thai", "adv_combining", "\u0e01\u0e49")
    add("combining_diaeresis", "adv_combining", "o\u0308\u0304")
    add("combining_zalgo", "adv_combining", "a" + "\u0301\u0302\u0303" * 5)
    # Rare / private-use / control code points (byte fallback).
    add("nul", "adv_rare", "\x00", "SPEC golden")
    add("ctrl_01", "adv_rare", "\x01")
    add("ctrl_1b", "adv_rare", "\x1b")
    add("del_7f", "adv_rare", "\x7f")
    add("ctrl_80", "adv_rare", "\x80")
    add("ctrl_9f", "adv_rare", "\x9f")
    add("nbsp", "adv_rare", "\u00a0")
    add("zwsp", "adv_rare", "\u200b")
    add("zwnj", "adv_rare", "\u200c")
    add("zwj", "adv_rare", "\u200d")
    add("line_sep", "adv_rare", "\u2028\u2029")
    add("bom", "adv_rare", "\ufeff")
    add("bom_text", "adv_rare", "\ufeffhello")
    add("replacement", "adv_rare", "\ufffd")
    add("pua_single", "adv_rare", "x\ue000y", "SPEC golden")
    add("pua_three", "adv_rare", "\ue000\ue001\ue002")
    add("pua_bounds", "adv_rare", "\ud7ff\ue000\uffff", "surrogate-adjacent")
    add("pua_plane15", "adv_rare", "\U000f0000\U000f0001", "private use plane 15")
    add("tag_char", "adv_rare", "\U000e0001")
    add("vs16", "adv_rare", "\ufe0f")
    add("byte_sweep", "adv_byte_fallback", "".join(chr(i) for i in range(256)),
        "all U+0000..U+00FF in one string")
    add("runic", "adv_byte_fallback", "ᚠᚢᚦᚨᚱᚲ")
    add("cherokee", "adv_byte_fallback", "ᎠᎡᎢ")
    add("deseret", "adv_byte_fallback", "\U00010400\U00010401")
    add("old_italic", "adv_byte_fallback", "\U00010300\U00010301")
    add("coptic", "adv_byte_fallback", "ⲀⲂ")
    add("glagolitic", "adv_byte_fallback", "ⰀⰁ")
    add("cuneiform", "adv_byte_fallback", "\U00012000\U00012001")
    add("egyptian", "adv_byte_fallback", "\U00013000")
    add("linear_b", "adv_byte_fallback", "\U00010000")
    add("gothic", "adv_byte_fallback", "\U00010330")
    add("unknown_script_sentence", "adv_byte_fallback",
        "ᚠᚢᚦ mixed 你好 \U00010400 e\u0301 \x01", "mix of byte-fallback and known text")
    # Added tokens (249 of them; matched longest-first in the raw input).
    for name in ("<mask>", "<bos>", "<pad>", "<eos>", "<unk>", "<cls>", "<sep>"):
        add("added_" + name.strip("<>"), "adv_added_token", name)
    add("added_word_bos", "adv_added_token", "a<bos>b", "SPEC golden")
    add("added_mask_hi", "adv_added_token", "<mask>hi", "SPEC golden; <mask> has lstrip")
    add("added_hi_mask", "adv_added_token", "hi<mask>")
    add("added_space_before_mask", "adv_added_token", "a <mask>b")
    add("added_spaces_before_mask", "adv_added_token", "a  <mask>b")
    add("added_mask_mask", "adv_added_token", "<mask><mask>")
    add("added_mask_words", "adv_added_token", "<mask> word <bos>")
    add("added_eos_word", "adv_added_token", "hello<eos>world")
    add("added_pad_word", "adv_added_token", "x<pad>y")
    add("added_mask_infix", "adv_added_token", "pre<mask>post")
    add("added_mask_multi_infix", "adv_added_token", "abc<mask>def<mask>ghi")
    add("added_unused_prefix", "adv_added_token", "<unused1>0", "longest-match: <unused1> then 0")
    add("added_unused10", "adv_added_token", "<unused10>")
    add("added_unused99", "adv_added_token", "<unused99>")
    add("added_2mass", "adv_added_token", "<2mass>")
    add("added_bos_tag", "adv_added_token", "[@BOS@]")
    add("added_turn", "adv_added_token", "<start_of_turn>user<end_of_turn>")
    add("added_code_tag", "adv_added_token", "<code>x</code>")
    add("added_blockquote", "adv_added_token", "<blockquote>q</blockquote>")
    add("added_toxicity", "adv_added_token", "[toxicity=0]")
    add("added_bold", "adv_added_token", "<b>bold</b>")
    add("added_inline_code", "adv_added_token", "text<code>inline</code>text")
    add("added_metaspace_mask", "adv_added_token", "\u2581\u2581<mask>")
    add("added_mask_with_spaces", "adv_added_token", " <mask> ")
    add("added_runs_mixed", "adv_added_token", "\n\n<mask>\t\t<eos>\n")
    add("added_repeat", "adv_added_token", "<mask>" * 20)
    # Long strings (truncation of state/head must not crash; option at 48).
    add("long_words", "adv_long", "hello " * 200)
    add("long_chars", "adv_long", "a" * 5000)
    add("long_cjk", "adv_long", "你好" * 1500)
    add("long_sentence", "adv_long", "The quick brown fox jumps over the lazy dog. " * 100)
    add("long_mask_repeat", "adv_long", " <mask>" * 100)
    return cases


def find_truncation_boundary_cases(tok):
    """Texts that tokenize to exactly 46..51 tokens (the 48-token option cap).

    build_head encodes " " + option with truncation=True, max_length=48, so the
    ids at 48/49 are exactly where a C++ truncation bug shows up.
    """
    out = []
    for target in range(46, 52):
        text = None
        for n in range(1, 300):
            candidate = " " + " ".join(["hello"] * n)
            if len(encode_ids(tok, candidate)) >= target:
                if len(encode_ids(tok, candidate)) == target:
                    text = candidate
                break
        if text is None:
            print(f"  WARNING: could not construct a {target}-token text", flush=True)
            continue
        ids = encode_ids(tok, text)
        ids48 = encode_ids(tok, text, max_length=48)
        assert len(ids) == target, (target, len(ids))
        assert len(ids48) == min(target, 48), (target, len(ids48))
        out.append({
            "id": f"adv:trunc48_len{target}",
            "kind": "adv_trunc48",
            "text": text,
            "note": f"exactly {target} tokens before max_length=48 truncation",
            "n_tokens": len(ids),
            "ids": ids,
            "ids_trunc48": ids48,
        })
    return out


def build_tokenizer_corpus(tok):
    """Build and return (records, stats) for tokenizer_corpus.jsonl."""
    records = []
    ids_seen = set()
    kind_counts = {}

    def emit(rec):
        rid = rec["id"]
        if rid in ids_seen:
            raise SystemExit(f"duplicate tokenizer corpus id: {rid}")
        ids_seen.add(rid)
        kind_counts[rec["kind"]] = kind_counts.get(rec["kind"], 0) + 1
        records.append(rec)

    # 0. Golden self-check first: a tokenizer mismatch here means the oracle is
    #    not looking at the right tokenizer at all.
    golden_ok = 0
    for text, expected in SPEC_TOKENIZER_GOLDENS.items():
        got = encode_ids(tok, text)
        if got != expected:
            raise SystemExit(
                "SPEC tokenizer golden mismatch for %r: got %r, expected %r"
                % (text, got, expected))
        golden_ok += 1

    # 1. All 1,680 dataset states.
    dataset_count = 0
    max_state_tokens = 0
    mask_states = 0
    for split, lineno, row in iter_dataset_rows():
        state = row["state"]
        if not isinstance(state, str):
            raise SystemExit(f"{split}:{lineno}: state is not a string: {type(state)!r}")
        ids = encode_ids(tok, state)
        ids48 = encode_ids(tok, state, max_length=48)
        dataset_count += 1
        max_state_tokens = max(max_state_tokens, len(ids))
        if MASK_TOKEN in state:
            mask_states += 1
        emit({
            "id": f"{split}:{lineno:04d}",
            "kind": "dataset_state",
            "source": f"finetune/data/{split}.jsonl:{lineno}",
            "text": state,
            "n_tokens": len(ids),
            "ids": ids,
            "ids_trunc48": ids48,
        })

    # 2. The real judge prompt texts and the exact strings build_head encodes.
    q = oc.load_questions()
    instruction = q["instructions"]
    levels = list(q["criteria"])
    rendered = [f"level {i}: {level}" for i, level in enumerate(levels)]
    emit(corpus_record("prompt:instruction", "prompt_instruction", instruction,
                       encode_ids(tok, instruction), encode_ids(tok, instruction, 48)))
    for i, level in enumerate(levels):
        emit(corpus_record(f"prompt:level{i}", "prompt_level", level,
                           encode_ids(tok, level), encode_ids(tok, level, 48)))
    for i, option in enumerate(rendered):
        emit(corpus_record(f"prompt:option{i}", "prompt_option", option,
                           encode_ids(tok, option), encode_ids(tok, option, 48)))
    for i, option in enumerate(rendered):
        text = " " + option.replace(MASK_TOKEN, " ")
        ids = encode_ids(tok, text)
        ids48 = encode_ids(tok, text, max_length=48)
        emit({
            "id": f"prompt:option_encoding{i}",
            "kind": "prompt_option_encoding",
            "text": text,
            "n_tokens": len(ids),
            "ids": ids,
            "ids_trunc48": ids48,
            "note": "exact string build_head passes to the tokenizer (trunc 48)",
        })
    head_text = "score question: " + instruction.replace(MASK_TOKEN, " ")
    emit(corpus_record("prompt:head_text", "prompt_head", head_text,
                       encode_ids(tok, head_text), encode_ids(tok, head_text, 48)))
    long_option = ("level 3: " + "This is a deliberately long level description used to "
                   "exercise the forty-eight token option truncation path in build_head. "
                   "It keeps going with more words so that the encoded length is comfortably "
                   "over the cap and the C++ tokenizer must truncate exactly like HF.")
    long_text = " " + long_option
    emit({
        "id": "prompt:option_long",
        "kind": "prompt_option_long",
        "text": long_text,
        "n_tokens": len(encode_ids(tok, long_text)),
        "ids": encode_ids(tok, long_text),
        "ids_trunc48": encode_ids(tok, long_text, max_length=48),
        "note": "long option: max_length=48 must truncate to exactly 48 ids",
    })
    assert len(encode_ids(tok, long_text)) > 48

    # 3. Every added token on its own (249 entries) plus all of them concatenated.
    with open(oc.TOKENIZER_DIR / "tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    added = tokenizer_json.get("added_tokens", [])
    added_mismatches = []
    for a in sorted(added, key=lambda x: int(x["id"])):
        text = a["content"]
        ids = encode_ids(tok, text)
        expected = [int(a["id"])]
        if ids != expected:
            added_mismatches.append({"id": int(a["id"]), "content": text,
                                     "expected": expected, "got": ids})
        flags = " ".join(k for k in ("special", "lstrip", "rstrip") if a.get(k))
        emit({
            "id": f"added:{int(a['id']):06d}",
            "kind": "added_token",
            "text": text,
            "note": f"added token id {int(a['id'])} flags=[{flags}]",
            "expected_ids": expected,
            "n_tokens": len(ids),
            "ids": ids,
            "ids_trunc48": ids[:48],
        })
    concatenated = "".join(a["content"] for a in sorted(added, key=lambda x: int(x["id"])))
    emit(corpus_record("added:concat_all", "added_token_concat", concatenated,
                       encode_ids(tok, concatenated), encode_ids(tok, concatenated, 48)))
    del tokenizer_json

    # 4. Adversarial corpus.
    for case in adversarial_cases():
        text = case["text"]
        ids = encode_ids(tok, text)
        ids48 = encode_ids(tok, text, max_length=48)
        emit({
            "id": "adv:" + case["id"],
            "kind": case["kind"],
            "text": text,
            "note": case["note"],
            "n_tokens": len(ids),
            "ids": ids,
            "ids_trunc48": ids48,
        })

    # 5. Exact 46..51-token texts around the option truncation boundary.
    for rec in find_truncation_boundary_cases(tok):
        emit(rec)

    stats = {
        "records": len(records),
        "dataset_states": dataset_count,
        "max_dataset_state_tokens": max_state_tokens,
        "dataset_states_containing_literal_<mask>": mask_states,
        "added_tokens": len(added),
        "added_token_mismatches": added_mismatches,
        "spec_tokenizer_goldens_passed": golden_ok,
        "kinds": dict(sorted(kind_counts.items())),
    }
    return records, stats


# ---------------------------------------------------------------------------
# eval reference
# ---------------------------------------------------------------------------
def decode_row(logits_row, k, t_scale):
    """Reproduce laya.onnx_agent.ONNXAgent._decode_answers for one score row.

    Same numpy dtypes/operations as the package: float32 logits / python float
    temperature, float32 exp/sum; score is computed in float64 because
    np.arange(k) * float32 promotes to float64, exactly as in the package.
    """
    from laya.common import answer_confidence, confidence_from_probs

    logits = np.asarray(logits_row, dtype=np.float32)
    z = logits[:k] / t_scale
    p = np.exp(z - z.max())
    p /= p.sum()
    index = int(p.argmax())
    expected_score = float((np.arange(k) * p).sum())
    entropy_conf = float(confidence_from_probs(p, k))
    answer_conf = float(answer_confidence(p, k))
    return {
        "probabilities": [float(v) for v in p],
        "probabilities_rounded4": [round(float(v), 4) for v in p],
        "index": index,
        "score": expected_score,
        "score_rounded4": round(expected_score, 4),
        "confidence": answer_conf,
        "answer_confidence": answer_conf,
        "answer_confidence_rounded4": round(answer_conf, 4),
        "entropy_confidence": entropy_conf,
        "confidence_entropy": entropy_conf,
        "confidence_entropy_rounded4": round(entropy_conf, 4),
    }


def build_sequence_parts(tok, state, q, max_len, head_max_len):
    """build_head + build_sequence with explicit invariants (SPEC section 5)."""
    from laya.common import build_head, build_sequence, encode_text

    state_ids = [int(i) for i in encode_text(
        tok, state.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]]
    head_ids, head_markers, head_stats = build_head(tok, q, head_max_len)
    seq, markers = build_sequence(tok, state, q, max_len, head_max_len, state_ids=state_ids)
    seq = [int(i) for i in seq]
    markers = [int(m) for m in markers]
    head_ids = [int(i) for i in head_ids]
    head_markers = [int(m) for m in head_markers]
    if markers != [m for m in head_markers if m < max_len]:
        raise SystemExit(
            f"build_sequence markers {markers} != filtered build_head markers {head_markers} "
            f"for state {state!r}")
    room = max(0, max_len - len(head_ids) - 1)
    expected_seq = (head_ids + state_ids[:room] + [int(tok.sep_token_id)])[:max_len]
    if seq != expected_seq:
        raise SystemExit(
            "build_sequence does not match [CLS]+head+[SEP]+state[:room]+[SEP] for "
            f"state {state!r}: got {seq[:12]}... expected {expected_seq[:12]}...")
    return {
        "state_ids": state_ids,
        "head_ids": head_ids,
        "head_len": len(head_ids),
        "markers": markers,
        "sequence": seq,
        "room": room,
        "state_tokens_used": max(0, len(seq) - len(head_ids) - 1),
        "head_stats": head_stats,
    }


def run_logits(agent, sequence, markers):
    """One single-row ONNX run, exactly the collation ONNXAgent uses."""
    from laya.onnx_agent import _collate_items_numpy

    items = [{"ids": list(sequence), "markers": list(markers), "qtype": oc.QTYPE_SCORE}]
    ort_inputs = _collate_items_numpy([items], agent.tok.pad_token_id)
    outs = agent.session.run(["logits", "act_logits"], ort_inputs)
    return np.asarray(outs[0][0], dtype=np.float32)


def temperature_for_score(agent, k):
    from laya.common import temp_bucket

    bucket = temp_bucket(oc.QTYPE_SCORE, k)
    if bucket in agent.temperature_by_options:
        return float(agent.temperature_by_options[bucket])
    return float(agent.temperature[oc.QTYPE_SCORE])


def check_eval_row_question(row, qdef, lineno):
    questions = row.get("questions")
    if not isinstance(questions, dict) or list(questions) != ["category"]:
        raise SystemExit(f"eval.jsonl:{lineno}: expected exactly one 'category' question")
    got = questions["category"]
    if got.get("type") != "score":
        raise SystemExit(f"eval.jsonl:{lineno}: expected a score question, got {got.get('type')!r}")
    if got.get("instructions") != qdef["instructions"]:
        raise SystemExit(f"eval.jsonl:{lineno}: instruction differs from laya/questions.json")
    if list(got.get("criteria") or []) != list(qdef["criteria"]):
        raise SystemExit(f"eval.jsonl:{lineno}: levels differ from laya/questions.json")
    expected = row.get("expected", {}).get("category")
    if not isinstance(expected, int) or not 0 <= expected < len(qdef["criteria"]):
        raise SystemExit(f"eval.jsonl:{lineno}: bad expected category {expected!r}")


def build_eval_reference(agent, limit=0):
    """Run the exact Python pipeline over eval.jsonl rows; return (records, contexts, stats)."""
    from laya.common import QTYPES

    cfg = agent.cfg
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 256))
    if (max_len, head_max_len) != (512, 256):
        raise SystemExit(f"unexpected config: max_len={max_len} head_max_len={head_max_len} "
                         "(SPEC expects 512/256)")
    qdef = oc.load_questions()
    q = {"t": "score", "ins": qdef["instructions"], "crit": list(qdef["criteria"])}
    rows = []
    contexts = []
    t0 = time.time()
    with open(oc.EVAL_JSONL, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            check_eval_row_question(row, qdef, lineno)
            state = row["state"]
            expected = int(row["expected"]["category"])
            parts = build_sequence_parts(agent.tok, state, q, max_len, head_max_len)
            logits = run_logits(agent, parts["sequence"], parts["markers"])
            k = len(parts["markers"])
            t_scale = temperature_for_score(agent, k)
            decoded = decode_row(logits, k, t_scale)
            predicted = decoded["index"]
            record = {
                "id": f"eval:{lineno:04d}",
                "source": f"finetune/data/eval.jsonl:{lineno}",
                "state": state,
                "instruction": q["ins"],
                "levels": list(q["crit"]),
                "qtype": int(QTYPES["score"]),
                "temperature": t_scale,
                "max_len": max_len,
                "head_max_len": head_max_len,
                "state_ids": parts["state_ids"],
                "head_ids": parts["head_ids"],
                "head_len": parts["head_len"],
                "markers": parts["markers"],
                "sequence": parts["sequence"],
                "seq_len": len(parts["sequence"]),
                "state_tokens": len(parts["state_ids"]),
                "state_tokens_used": parts["state_tokens_used"],
                "state_truncated": parts["state_tokens_used"] < len(parts["state_ids"]),
                "logits": [float(v) for v in logits[:k]],
                "probabilities": decoded["probabilities"],
                "probabilities_rounded4": decoded["probabilities_rounded4"],
                "predicted_index": predicted,
                "predicted_label": q["crit"][predicted],
                "score": decoded["score"],
                "score_rounded4": decoded["score_rounded4"],
                "confidence": decoded["confidence"],
                "confidence_entropy": decoded["confidence_entropy"],
                "entropy_confidence": decoded["entropy_confidence"],
                "answer_confidence": decoded["answer_confidence"],
                "answer_confidence_rounded4": decoded["answer_confidence_rounded4"],
                "confidence_entropy_rounded4": decoded["confidence_entropy_rounded4"],
                "expected_index": expected,
                "expected_label": q["crit"][expected],
                "correct": predicted == expected,
            }
            rows.append(record)
            contexts.append({"record": record, "state": state, "expected": expected, "qdef": qdef})
            done = len(rows)
            if done % 25 == 0 or done == 330:
                elapsed = time.time() - t0
                print(f"  [{done:3d}/330] {elapsed:.1f}s ({elapsed / done:.2f}s/row)", flush=True)
            if limit and done >= limit:
                break
    stats = {
        "items": len(rows),
        "max_state_tokens": max((r["state_tokens"] for r in rows), default=0),
        "max_seq_len": max((r["seq_len"] for r in rows), default=0),
        "truncated_rows": [r["id"] for r in rows if r["state_truncated"]],
        "head_len_values": sorted({r["head_len"] for r in rows}),
        "marker_layouts": sorted({tuple(r["markers"]) for r in rows}),
    }
    return rows, contexts, stats


def cross_check_agent(agent, contexts, sample_count=6):
    """Verify the reference path against the official ONNXAgent.predict on a few rows."""
    from laya.onnx_agent import ONNXAgent

    if not contexts:
        return {"checked": 0}
    # shortest, longest and a few fixed rows give a representative spread.
    by_len = sorted(range(len(contexts)), key=lambda i: len(contexts[i]["record"]["sequence"]))
    picks = []
    for idx in (0, 1, 2, len(contexts) // 2, by_len[0], by_len[-1]):
        if idx not in picks:
            picks.append(idx)
    picks = picks[:max(1, sample_count)]

    results = []
    max_prob_diff = 0.0
    max_answer_conf_diff = 0.0
    max_entropy_conf_diff = 0.0
    max_score_diff = 0.0
    seq_exact = True
    predicted_argmax_mismatch = []
    official_round_mismatch = []
    for idx in picks:
        ctx = contexts[idx]
        rec = ctx["record"]
        state, qdef = ctx["state"], ctx["qdef"]
        internal = ONNXAgent._to_internal(qdef)
        encoded = agent._encode_state(
            state, ["category"], {"category": internal},
            int(agent.cfg.get("max_len", 512)), int(agent.cfg.get("head_max_len", 256)))
        item = encoded[0]
        seq_exact = seq_exact and list(item["ids"]) == rec["sequence"]             and list(item["markers"]) == rec["markers"]

        answer = agent.predict(state, {"category": qdef})["answers"]["category"]
        official_probs = [float(answer["probabilities"][str(i)]) for i in range(len(rec["probabilities"]))]
        official_score = float(answer["score"])
        official_answer_conf = float(answer["answer_confidence"])
        official_entropy = float(answer["confidence"])
        # verify_onnx.py derives the class from round(score); SPEC section 6 uses
        # argmax(p).  Both are tracked so a difference is visible rather than silent.
        official_argmax = int(np.argmax(np.asarray(official_probs)))
        official_from_score = int(round(official_score))
        if official_argmax != rec["predicted_index"]:
            predicted_argmax_mismatch.append(rec["id"])
        if official_from_score != rec["predicted_index"]:
            official_round_mismatch.append({
                "id": rec["id"], "argmax": rec["predicted_index"],
                "round_score": official_from_score, "score": official_score})

        prob_diff = max(abs(a - b) for a, b in zip(official_probs, rec["probabilities_rounded4"]))
        max_prob_diff = max(max_prob_diff, prob_diff)
        max_answer_conf_diff = max(
            max_answer_conf_diff, abs(official_answer_conf - rec["answer_confidence_rounded4"]))
        max_entropy_conf_diff = max(
            max_entropy_conf_diff, abs(official_entropy - rec["confidence_entropy_rounded4"]))
        max_score_diff = max(max_score_diff, abs(official_score - rec["score_rounded4"]))
        results.append({
            "id": rec["id"],
            "predicted_index": rec["predicted_index"],
            "official_argmax": official_argmax,
            "official_round_score": official_from_score,
            "max_prob_diff": prob_diff,
            "answer_confidence_diff": abs(official_answer_conf - rec["answer_confidence_rounded4"]),
            "entropy_confidence_diff": abs(official_entropy - rec["confidence_entropy_rounded4"]),
            "score_diff": abs(official_score - rec["score_rounded4"]),
        })
    return {
        "checked": len(picks),
        "rows": results,
        "max_prob_abs_diff": max_prob_diff,
        "max_answer_confidence_abs_diff": max_answer_conf_diff,
        "max_entropy_confidence_abs_diff": max_entropy_conf_diff,
        "max_score_abs_diff": max_score_diff,
        "sequences_exact": seq_exact,
        "predicted_index_mismatches": predicted_argmax_mismatch,
        "round_score_vs_argmax_mismatches": official_round_mismatch,
        "pass": (seq_exact and not predicted_argmax_mismatch
                 and max_prob_diff <= 1e-4 and max_answer_conf_diff <= 1e-4),
    }


def golden_sample_check(agent):
    """The SPEC section 6 worked example, end to end."""
    ins, levels = oc.instruction_and_levels()
    q = {"t": "score", "ins": ins, "crit": levels}
    max_len, head_max_len = int(agent.cfg["max_len"]), int(agent.cfg["head_max_len"])
    parts = build_sequence_parts(agent.tok, SPEC_SAMPLE_STATE, q, max_len, head_max_len)
    logits = run_logits(agent, parts["sequence"], parts["markers"])
    k = len(parts["markers"])
    t_scale = temperature_for_score(agent, k)
    decoded = decode_row(logits, k, t_scale)
    logit_diff = max(abs(a - b) for a, b in zip([float(v) for v in logits[:k]], SPEC_SAMPLE_LOGITS))
    prob_diff = max(abs(a - b) for a, b in zip(decoded["probabilities"], SPEC_SAMPLE_PROBS))
    layout_ok = (parts["head_len"] == SPEC_SAMPLE_HEAD_LEN
                 and parts["markers"] == SPEC_SAMPLE_MARKERS
                 and parts["state_ids"] == SPEC_SAMPLE_STATE_IDS
                 and len(parts["sequence"]) == SPEC_SAMPLE_SEQ_LEN)
    return {
        "state": SPEC_SAMPLE_STATE,
        "head_len": parts["head_len"],
        "markers": parts["markers"],
        "state_ids": parts["state_ids"],
        "seq_len": len(parts["sequence"]),
        "expected_head_len": SPEC_SAMPLE_HEAD_LEN,
        "expected_markers": SPEC_SAMPLE_MARKERS,
        "expected_state_ids": SPEC_SAMPLE_STATE_IDS,
        "expected_seq_len": SPEC_SAMPLE_SEQ_LEN,
        "layout_matches_spec": layout_ok,
        "logits": [float(v) for v in logits[:k]],
        "logits_expected": SPEC_SAMPLE_LOGITS,
        "logits_max_abs_diff": logit_diff,
        "probabilities": decoded["probabilities"],
        "probabilities_expected": SPEC_SAMPLE_PROBS,
        "probabilities_max_abs_diff": prob_diff,
        "temperature": t_scale,
        "index": decoded["index"],
        "score": decoded["score"],
        "confidence": decoded["confidence"],
        "confidence_entropy": decoded["confidence_entropy"],
        "pass": layout_ok and logit_diff <= 1e-4 and prob_diff <= 1e-4
                and decoded["index"] == 0,
    }


def batch_invariance_check(agent, rows, n=8):
    """A right-padded batch must not move logits vs the single-row reference."""
    from laya.onnx_agent import _collate_items_numpy

    n = min(n, len(rows))
    if n <= 1:
        return {"rows": 0}
    groups = [[{"ids": r["sequence"], "markers": r["markers"], "qtype": oc.QTYPE_SCORE}]
              for r in rows[:n]]
    ort_inputs = _collate_items_numpy(groups, agent.tok.pad_token_id)
    outs = agent.session.run(["logits", "act_logits"], ort_inputs)
    logits = np.asarray(outs[0], dtype=np.float32)
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    for i, row in enumerate(rows[:n]):
        k = len(row["markers"])
        max_logit_diff = max(max_logit_diff,
                             max(abs(float(a) - b) for a, b in zip(logits[i][:k], row["logits"])))
        t_scale = temperature_for_score(agent, k)
        decoded = decode_row(logits[i], k, t_scale)
        max_prob_diff = max(max_prob_diff,
                            max(abs(a - b) for a, b in zip(decoded["probabilities"], row["probabilities"])))
    return {
        "rows": n,
        "padded_to_len": int(logits.shape[1]),
        "max_logit_abs_diff": max_logit_diff,
        "max_prob_abs_diff": max_prob_diff,
        "within_1e-4": max_logit_diff <= 1e-4 and max_prob_diff <= 1e-4,
        "informational_only": True,
        "note": ("The int8 graph has 98 DynamicQuantizeLinear nodes; a per-tensor "
                 "activation scale is shared across the whole batch, so multi-row "
                 "batches change every row's logits (verified even for equal-length, "
                 "unpadded batches). The reference is therefore produced one row per "
                 "session run; Engine::score_batch output is not expected to match it "
                 "unless the C++ batch path runs one row at a time."),
    }


def compute_metrics(rows):
    """Accuracy, per-class accuracy, confusion, confidence and calibration metrics."""
    from laya.common import ece_score

    labels = list(oc.LABELS)
    k = len(labels)
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    confusion = [[0] * k for _ in range(k)]
    for r in rows:
        confusion[int(r["expected_index"])][int(r["predicted_index"])] += 1

    per_class = {}
    for c in range(k):
        idx = [r for r in rows if int(r["expected_index"]) == c]
        n = len(idx)
        c_ok = sum(1 for r in idx if r["correct"])
        per_class[c] = {
            "label": labels[c],
            "items": n,
            "correct": c_ok,
            "accuracy": (c_ok / n) if n else None,
            "mean_answer_confidence": (sum(r["answer_confidence"] for r in idx) / n) if n else None,
        }

    conf = np.asarray([r["answer_confidence"] for r in rows], dtype=np.float64)
    ent = np.asarray([r["confidence_entropy"] for r in rows], dtype=np.float64)
    correct_arr = np.asarray([1.0 if r["correct"] else 0.0 for r in rows], dtype=np.float64)
    probs = np.asarray([r["probabilities"] for r in rows], dtype=np.float64)
    expected = np.asarray([int(r["expected_index"]) for r in rows], dtype=np.int64)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    true_p = np.clip(probs[np.arange(total), expected], 1e-12, 1.0)
    onehot = np.zeros_like(probs)
    onehot[np.arange(total), expected] = 1.0

    # Weighted ECE, the same formula laya.common.ece_score uses (default 15 bins).
    ece = float(ece_score(conf, correct_arr, bins=15))
    ece_10 = float(ece_score(conf, correct_arr, bins=10))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    brier_top1 = float(np.mean((1.0 - true_p) ** 2))
    nll = float(np.mean(-np.log(true_p)))
    accuracy_by_round = sum(
        1 for r in rows if int(round(r["score"])) == int(r["expected_index"])) / max(total, 1)

    return {
        "items": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_by_round_score": accuracy_by_round,
        "round_score_vs_argmax_mismatches": [
            {"id": r["id"], "expected": int(r["expected_index"]),
             "predicted_argmax": int(r["predicted_index"]),
             "round_score": int(round(r["score"])), "score": r["score"]}
            for r in rows if int(round(r["score"])) != int(r["predicted_index"])
        ],
        "per_class": per_class,
        "confusion_matrix": {
            "rows_expected_cols_predicted": True,
            "labels": labels,
            "matrix": confusion,
        },
        "mean_confidence": float(conf.mean()),
        "mean_answer_confidence": float(conf.mean()),
        "mean_entropy_confidence": float(ent.mean()),
        "mean_score": float(scores.mean()),
        "nll": nll,
        "ece": ece,
        "ece_15bin": ece,
        "ece_10bin": ece_10,
        "brier": brier,
        "brier_top1": brier_top1,
        "expected_index_distribution": {
            labels[c]: sum(1 for r in rows if int(r["expected_index"]) == c) for c in range(k)
        },
    }


def train_report_reference():
    """The training-time held-out report from laya/rl_agent_config.json."""
    with open(oc.CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    report = cfg.get("training", {}).get("train_report", {})
    return {
        "after": report.get("after"),
        "before": report.get("before"),
        "calibration": report.get("calibration"),
        "abstention_thresholds": report.get("abstention_thresholds"),
        "eval_mode": report.get("eval_mode"),
        "eval_items": report.get("eval_items"),
    }


def compare_train_report(metrics, ref):
    """Round-to-4 comparison against the training-time report (definition notes kept)."""
    after = (ref or {}).get("after") or {}
    pairs = {
        "accuracy": (metrics["accuracy"], after.get("accuracy")),
        "mean_confidence": (metrics["mean_confidence"], after.get("mean_confidence")),
        "nll_loss": (metrics["nll"], after.get("loss")),
        "ece_15bin": (metrics["ece_15bin"], after.get("ece")),
        "brier": (metrics["brier"], after.get("brier")),
        "brier_top1": (metrics["brier_top1"], after.get("brier_top1")),
    }
    out = {}
    for name, (ours, reported) in pairs.items():
        if reported is None:
            out[name] = {"oracle": ours, "reported": None}
            continue
        out[name] = {
            "oracle": ours,
            "reported": reported,
            "delta": ours - reported,
            "rounded4_matches": round(ours, 4) == round(reported, 4),
        }
    for name in ("ece_15bin", "brier", "brier_top1", "nll_loss"):
        if name in out:
            out[name]["note"] = ("definition may differ from the training harness; "
                                 "accuracy/mean_confidence are the gate-relevant values")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the independent Python reference oracle for laya.")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="only write tokenizer_corpus.jsonl (no ONNX model load)")
    parser.add_argument("--eval-only", action="store_true",
                        help="only write eval_reference.jsonl + baseline_report.json")
    parser.add_argument("--out-dir", default=str(oc.REFERENCE_DIR),
                        help="output directory (default: tools/laya_oracle/reference)")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke-run only the first N eval rows; writes *.partial files")
    parser.add_argument("--agent-samples", type=int, default=6,
                        help="rows to cross-check against ONNXAgent.predict (default 6)")
    parser.add_argument("--batch-check", type=int, default=8,
                        help="rows to re-run as a padded batch (default 8, 0 disables)")
    args = parser.parse_args(argv)
    if args.tokenizer_only and args.eval_only:
        parser.error("--tokenizer-only and --eval-only are mutually exclusive")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".partial" if args.limit else ""
    corpus_path = out_dir / ("tokenizer_corpus" + suffix + ".jsonl")
    eval_path = out_dir / ("eval_reference" + suffix + ".jsonl")
    report_path = out_dir / ("baseline_report" + suffix + ".json")
    started = time.time()
    commands = [" ".join([str(oc.VENV_PYTHON), "tools/laya_oracle/dump_reference.py"]
                         + (sys.argv[1:] if argv is None else list(argv)))]
    print("laya Python oracle")
    print(f"  repo:      {oc.REPO_ROOT}")
    print(f"  python:    {sys.executable}")
    print(f"  out dir:   {out_dir}")
    print(f"  commands:  {commands[0]}", flush=True)

    corpus_stats = None
    if not args.eval_only:
        print("\n[1/2] tokenizer corpus", flush=True)
        t0 = time.time()
        tok = load_tokenizer()
        records, corpus_stats = build_tokenizer_corpus(tok)
        oc.write_jsonl(corpus_path, records)
        print(f"  wrote {corpus_path} ({len(records)} records, "
              f"{oc.human_bytes(corpus_path.stat().st_size)}) in {time.time() - t0:.1f}s", flush=True)
        print(f"  dataset states: {corpus_stats['dataset_states']} "
              f"(max {corpus_stats['max_dataset_state_tokens']} tokens); "
              f"added tokens: {corpus_stats['added_tokens']} "
              f"({len(corpus_stats['added_token_mismatches'])} mismatches); "
              f"SPEC goldens: {corpus_stats['spec_tokenizer_goldens_passed']} passed", flush=True)
        print(f"  kinds: {corpus_stats['kinds']}", flush=True)
        if corpus_stats["added_token_mismatches"]:
            print("  ERROR: added-token self-check mismatches: "
                  f"{corpus_stats['added_token_mismatches'][:5]}", flush=True)
            return 3
        del records, tok
        if args.tokenizer_only:
            print(f"\ntokenizer-only run done in {time.time() - started:.1f}s", flush=True)
            return 0

    print("\n[2/2] eval reference (single model load)", flush=True)
    from laya.onnx_agent import ONNXAgent

    t_load = time.time()
    agent = ONNXAgent(str(oc.MODEL_DIR), onnx_path=str(oc.ONNX_PATH))
    print(f"  model loaded in {time.time() - t_load:.1f}s; "
          f"max_len={agent.cfg.get('max_len')} head_max_len={agent.cfg.get('head_max_len')} "
          f"temperature={agent.cfg.get('temperature')}", flush=True)

    rows, contexts, stats = build_eval_reference(agent, limit=args.limit)
    t_infer = time.time() - t_load
    metrics = compute_metrics(rows)
    ref = train_report_reference()
    comparison = compare_train_report(metrics, ref)
    golden = golden_sample_check(agent)
    cross = cross_check_agent(agent, contexts, sample_count=args.agent_samples)
    batch = batch_invariance_check(agent, rows, n=args.batch_check) if args.batch_check else {"rows": 0}

    oc.write_jsonl(eval_path, rows)
    files = {}
    if corpus_path.exists():
        files[corpus_path.name] = {
            "bytes": corpus_path.stat().st_size,
            "sha256": oc.sha256_file(corpus_path),
            "records": (corpus_stats or {}).get("records"),
        }
    files[eval_path.name] = {
        "bytes": eval_path.stat().st_size,
        "sha256": oc.sha256_file(eval_path),
        "records": len(rows),
    }
    import onnxruntime
    import transformers

    report = {
        "schema": "laya-oracle-baseline/1",
        "generated_at": oc.now_iso(),
        "generated_by": "tools/laya_oracle/dump_reference.py",
        "partial": bool(args.limit),
        "commands": commands,
        "environment": {
            "repo_root": str(oc.REPO_ROOT),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "onnxruntime": onnxruntime.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "git_commit": oc.git_commit(),
        },
        "model": {
            "dir": str(oc.MODEL_DIR.relative_to(oc.REPO_ROOT)),
            "onnx": str(oc.ONNX_PATH.relative_to(oc.REPO_ROOT)),
            "onnx_bytes": oc.ONNX_PATH.stat().st_size,
            "max_len": int(agent.cfg.get("max_len", 512)),
            "head_max_len": int(agent.cfg.get("head_max_len", 256)),
            "temperature": agent.cfg.get("temperature"),
            "temperature_clamped": agent.temperature,
            "temperature_by_options": agent.temperature_by_options,
            "applied_score_temperature": temperature_for_score(agent, 3),
            "qtype": {"name": "score", "value": oc.QTYPE_SCORE},
            "option_layout": agent.cfg.get("option_layout", "sequential"),
        },
        "pipeline": {
            "sequence": "[CLS] head [SEP] option0 [SEP] ... state[:room] [SEP]; "
                        "head = 'score question: ' + instruction, options = 'level i: level'",
            "option_truncation": "truncation=True max_length=48 on ' ' + option",
            "state_truncation": "right truncation to max_len - len(head) - 1 tokens",
            "probabilities": "softmax(float32 logits / temperature[1]), float32 numpy; "
                             "rounded4 variants mirror ONNXAgent output",
            "confidence": "max(p) (laya.answer_confidence / C++ ScoreResult.confidence)",
            "entropy_confidence": "1 - H(p)/log(k) (laya.confidence_from_probs)",
            "reference_run_mode": "one row per ONNX session run (no padding)",
            "inference_seconds": t_infer,
            "seconds_per_row": t_infer / max(1, len(rows)),
        },
        "dataset": {
            "eval_source": str(oc.EVAL_JSONL.relative_to(oc.REPO_ROOT)),
            "eval_rows": len(rows),
            "unique_states": len({r["state"] for r in rows}),
            "max_state_tokens": stats["max_state_tokens"],
            "max_seq_len": stats["max_seq_len"],
            "head_len_values": stats["head_len_values"],
            "marker_layouts": [list(m) for m in stats["marker_layouts"]],
            "truncated_rows": stats["truncated_rows"],
        },
        **metrics,
        "train_report": ref,
        "train_report_comparison": comparison,
        "sanity": {
            "spec_golden_sample": golden,
            "onnx_agent_cross_check": cross,
            "batch_invariance": batch,
        },
        "files": files,
        "notes": [
            "Reference rows are produced one at a time (no padding), matching SPEC section 6 "
            "and finetune/verify_onnx.py. Engine::score() must reproduce them directly.",
            "The int8 model is NOT batch-composition invariant: ORT's DynamicQuantizeLinear "
            "uses one activation scale per tensor (98 such nodes), so a multi-row batch shifts "
            "every row's logits (reproduced with equal-length, unpadded batches). "
            "sanity.batch_invariance records the measured drift; it is informational.",
            "Any batching implementation (Python ONNXAgent.predict_batch or C++ "
            "Engine::score_batch) therefore produces different numbers from these single-row "
            "reference rows; a batch path can only match another run with the identical batch "
            "layout on the identical ORT build.",
            "Empty result lists in this report are not failures; see the *_mismatches keys.",
        ],
    }
    oc.write_json(report_path, report)

    # ---------------------------------------------------------------- summary
    acc = metrics["accuracy"]
    train_acc = (ref.get("after") or {}).get("accuracy")
    print("\nbaseline report", flush=True)
    print(f"  items:     {metrics['items']}", flush=True)
    print(f"  accuracy:  {metrics['correct']}/{metrics['items']} = {acc:.4%}"
          + (f"   (train report {train_acc:.4%})" if train_acc is not None else ""), flush=True)
    print(f"  confusion (rows=expected, cols=predicted): {metrics['confusion_matrix']['matrix']}",
          flush=True)
    for c, pc in metrics["per_class"].items():
        print(f"    {pc['label']:<8} {pc['correct']}/{pc['items']} = "
              f"{pc['accuracy']:.4f}  mean_conf={pc['mean_answer_confidence']:.4f}", flush=True)
    print(f"  mean confidence (max p): {metrics['mean_confidence']:.6f}", flush=True)
    print(f"  mean entropy confidence: {metrics['mean_entropy_confidence']:.6f}", flush=True)
    print(f"  mean score:              {metrics['mean_score']:.6f}", flush=True)
    print(f"  nll / ece15 / ece10 / brier / brier_top1: {metrics['nll']:.6f} / "
          f"{metrics['ece_15bin']:.6f} / {metrics['ece_10bin']:.6f} / "
          f"{metrics['brier']:.6f} / {metrics['brier_top1']:.6f}", flush=True)
    print(f"  accuracy by round(score): {metrics['accuracy_by_round_score']:.4%} "
          f"({len(metrics['round_score_vs_argmax_mismatches'])} rows differ from argmax)", flush=True)
    print(f"  max state tokens: {stats['max_state_tokens']}; "
          f"truncated rows: {len(stats['truncated_rows'])}", flush=True)
    print(f"  SPEC golden sample: {'PASS' if golden['pass'] else 'FAIL'} "
          f"(logits max|diff|={golden['logits_max_abs_diff']:.2e}, "
          f"probs max|diff|={golden['probabilities_max_abs_diff']:.2e})", flush=True)
    print(f"  ONNXAgent cross-check: {cross['checked']} rows, "
          f"{'PASS' if cross['pass'] else 'FAIL'} "
          f"(prob max|diff|={cross['max_prob_abs_diff']:.2e}, "
          f"seq exact={cross['sequences_exact']})", flush=True)
    print(f"  batch sensitivity (informational): {batch.get('rows', 0)} rows, "
          + ("matches single-row" if batch.get("within_1e-4")
             else "differs from single-row")
          + f" (logit max|diff|={batch.get('max_logit_abs_diff', 0.0):.2e}; int8 "
            "per-tensor activation quantization is batch-composition dependent)", flush=True)
    print(f"  wrote {eval_path} ({oc.human_bytes(eval_path.stat().st_size)})", flush=True)
    print(f"  wrote {report_path}", flush=True)
    print(f"  done in {time.time() - started:.1f}s", flush=True)

    failures = []
    if not golden["pass"]:
        failures.append("SPEC golden sample")
    if not cross.get("pass"):
        failures.append("ONNXAgent cross-check")
    if args.limit:
        print(f"  smoke run: only {len(rows)} rows; *.partial files written", flush=True)
    if failures:
        print(f"  FAILED sanity checks: {failures}", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())

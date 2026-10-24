# 针对本任务微调 laya

LodeRouter 的裁判不是原版 laya 的难度模型，而是在 `convaiinnovations/laya-multilingual` 上按本任务的三分类微调出来的 ONNX 模型：

| score level | 类别 | 路由挡位 |
| --- | --- | --- |
| 0 | `chitchat` 闲聊 | `easy` |
| 1 | `math` 数学 | `middle` |
| 2 | `code` 代码 | `hard` |

`src/judge_laya.h` 里的 `instruction` 与三个 level 文本必须和这里的训练渲染逐字一致，否则概率会错位。

## 1. 环境

```bash
python3 -m venv .cache/laya-ft/venv
.cache/laya-ft/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.cache/laya-ft/venv/bin/pip install 'laya[onnx]'
```

CPU 就够：下面的配置只训练 2 层 decision head（`--freeze-encoder`），1350 条样本约 20–30 分钟。

## 2. 下载基础权重

从 hf-mirror.com 取 `laya-multilingual`（约 643 MB）：

```bash
BASE=https://hf-mirror.com/convaiinnovations/laya-multilingual/resolve/main
mkdir -p .cache/laya-ft/laya-multilingual/{encoder,tokenizer}
cd .cache/laya-ft/laya-multilingual
for f in model.safetensors config.json encoder/config.json rl_agent_config.json \
         tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
  curl -L -C - --retry 5 -o "$f" "$BASE/$f"
done
```

## 3. 生成数据

```bash
python3 finetune/build_dataset.py
# finetune/data/train.jsonl  1350 条
# finetune/data/eval.jsonl    330 条（评测句式与训练不重叠，另有 30 条边界样本）
```

`build_dataset.py` 里的 `INSTRUCTION` 和 `LEVELS` 就是训练与推理共用的问法。改它们时同步改 `src/judge_laya.h`。

## 4. 微调

```bash
HF_HUB_OFFLINE=1 .cache/laya-ft/venv/bin/laya-train \
  --data finetune/data/train.jsonl \
  --eval-data finetune/data/eval.jsonl \
  --base .cache/laya-ft/laya-multilingual \
  --out .cache/laya-ft/laya \
  --freeze-encoder --loss soft-ce \
  --epochs 12 --micro-batch 8 --grad-accum 4 \
  --device cpu --max-len 512 --head-max-len 256
```

跑完会打印 before/after 的 accuracy 与 ECE；`--eval-data` 是独立的 330 条评测集，不是训练集切片。

## 5. 导出 ONNX（int8）

```bash
curl -L -o .cache/laya-ft/export_onnx.py \
  https://raw.githubusercontent.com/NandhaKishorM/laya/main/scripts/export_onnx.py
.cache/laya-ft/venv/bin/python .cache/laya-ft/export_onnx.py \
  --model .cache/laya-ft/laya \
  --output .cache/laya-ft/laya/model.fp32.onnx
```

交付只要 int8：对导出的 fp32 图做动态量化（只量化 `MatMul` 权重，per-tensor），写成单文件 `model.onnx`，不保留 fp32 权重：

```bash
.cache/laya-ft/venv/bin/python - <<'PY'
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model = onnx.load(".cache/laya-ft/laya/model.fp32.onnx")
del model.graph.value_info[:]
quantize_dynamic(
    model_input=model,
    model_output=".cache/laya-ft/laya/model.onnx",
    op_types_to_quantize=["MatMul"],
    weight_type=QuantType.QInt8,
    per_channel=False,
)
PY
rm -f .cache/laya-ft/laya/model.fp32.onnx \
      .cache/laya-ft/laya/model.fp32.onnx.data
```

`export_onnx.py` 来自 laya 仓库（`scripts/export_onnx.py`），输入 `input_ids/attention_mask/marker_pos/marker_mask/qtype`，输出 `logits/act_logits`。量化后的 `model.onnx` 约 925 MB，单文件、无外置 `.data`。再把 tokenizer 放到 C++ 端期望的根目录：

```bash
cp .cache/laya-ft/laya/tokenizer/tokenizer.json \
   .cache/laya-ft/laya/tokenizer.json
```

最终目录（只有 int8，运行时不需要 PyTorch 权重）：

```text
laya/
├── model.onnx              # int8，约 925 MB，单文件
├── tokenizer.json
├── rl_agent_config.json
├── questions.json          # laya-train 写出的训练问法
├── encoder/config.json
└── tokenizer/
```

## 6. 让 LodeRouter 用它

把整个目录复制到以下任一位置（按优先级）：

1. `judge_model_dir` 配置或 `LODE_JUDGE_MODEL_DIR` 环境变量
2. 编译时 `-DLAYA_MODEL_DIR=...` 指定的路径（默认仓库内 `laya/`）
3. 可执行文件旁的 `laya/`
4. 工作目录下的 `laya/` 或 `.cache/laya-ft/laya/`

默认路径已经包含工作目录下的 `laya/` 和 `.cache/laya-ft/laya/`。本仓库已经把导出结果放在根目录 `laya/`，所以按上面的命令跑完就能直接启动。

## 备注

- 原版 laya 只分难度（trivial/easy/moderate/hard），没有闲聊/数学/代码的区分；`judge_auto_download` 默认关闭，避免误加载。
- 这里的 C++ 裁判用 laya 的 `score` 接口（`instruction` + 有序 levels），因此微调也必须用 `score` 问题，而不是 `choice`。
- 数据是模板生成的。要提高真实流量上的准确率，把线上日志里的请求按三类标注后加入 `train.jsonl`，再重跑第 4、5 步。

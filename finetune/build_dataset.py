#!/usr/bin/env python3
"""Build the chitchat / math / code training and evaluation sets for LodeRouter's judge.

The judge is a laya ``score`` question with three levels, in this order:

    0 chitchat  -> router easy   (闲聊)
    1 math      -> router middle (数学)
    2 code      -> router hard   (代码)

The level texts and the instruction below must stay byte-identical to the ones in
``src/judge_laya.h``; the model was trained with this exact rendering.

Run from the repository root:

    python3 finetune/build_dataset.py

Writes ``finetune/data/train.jsonl`` and ``finetune/data/eval.jsonl``.
Deterministic: the same seed always produces the same files.
"""

import json
import random
from pathlib import Path

INSTRUCTION = "Which category does this request belong to?"
LEVELS = [
    "chitchat: casual conversation, greetings, small talk, jokes, feelings, opinions",
    "math: mathematics, arithmetic, algebra, geometry, calculus, probability, proofs",
    "code: programming, debugging, refactoring, software design, APIs, databases, tooling",
]

# ───────── chitchat ─────────
CHITCHAT_TOPICS = [
    "你好呀",
    "在吗",
    "早上好",
    "晚上好",
    "今天心情不错",
    "我最近有点累，想找人聊聊",
    "周末一般都干嘛",
    "推荐一部轻松的电影吧",
    "给我讲个笑话",
    "你觉得猫和狗哪个更可爱",
    "你吃饭了吗",
    "最近有什么好看的剧",
    "我睡不着，陪我聊会儿",
    "生日快乐",
    "谢谢你帮我这么多",
    "再见，明天见",
    "你觉得夏天好还是冬天好",
    "我养的多肉又死了，好难过",
    "今天上班被老板夸了，开心",
    "你喜欢什么颜色",
    "下雨天适合做什么",
    "有什么解压的办法吗",
    "我朋友最近不理我了",
    "推荐一首适合跑步听的歌",
    "你觉得人应该早睡吗",
    "好无聊啊",
    "我刚搬到一个新城市，有点不习惯",
    "你平时会做梦吗",
    "周末想去公园野餐",
    "我好像感冒了，难受",
    "你觉得旅行最有趣的是什么",
    "给我起个可爱的昵称",
    "今天是我的生日",
    "你最喜欢哪个季节",
    "我想养一只猫，有什么建议",
    "好久没聊天了",
    "晚上吃什么好呢",
    "你相信星座吗",
    "最近压力有点大",
    "能夸夸我吗",
    "hey there",
    "how's your day going",
    "tell me a joke",
    "good morning",
    "I'm feeling a bit down today",
    "what's your favorite movie",
    "any plans for the weekend",
    "do you like tea or coffee",
    "I just adopted a puppy",
    "good night, talk tomorrow",
]

CHITCHAT_TEMPLATES = [
    "{t}",
    "{t}，随便聊聊",
    "在吗？{t}",
    "想问个轻松的：{t}",
    "{t}，说说你的看法",
    "有点想聊天，{t}",
    "{t}～",
    "嘿，{t}",
    "跟你唠两句：{t}",
    "{t}，不用太认真回答",
]

# ───────── math ─────────
MATH_TOPICS = [
    "帮我算一下 17×23 等于多少",
    "1234+5678 是多少",
    "解方程 2x+5=13",
    "求 x^2-4x+3=0 的两个根",
    "圆的半径是 3，面积是多少",
    "求 f(x)=x^3 的导数",
    "计算 ∫x^2 dx",
    "抛两次硬币，至少一次正面的概率是多少",
    "证明根号 2 是无理数",
    "甲乙两人相向而行，甲每小时 5 公里，乙每小时 4 公里，两地相距 18 公里，多久相遇",
    "一个班 40 人，男生占 60%，女生多少人",
    "sin(30°) 等于多少",
    "log2(1024) 是多少",
    "求 1 到 100 的和",
    "3 的 5 次方是多少",
    "把 0.375 化成分数",
    "已知三角形两边长 3 和 4，夹角 90°，斜边多长",
    "求等差数列 2,5,8,... 的第 20 项",
    "解不等式 3x-7>5",
    "矩阵 [[1,2],[3,4]] 的行列式是多少",
    "求极限 lim(x→0) sin(x)/x",
    "一个正方体边长 2，体积和表面积分别是多少",
    "从 5 个人里选 2 个，有多少种组合",
    "求函数 y=x^2-4x+5 的最小值",
    "计算 15% 的 240 是多少",
    "证明勾股定理",
    "求 2/3 + 3/4 等于多少",
    "一个数的 3 倍减去 7 等于 11，这个数是多少",
    "掷一个骰子，点数是偶数的概率",
    "求圆的周长，直径是 10",
    "因式分解 x^2-9",
    "计算 7! 的值",
    "求过点 (1,2) 且斜率为 3 的直线方程",
    "解方程组 x+y=10, x-y=2",
    "一个长方形长 8 宽 5，对角线多长",
    "求 cos(60°)",
    "将 101101 二进制转成十进制",
    "求 1/2 + 1/6 + 1/12 + 1/20 的和",
    "证明任意三个连续整数中必有一个是 3 的倍数",
    "已知 f(x)=2x+1，求 f(f(3))",
    "solve for x: 3x - 7 = 11",
    "what's the integral of sin(x)",
    "calculate 18% of 250",
    "find the derivative of ln(x)",
    "what is the probability of rolling a 7 with two dice",
    "simplify (x^2 - 1)/(x - 1)",
    "how many ways to arrange the letters in BANANA",
    "find the area of a circle with radius 5",
    "solve the system 2x + y = 7, x - y = 2",
    "what is the sum of the first 50 positive integers",
]

MATH_TEMPLATES = [
    "{t}",
    "帮我算算：{t}",
    "求解：{t}",
    "{t}，要步骤",
    "这道题怎么做：{t}",
    "数学题：{t}",
    "{t}，请给出推导过程",
    "我想知道 {t} 的结果",
    "请计算 {t}",
    "{t}，答案是多少",
]

# Eval uses different phrasings from training, so a high score is not a memorised template.
EVAL_CHITCHAT_TEMPLATES = [
    "{t}，想听听你的想法",
    "能陪我聊聊吗？{t}",
    "问个轻松的：{t}",
    "随便说说，{t}",
    "{t} 哈哈",
]

EVAL_MATH_TEMPLATES = [
    "{t}，请一步步算",
    "数学求助：{t}",
    "这道题的答案是？{t}",
    "帮我验证一下：{t}",
    "{t} 怎么推导",
]

EVAL_CODE_TEMPLATES = [
    "{t}，代码怎么写",
    "遇到技术问题：{t}",
    "请教一个编程问题：{t}",
    "{t} 报错了",
    "帮我实现：{t}",
]

# ───────── code ─────────
CODE_TOPICS = [
    "用 Python 写一个快速排序",
    "写个 SQL 查询，按部门统计平均工资",
    "这段 C++ 编译报 undefined reference to `parse_config`，怎么解决",
    "TypeError: 'NoneType' object is not subscriptable 是什么原因",
    "解释一下 React 的 useEffect",
    "什么是闭包",
    "帮我重构这个函数，它太长了",
    "设计一个高并发的短链接服务",
    "git 怎么撤销上一次 commit",
    "docker compose 起不来，日志显示 port is already allocated",
    "用 Go 写一个 HTTP 健康检查接口",
    "Java 的 HashMap 和 ConcurrentHashMap 有什么区别",
    "Rust 的所有权规则怎么理解",
    "给这段代码加单元测试",
    "为什么我的 CSS flex 布局不居中",
    "Vue 3 的 ref 和 reactive 区别",
    "Python 里怎么读取一个 10GB 的日志文件",
    "实现一个 LRU 缓存",
    "MySQL 索引失效的常见原因",
    "Kubernetes Pod 一直 CrashLoopBackOff 怎么排查",
    "写一个正则匹配邮箱地址",
    "解释 TCP 三次握手",
    "这段 JavaScript 为什么会内存泄漏",
    "用 TypeScript 定义一个泛型函数",
    "Redis 缓存雪崩怎么处理",
    "Linux 怎么查看某个端口被哪个进程占用",
    "写一个 shell 脚本批量重命名文件",
    "这个 Python 脚本跑得太慢，怎么优化",
    "解释一下 REST 和 GraphQL 的区别",
    "实现二叉树的层序遍历",
    "C++ 的 unique_ptr 和 shared_ptr 怎么选",
    "Nginx 反向代理配置怎么写",
    "前端怎么实现无限滚动",
    "这段代码的复杂度是多少",
    "怎么写一个 Dockerfile 部署 Node 服务",
    "Git 合并冲突怎么解决",
    "解释一下数据库事务的隔离级别",
    "写一个定时任务每天凌晨清理临时文件",
    "我的 React 组件重复渲染，怎么定位",
    "用 C 写一个简单的链表",
    "write a bash script that renames all jpg files",
    "why does my Rust borrow checker complain here",
    "implement a binary search in Go",
    "how do I center a div with CSS",
    "explain the difference between process and thread",
    "my pytest fixture is not found, how to fix it",
    "write a SQL query to find duplicate emails",
    "how to undo the last git commit",
    "implement debounce in JavaScript",
    "what's the time complexity of this loop",
    "fix this Python recursion error",
]

CODE_TEMPLATES = [
    "{t}",
    "代码问题：{t}",
    "帮我写代码：{t}",
    "{t}，给个示例",
    "请问 {t}",
    "{t}，最好带注释",
    "开发时遇到：{t}",
    "{t}，怎么实现",
    "我需要 {t}",
    "{t}，给出可运行的代码",
]

# ───────── hand-written boundary cases (eval only) ─────────
EVAL_BOUNDARY = [
    ("今天好热，35 摄氏度换成华氏度是多少", 1),
    ("帮我写个程序算 1 到 100 的和", 2),
    ("用一句话解释勾股定理", 1),
    ("解释一下快速排序的数学原理", 2),
    ("这道数学题用 Python 怎么写代码解", 2),
    ("聊天机器人是怎么实现的", 2),
    ("1+1 等于几", 1),
    ("推荐几个学线性代数的网站", 1),
    ("给我讲讲你最喜欢的一门编程语言", 2),
    ("在吗？帮我看看这段报错", 2),
    ("你觉得数学难还是编程难", 0),
    ("周末写代码还是出去爬山", 0),
    ("我数学不好，怎么才能学好", 1),
    ("怎么用 Excel 算平均数", 1),
    ("用代码画一个心形图案", 2),
    ("这个概率题我完全没思路", 1),
    ("帮我解释一下这个报错是什么意思", 2),
    ("好累啊，写了一天代码", 0),
    ("推荐一本适合入门的算法书", 2),
    ("求推荐一个好看的电影，不要烧脑的", 0),
    ("帮我算算这个月还剩多少天", 1),
    ("Python 的列表和元组有什么区别", 2),
    ("怎么证明一个数是质数", 1),
    ("写一个函数判断回文数", 2),
    ("今天股市涨了多少", 0),
    ("矩阵乘法怎么手算", 1),
    ("git rebase 和 merge 有什么区别", 2),
    ("给我讲个冷笑话", 0),
    ("怎么把十进制转成二进制", 1),
    ("这段 SQL 为什么查不出数据", 2),
]


def _make_row(state: str, level: int) -> dict:
    return {
        "state": state,
        "questions": {
            "category": {
                "type": "score",
                "instructions": INSTRUCTION,
                "criteria": LEVELS,
            }
        },
        "expected": {"category": level},
    }


def _generate(topics, templates, count, rng, used):
    rows = []
    attempts = 0
    while len(rows) < count and attempts < count * 40:
        attempts += 1
        topic = rng.choice(topics)
        template = rng.choice(templates)
        state = template.format(t=topic).strip()
        if state in used:
            continue
        used.add(state)
        rows.append(state)
    return rows


def build_split(seed: int, per_class: int, boundary: list, eval_templates: bool = False) -> list:
    rng = random.Random(seed)
    used = set()
    rows = []
    for level, (topics, templates) in enumerate(
        [
            (CHITCHAT_TOPICS, EVAL_CHITCHAT_TEMPLATES if eval_templates else CHITCHAT_TEMPLATES),
            (MATH_TOPICS, EVAL_MATH_TEMPLATES if eval_templates else MATH_TEMPLATES),
            (CODE_TOPICS, EVAL_CODE_TEMPLATES if eval_templates else CODE_TEMPLATES),
        ]
    ):
        for state in _generate(topics, templates, per_class, rng, used):
            rows.append(_make_row(state, level))
    for state, level in boundary:
        rows.append(_make_row(state, level))
    rng.shuffle(rows)
    return rows


def main() -> None:
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(parents=True, exist_ok=True)

    # Train and eval draw from the same topic pools but use disjoint templates and
    # different seeds, so an eval score is not a memorised template.
    train = build_split(seed=20261024, per_class=450, boundary=[])
    evals = build_split(seed=7, per_class=100, boundary=EVAL_BOUNDARY, eval_templates=True)

    train_names = {row["state"] for row in train}
    evals = [row for row in evals if row["state"] not in train_names]

    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        path = out / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):4d} rows to {path}")


if __name__ == "__main__":
    main()

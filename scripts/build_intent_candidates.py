"""从 dataset/ 里的 CSDS 客服对话（京东 JDDC）抽取意图评测候选样本。

数据集的 211 个细粒度标签和我们的 11 类对不上，所以这里只做两件事：
把真实用户的原话抽出来，并按映射表给一个"建议意图"。最终标签必须人工确认。

用法（本机没有 python 命令，用 uv run python）：
    uv run python scripts/build_intent_candidates.py extract                 # 生成 evals/candidates/<意图>.jsonl
    uv run python scripts/build_intent_candidates.py show refund             # 带编号打印候选：原话、上文、源标签
    uv run python scripts/build_intent_candidates.py pick refund 3 7 12      # 按编号选入 picked.jsonl
    uv run python scripts/build_intent_candidates.py pick refund 5 --as order_logistics --note "见指南第 1 条"
    uv run python scripts/build_intent_candidates.py merge                   # 把 picked.jsonl 并入评测集

人工流程：show 看候选，pick 选入。建议意图不对的用 --as 改掉，要改 text 的直接编辑 picked.jsonl，
最后 merge。下划线开头的字段只给人看，merge 时会去掉。
编号就是候选文件里的行号。extract 只在文件末尾追加，已有的行和编号不变。pick 会把选中的原话打印出来供核对。
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "evals" / "candidates"
PICKED_PATH = OUT_DIR / "picked.jsonl"
CASES_PATH = ROOT / "evals" / "intent_cases.jsonl"

PER_CLASS = 120          # 每类的候选总数。目标是每类留 20 到 35 条，实际能留下的大约三分之一
MIN_LEN, MAX_LEN = 4, 80
ELLIPSIS_MAX_LEN = 10    # 不超过这个长度且有上文的，当作省略句，带上 history
HISTORY_TURNS = 4
SEED = 20260921

INTENTS = [
    "greeting", "order_logistics", "complaint", "refund", "invoice", "payment_issue",
    "tech_login", "tech_error", "account_security", "human_handoff", "other",
]

# 源标签 -> 建议意图。只收对应关系比较明确的，其余源标签直接丢弃
SOURCE_MAP = {
    "order_logistics": [
        "配送周期", "修改订单", "订单状态解释", "什么时间出库", "联系配送", "物流全程跟踪",
        "订单签收异常", "预约配送时间", "物流信息不正确", "快递单号不正确", "提前配送",
        "订单无故取消", "怎么确认收货", "下单地址填写", "能否自提", "代收",
    ],
    "refund": [
        "正常退款周期", "申请退款", "退款到哪儿", "返回方式", "返修退换货处理周期", "退款异常",
        "在哪里查询退款", "取消退款", "部分商品退款", "审核时效", "返回地址", "填写返件运单号",
        "返修退换货审核不通过", "余额退款", "退款通知", "退款手续费",
        # refund 的范围是整个售后，换货、维修、补发都算
        "保修返修及退换货政策", "售后运费", "服务单查询", "服务单修改", "关闭服务单", "少商品与少配件",
        "发错货", "补发", "联系售后", "返修退换货拆包装", "检测单咨询", "售后维修点查询",
    ],
    "invoice": [
        "发票退换修改", "是否提供发票", "查看发票", "增票相关", "电子发票", "填写发票信息",
        "补发票", "返修退换货发票",
    ],
    # "在线支付""微信支付"两个源标签实际是支付方式咨询，抽出来几乎全是噪声，已移除
    "payment_issue": ["支付异常", "支付到账时间"],
    "tech_login": ["登录问题", "忘记账户名"],
    # "无法使用优惠券""无法申请价保"问的是规则而不是故障，按指南第 4 条属于 other，已移除
    "tech_error": ["无法购买提交", "无法加入购物车", "下单后无记录", "评价晒单异常"],
    "account_security": ["账户安全", "账户注销", "解锁锁定", "修改账户信息", "实名认证与解除", "解绑和绑定微信"],
    "human_handoff": ["联系客服"],
    "other": ["属性咨询", "使用咨询", "近期活动咨询", "商家入驻条件", "手机回收流程", "商品比较", "正品保障", "家电安装"],
}

# 两类之间的边界源标签：给出建议意图，同时在 note 里提醒这是需要自己定规则的案例
BOUNDARY_MAP = {
    "如何取消订单": ("order_logistics", "见指南第 6 条：取消或拦截标 order_logistics，取消后问退款标 refund"),
    "查询取消是否成功": ("order_logistics", "见指南第 6 条：取消或拦截标 order_logistics，取消后问退款标 refund"),
    "拒收": ("order_logistics", "见指南第 7 条：能否拒收、怎么拒收标 order_logistics，拒收后问退款标 refund"),
    "物流损": ("refund", "边界：运输损坏，要退换是 refund，只是查件是 order_logistics"),
    "保修期保质期": ("refund", "边界：问保修多久是售后，问食品保质期是售前咨询"),
    "找回密码": ("tech_login", "见指南第 9 条：忘记密码、收不到验证码标 tech_login，绑定手机已换需改绑标 account_security"),
    "手机邮件相关问题": ("account_security", "见指南第 9 条：收不到验证码标 tech_login，换绑手机标 account_security"),
    "支付密码": ("account_security", "见指南第 8 条：改或找回支付密码标 account_security"),
    "充值未到账充值到账时间": ("payment_issue", "见指南第 5 条：钱付了没到账标 payment_issue；还没付款的咨询标 other"),
    "优惠券退回": ("refund", "边界：退的是券不是钱"),
}

# 数据集没有对应标签的类，用正则从所有用户发言里捞。
# 注意这样捞出来的全是带关键词的说法，不含关键词的那一半仍然要自己写
KEYWORD_POOLS = {
    "greeting": r"^(你好|您好|在吗|在不在|有人吗|有人在吗|hi|hello|哈喽|嗨)[呀啊吗么?？!！~。，]*$",
    "complaint": r"投诉|太差|什么态度|无语|坑人|骗人|忽悠|差劲|没人管|不负责",
    "human_handoff": r"人工|真人|机器人|找.*(主管|领导|经理)",
    "payment_issue": r"扣了两|重复(扣|支付|付款)|多扣|付了两|乱扣|(支付|付款)(失败|不了|不成功)|付不了",
    "tech_login": r"登[录陆]不|登不[上了]|无法登[录陆]|收不到验证码|验证码.*(收不到|没收到|不对|错误)",
    "tech_error": r"闪退|白屏|报错|点不了|提交不了|系统(错误|繁忙|异常)|bug|一直转圈|加载不",
    "account_security": r"被盗|盗号|异地登|不是我(本人)?(下的|买的|操作)|改密码|修改密码|换绑|改绑|注销账",
}

_BRAND_RE = re.compile(r"京东|京豆|白条|jd|plus|小金库|东券", re.I)
_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]{1,8}\]")
_ORDER_PLACEHOLDER = "[订单编号]"


def detokenize(s: str) -> str:
    """去掉分词空格，但保留两个英文或数字单词之间的空格。"""
    s = re.sub(r"\s+", " ", s.strip())
    return re.sub(r"(?<![A-Za-z0-9]) | (?![A-Za-z0-9])", "", s)


def fake_order_id(rng: random.Random) -> str:
    return rng.choice("ABCDEFGH") + str(rng.randint(10000, 99999))


def fill_order_ids(s: str, rng: random.Random) -> str:
    while _ORDER_PLACEHOLDER in s:
        s = s.replace(_ORDER_PLACEHOLDER, fake_order_id(rng), 1)
    return s


def usable(text: str) -> bool:
    if not MIN_LEN <= len(text) <= MAX_LEN:
        return False
    if _BRAND_RE.search(text):
        return False
    # 订单号以外的脱敏占位符（[数字]、[姓名] 等）没法还原成自然的句子
    return not _PLACEHOLDER_RE.search(text.replace(_ORDER_PLACEHOLDER, ""))


def join_utterances(parts: list[str]) -> str:
    out = ""
    for p in parts:
        if out and out[-1] not in "，。？！?!,.~…":
            out += "，"
        out += p
    return out


def context_before(dialogue: list[dict], first_turn: int, rng: random.Random) -> list[dict]:
    """first_turn 之前的对话，连续同一方的发言合并成一轮，只留最后几轮。"""
    merged: list[dict] = []
    for u in dialogue:
        if u["turn"] >= first_turn:
            break
        role = "user" if u["speaker"] == "Q" else "assistant"
        content = fill_order_ids(detokenize(u["utterance"]), rng)
        if not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] = join_utterances([merged[-1]["content"], content])
        else:
            merged.append({"role": role, "content": content})
    return merged[-HISTORY_TURNS:]


def load_dialogues() -> list[dict]:
    dialogues = []
    for name in ("train.json", "val.json", "test.json"):
        dialogues += json.loads((DATASET_DIR / name).read_text(encoding="utf-8"))
    # 商家、配送师傅发起的对话不是我们的场景
    return [d for d in dialogues if d["QRole"] == "用户"]


def candidates_from_labels(dialogues: list[dict], rng: random.Random) -> dict[str, dict[str, list[dict]]]:
    """按源标签映射。返回 {建议意图: {源标签: [候选]}}，保留源标签是为了后面分层抽样。"""
    lookup = {src: (intent, None) for intent, srcs in SOURCE_MAP.items() for src in srcs}
    lookup.update(BOUNDARY_MAP)
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for d in dialogues:
        by_turn = {u["turn"]: u for u in d["Dialogue"]}
        for i, qa in enumerate(d["QA"]):
            if qa["intent"] not in lookup or not qa["QueSummUttIDs"]:
                continue
            intent, note = lookup[qa["intent"]]
            turns = [t for t in sorted(qa["QueSummUttIDs"]) if t in by_turn and by_turn[t]["speaker"] == "Q"]
            if not turns:
                continue
            text = join_utterances([detokenize(by_turn[t]["utterance"]) for t in turns])
            if not usable(text):
                continue
            ctx = context_before(d["Dialogue"], turns[0], rng)
            case = {"text": fill_order_ids(text, rng), "intent": intent}
            if len(text) <= ELLIPSIS_MAX_LEN and ctx:
                case["history"] = ctx
                note = "；".join(filter(None, [note, "省略句，需要上文"]))
            if note:
                case["note"] = note
            case.update({"author": "csds", "_src": qa["intent"], "_summary": qa["QueSumm"],
                         "_ctx": ctx, "_id": f"{d['DialogueID']}#{i}"})
            pools[intent][qa["intent"]].append(case)
    return pools


def candidates_from_keywords(dialogues: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    compiled = {k: re.compile(p, re.I) for k, p in KEYWORD_POOLS.items()}
    pools: dict[str, list[dict]] = defaultdict(list)
    for d in dialogues:
        for u in d["Dialogue"]:
            if u["speaker"] != "Q":
                continue
            text = detokenize(u["utterance"])
            for intent, pattern in compiled.items():
                min_len = 2 if intent == "greeting" else MIN_LEN
                if pattern.search(text) and len(text) >= min_len and (intent == "greeting" or usable(text)):
                    pools[intent].append({
                        "text": fill_order_ids(text, rng), "intent": intent, "author": "csds",
                        "_src": "关键词匹配", "_ctx": context_before(d["Dialogue"], u["turn"], rng),
                        "_id": f"{d['DialogueID']}@{u['turn']}",
                    })
    return pools


def stratified_sample(by_src: dict[str, list[dict]], k: int, rng: random.Random) -> list[dict]:
    """按源标签分层抽样，名额与该标签样本数的平方根成正比。

    完全按比例，"配送周期"这种大类会占满名额；完全平均，1800 条的大类和 5 条的小类各取一条，
    又看不到主流问法。平方根是两者的折中。
    """
    weights = {src: len(items) ** 0.5 for src, items in by_src.items()}
    total = sum(weights.values())
    picked, leftovers = [], []
    for src in sorted(by_src):
        items = by_src[src][:]
        rng.shuffle(items)
        quota = max(1, round(k * weights[src] / total))
        picked += items[:quota]
        leftovers += items[quota:]
    rng.shuffle(picked)
    rng.shuffle(leftovers)
    return (picked + leftovers)[:k] if len(picked) < k else picked[:k]


def dedupe(cases: list[dict], seen: set[str]) -> list[dict]:
    out = []
    for c in cases:
        if c["text"] not in seen:
            seen.add(c["text"])
            out.append(c)
    return out


def extract() -> None:
    """生成候选。只追加不覆盖：已有的行原样保留，包括人工改过的 intent，编号也不会变。"""
    rng = random.Random(SEED)
    dialogues = load_dialogues()
    labeled = candidates_from_labels(dialogues, rng)
    keyword = candidates_from_keywords(dialogues, rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {intent: read_jsonl(OUT_DIR / f"{intent}.jsonl") for intent in INTENTS}
    seen = {c["text"] for rows in existing.values() for c in rows} | {c["text"] for c in read_jsonl(CASES_PATH)}
    print(f"用户对话 {len(dialogues)} 段")
    for intent in INTENTS:
        room = max(0, PER_CLASS - len(existing[intent]))
        kw = keyword.get(intent, [])
        rng.shuffle(kw)
        # 两个来源都有的类，给关键词留三分之一名额，否则会被数量大的源标签挤掉。
        # 转人工、打招呼的说法高度重复，所以先去重再截断
        has_labels = bool(labeled.get(intent))
        from_kw = dedupe(kw, seen)[: room // 3 if has_labels else room]
        label_room = room - len(from_kw)
        from_labels = dedupe(stratified_sample(labeled.get(intent, {}), len(seen) + label_room, rng), seen)[:label_room]
        with (OUT_DIR / f"{intent}.jsonl").open("a", encoding="utf-8") as f:
            for r in from_labels + from_kw:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {intent:<17} 原有 {len(existing[intent]):>3}，新增 {len(from_labels) + len(from_kw):>3}"
              f"（标签映射 {len(from_labels)}，关键词 {len(from_kw)}）")
    if not PICKED_PATH.exists():
        PICKED_PATH.touch()
    print(f"候选在 {OUT_DIR}。用 show 查看、pick 选入，最后 merge")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_candidates(intent: str) -> list[dict]:
    rows = read_jsonl(OUT_DIR / f"{intent}.jsonl")
    if not rows:
        raise SystemExit(f"没有 {intent} 的候选，先运行 extract")
    return rows


def history_has_placeholder(case: dict) -> bool:
    """上文里的 [数字]、[电话] 这类脱敏占位符会原样进入评测集，需要人工改成自然的内容或删掉那一轮。"""
    return any(_PLACEHOLDER_RE.search(turn["content"]) for turn in case.get("history", []))


def show(intent: str) -> None:
    rows = load_candidates(intent)
    picked_ids = {c.get("_id") for c in read_jsonl(PICKED_PATH)}
    merged_texts = {c["text"] for c in read_jsonl(CASES_PATH)}
    for n, c in enumerate(rows, start=1):
        mark = "  [已选]" if c["_id"] in picked_ids else "  [已在评测集]" if c["text"] in merged_texts else ""
        print(f"[{n}]{mark} {c['text']}")
        detail = f"     源标签: {c['_src']}"
        if c.get("_summary"):
            detail += f" | 摘要: {c['_summary']}"
        print(detail)
        if c.get("note"):
            print(f"     提示: {c['note']}")
        if history_has_placeholder(c):
            print("     注意: history 里有脱敏占位符，选入后要在 picked.jsonl 里改掉")
        for turn in c.get("_ctx", []):
            who = "用户" if turn["role"] == "user" else "客服"
            flag = "*" if "history" in c else " "   # 带 * 的上文会作为 history 进入评测集
            print(f"    {flag}{who}: {turn['content'][:70]}")
        print()
    print(f"共 {len(rows)} 条，建议意图 {intent}。带 * 的上文会作为 history 一起选入")


def pick(intent: str, numbers: list[int], as_intent: str | None, note: str | None) -> None:
    rows = load_candidates(intent)
    bad = [n for n in numbers if not 1 <= n <= len(rows)]
    if bad:
        raise SystemExit(f"编号超出范围 1-{len(rows)}: {bad}")
    picked_ids = {c.get("_id") for c in read_jsonl(PICKED_PATH)}
    added = []
    for n in dict.fromkeys(numbers):   # 去掉重复编号，同时保持输入顺序
        case = dict(rows[n - 1])
        if case["_id"] in picked_ids:
            print(f"  [{n}] 已经选过，跳过: {case['text']}")
            continue
        if as_intent:
            case["intent"] = as_intent
        if note:
            case["note"] = note
        added.append(case)
        print(f"  [{n}] -> {case['intent']}: {case['text']}")
        if history_has_placeholder(case):
            print("       注意: history 里有脱敏占位符，请在 picked.jsonl 里改掉")
    content = PICKED_PATH.read_text(encoding="utf-8") if PICKED_PATH.exists() else ""
    with PICKED_PATH.open("a", encoding="utf-8") as f:
        if content and not content.endswith("\n"):
            f.write("\n")
        for case in added:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    by_intent = Counter(c["intent"] for c in read_jsonl(PICKED_PATH))
    print(f"选入 {len(added)} 条。picked.jsonl 现有: {dict(by_intent)}")


def merge() -> None:
    existing = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {(c["text"], json.dumps(c.get("history"), ensure_ascii=False)) for c in existing}
    added, skipped = [], 0
    for n, line in enumerate(PICKED_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = {k: v for k, v in json.loads(line).items() if not k.startswith("_")}
        if case.get("intent") not in INTENTS:
            raise SystemExit(f"picked.jsonl 第 {n} 行的 intent={case.get('intent')!r} 不合法")
        key = (case["text"], json.dumps(case.get("history"), ensure_ascii=False))
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        added.append(case)
    ends_with_newline = CASES_PATH.read_text(encoding="utf-8").endswith("\n")
    with CASES_PATH.open("a", encoding="utf-8") as f:
        if added and not ends_with_newline:
            f.write("\n")
        for case in added:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    total = Counter(c["intent"] for c in existing + added)
    print(f"新增 {len(added)} 条，跳过重复 {skipped} 条。当前各类数量：")
    for intent in INTENTS:
        print(f"  {intent:<17} {total[intent]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("extract", help="从数据集生成候选")
    commands.add_parser("merge", help="把 picked.jsonl 并入评测集")
    show_cmd = commands.add_parser("show", help="带编号打印某一类的候选")
    show_cmd.add_argument("intent", choices=INTENTS)
    pick_cmd = commands.add_parser("pick", help="按编号把候选选入 picked.jsonl")
    pick_cmd.add_argument("intent", choices=INTENTS, help="候选文件对应的意图")
    pick_cmd.add_argument("numbers", type=int, nargs="+", help="show 里显示的编号")
    pick_cmd.add_argument("--as", dest="as_intent", choices=INTENTS, help="建议意图不对时，改标成这个")
    pick_cmd.add_argument("--note", help="写入 note 字段，比如：边界案例，见指南第 3 条")
    args = parser.parse_args()

    if args.command == "extract":
        extract()
    elif args.command == "merge":
        merge()
    elif args.command == "show":
        show(args.intent)
    else:
        pick(args.intent, args.numbers, args.as_intent, args.note)

"""意图识别评测。

运行：uv run python scripts/eval_intent.py [--split dev|test|all]

调参数（LLM_TRUST、MIN_CONF、提示词等）只看 dev，定下来以后在 test 上跑一次，简历和 README 里报 test 的数字。
在同一批样本上反复调参再报这批样本的成绩，数字会虚高，这和训练集上报准确率是一个问题。

做三件事：
  1. 在 evals/intent_cases.jsonl 上跑完整的意图识别，计算准确率、宏平均 F1 和各类别指标
  2. 消融对比：只用规则、只用 LLM、两路融合，各自的指标。用来回答"两路融合到底有没有用"
  3. 列出最常见的误判和每一条错例，指导下一步改进

每条样本只调用一次模型，三种模式的结果都从同一次调用里推导，保证对比公平，也省钱。
报告写入 evals/reports/intent_latest.json，随代码一起提交，作为简历数字的依据。
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.evals.metrics import classification_report, confusion_pairs  # noqa: E402
from app.intent.llm_classifier import FEW_SHOTS, llm_vote  # noqa: E402
from app.intent.recognizer import fuse  # noqa: E402
from app.intent.rules import rule_vote  # noqa: E402
from app.intent.rules import detect_urgency, extract_entities  # noqa: E402
from app.intent.schema import INTENT_GROUP, Intent, IntentResult  # noqa: E402
from app.routing.router import decide  # noqa: E402
from app.llm import LLMClient  # noqa: E402

CASES_PATH = ROOT / "evals" / "intent_cases.jsonl"
REPORT_PATH = ROOT / "evals" / "reports" / "intent_latest.json"
CONCURRENCY = 5          # 同时进行的模型调用数。太高会被中转站限流
DEV_SHARE = 0.4          # 开发集占比。按文本哈希划分，新增样本不会让旧样本换边
MIN_PER_CLASS = 8        # 每类少于这个数，该类的指标波动太大，没有参考价值


def load_cases() -> list[dict]:
    cases, valid = [], {i.value for i in Intent}
    for n, line in enumerate(CASES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if case["intent"] not in valid:
            raise SystemExit(f"第 {n} 行的 intent={case['intent']!r} 不是合法的意图，可选: {sorted(valid)}")
        cases.append(case)
    return cases


def split_of(case: dict) -> str:
    """按文本哈希决定样本属于 dev 还是 test。不用随机数，保证每次运行、每台机器的划分都一样。"""
    bucket = int(hashlib.md5(case["text"].encode("utf-8")).hexdigest(), 16) % 100
    return "dev" if bucket < DEV_SHARE * 100 else "test"


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """准确率的 95% 置信区间。样本越少区间越宽，用来判断两次评测的差距是不是噪声。"""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half


def check_dataset(cases: list[dict]) -> None:
    few_shot_texts = {text for text, _ in FEW_SHOTS}
    leaked = [c["text"] for c in cases if c["text"] in few_shot_texts]
    if leaked:
        raise SystemExit(f"数据泄漏：以下样本同时出现在提示词的 few-shot 示例里，必须移除: {leaked}")

    texts = Counter((c["text"], json.dumps(c.get("history"), ensure_ascii=False)) for c in cases)
    dupes = [t for (t, _), n in texts.items() if n > 1]
    if dupes:
        raise SystemExit(f"存在重复样本: {dupes}")

    per_class = Counter(c["intent"] for c in cases)
    authors = Counter(c.get("author", "me") for c in cases)
    print(f"样本总数 {len(cases)}，来源 {dict(authors)}")
    print("各类数量: " + "  ".join(f"{k}={per_class.get(k, 0)}" for k in sorted(i.value for i in Intent)))
    thin = [k for k in (i.value for i in Intent) if per_class.get(k, 0) < MIN_PER_CLASS]
    if thin:
        print(f"⚠ 以下类别不足 {MIN_PER_CLASS} 条，指标仅供参考: {thin}")
    print()


def expected_primary(intent: Intent) -> str:
    """真实标签对应的主 Agent。other 应当反问，用 clarify 表示；转人工用 escalation。"""
    group = INTENT_GROUP[intent].value
    return {"none": "clarify", "escalation": "escalation"}.get(group, group)


def routed_primary(text: str, intent: Intent, conf: float) -> str:
    """把识别结果送进真实的路由函数，看主 Agent 选了谁。路由还看关键词和实体，所以不等于意图组映射。"""
    result = IntentResult(intent, INTENT_GROUP[intent], conf, "eval", detect_urgency(text, intent), extract_entities(text))
    d = decide(result, text)
    return "clarify" if d.action == "clarify" else d.primary


async def main(split: str, with_notes: bool = True) -> None:
    cases = load_cases()
    if not with_notes:
        print("消融模式：提示词不含意图说明\n")
    check_dataset(cases)
    if split != "all":
        cases = [c for c in cases if split_of(c) == split]
        print(f"只评测 {split} 部分，共 {len(cases)} 条\n")
    llm = LLMClient(settings)
    gate = asyncio.Semaphore(CONCURRENCY)

    async def run_one(case: dict) -> dict:
        async with gate:   # 信号量：最多同时放 CONCURRENCY 个进去，其余排队
            t0 = time.monotonic()
            llm_v = await llm_vote(llm, case["text"], case.get("history"), with_notes=with_notes)
            ms = (time.monotonic() - t0) * 1000
        rule_v = rule_vote(case["text"])
        fused, conf, source = fuse(llm_v, rule_v)
        return {
            **case,
            "pred_fused": fused.value, "confidence": round(conf, 3), "source": source,
            "pred_llm": fuse(llm_v, None)[0].value,
            "pred_rule": fuse(None, rule_v)[0].value,
            "llm_failed": llm_v is None, "latency_ms": round(ms),
            "pred_primary": routed_primary(case["text"], fused, conf),
            "gold_primary": expected_primary(Intent(case["intent"])),
        }

    t0 = time.monotonic()
    rows = await asyncio.gather(*[run_one(c) for c in cases])
    wall = time.monotonic() - t0

    y_true = [r["intent"] for r in rows]
    reports = {mode: classification_report(y_true, [r[f"pred_{mode}"] for r in rows]) for mode in ("rule", "llm", "fused")}

    print(f"{'模式':<10}{'准确率':>10}{'宏平均F1':>12}")
    for mode, name in (("rule", "仅规则"), ("llm", "仅 LLM"), ("fused", "两路融合")):
        print(f"{name:<10}{reports[mode]['accuracy']:>12.1%}{reports[mode]['macro_f1']:>12.4f}")

    lo, hi = wilson_interval(sum(t == p for t, p in zip(y_true, [r["pred_fused"] for r in rows])), len(rows))
    print(f"两路融合准确率的 95% 置信区间: {lo:.1%} ~ {hi:.1%}。两次评测的差距落在这个范围内就不能算改进")

    print(f"\n两路融合的各类别指标:\n{'类别':<20}{'精确率':>8}{'召回率':>8}{'F1':>8}{'样本数':>8}")
    for label, m in reports["fused"]["per_class"].items():
        print(f"{label:<20}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1']:>8.2f}{m['support']:>8}")

    y_fused = [r["pred_fused"] for r in rows]
    pairs = confusion_pairs(y_true, y_fused)
    if pairs:
        print("\n最常见的误判，真实 -> 预测:")
        for t, p, n in pairs:
            print(f"  {t} -> {p}  × {n}")
        print("\n全部错例:")
        for r in rows:
            if r["pred_fused"] != r["intent"]:
                print(f"  「{r['text']}」 真实={r['intent']} 预测={r['pred_fused']} "
                      f"(llm={r['pred_llm']}, rule={r['pred_rule']}, 置信度={r['confidence']})")

    sources = Counter(r["source"] for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    failed = sum(r["llm_failed"] for r in rows)
    routing_ok = sum(r["pred_primary"] == r["gold_primary"] for r in rows)
    print(f"\n路由准确率（主 Agent 选对）: {routing_ok}/{len(rows)} = {routing_ok / len(rows):.1%}")
    print(f"\n结论来源: {dict(sources)}   LLM 调用失败: {failed}")
    print(f"单次 LLM 延迟: 中位数 {latencies[len(latencies) // 2]}ms，最大 {latencies[-1]}ms；总耗时 {wall:.0f}s")

    variant = "" if with_notes else "_no_notes"
    stem = "intent_latest" if split == "all" else f"intent_{split}"
    report_path = REPORT_PATH.with_name(f"{stem}{variant}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "split": split, "accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": settings.llm_model, "n_cases": len(rows),
        "per_class_counts": dict(Counter(y_true)),
        "reports": reports, "confusions": pairs, "sources": dict(sources), "llm_failures": failed,
        "with_notes": with_notes, "routing_accuracy": round(routing_ok / len(rows), 4),
        "errors": [r for r in rows if r["pred_fused"] != r["intent"]],
        "rows": rows,   # 全部样本的预测结果，用于事后分析，例如规则和 LLM 不一致时谁更可靠
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--no-notes", action="store_true", help="消融：提示词里去掉意图说明")
    args = parser.parse_args()
    asyncio.run(main(args.split, with_notes=not args.no_notes))

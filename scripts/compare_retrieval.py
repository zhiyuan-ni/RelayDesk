"""对比几种检索模式的命中率和延迟：纯向量、LLM 重排、jev 重排、改写加重排。

运行：uv run python scripts/compare_retrieval.py [--runs 3]

每条用例是 (用户问法, 期望命中的小节)。问法刻意避开文档里的原词，模拟真实用户的口语表达。
指标：
  hit@1  正确小节排在第 1 位的比例
  hit@3  正确小节出现在前 3 位的比例
向量库建在临时目录，不影响正在运行的服务。
结果同时写入 evals/reports/retrieval_compare.json，README 里的命中率数字以该文件为准。
"""
import argparse
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402
from app.jev import JevClient  # noqa: E402
from app.knowledge.embedder import Embedder  # noqa: E402
from app.knowledge.kb import KnowledgeBase  # noqa: E402
from app.knowledge.retriever import Retriever  # noqa: E402
from app.llm import LLMClient  # noqa: E402

CASES = [
    ("换手机了验证器登不上", "两步验证"),
    ("快递显示签收了但我没拿到", "物流异常处理"),
    ("开会员后悔了能退吗", "年度会员的退订"),
    ("银行短信说扣了两笔，你们这边查只有一笔", "重复扣款处理"),
    ("付款的时候页面卡死了，我要不要再付一次", "支付页面异常"),
    ("买了三样东西只想退一样，券怎么算", "部分退款与优惠分摊"),
    ("钱退到信用卡要多久", "退款流程与时效"),
    ("公司财务说要专票，需要给你们什么资料", "发票类型"),
    ("发票上的公司名字写错了", "抬头和信息修改"),
    ("半夜收到短信说有人在国外登我的号", "异常登录提醒"),
    ("有人打电话说是你们客服，让我报验证码", "防范诈骗"),
    ("包裹寄出去了才发现地址填错了", "修改地址与拦截"),
    ("衣服买小了想换大一号", "换货"),
    ("耳机用了半年一边没声音了，能修吗", "保修与维修"),
    ("盒子里少了一根充电线", "少件、错发与补发"),
]


async def evaluate(name: str, search, uses_rerank: bool) -> dict:
    hit1 = hit3 = 0
    total_ms = 0.0
    misses = []
    fallbacks = 0
    for query, expected in CASES:
        t0 = time.monotonic()
        res = await search(query)
        total_ms += (time.monotonic() - t0) * 1000
        hits = res.hits if hasattr(res, "hits") else res
        fell_back = bool(uses_rerank and not getattr(res, "reranked", True))   # 重排超时或失败，退回了向量顺序
        fallbacks += fell_back
        sections = [h["section"] for h in hits[:3]]
        hit1 += sections[:1] == [expected]
        hit3 += expected in sections
        if sections[:1] != [expected]:
            misses.append({"query": query, "expected": expected, "top3": sections, "rerank_fallback": fell_back})
    n = len(CASES)
    extra = f"   重排回退 {fallbacks} 次" if uses_rerank else ""
    print(f"{name:<14} hit@1 {hit1}/{n} = {hit1 / n:.0%}   hit@3 {hit3}/{n} = {hit3 / n:.0%}   平均 {total_ms / n:.0f}ms{extra}")
    for m in misses:
        flag = "（重排回退）" if m["rerank_fallback"] else ""
        print(f"      「{m['query']}」期望 {m['expected']}，实际前三 {m['top3']}{flag}")
    return {"mode": name, "hit1": hit1, "hit3": hit3, "n": n,
            "hit1_rate": round(hit1 / n, 4), "hit3_rate": round(hit3 / n, 4),
            "avg_ms": round(total_ms / n),
            "rerank_fallbacks": fallbacks if uses_rerank else None, "misses": misses}


def summarize(runs: list[list[dict]]) -> list[dict]:
    """按模式汇总多次运行：hit@1 的最小 / 最大 / 平均。中转站延迟波动大，结果应写成范围而不是单个值。"""
    by_mode: dict[str, list[dict]] = {}
    for results in runs:
        for r in results:
            by_mode.setdefault(r["mode"], []).append(r)
    summary = []
    for mode, rs in by_mode.items():
        h1 = [r["hit1_rate"] for r in rs]
        h3 = [r["hit3_rate"] for r in rs]
        summary.append({"mode": mode, "runs": len(rs),
                        "hit1_min": min(h1), "hit1_max": max(h1), "hit1_mean": round(sum(h1) / len(h1), 4),
                        "hit3_min": min(h3), "hit3_max": max(h3),
                        "avg_ms_mean": round(sum(r["avg_ms"] for r in rs) / len(rs)),
                        "rerank_fallbacks_total": sum(r["rerank_fallbacks"] or 0 for r in rs)})
    return summary


async def main(n_runs: int) -> None:
    kb = KnowledgeBase(Embedder(settings), tempfile.mkdtemp())
    await kb.ensure_ready()
    llm = LLMClient(settings).for_model(settings.rerank_model)
    jev = JevClient(settings)
    await jev.warmup()   # 预热，否则第一条的延迟里含约 5 秒建连时间
    print(f"LLM 重排模型: {llm.model}，jev 模型: {jev.model}")
    print(f"片段数 {await kb.count()}，向量模型 {settings.embedding_model}，LLM {settings.llm_model}\n")

    async def vector_only(q):
        return await kb.search(q, 4)

    async def rerank_only(q):
        return await Retriever(kb, llm, rewrite=False, rerank=True).retrieve(q, 4)

    async def jev_rerank(q):
        return await Retriever(kb, llm, rewrite=False, rerank=True, jev=jev).retrieve(q, 4)

    async def full(q):
        return await Retriever(kb, llm, rewrite=True, rerank=True).retrieve(q, 4)

    modes = [("纯向量", vector_only, False), ("向量+重排", rerank_only, True),
             ("向量+jev重排", jev_rerank, True), ("改写+向量+重排", full, True)]
    runs: list[list[dict]] = []
    for i in range(n_runs):
        if n_runs > 1:
            print(f"── 第 {i + 1}/{n_runs} 次运行 ──")
        runs.append([await evaluate(name, fn, uses_rerank) for name, fn, uses_rerank in modes])
        print()
    await jev.aclose()

    summary = summarize(runs)
    if n_runs > 1:
        print("汇总（hit@1 范围）")
        for item in summary:
            print(f"{item['mode']:<14} {item['hit1_min']:.0%} – {item['hit1_max']:.0%}"
                  f"（均值 {item['hit1_mean']:.0%}），重排回退共 {item['rerank_fallbacks_total']} 次")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "rerank_model": llm.model,
        "rewrite_model": settings.llm_model,
        "jev_model": jev.model,
        "chunks": await kb.count(),
        "n_cases": len(CASES),
        "n_runs": n_runs,
        "cases": [{"query": q, "expected": e} for q, e in CASES],
        "summary": summary,
        "runs": [{"run": i + 1, "results": r} for i, r in enumerate(runs)],
    }
    out = Path(__file__).parent.parent / "evals" / "reports" / "retrieval_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比几种检索模式的命中率和延迟")
    parser.add_argument("--runs", type=int, default=1, help="重复运行次数；多次运行时汇总 hit@1 的范围（默认 1）")
    asyncio.run(main(parser.parse_args().runs))

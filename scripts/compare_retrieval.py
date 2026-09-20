"""对比三种检索模式的命中率和延迟。

运行：uv run python scripts/compare_retrieval.py

每条用例是 (用户问法, 期望命中的小节)。问法刻意避开文档里的原词，模拟真实用户的口语表达。
指标：
  hit@1  正确小节排在第 1 位的比例
  hit@3  正确小节出现在前 3 位的比例
向量库建在临时目录，不影响正在运行的服务。
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402
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
]


async def evaluate(name: str, search) -> None:
    hit1 = hit3 = 0
    total_ms = 0.0
    misses = []
    for query, expected in CASES:
        t0 = time.monotonic()
        hits = await search(query)
        total_ms += (time.monotonic() - t0) * 1000
        sections = [h["section"] for h in hits[:3]]
        hit1 += sections[:1] == [expected]
        hit3 += expected in sections
        if sections[:1] != [expected]:
            misses.append(f"      「{query}」期望 {expected}，实际前三 {sections}")
    n = len(CASES)
    print(f"{name:<14} hit@1 {hit1}/{n} = {hit1 / n:.0%}   hit@3 {hit3}/{n} = {hit3 / n:.0%}   平均 {total_ms / n:.0f}ms")
    print("\n".join(misses))


async def main() -> None:
    kb = KnowledgeBase(Embedder(settings), tempfile.mkdtemp())
    await kb.ensure_ready()
    llm = LLMClient(settings)
    print(f"片段数 {await kb.count()}，向量模型 {settings.embedding_model}，LLM {settings.llm_model}\n")

    async def vector_only(q):
        return await kb.search(q, 4)

    async def rerank_only(q):
        return (await Retriever(kb, llm, rewrite=False, rerank=True).retrieve(q, 4)).hits

    async def full(q):
        return (await Retriever(kb, llm, rewrite=True, rerank=True).retrieve(q, 4)).hits

    await evaluate("纯向量", vector_only)
    await evaluate("向量+重排", rerank_only)
    await evaluate("改写+向量+重排", full)


if __name__ == "__main__":
    asyncio.run(main())

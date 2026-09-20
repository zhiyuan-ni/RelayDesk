"""检索器测试：merge_hits 的规格，加上改写、重排和各级降级。"""
import json

from app.knowledge.retriever import Retriever, apply_order, merge_hits


def hit(name: str, score: float) -> dict:
    return {"title": "文档", "section": name, "text": f"{name}的内容", "score": score}


# ── merge_hits 规格 ──

def test_1_empty():
    assert merge_hits([]) == [] and merge_hits([[], []]) == []


def test_2_and_4_union_sorted_by_score():
    merged = merge_hits([[hit("甲", 0.5)], [hit("乙", 0.9), hit("丙", 0.7)]])
    assert [h["section"] for h in merged] == ["乙", "丙", "甲"]


def test_3_duplicate_keeps_highest_score():
    merged = merge_hits([[hit("甲", 0.5), hit("乙", 0.6)], [hit("甲", 0.8)], [hit("甲", 0.3)]])
    assert [(h["section"], h["score"]) for h in merged] == [("甲", 0.8), ("乙", 0.6)]


def test_2_same_section_different_text_are_different_chunks():
    a, b = hit("甲", 0.5), hit("甲", 0.6)
    b["text"] = "同一小节的第二个片段"
    assert len(merge_hits([[a], [b]])) == 2


# ── apply_order：不信任模型输出 ──

def test_apply_order_ignores_garbage_and_backfills():
    cands = [hit("甲", 0.9), hit("乙", 0.8), hit("丙", 0.7)]
    ordered = apply_order(cands, [2, 2, 99, "x", True, 0], top_k=3)
    assert [h["section"] for h in ordered] == ["丙", "甲", "乙"]   # 乙被模型漏掉，补在最后


# ── Retriever ──

class FakeKB:
    """按查询里的关键词返回预设结果，并记录被查了哪些查询。"""
    def __init__(self):
        self.queries = []

    async def search(self, query, top_k):
        self.queries.append(query)
        if "坏" in query:
            raise RuntimeError("向量服务超时")
        if "两步验证" in query:
            return [hit("两步验证", 0.80), hit("短信验证码", 0.60)]
        return [hit("短信验证码", 0.70), hit("两步验证", 0.65), hit("修改手机", 0.55)]


class FakeLLM:
    def __init__(self, variants=None, order=None, fail=()):
        self.variants, self.order, self.fail = variants or [], order, set(fail)

    async def chat_text(self, prompt, **kwargs):
        step = "rewrite" if "改写" in prompt else "rerank"
        if step in self.fail:
            raise TimeoutError(step)
        return "结果：" + json.dumps(self.variants if step == "rewrite" else self.order, ensure_ascii=False)


Q = "换手机了验证器登不上"


async def test_vector_only_mode():
    kb = FakeKB()
    r = await Retriever(kb, None, rewrite=False, rerank=False).retrieve(Q, top_k=2)
    assert kb.queries == [Q] and r.queries == [Q] and not r.reranked
    assert [h["section"] for h in r.hits] == ["短信验证码", "两步验证"]      # 向量顺序，正确答案排第二


async def test_rewrite_adds_recall_and_rerank_fixes_order():
    kb = FakeKB()
    llm = FakeLLM(variants=["更换手机后两步验证无法登录", Q], order=[0, 1, 2])
    r = await Retriever(kb, llm).retrieve(Q, top_k=2)
    assert sorted(kb.queries) == sorted([Q, "更换手机后两步验证无法登录"])    # 与原问题重复的变体被去掉
    assert r.n_candidates == 3 and r.reranked
    assert r.hits[0]["section"] == "两步验证" and r.hits[0]["score"] == 0.80  # 改写后的查询给了它更高的分


async def test_rerank_can_override_vector_order():
    llm = FakeLLM(order=[1, 0])
    r = await Retriever(FakeKB(), llm, rewrite=False).retrieve(Q, top_k=2)
    assert [h["section"] for h in r.hits] == ["两步验证", "短信验证码"] and r.reranked


async def test_rewrite_failure_degrades_to_original_query():
    kb = FakeKB()
    r = await Retriever(kb, FakeLLM(order=[0], fail={"rewrite"})).retrieve(Q)
    assert kb.queries == [Q] and r.queries == [Q] and r.hits


async def test_rerank_failure_degrades_to_vector_order():
    r = await Retriever(FakeKB(), FakeLLM(fail={"rerank"}), rewrite=False).retrieve(Q, top_k=2)
    assert not r.reranked and [h["section"] for h in r.hits] == ["短信验证码", "两步验证"]


async def test_one_failed_recall_does_not_break_the_rest():
    llm = FakeLLM(variants=["坏查询", "两步验证怎么恢复"], order=[0])
    r = await Retriever(FakeKB(), llm).retrieve(Q)
    assert r.hits and r.hits[0]["section"] == "两步验证"


async def test_slow_rerank_times_out_but_recall_results_survive():
    import asyncio

    class SlowLLM:
        async def chat_text(self, prompt, **kwargs):
            await asyncio.sleep(5)
            return "[1, 0]"

    r = await Retriever(FakeKB(), SlowLLM(), rewrite=False, rerank_timeout_s=0.05).retrieve(Q, top_k=2)
    assert not r.reranked and [h["section"] for h in r.hits] == ["短信验证码", "两步验证"] and r.latency_ms < 1000

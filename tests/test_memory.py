"""会话记忆测试。用 fakeredis 在内存里模拟 Redis，不需要启动真实的 Redis。"""
import fakeredis
import pytest

from app.memory.manager import COMPRESS_AT, KEEP_RECENT, MemoryManager, select_within_budget
from app.memory.store import TTL_SECONDS, ConversationStore


def msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


# ── select_within_budget 规格 ──

def test_1_empty():
    assert select_within_budget([], 100) == []


def test_2_takes_newest_until_budget_is_hit():
    messages = [msg("aaaa"), msg("bbbb"), msg("cccc"), msg("dddd")]       # 每条 4 个字符
    assert select_within_budget(messages, 8) == [msg("cccc"), msg("dddd")]
    assert select_within_budget(messages, 11) == [msg("cccc"), msg("dddd")]   # 第三条放进去会到 12，超了
    assert select_within_budget(messages, 100) == messages


def test_2_stops_at_first_overflow_and_does_not_skip():
    # 倒数第二条太长放不下。不能跳过它去拿更早的短消息，否则对话中间缺一段，上下文就断了
    messages = [msg("短"), msg("x" * 50), msg("新")]
    assert select_within_budget(messages, 10) == [msg("新")]


def test_3_newest_message_is_always_kept():
    messages = [msg("旧"), msg("x" * 500)]
    assert select_within_budget(messages, 10) == [msg("x" * 500)]


def test_4_chronological_order_is_preserved():
    messages = [msg("一"), msg("二"), msg("三")]
    assert [m["content"] for m in select_within_budget(messages, 100)] == ["一", "二", "三"]


# ── ConversationStore ──

@pytest.fixture
def store():
    return ConversationStore(fakeredis.FakeAsyncRedis(decode_responses=True))


async def test_append_and_read_back_in_order(store):
    await store.append("u1", "c1", "user", "你好")
    assert await store.append("u1", "c1", "assistant", "您好") == 2
    assert [(m["role"], m["content"]) for m in await store.messages("u1", "c1")] == [("user", "你好"), ("assistant", "您好")]


async def test_conversations_are_isolated_by_user(store):
    await store.append("u1", "same-conv", "user", "甲的秘密")
    assert await store.messages("u2", "same-conv") == []      # 乙用同一个会话号，读不到甲的内容


async def test_ttl_is_set(store):
    await store.append("u1", "c1", "user", "hi")
    ttl = await store._r.ttl("conv:u1:c1:messages")
    assert 0 < ttl <= TTL_SECONDS


async def test_drop_oldest_keeps_messages_appended_meanwhile(store):
    for i in range(6):
        await store.append("u1", "c1", "user", f"m{i}")
    await store.append("u1", "c1", "user", "压缩期间新来的")
    await store.drop_oldest("u1", "c1", 4)
    assert [m["content"] for m in await store.messages("u1", "c1")] == ["m4", "m5", "压缩期间新来的"]


# ── MemoryManager ──

class FakeLLM:
    def __init__(self, reply="用户想退订单 A12345 的货，已确认在 7 天内。", error=None):
        self.reply, self.error, self.prompts = reply, error, []

    async def chat_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


async def fill(manager, turns: int):
    for i in range(turns):
        await manager.save_turn("u1", "c1", f"问题{i}", f"回答{i}")
    await manager.wait_background()


async def test_no_compression_below_threshold(store):
    llm = FakeLLM(); m = MemoryManager(store, llm)
    await fill(m, COMPRESS_AT // 2 - 1)
    assert llm.prompts == [] and len(await store.messages("u1", "c1")) == COMPRESS_AT - 2


async def test_compression_summarizes_old_and_keeps_recent(store):
    llm = FakeLLM(); m = MemoryManager(store, llm)
    await fill(m, COMPRESS_AT // 2)

    remaining = await store.messages("u1", "c1")
    assert len(remaining) == KEEP_RECENT and remaining[-1]["content"] == f"回答{COMPRESS_AT // 2 - 1}"
    assert await store.get_summary("u1", "c1") == llm.reply
    assert "问题0" in llm.prompts[0] and f"问题{COMPRESS_AT // 2 - 1}" not in llm.prompts[0]   # 最近的不进摘要

    ctx = await m.load("u1", "c1")
    assert ctx.summary == llm.reply and len(ctx.history()) == KEEP_RECENT and ctx.total == KEEP_RECENT
    assert set(ctx.history()[0]) == {"role", "content"}


async def test_second_compression_merges_previous_summary(store):
    llm = FakeLLM(); m = MemoryManager(store, llm)
    await fill(m, COMPRESS_AT // 2)
    await fill(m, COMPRESS_AT // 2)
    assert len(llm.prompts) >= 2 and llm.reply in llm.prompts[-1]    # 旧摘要被带进了新一轮压缩


async def test_failed_compression_loses_nothing(store):
    m = MemoryManager(store, FakeLLM(error=TimeoutError("模型超时")))
    await fill(m, COMPRESS_AT // 2)
    assert len(await store.messages("u1", "c1")) == COMPRESS_AT and await store.get_summary("u1", "c1") == ""


# ── 长期记忆召回的结果记录 ──

class StubLongTerm:
    def __init__(self, hits=None, delay=0.0, error=None):
        self.hits, self.delay, self.error = hits or [], delay, error

    async def recall(self, user_id, query, exclude_conv_id=""):
        import asyncio
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.hits


@pytest.mark.parametrize("longterm, status, expected", [
    (StubLongTerm(hits=[{"text": "上次问过重复扣款"}]), "hit", ["上次问过重复扣款"]),
    (StubLongTerm(), "miss", []),
    (StubLongTerm(delay=1.0), "timeout", []),
    (StubLongTerm(error=ConnectionError("连不上")), "error", []),
])
async def test_recall_outcome_is_recorded_on_the_timeline(store, monkeypatch, longterm, status, expected):
    # 后三种都返回空列表，只有时间线上的 status 能区分"真没有"和"没查成"
    from app.memory import manager
    from app.observability import tracer
    monkeypatch.setattr(manager, "RECALL_TIMEOUT_S", 0.05)
    tl = tracer.start_timeline("r")
    assert await MemoryManager(store, None, longterm).recall("u1", "c1", "扣款") == expected
    [span] = tl.summary()
    assert span["name"] == "memory_recall" and span["status"] == status

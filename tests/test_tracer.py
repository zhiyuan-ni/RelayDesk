"""请求时间线：记录每一步的耗时，以及执行过程中补充的结果。"""
import pytest

from app.observability import tracer


async def test_span_records_meta_added_while_running():
    tl = tracer.start_timeline("r1")
    async with tracer.span("memory_recall", k=2) as meta:
        meta["status"] = "timeout"
    [s] = tl.summary()
    assert s["name"] == "memory_recall" and s["k"] == 2 and s["status"] == "timeout"


async def test_span_is_recorded_even_when_the_step_raises():
    tl = tracer.start_timeline("r2")
    with pytest.raises(ValueError):
        async with tracer.span("boom"):
            raise ValueError
    assert [s["name"] for s in tl.summary()] == ["boom"]


async def test_span_without_timeline_still_yields_a_dict():
    tracer._current.set(None)
    async with tracer.span("anything") as meta:
        meta["status"] = "ok"   # 不应该报错


async def test_annotate_writes_to_the_innermost_span():
    tl = tracer.start_timeline("r3")
    async with tracer.span("outer"):
        async with tracer.span("inner"):
            tracer.annotate(completion_tokens=95)
        tracer.annotate(after_inner=True)   # 内层结束后，写回外层
    spans = {s["name"]: s for s in tl.summary()}
    assert spans["inner"]["completion_tokens"] == 95 and "after_inner" not in spans["inner"]
    assert spans["outer"]["after_inner"] is True


async def test_annotate_outside_any_span_does_nothing():
    tracer.start_timeline("r4")
    tracer.annotate(x=1)   # 不应该报错


async def test_llm_client_records_token_usage():
    from types import SimpleNamespace as NS
    from app.config import Settings
    from app.llm import LLMClient

    class FakeCompletions:
        async def create(self, **kwargs):
            return NS(usage=NS(prompt_tokens=800, completion_tokens=95),
                      choices=[NS(message=NS(content="好的", tool_calls=None))])

    llm = LLMClient(Settings("k", "https://example.com/v1", "qwen3.8-flash"))
    llm._client = NS(chat=NS(completions=FakeCompletions()))
    tl = tracer.start_timeline("r5")
    async with tracer.span("compose"):
        await llm.chat_text("合并")
    [s] = tl.summary()
    assert (s["model"], s["prompt_tokens"], s["completion_tokens"]) == ("qwen3.8-flash", 800, 95)


async def test_raw_output_of_every_call_is_kept():
    # 同一步里调两次模型，两次的原始输出都要留下，不能后一次覆盖前一次
    from types import SimpleNamespace as NS
    from app.config import Settings
    from app.llm import LLMClient

    replies = iter([
        NS(content="回复正文\n[系统提示] 模型自己编的", tool_calls=None),
        NS(content="", tool_calls=[NS(id="c1", function=NS(name="get_order_status", arguments='{"order_id": "A1'))]),
    ])

    class FakeCompletions:
        async def create(self, **kwargs):
            return NS(usage=None, choices=[NS(finish_reason="stop", message=next(replies))])

    llm = LLMClient(Settings("k", "https://example.com/v1", "qwen3.8-flash"))
    llm._client = NS(chat=NS(completions=FakeCompletions()))
    tl = tracer.start_timeline("r6")
    async with tracer.span("llm:general"):
        await llm.chat([{"role": "user", "content": "你好"}])
        await llm.chat([{"role": "user", "content": "查订单"}])
    first, second = tl.summary()[0]["raw_output"]
    assert "[系统提示]" in first["content"] and first["finish_reason"] == "stop"   # 清理之前的原文
    assert second["tool_calls"][0]["arguments"] == '{"order_id": "A1'               # 写坏的参数原样保留

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

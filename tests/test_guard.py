"""guarded() 测试：缓存、熔断、超时三层保护如何配合。"""
import asyncio

from app.agents.tools import ToolContext, ToolSpec
from app.reliability.breaker import CircuitBreaker, State
from app.reliability.cache import TTLCache
from app.reliability.guard import guarded

U1, U2 = ToolContext("u1001"), ToolContext("u1002")
SCHEMA = {"type": "object", "properties": {}, "required": []}


def fallback(args):
    return {"found": False, "degraded": True}


class Upstream:
    """可控的假上游：可以设置耗时和是否报错，并记录被真正调用了几次。"""
    def __init__(self, delay=0.0, error=None):
        self.delay, self.error, self.calls = delay, error, 0

    async def __call__(self, ctx, args):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return {"found": True, "user": ctx.user_id, "q": args.get("query")}

    def spec(self):
        return ToolSpec("search", "检索", SCHEMA, self)


async def test_success_passes_through_and_keeps_tool_identity():
    up = Upstream()
    tool, stats = guarded(up.spec(), timeout_s=1, fallback=fallback)
    assert (tool.name, tool.description, tool.parameters) == ("search", "检索", SCHEMA)
    assert (await tool.handler(U1, {"query": "a"}))["found"] and stats.successes == 1


async def test_cache_hit_skips_upstream():
    up = Upstream()
    tool, stats = guarded(up.spec(), timeout_s=1, fallback=fallback, cache=TTLCache(60))
    first = await tool.handler(U1, {"query": "a"})
    assert await tool.handler(U2, {"query": "a"}) == first          # 不区分用户，乙命中甲的缓存
    await tool.handler(U1, {"query": "b"})
    assert up.calls == 2 and stats.cache_hits == 1


async def test_per_user_cache_never_leaks_across_users():
    up = Upstream()
    tool, _ = guarded(up.spec(), timeout_s=1, fallback=fallback, cache=TTLCache(60), per_user=True)
    assert (await tool.handler(U1, {"query": "a"}))["user"] == "u1001"
    assert (await tool.handler(U2, {"query": "a"}))["user"] == "u1002" and up.calls == 2


async def test_timeout_returns_fallback_quickly():
    up = Upstream(delay=5.0)
    tool, stats = guarded(up.spec(), timeout_s=0.05, fallback=fallback)
    t0 = asyncio.get_event_loop().time()
    assert await tool.handler(U1, {}) == {"found": False, "degraded": True}
    assert asyncio.get_event_loop().time() - t0 < 0.5 and stats.timeouts == 1


async def test_fallback_is_never_cached():
    up = Upstream(error=RuntimeError("上游挂了"))
    cache = TTLCache(60)
    tool, stats = guarded(up.spec(), timeout_s=1, fallback=fallback, cache=cache)
    await tool.handler(U1, {"query": "a"})
    up.error = None                                                   # 上游恢复
    assert (await tool.handler(U1, {"query": "a"}))["found"] is True  # 立刻拿到真实结果，而不是缓存的降级
    assert stats.errors == 1 and stats.successes == 1


async def test_breaker_opens_then_upstream_is_no_longer_called():
    up = Upstream(error=RuntimeError("上游挂了"))
    breaker = CircuitBreaker(failure_threshold=2, recovery_s=60)
    tool, stats = guarded(up.spec(), timeout_s=1, fallback=fallback, breaker=breaker)
    for _ in range(5):
        assert (await tool.handler(U1, {}))["degraded"]
    assert breaker.state == State.OPEN and up.calls == 2 and stats.rejected == 3


async def test_cache_still_serves_while_breaker_is_open():
    up = Upstream()
    breaker = CircuitBreaker(failure_threshold=1, recovery_s=60)
    tool, _ = guarded(up.spec(), timeout_s=1, fallback=fallback, cache=TTLCache(60), breaker=breaker)
    good = await tool.handler(U1, {"query": "a"})
    breaker.record_failure()
    assert breaker.state == State.OPEN
    assert await tool.handler(U1, {"query": "a"}) == good             # 熔断期间，缓存里有的照常返回
    assert (await tool.handler(U1, {"query": "新问题"}))["degraded"]

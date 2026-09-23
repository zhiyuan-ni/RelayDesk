"""请求级的耗时追踪。

一个请求进来时开一条时间线，链路上每一步用 span 包住：
    async with trace.span("intent"):
        ...
结束时时间线上就有每一步的名字、开始时刻和耗时。并行的步骤会有重叠的时间段，
所以时间线记的是每一步各自的耗时，而不是简单相加。

时间线用 contextvars 传递，不用一层层往函数里传参数。
contextvars 和全局变量的区别：每个请求（每个 asyncio 任务）看到的是自己那份，互不干扰。
asyncio.gather 创建的子任务会继承父任务的值，所以并行的 Agent 也写在同一条时间线上。
"""
import contextvars
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Span:
    name: str
    start_ms: float          # 相对于时间线起点
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Timeline:
    request_id: str
    _t0: float = field(default_factory=time.monotonic)
    spans: list[Span] = field(default_factory=list)

    @asynccontextmanager
    async def span(self, name: str, **meta: Any):
        start = time.monotonic()
        try:
            yield
        finally:   # 出异常也要记录，否则失败的那一步恰好从时间线上消失
            self.spans.append(Span(name, round((start - self._t0) * 1000, 1),
                                   round((time.monotonic() - start) * 1000, 1), meta))

    def total_ms(self) -> float:
        return round((time.monotonic() - self._t0) * 1000, 1)

    def summary(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "start_ms": s.start_ms, "duration_ms": s.duration_ms, **s.meta} for s in self.spans]


_current: contextvars.ContextVar[Optional[Timeline]] = contextvars.ContextVar("timeline", default=None)


def start_timeline(request_id: str) -> Timeline:
    tl = Timeline(request_id)
    _current.set(tl)
    return tl


def current() -> Optional[Timeline]:
    return _current.get()


@asynccontextmanager
async def span(name: str, **meta: Any):
    """在任何地方给一段代码计时。当前没有时间线时什么都不做，所以库代码可以放心加。"""
    tl = _current.get()
    if tl is None:
        yield
        return
    async with tl.span(name, **meta):
        yield

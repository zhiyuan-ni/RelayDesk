"""给工具加三层保护：缓存、熔断、超时，全部失败时返回降级结果。

  请求 ─► 缓存命中？──是──► 直接返回，零延迟
            │否
            ▼
          熔断器放行？──否──► 降级结果，立即返回
            │是
            ▼
          限时执行 ──超时或异常──► 记一次失败 ─► 降级结果
            │成功
            ▼
          记一次成功，写缓存，返回

顺序有讲究：缓存放在最前，缓存命中时既不受熔断影响，也不消耗上游；
降级结果绝不写入缓存，否则上游恢复之后用户还会继续拿到"服务不可用"。
"""
import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.agents.tools import ToolContext, ToolSpec
from app.reliability.breaker import CircuitBreaker
from app.reliability.cache import TTLCache

logger = logging.getLogger(__name__)


@dataclass
class GuardStats:
    """运行统计。阶段 6 的监控接口会把它暴露出去。"""
    calls: int = 0
    cache_hits: int = 0
    successes: int = 0
    timeouts: int = 0
    errors: int = 0
    rejected: int = 0   # 被熔断器挡掉的次数


def guarded(
    tool: ToolSpec,
    *,
    timeout_s: float,
    fallback: Callable[[dict[str, Any]], Any],
    cache: Optional[TTLCache] = None,
    breaker: Optional[CircuitBreaker] = None,
    per_user: bool = False,
) -> tuple[ToolSpec, GuardStats]:
    """返回一个带保护的新工具，名称、描述、参数和原工具完全一样，Agent 无感知。

    per_user: 结果是否因用户而异。知识库对所有人一样，设 False，不同用户可以共享缓存。
              如果给订单查询这类工具加缓存，必须设 True，否则甲会看到乙的订单。
    """
    stats = GuardStats()

    def cache_key(ctx: ToolContext, args: dict[str, Any]) -> str:
        key = tool.name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)
        return f"{ctx.user_id}|{key}" if per_user else key

    async def handler(ctx: ToolContext, args: dict[str, Any]) -> Any:
        stats.calls += 1
        key = cache_key(ctx, args)

        if cache is not None and (cached := cache.get(key)) is not None:
            stats.cache_hits += 1
            return cached

        if breaker is not None and not breaker.allow():
            stats.rejected += 1
            return fallback(args)

        try:
            result = tool.handler(ctx, args)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=timeout_s)
        except asyncio.TimeoutError:
            stats.timeouts += 1
            logger.warning("工具 %s 超时，超过 %.1fs", tool.name, timeout_s)
            if breaker is not None:
                breaker.record_failure()
            return fallback(args)
        except Exception as ex:
            stats.errors += 1
            logger.warning("工具 %s 失败: %r", tool.name, ex)
            if breaker is not None:
                breaker.record_failure()
            return fallback(args)

        stats.successes += 1
        if breaker is not None:
            breaker.record_success()
        if cache is not None:
            cache.set(key, result)
        return result

    return ToolSpec(tool.name, tool.description, tool.parameters, handler), stats

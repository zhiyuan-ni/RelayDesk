"""运行期统计：每个 Agent、每个工具、每个环节的调用次数、成功率和延迟分位数。

延迟为什么看分位数而不是平均值：
  100 次请求里 99 次 1 秒、1 次 60 秒，平均值是 1.6 秒，看起来没问题，但那个用户等了一分钟。
  P95 表示 95% 的请求都比它快，P99 同理。线上监控看的是 P95 和 P99，平均值几乎没人看。

样本保存在固定长度的环形缓冲里，只统计最近 N 次，反映的是"现在"的状况而不是从启动到现在的累计。
"""
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

WINDOW = 500


def percentile(values: list[float], p: float) -> float:
    """最近邻法计算分位数。values 可以是无序的，p 取 0 到 100，空列表返回 0.0，不修改传入的列表。

    位置 = ceil(p / 100 * n) - 1，再限制在 0 到 n-1。P95 就是"排在前 95% 的那个位置上的值"。
    """
    n = len(values)
    if n == 0:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(p / 100 * n) - 1
    index = max(0, min(n - 1, index))
    return sorted_values[index]


@dataclass
class Series:
    total: int = 0
    failures: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def record(self, latency_ms: float, ok: bool = True) -> None:
        self.total += 1
        self.failures += not ok
        self.latencies.append(latency_ms)

    def summary(self) -> dict[str, Any]:
        recent = list(self.latencies)
        return {
            "total": self.total,
            "failures": self.failures,
            "success_rate": round(1 - self.failures / self.total, 4) if self.total else None,
            "recent_n": len(recent),
            "p50_ms": round(percentile(recent, 50), 1),
            "p95_ms": round(percentile(recent, 95), 1),
            "p99_ms": round(percentile(recent, 99), 1),
            "max_ms": round(max(recent), 1) if recent else 0.0,
        }


class StatsRegistry:
    """按 类别/名字 两级组织，例如 agent/billing、tool/get_order_status、stage/intent。"""

    def __init__(self):
        self._series: dict[str, dict[str, Series]] = defaultdict(lambda: defaultdict(Series))

    def record(self, kind: str, name: str, latency_ms: float, ok: bool = True) -> None:
        self._series[kind][name].record(latency_ms, ok)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {kind: {name: s.summary() for name, s in names.items()} for kind, names in self._series.items()}


stats = StatsRegistry()   # 进程内唯一的一份，各模块直接 import 使用

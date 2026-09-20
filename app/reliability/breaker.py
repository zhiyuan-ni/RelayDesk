"""熔断器。名字来自电路里的保险丝：电流异常时自动断开，保护后面的电器。

要解决的问题：
  上游服务挂了的时候，每个请求都要傻等到超时才失败。假设超时是 8 秒，
  这期间所有用户的每次检索都白等 8 秒，线程和连接被占满，故障会从一个服务蔓延到整个系统。
  熔断器的做法是：连续失败几次之后，直接判定"它现在不行"，后续请求不再发出，立即走降级，
  过一会儿再小心地放一个请求去试探它恢复了没有。

三个状态：

    CLOSED 关闭 ──连续失败达到阈值──► OPEN 打开 ──冷却时间到──► HALF_OPEN 半开
      正常放行                          全部拒绝                     只放行一个探测请求
        ▲                                  ▲                              │
        │                                  └────────探测失败──────────────┤
        └──────────────────────────────探测成功───────────────────────────┘

  注意"关闭"是正常状态，和直觉相反。这是沿用电路的说法：开关闭合，电流才能通过。
"""
import time
from enum import Enum
from typing import Callable


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_s: float = 30.0,
                 now: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold   # 连续失败多少次后打开
        self.recovery_s = recovery_s                 # 打开后冷却多久再试探
        self._now = now                              # 可注入的时钟，测试时不用真的等 30 秒

        self.state = State.CLOSED
        self._failures = 0              # 当前的连续失败次数
        self._opened_at = 0.0           # 最近一次进入 OPEN 的时刻
        self._probe_in_flight = False   # HALF_OPEN 下，探测请求是否已经放出去、还没回来

    # 使用方式：
    #     if breaker.allow():
    #         try:    调用上游;  breaker.record_success()
    #         except: breaker.record_failure()
    #     else:
    #         直接降级，不调用上游

    def allow(self) -> bool:
        """现在能不能放一个请求过去。可能顺带把状态从 OPEN 推进到 HALF_OPEN。"""
        if self.state == State.CLOSED:
            return True

        if self.state == State.OPEN and self._cooldown_elapsed():
            # 冷却结束：进入半开，当前这个请求就是探测请求
            self.state = State.HALF_OPEN
            self._probe_in_flight = True
            return True

        # 剩下两种情况都拒绝：
        #   OPEN 且冷却未结束
        #   HALF_OPEN，探测请求已经放出去还没回来。半开状态只放一个请求去试探，
        #   否则上游刚要恢复，就又被积压的请求冲垮
        return False

    def record_success(self) -> None:
        """一次调用成功。无论之前是什么状态，都回到 CLOSED 并清零连续失败计数。"""
        self.state = State.CLOSED
        self._failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> None:
        """一次调用失败。"""
        if self.state == State.HALF_OPEN:
            # 探测失败，说明上游还没恢复：重新打开，冷却重新计时
            self._trip()
        elif self.state == State.CLOSED:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._trip()
        # OPEN 状态下不会有请求被放行，正常情况走不到这里，所以不处理

    # ── 内部辅助 ──

    def _cooldown_elapsed(self) -> bool:
        return self._now() - self._opened_at >= self.recovery_s

    def _trip(self) -> None:
        """跳闸：进入 OPEN，并记下时刻作为冷却的起点。"""
        self.state = State.OPEN
        self._opened_at = self._now()
        self._probe_in_flight = False

"""CircuitBreaker 的规格测试。用假时钟，不需要真的等待冷却时间。"""
from app.reliability.breaker import CircuitBreaker, State


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def make(threshold=3, recovery=30.0):
    clock = Clock()
    return CircuitBreaker(threshold, recovery, now=clock), clock


def trip(b):
    for _ in range(b.failure_threshold):
        b.record_failure()


def test_starts_closed_and_allows():
    b, _ = make()
    assert b.state == State.CLOSED and b.allow() and b.allow()


def test_opens_only_after_consecutive_failures_reach_threshold():
    b, _ = make(threshold=3)
    b.record_failure(); b.record_failure()
    assert b.state == State.CLOSED and b.allow()
    b.record_failure()
    assert b.state == State.OPEN and not b.allow()


def test_success_resets_the_consecutive_count():
    b, _ = make(threshold=3)
    b.record_failure(); b.record_failure(); b.record_success()
    b.record_failure(); b.record_failure()
    assert b.state == State.CLOSED       # 中间成功过一次，不算连续 4 次失败


def test_open_rejects_until_recovery_time_passes():
    b, clock = make(recovery=30.0)
    trip(b)
    clock.t = 29.9
    assert not b.allow() and b.state == State.OPEN


def test_half_open_lets_exactly_one_probe_through():
    b, clock = make(recovery=30.0)
    trip(b)
    clock.t = 30.0
    assert b.allow() and b.state == State.HALF_OPEN     # 第一个请求作为探测放行
    assert not b.allow() and not b.allow()              # 探测没回来之前，其余全部拒绝


def test_probe_success_closes_the_breaker():
    b, clock = make(threshold=3, recovery=30.0)
    trip(b); clock.t = 30.0; b.allow()
    b.record_success()
    assert b.state == State.CLOSED and b.allow() and b.allow()
    b.record_failure(); b.record_failure()
    assert b.state == State.CLOSED                      # 失败计数已清零，要重新攒够 3 次才会再打开


def test_probe_failure_reopens_and_restarts_the_timer():
    b, clock = make(recovery=30.0)
    trip(b); clock.t = 30.0; b.allow()
    b.record_failure()
    assert b.state == State.OPEN
    clock.t = 59.9; assert not b.allow()                # 冷却从 30.0 重新计时，到 60.0 才够
    clock.t = 60.0; assert b.allow() and b.state == State.HALF_OPEN

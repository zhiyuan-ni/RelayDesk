from app.reliability.cache import TTLCache


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def test_hit_then_expire():
    clock = Clock(); c = TTLCache(ttl_s=10, now=clock)
    c.set("k", {"v": 1})
    clock.t = 9.9;  assert c.get("k") == {"v": 1}
    clock.t = 10.0; assert c.get("k") is None and len(c) == 0   # 过期后被惰性删除


def test_evicts_oldest_when_full():
    c = TTLCache(ttl_s=10, max_items=2, now=Clock())
    c.set("a", 1); c.set("b", 2); c.set("c", 3)
    assert c.get("a") is None and c.get("b") == 2 and c.get("c") == 3

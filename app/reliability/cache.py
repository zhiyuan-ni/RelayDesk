"""带过期时间的内存缓存。

为什么自己写而不用 functools.lru_cache：lru_cache 没有过期时间，政策文档更新后旧答案会一直被命中。
为什么不用 Redis：这里缓存的是可以随时重算的检索结果，丢了没有任何损失，进程内字典最简单也最快。
阶段 4 的会话记忆才需要 Redis，因为那是丢不得、并且要跨进程共享的数据。
"""
import time
from typing import Any, Callable, Optional


class TTLCache:
    def __init__(self, ttl_s: float, max_items: int = 1000, now: Callable[[], float] = time.monotonic):
        self._ttl, self._max, self._now = ttl_s, max_items, now   # now 可注入，测试时用假时钟，不用真的等
        self._data: dict[str, tuple[float, Any]] = {}              # key -> (过期时刻, 值)

    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if self._now() >= expires_at:
            del self._data[key]      # 惰性删除：不开后台线程定时清理，读到过期的才删
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._data) >= self._max and key not in self._data:
            # 满了就删最早写入的那一条。Python 的字典保持插入顺序，第一个键就是最老的
            del self._data[next(iter(self._data))]
        self._data[key] = (self._now() + self._ttl, value)

    def __len__(self) -> int:
        return len(self._data)

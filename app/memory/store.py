"""会话存储：Redis 里的对话记录和摘要。

数据结构，每个会话两个键：
  conv:{user_id}:{conv_id}:messages   列表，按时间顺序追加，每个元素是一条消息的 JSON
  conv:{user_id}:{conv_id}:summary    字符串，被压缩掉的旧消息的摘要

键里带 user_id 的原因：conv_id 是客户端传来的，不可信。
如果只用 conv_id 当键，甲只要猜到乙的会话号，就能读到乙的对话。带上 user_id，最多只能读到自己的。

每次写入都刷新 24 小时的过期时间。用户一天不来，这段对话就自动清理，不需要写定时任务。
"""
import json
import time
from typing import Any

TTL_SECONDS = 24 * 3600


class ConversationStore:
    def __init__(self, redis):
        self._r = redis

    @staticmethod
    def _key(user_id: str, conv_id: str, kind: str) -> str:
        return f"conv:{user_id}:{conv_id}:{kind}"

    async def append(self, user_id: str, conv_id: str, role: str, content: str) -> int:
        """追加一条消息，返回追加后的消息总数。"""
        key = self._key(user_id, conv_id, "messages")
        item = json.dumps({"role": role, "content": content, "ts": round(time.time(), 3)}, ensure_ascii=False)
        n = await self._r.rpush(key, item)
        await self._r.expire(key, TTL_SECONDS)
        return n

    async def messages(self, user_id: str, conv_id: str) -> list[dict[str, Any]]:
        raw = await self._r.lrange(self._key(user_id, conv_id, "messages"), 0, -1)
        return [json.loads(x) for x in raw]

    async def drop_oldest(self, user_id: str, conv_id: str, n: int) -> None:
        """删除最早的 n 条。LTRIM 的含义是只保留下标 n 到末尾的部分。

        用"删掉前 n 条"而不是"重写整个列表"：压缩要调模型，耗时几秒，
        这期间用户可能又发了新消息。只删前 n 条，新追加的消息不受影响。
        """
        await self._r.ltrim(self._key(user_id, conv_id, "messages"), n, -1)

    async def get_summary(self, user_id: str, conv_id: str) -> str:
        return await self._r.get(self._key(user_id, conv_id, "summary")) or ""

    async def set_summary(self, user_id: str, conv_id: str, summary: str) -> None:
        await self._r.set(self._key(user_id, conv_id, "summary"), summary, ex=TTL_SECONDS)

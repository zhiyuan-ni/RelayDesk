"""长期记忆：跨会话的历史，按语义检索。

工作记忆回答"这次对话刚才说了什么"，存在 Redis，24 小时过期。
长期记忆回答"这个用户以前来问过什么"，存在 ChromaDB，不过期。
用户下周再来说"上次那个重复扣款的事怎么样了"，靠的就是它。

每个会话在这里只占一条记录，ID 由用户和会话号决定，每轮对话后更新。
这样做的原因：大多数客服对话只有两三轮，远远到不了压缩阈值。如果只在压缩时才写入，
长期记忆里几乎不会有东西。

排序不只看语义相似度，还乘上一个时间衰减系数。三个月前聊过的退款，和昨天聊过的退款，
对理解用户现在的问题，价值是不一样的。
"""
import asyncio
import hashlib
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

COLLECTION = "episodic_memory"
MIN_SIMILARITY = 0.5     # 相似度低于它的记忆视为无关。宁可想不起来，也不要把不相干的往事塞给模型
HALF_LIFE_DAYS = 30.0    # 半衰期：每过 30 天，一条记忆的权重减半
SECONDS_PER_DAY = 86400.0


def rank_memories(
    hits: list[dict[str, Any]],
    now_ts: float,
    exclude_conv_id: str = "",
    top_k: int = 2,
    min_similarity: float = MIN_SIMILARITY,
    half_life_days: float = HALF_LIFE_DAYS,
) -> list[dict[str, Any]]:
    """过滤并排序候选记忆。

    去掉当前会话的记忆（它已经在工作记忆里）和相似度低于门槛的记忆，
    其余按 相似度 × 0.5^(天数 / 半衰期) 排序，取前 top_k 条，并在每条上附加 final_score。
    不修改传入的字典。
    """
    results = []
    for hit in hits:
        if hit["conv_id"] != exclude_conv_id and hit["similarity"] >= min_similarity:
            age_days = (now_ts - hit["ts"]) / SECONDS_PER_DAY
            final = hit["similarity"] * 0.5 ** (age_days / half_life_days)
            results.append({**hit, "final_score": round(final, 4)})

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]




class LongTermMemory:
    def __init__(self, embedder, client, now: Callable[[], float] = time.time,
                 min_similarity: float = MIN_SIMILARITY):
        self._embedder, self._client, self._now = embedder, client, now
        # 相似度的绝对值取决于向量模型，换模型后这个门槛要重新标定，所以做成参数
        self._min_similarity = min_similarity
        self._col = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine", "embedding_model": self._embedder.model},
            embedding_function=None)

    async def ensure_ready(self) -> str:
        """向量模型换了的话，用存着的原文重新向量化。和知识库不同，这里的内容无法从文件重建，所以必须就地迁移。"""
        built_with = (self._col.metadata or {}).get("embedding_model")
        if built_with == self._embedder.model:
            return "reused"
        data = await asyncio.to_thread(self._col.get, include=["documents", "metadatas"])
        await asyncio.to_thread(self._client.delete_collection, COLLECTION)
        self._col = await asyncio.to_thread(self._open)
        if data["ids"]:
            vectors = await self._embedder.embed(data["documents"])
            await asyncio.to_thread(self._col.upsert, ids=data["ids"], embeddings=vectors,
                                    documents=data["documents"], metadatas=data["metadatas"])
        logger.warning("长期记忆由 %s 迁移到 %s，共 %d 条", built_with, self._embedder.model, len(data["ids"]))
        return "migrated"

    async def remember(self, user_id: str, conv_id: str, text: str) -> None:
        """写入或更新一个会话的记忆。同一个会话反复调用只会更新同一条记录。"""
        text = text.strip()
        if not text:
            return
        [vector] = await self._embedder.embed([text])
        record_id = hashlib.md5(f"{user_id}|{conv_id}".encode()).hexdigest()
        await asyncio.to_thread(
            self._col.upsert, ids=[record_id], embeddings=[vector], documents=[text],
            metadatas=[{"user_id": user_id, "conv_id": conv_id, "ts": self._now()}])

    async def recall(self, user_id: str, query: str, exclude_conv_id: str = "", top_k: int = 2) -> list[dict[str, Any]]:
        [vector] = await self._embedder.embed([query])
        # where 条件是硬性过滤：只在这个用户自己的记忆里找。绝不能靠相似度来"顺便"隔离用户
        res = await asyncio.to_thread(
            self._col.query, query_embeddings=[vector], n_results=top_k * 4, where={"user_id": user_id})
        hits = [
            {"conv_id": meta["conv_id"], "text": doc, "similarity": round(1.0 - dist, 4), "ts": meta["ts"]}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        return rank_memories(hits, self._now(), exclude_conv_id, top_k, self._min_similarity)

    async def count(self) -> int:
        return await asyncio.to_thread(self._col.count)

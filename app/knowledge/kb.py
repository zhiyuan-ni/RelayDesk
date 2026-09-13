"""知识库：文档入库和语义检索，底层是 ChromaDB。

两个关键决定：
  1. 向量由我们自己调 Embedding 模型生成，再交给 ChromaDB 存储和检索。
     ChromaDB 自带的默认模型主要面向英文，中文同义句的区分度不够。
  2. ChromaDB 的 Python 接口是同步的。在 async 函数里直接调用会卡住整个事件循环，
     其他用户的请求全得等着。所以统一用 asyncio.to_thread 丢到线程池里执行。
"""
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb

from app.knowledge.chunker import Chunk, chunk_markdown
from app.knowledge.embedder import EmbeddingUnavailable

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "docs"
COLLECTION = "knowledge_base"


class KnowledgeBase:
    def __init__(self, embedder, path: str):
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=path, settings=chromadb.Settings(anonymized_telemetry=False))
        self._col = self._open_collection()

    @property
    def embedder(self):
        return self._embedder

    @property
    def client(self):
        """底层的 ChromaDB 客户端。长期记忆复用它，两者存在同一个本地库的不同集合里。"""
        return self._client

    def _open_collection(self):
        # embedding_function=None：明确告诉 ChromaDB 不要用它自带的模型
        # hnsw:space=cosine：用余弦距离比较向量，只看方向不看长度，是文本检索的标准选择
        # embedding_model：记下建库用的向量模型。元数据只在集合首次创建时写入，之后打开不会被覆盖
        return self._client.get_or_create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedder.model},
            embedding_function=None,
        )

    async def ensure_ready(self, docs_dir: Path = DOCS_DIR) -> str:
        """服务启动时调用。返回 "reused" / "ingested" / "rebuilt" / "unverified"，说明这次做了什么。

        三个保护：
          1. 配置错误快速失败。用向量模型做一次探测调用，模型名配错、密钥失效时抛
             EmbeddingConfigError，让启动直接失败。这类错误重试没用，越早暴露越好。
          2. 暂时不可用时不阻止启动。网络抖动、上游限流是会自己恢复的。只要本地向量库已经用
             当前模型建好了，就返回 "unverified" 照常启动。检索工具自带超时和熔断，恢复前走降级即可。
             向量库还是空的、或者需要重建时没有退路，只能把 EmbeddingUnavailable 继续向上抛。
          3. 一致性。不同向量模型的维度和语义空间都不同，不能混用。建库时把模型名记在元数据里，
             和当前配置不一致就重建。知识库内容全部来自 docs 目录，重建不会丢数据。
        """
        count = await self.count()
        built_with = (self._col.metadata or {}).get("embedding_model")
        usable_as_is = count > 0 and built_with == self._embedder.model

        try:
            await self._embedder.embed(["启动探测"])
        except EmbeddingUnavailable as ex:
            if not usable_as_is:
                raise
            logger.warning("向量服务暂时不可用，沿用本地向量库启动，检索在恢复前会走降级: %s", ex)
            return "unverified"

        if count == 0:
            await self.ingest_dir(docs_dir)
            return "ingested"
        if not usable_as_is:
            logger.warning("向量模型由 %s 变为 %s，重建知识库", built_with, self._embedder.model)
            await asyncio.to_thread(self._client.delete_collection, COLLECTION)
            self._col = await asyncio.to_thread(self._open_collection)
            await self.ingest_dir(docs_dir)
            return "rebuilt"
        return "reused"

    async def count(self) -> int:
        return await asyncio.to_thread(self._col.count)

    async def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = await self._embedder.embed([c.for_embedding() for c in chunks])
        # 用内容哈希当 ID，再配合 upsert：同一段内容重复导入不会产生重复记录
        ids = [hashlib.md5(f"{c.title}|{c.section}|{c.text}".encode()).hexdigest() for c in chunks]
        await asyncio.to_thread(
            self._col.upsert,
            ids=ids,
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"title": c.title, "section": c.section} for c in chunks],
        )
        return len(chunks)

    async def ingest_dir(self, docs_dir: Path = DOCS_DIR) -> int:
        chunks = [c for p in sorted(docs_dir.glob("*.md")) for c in chunk_markdown(p.read_text(encoding="utf-8"))]
        n = await self.add_chunks(chunks)
        logger.info("知识库导入 %d 个片段", n)
        return n

    async def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        [vector] = await self._embedder.embed([query])
        res = await asyncio.to_thread(self._col.query, query_embeddings=[vector], n_results=top_k)
        return [
            {"title": meta["title"], "section": meta["section"], "text": doc,
             "score": round(1.0 - dist, 4)}   # ChromaDB 返回的是距离，越小越近；转成相似度，越大越相关
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

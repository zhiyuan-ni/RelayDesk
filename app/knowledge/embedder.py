"""文本向量化。把一段文字变成一串数字，语义相近的文字，数字也相近。"""
from openai import AsyncOpenAI

from app.config import Settings

_BATCH = 10  # 单次请求最多送多少条文本。不同模型上限不同，10 是各家都能接受的保守值


class Embedder:
    def __init__(self, cfg: Settings):
        self._client = AsyncOpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, timeout=30.0, max_retries=1)
        self.model = cfg.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            resp = await self._client.embeddings.create(model=self.model, input=texts[i:i + _BATCH])
            vectors.extend(d.embedding for d in resp.data)
        return vectors

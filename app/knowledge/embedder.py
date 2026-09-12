"""文本向量化。把一段文字变成一串数字，语义相近的文字，数字也相近。"""
from typing import Optional

from openai import APIError, APIStatusError, AsyncOpenAI

from app.config import Settings

class EmbeddingConfigError(Exception):
    """配置有问题：模型名不存在、密钥无效、无权限。重试没有意义，需要人来修改配置。"""


class EmbeddingUnavailable(Exception):
    """暂时不可用：网络不通、超时、限流、上游 5xx。过一会儿可能自己恢复。"""


_CONFIG_STATUS = {400, 401, 403, 404}   # 这几类状态码说明请求本身有问题，而不是上游一时繁忙

_BATCH = 10  # 单次请求最多送多少条文本。不同模型上限不同，10 是各家都能接受的保守值


class Embedder:
    def __init__(self, cfg: Settings, client: Optional[AsyncOpenAI] = None):
        """client: 传入对话模型正在用的客户端，两者共享一个连接池。

        实测：连接闲置一段时间后会被关闭，下一次请求要重新建立加密连接，经过代理时要 5 到 6 秒，
        而热连接上的同一个请求只要 1 秒。向量接口只在检索时才调用，单独一个客户端几乎每次都撞上冷连接。
        对话接口在每次检索前几秒刚被意图识别和 Agent 用过，共用它的连接池就总是热的。
        """
        self._client = client or AsyncOpenAI(
            api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, timeout=30.0, max_retries=1)
        self.model = cfg.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(texts), _BATCH):
                resp = await self._client.embeddings.create(model=self.model, input=texts[i:i + _BATCH])
                vectors.extend(d.embedding for d in resp.data)
        except APIStatusError as ex:
            # 把 SDK 的异常翻译成两类业务含义明确的异常，上层不需要了解 HTTP 状态码
            kind = EmbeddingConfigError if ex.status_code in _CONFIG_STATUS else EmbeddingUnavailable
            raise kind(f"HTTP {ex.status_code}: {ex.message}") from ex
        except APIError as ex:   # 连接失败、超时等没有状态码的情况
            raise EmbeddingUnavailable(str(ex) or type(ex).__name__) from ex
        return vectors

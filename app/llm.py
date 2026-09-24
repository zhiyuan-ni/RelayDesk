"""LLM 客户端封装。

为什么要封装一层，而不是到处直接用 openai SDK：
  1. 后面意图识别、Agent、重排、评测都要调模型，统一入口便于加超时、重试、统计。
  2. 测试时可以用一个假的客户端替换它，不花钱也不依赖网络。

aihubmix 是 OpenAI 协议兼容的中转站，所以用 openai SDK，只把 base_url 指过去。
"""
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from app.config import Settings


def connection_limits(cfg: Settings) -> httpx.Limits:
    """对话、向量、jev 客户端共用的连接池参数。只改闲置保持时间，其余沿用 httpx 的默认值。"""
    return httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=cfg.http_keepalive_s)


class LLMClient:
    def __init__(self, cfg: Settings):
        if not cfg.llm_api_key:
            raise RuntimeError("未设置 LLM_API_KEY，请先把 .env.example 复制为 .env 并填写")
        self._client = AsyncOpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            timeout=30.0,   # 单次请求最长等 30 秒
            max_retries=1,  # 网络抖动时 SDK 自动重试 1 次
            # 用 SDK 提供的默认客户端，只换连接池参数。它和 httpx 一样会读取系统代理和 HTTPS_PROXY
            http_client=DefaultAsyncHttpxClient(limits=connection_limits(cfg)),
        )
        self.model = cfg.llm_model
        self._enable_thinking = cfg.llm_enable_thinking

    def for_model(self, model: str) -> "LLMClient":
        """同一个连接池、不同的模型。model 为空时返回自己。

        某些环节只做分类或排序，用小模型又快又便宜。复用底层客户端是为了共享连接池，
        否则每个环节各自冷启动连接，之前实测冷连接要多等 5 秒。
        """
        if not model or model == self.model:
            return self
        other = object.__new__(LLMClient)          # 跳过 __init__，不再新建客户端
        other._client = self._client
        other.model = model
        # 思考开关是 Qwen 系列的参数，换成别家模型时不能带上，否则可能报错
        other._enable_thinking = self._enable_thinking if model.startswith("qwen") else None
        return other

    @property
    def client(self) -> AsyncOpenAI:
        """底层客户端。向量模型复用它，共享同一个连接池，原因见 Embedder。"""
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Any:
        """发起一次对话请求，返回模型的 message 对象。

        返回整个 message 而不是纯文本，是因为阶段 2 做工具调用时
        需要读取 message.tool_calls。
        """
        full_messages = list(messages)
        if system:
            full_messages.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self._enable_thinking is not None:
            # extra_body 里的字段会原样放进请求体，用来传 OpenAI 标准协议之外的厂商参数
            kwargs["extra_body"] = {"enable_thinking": self._enable_thinking}

        # await 的含义：在等网络返回的这几秒里，把 CPU 让给别的请求
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message

    async def chat_text(self, prompt: str, **kwargs: Any) -> str:
        """最常用的简化形式：给一段 prompt，拿回一段文本。"""
        message = await self.chat([{"role": "user", "content": prompt}], **kwargs)
        return (message.content or "").strip()

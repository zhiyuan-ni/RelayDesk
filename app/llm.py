"""LLM 客户端封装。

为什么要封装一层，而不是到处直接用 openai SDK：
  1. 后面意图识别、Agent、重排、评测都要调模型，统一入口便于加超时、重试、统计。
  2. 测试时可以用一个假的客户端替换它，不花钱也不依赖网络。

aihubmix 是 OpenAI 协议兼容的中转站，所以用 openai SDK，只把 base_url 指过去。
"""
from typing import Any, Optional

from openai import AsyncOpenAI

from app.config import Settings


class LLMClient:
    def __init__(self, cfg: Settings):
        if not cfg.llm_api_key:
            raise RuntimeError("未设置 LLM_API_KEY，请先把 .env.example 复制为 .env 并填写")
        self._client = AsyncOpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            timeout=30.0,   # 单次请求最长等 30 秒
            max_retries=1,  # 网络抖动时 SDK 自动重试 1 次
        )
        self.model = cfg.llm_model

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

        # await 的含义：在等网络返回的这几秒里，把 CPU 让给别的请求
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message

    async def chat_text(self, prompt: str, **kwargs: Any) -> str:
        """最常用的简化形式：给一段 prompt，拿回一段文本。"""
        message = await self.chat([{"role": "user", "content": prompt}], **kwargs)
        return (message.content or "").strip()

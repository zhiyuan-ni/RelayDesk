"""jev 客户端。

jev 是 TypeSafe AI 的 System One 模型：不生成文字，只回答预先定义好的选择题、打分题和是非题，
返回值一定落在你给的选项里，并附带整个概率分布和校准过的置信度。

aihubmix 转发的是 TypeSafe 的原生接口 POST /v1/systemone，不是 OpenAI 的 chat 协议，
所以不能复用 openai SDK，这里直接用 httpx 发请求。key 和中转站地址与对话模型共用。
"""
import logging
from typing import Any, Optional

import httpx

from app.config import Settings
from app.llm import connection_limits

logger = logging.getLogger(__name__)


class JevClient:
    def __init__(self, cfg: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        if not cfg.llm_api_key:
            raise RuntimeError("未设置 LLM_API_KEY，jev 通过 aihubmix 调用，和对话模型共用同一个 key")
        self.model = cfg.jev_model
        # transport 只在测试时传入假的传输层。线上不能传：httpx 一旦拿到自定义 transport，
        # 就不再读取系统代理和 HTTPS_PROXY 环境变量，而本机和容器访问中转站都要走代理。
        # 曾为了加连接重试传过 AsyncHTTPTransport(retries=1)，结果所有请求连接失败，意图识别全部退回规则路
        extra = {"transport": transport} if transport else {}
        self._http = httpx.AsyncClient(
            base_url=cfg.llm_base_url,
            headers={"Authorization": f"Bearer {cfg.llm_api_key}"},
            timeout=cfg.jev_timeout_s,
            limits=connection_limits(cfg),   # 不设的话闲置 5 秒就断开，下一次意图识别要重新握手
            **extra,
        )

    async def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, dict]:
        """一次请求问多道题，返回 {题目 id: 答案}。所有题目在服务端并行作答，多问几道几乎不增加延迟。"""
        resp = await self._http.post("/systemone", json={"model": self.model, "state": state, "questions": questions})
        resp.raise_for_status()   # 4xx/5xx 抛 HTTPStatusError，由调用方决定怎么兜底
        return resp.json()["answers"]

    async def warmup(self) -> bool:
        """启动时先发一道小题，把连接建好，顺便检查 key 和模型名是否可用。

        经代理的冷连接光 TLS 握手就要约 5 秒，热连接的一次请求约 0.5 秒。
        不预热的话，服务启动后第一位用户要替所有人付这 5 秒。失败只记警告，不阻止启动：
        意图识别在 jev 不可用时会退回规则路。
        """
        try:
            await self.ask("你好", {"ping": {"type": "noul", "instructions": "这是一句问候"}})
            return True
        except Exception as ex:
            logger.warning("jev 预热失败，意图识别将在调用失败时退回规则路: %r", ex)
            return False

    async def aclose(self) -> None:
        await self._http.aclose()


async def open_jev(cfg: Settings) -> Optional[JevClient]:
    """任一环节选了 jev 就建一个客户端并预热，各环节共用它的连接池；都没选返回 None。"""
    if "jev" not in (cfg.intent_backend, cfg.rerank_backend):
        return None
    jev = JevClient(cfg)
    await jev.warmup()
    return jev

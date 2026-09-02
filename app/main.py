"""RelayDesk 服务入口。

阶段 0 只做一件事：把"用户消息 -> 模型 -> 回复"跑通。
后面每个阶段都是往 chat() 这条流水线里插入新环节。
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

from app.config import settings
from app.intent.recognizer import IntentRecognizer
from app.llm import LLMClient

SYSTEM_PROMPT = "你是 RelayDesk 智能客服。回答要简洁、友好；不确定的事情明确说不确定。"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # yield 之前的代码在服务启动时执行一次，之后的在关闭时执行
    app.state.llm = LLMClient(settings)
    yield


app = FastAPI(title="RelayDesk", version="0.1.0", lifespan=lifespan)


def get_llm(request: Request) -> LLMClient:
    """依赖注入：接口通过它拿到 LLM 客户端，测试时可以整体替换成假的。"""
    return request.app.state.llm


def get_recognizer(llm: LLMClient = Depends(get_llm)) -> IntentRecognizer:
    """依赖可以嵌套：识别器依赖 LLM 客户端。测试只要替换 get_llm，这里自动跟着用假的。"""
    return IntentRecognizer(llm)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None


class ChatResponse(BaseModel):
    conv_id: str
    response: str
    latency_ms: float
    # 以下是意图识别的结构化输出，阶段 2 的路由会用到它们
    intent: str
    intent_group: str
    intent_confidence: float
    intent_source: str
    urgency: str
    entities: dict[str, list[str]]


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    llm: LLMClient = Depends(get_llm),
    recognizer: IntentRecognizer = Depends(get_recognizer),
):
    t0 = time.monotonic()  # monotonic 不受系统改时间影响，适合算耗时
    conv_id = req.conv_id or uuid.uuid4().hex[:12]

    # 第 1 步：先理解用户想干什么。现在只是把结果返回出去，阶段 2 会据此选 Agent。
    intent = await recognizer.recognize(req.message)

    # 第 2 步：生成回复。这两步目前是串行的，因为之后回复要依赖意图结果。
    answer = await llm.chat_text(req.message, system=SYSTEM_PROMPT)

    return ChatResponse(
        conv_id=conv_id,
        response=answer,
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
        intent=intent.intent.value,
        intent_group=intent.group.value,
        intent_confidence=intent.confidence,
        intent_source=intent.source,
        urgency=intent.urgency.name,
        entities=intent.entities,
    )

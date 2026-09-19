"""RelayDesk 服务入口。接口层只做三件事：收参数、调编排器、整理返回值。"""
from contextlib import asynccontextmanager
from typing import Any, Optional
import uuid

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

from app.config import settings
from app.llm import LLMClient
from app.orchestrator import Orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 编排器全局只建一个。它内部持有 Agent 等对象，后面还会挂上统计和记忆，不能每个请求新建。
    app.state.orchestrator = Orchestrator(LLMClient(settings))
    yield


app = FastAPI(title="RelayDesk", version="0.2.0", lifespan=lifespan)


def get_orchestrator(request: Request) -> Orchestrator:
    """依赖注入入口。测试时替换成装了假模型的编排器。"""
    return request.app.state.orchestrator


class ChatRequest(BaseModel):
    message: str
    user_id: str = "u1001"   # 默认用模拟库里的一个用户，方便在 /docs 页面直接试
    conv_id: Optional[str] = None


class ChatResponse(BaseModel):
    conv_id: str
    request_id: str
    response: str
    latency_ms: float
    # 意图
    intent: str
    intent_group: str
    intent_confidence: float
    intent_source: str
    urgency: str
    entities: dict[str, list[str]]
    # 路由
    action: str
    primary_agent: str
    supporting_agents: list[str]
    agents_used: list[str]
    routing_scores: dict[str, float]
    routing_reason: str
    # 工具
    tool_traces: list[dict[str, Any]]


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, orch: Orchestrator = Depends(get_orchestrator)):
    result = await orch.handle(req.message, req.user_id)
    intent, decision = result.intent, result.decision
    return ChatResponse(
        conv_id=req.conv_id or uuid.uuid4().hex[:12],
        request_id=result.request_id,
        response=result.response,
        latency_ms=result.latency_ms,
        intent=intent.intent.value,
        intent_group=intent.group.value,
        intent_confidence=intent.confidence,
        intent_source=intent.source,
        urgency=intent.urgency.name,
        entities=intent.entities,
        action=decision.action,
        primary_agent=decision.primary,
        supporting_agents=decision.supporting,
        agents_used=result.agents_used,
        routing_scores=decision.scores,
        routing_reason=decision.reason,
        tool_traces=result.tool_traces,
    )

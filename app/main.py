"""RelayDesk 服务入口。接口层只做三件事：收参数、调编排器、整理返回值。"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.config import settings
from app.knowledge.embedder import Embedder
from app.knowledge.kb import KnowledgeBase
from app.knowledge.retriever import Retriever
from app.knowledge.tool import build_knowledge_tool
from app.llm import LLMClient
from app.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 编排器全局只建一个。它内部持有 Agent 等对象，后面还会挂上统计和记忆，不能每个请求新建。
    kb = KnowledgeBase(Embedder(settings), settings.kb_path)
    try:
        state = await kb.ensure_ready()   # 首次启动导入文档；换了向量模型则重建；否则复用
    except Exception as ex:
        # 让启动直接失败并给出可读的原因，好过带着坏配置运行
        raise RuntimeError(f"知识库初始化失败，请检查 EMBEDDING_MODEL={settings.embedding_model!r} 是否正确: {ex}") from ex
    logger.info("知识库就绪: %s, 片段数 %d, 向量模型 %s", state, await kb.count(), settings.embedding_model)
    llm = LLMClient(settings)
    retriever = Retriever(kb, llm, rewrite=settings.retrieval_rewrite, rerank=settings.retrieval_rerank)
    tool = build_knowledge_tool(retriever)

    app.state.kb = kb
    app.state.retriever = retriever
    app.state.orchestrator = Orchestrator(llm, shared_tools={tool.name: tool})
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


@app.post("/search")
async def search(query: str, top_k: int = 4, enhanced: bool = True, request: Request = None):
    """查看检索结果，调试检索质量用。enhanced=false 时只做纯向量检索，方便对比改写和重排的效果。"""
    try:
        if not enhanced:
            return {"query": query, "mode": "vector_only", "results": await request.app.state.kb.search(query, top_k)}
        r = await request.app.state.retriever.retrieve(query, top_k)
    except Exception as ex:
        # 502 表示"我依赖的上游服务出错了"，比笼统的 500 更准确，错误原因也一并返回
        raise HTTPException(status_code=502, detail=f"检索失败: {ex}") from ex
    return {"query": query, "mode": "enhanced", "queries": r.queries, "n_candidates": r.n_candidates,
            "reranked": r.reranked, "latency_ms": r.latency_ms, "results": r.hits}


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

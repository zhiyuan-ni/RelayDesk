"""RelayDesk 服务入口。接口层只做三件事：收参数、调编排器、整理返回值。"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional
import uuid

import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.config import settings
from app.knowledge.embedder import Embedder, EmbeddingConfigError, EmbeddingUnavailable
from app.knowledge.kb import KnowledgeBase
from app.knowledge.retriever import Retriever
from app.knowledge.tool import build_knowledge_tool, knowledge_fallback
from app.llm import LLMClient
from app.memory.manager import MemoryManager
from app.memory.store import ConversationStore
from app.orchestrator import Orchestrator
from app.reliability.breaker import CircuitBreaker
from app.reliability.cache import TTLCache
from app.reliability.guard import guarded

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def _connect_memory(llm):
    """连接 Redis。连不上时不阻止启动，系统退化为无记忆模式并给出明确警告。

    和知识库初始化的处理不同：那里选择启动失败，因为向量模型配错了就什么都检索不到，属于配置错误。
    这里 Redis 没启动是环境问题，而且没有记忆时系统仍然能单轮回答，属于可以接受的降级。
    """
    client = redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await client.ping()
    except Exception as ex:
        logger.warning("Redis 不可用 (%s)，以无记忆模式运行。请执行 docker compose up -d redis", ex)
        await client.aclose()
        return None, None
    logger.info("会话记忆已启用: %s", settings.redis_url)
    return MemoryManager(ConversationStore(client), llm), client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 编排器全局只建一个。它内部持有 Agent 等对象，后面还会挂上统计和记忆，不能每个请求新建。
    llm = LLMClient(settings)
    kb = KnowledgeBase(Embedder(settings, client=llm.client), settings.kb_path)
    try:
        state = await kb.ensure_ready()   # 首次启动导入文档；换了向量模型则重建；否则复用
    except EmbeddingConfigError as ex:
        # 让启动直接失败并给出可读的原因，好过带着坏配置运行
        raise RuntimeError(
            f"向量模型配置有误，请检查 EMBEDDING_MODEL={settings.embedding_model!r} 和 LLM_API_KEY: {ex}") from ex
    except EmbeddingUnavailable as ex:
        raise RuntimeError(f"向量服务连不上，而本地还没有可用的向量库，无法完成首次建库。请检查网络或代理: {ex}") from ex
    logger.info("知识库就绪: %s, 片段数 %d, 向量模型 %s", state, await kb.count(), settings.embedding_model)
    retriever = Retriever(kb, llm, rewrite=settings.retrieval_rewrite, rerank=settings.retrieval_rerank,
                          rerank_timeout_s=settings.kb_rerank_timeout_s)
    # 知识库检索依赖两个外部接口，延迟波动大，所以包上缓存、熔断和超时。订单等本地工具不需要
    tool, kb_stats = guarded(
        build_knowledge_tool(retriever),
        timeout_s=settings.kb_timeout_s,
        fallback=knowledge_fallback,
        cache=TTLCache(settings.kb_cache_ttl_s),
        breaker=CircuitBreaker(settings.kb_breaker_failures, settings.kb_breaker_recovery_s),
    )
    app.state.kb_stats = kb_stats

    app.state.kb = kb
    app.state.retriever = retriever
    memory, redis_client = await _connect_memory(llm)
    app.state.memory = memory
    app.state.orchestrator = Orchestrator(llm, shared_tools={tool.name: tool}, memory=memory)
    yield
    if memory is not None:
        await memory.wait_background()   # 让还在进行的压缩任务跑完，再断开连接
        await redis_client.aclose()


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
    conv_id = req.conv_id or uuid.uuid4().hex[:12]   # 没带会话号就开一个新会话，客户端下次请求要带上它
    result = await orch.handle(req.message, req.user_id, conv_id)
    intent, decision = result.intent, result.decision
    return ChatResponse(
        conv_id=conv_id,
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

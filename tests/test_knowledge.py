"""知识库测试。用假的向量模型，不联网、不花钱，向量库建在临时目录里。"""
import hashlib

import pytest

from app.agents.base import AgentProfile, BaseAgent, execute_tool_call
from app.agents.tools import ToolContext
from app.knowledge.chunker import Chunk
from app.knowledge.kb import KnowledgeBase
from app.knowledge.tool import TOOL_NAME, build_knowledge_tool
from tests.fakes import FakeEmbedder


CHUNKS = [
    Chunk("退款政策", "退款时效", "审核通过后原路退回，支付宝一到三个工作日到账。"),
    Chunk("配送说明", "配送时效", "标准配送三到五个工作日送达，满九十九元免运费。"),
    Chunk("发票说明", "抬头修改", "发票开具后三十天内可以申请修改一次抬头。"),
]


async def make_kb(tmp_path):
    kb = KnowledgeBase(FakeEmbedder(), str(tmp_path))
    await kb.add_chunks(CHUNKS)
    return kb


async def test_search_returns_most_similar_first(tmp_path):
    kb = await make_kb(tmp_path)
    hits = await kb.search("发票抬头可以修改吗", top_k=2)
    assert len(hits) == 2 and hits[0]["title"] == "发票说明"
    assert hits[0]["score"] >= hits[1]["score"] and set(hits[0]) == {"title", "section", "text", "score"}


async def test_reimport_is_idempotent(tmp_path):
    kb = await make_kb(tmp_path)
    await kb.add_chunks(CHUNKS)
    assert await kb.count() == 3


async def test_ingest_real_docs(tmp_path):
    kb = KnowledgeBase(FakeEmbedder(), str(tmp_path))
    assert await kb.ingest_dir() >= 18 and await kb.count() >= 18


async def make_retriever(tmp_path):
    from app.knowledge.retriever import Retriever
    return Retriever(await make_kb(tmp_path), llm=None, rewrite=False, rerank=False)   # 纯向量模式，不需要模型


async def test_async_tool_runs_through_execute_tool_call(tmp_path):
    tool = build_knowledge_tool(await make_retriever(tmp_path))
    trace = await execute_tool_call({tool.name: tool}, TOOL_NAME, '{"query": "运费多少"}', ToolContext("u1001"))
    assert trace["success"] and trace["data"]["found"] and trace["data"]["results"][0]["title"] == "配送说明"


async def test_shared_tool_is_added_on_top_of_whitelist(tmp_path):
    tool = build_knowledge_tool(await make_retriever(tmp_path))
    profile = AgentProfile("technical", "技术支持", ("规则",), ("lookup_error_code",))
    agent = BaseAgent(None, profile, shared_tools={tool.name: tool})
    assert set(agent._tools) == {"lookup_error_code", TOOL_NAME}
    assert set(BaseAgent(None, profile)._tools) == {"lookup_error_code"}


async def test_ensure_ready_ingests_then_reuses(tmp_path):
    assert await KnowledgeBase(FakeEmbedder(), str(tmp_path)).ensure_ready() == "ingested"
    again = FakeEmbedder()
    assert await KnowledgeBase(again, str(tmp_path)).ensure_ready() == "reused"
    assert again.calls == 1   # 只有一次启动探测，没有重新向量化文档


async def test_ensure_ready_rebuilds_when_embedding_model_changes(tmp_path):
    await KnowledgeBase(FakeEmbedder("fake-embed-a"), str(tmp_path)).ensure_ready()
    kb = KnowledgeBase(FakeEmbedder("fake-embed-b"), str(tmp_path))
    assert await kb.ensure_ready() == "rebuilt" and await kb.count() >= 18
    assert await KnowledgeBase(FakeEmbedder("fake-embed-b"), str(tmp_path)).ensure_ready() == "reused"


async def test_config_error_fails_fast_even_when_store_is_ready(tmp_path):
    from app.knowledge.embedder import EmbeddingConfigError
    await KnowledgeBase(FakeEmbedder(), str(tmp_path)).ensure_ready()
    kb = KnowledgeBase(FakeEmbedder(error=EmbeddingConfigError("模型不存在")), str(tmp_path))
    with pytest.raises(EmbeddingConfigError):
        await kb.ensure_ready()


async def test_transient_outage_does_not_block_startup_when_store_is_ready(tmp_path):
    from app.knowledge.embedder import EmbeddingUnavailable
    await KnowledgeBase(FakeEmbedder(), str(tmp_path)).ensure_ready()
    kb = KnowledgeBase(FakeEmbedder(error=EmbeddingUnavailable("网络不通")), str(tmp_path))
    assert await kb.ensure_ready() == "unverified" and await kb.count() >= 18


async def test_transient_outage_still_fails_when_there_is_nothing_to_fall_back_on(tmp_path):
    from app.knowledge.embedder import EmbeddingUnavailable
    empty = KnowledgeBase(FakeEmbedder(error=EmbeddingUnavailable("网络不通")), str(tmp_path / "empty"))
    with pytest.raises(EmbeddingUnavailable):
        await empty.ensure_ready()

    await KnowledgeBase(FakeEmbedder("fake-embed-a"), str(tmp_path / "old")).ensure_ready()
    needs_rebuild = KnowledgeBase(FakeEmbedder("fake-embed-b", error=EmbeddingUnavailable("网络不通")), str(tmp_path / "old"))
    with pytest.raises(EmbeddingUnavailable):
        await needs_rebuild.ensure_ready()

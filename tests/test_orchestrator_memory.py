"""编排器与记忆的集成：多轮上下文是否真的传到了意图识别和 Agent。"""
import fakeredis

from app.memory.manager import MemoryManager
from app.memory.store import ConversationStore
from app.orchestrator import Orchestrator
from tests.fakes import FakeLLM


def make():
    llm = FakeLLM("refund")
    store = ConversationStore(fakeredis.FakeAsyncRedis(decode_responses=True))
    return Orchestrator(llm, memory=MemoryManager(store, llm)), llm, store


async def test_second_turn_sees_the_first_turn():
    orch, llm, store = make()
    await orch.handle("我想退款", "u1001", "c1")
    await orch.handle("订单号是 A12345", "u1001", "c1")

    seen = llm.agent_inputs["billing"]                       # 第二轮时账单 Agent 收到的全部消息
    assert "我想退款" in seen and "billing 的回答" in seen and "订单号是 A12345" in seen
    assert [m["content"] for m in await store.messages("u1001", "c1")] == [
        "我想退款", "billing 的回答", "订单号是 A12345", "billing 的回答"]


async def test_other_conversation_and_other_user_see_nothing():
    orch, llm, _ = make()
    await orch.handle("我想退款", "u1001", "c1")
    await orch.handle("帮我开发票", "u1001", "c2")
    assert "我想退款" not in llm.agent_inputs["billing"]
    await orch.handle("帮我开发票", "u1002", "c1")
    assert "我想退款" not in llm.agent_inputs["billing"]


async def test_memory_failure_does_not_break_the_request():
    class BrokenMemory:
        async def load(self, *a): raise ConnectionError("redis down")
        async def save_turn(self, *a): raise ConnectionError("redis down")

    r = await Orchestrator(FakeLLM("invoice"), memory=BrokenMemory()).handle("帮我开发票", "u1001", "c1")
    assert r.response == "billing 的回答"


async def test_without_conv_id_nothing_is_stored():
    orch, _, store = make()
    await orch.handle("你好", "u1001")
    assert await store.messages("u1001", "") == []


async def test_tool_context_only_contains_what_the_user_said():
    """助手的回复不能算进去，否则模型上一轮编造的订单号，下一轮就成了"出现过的"。"""
    captured = {}

    class SpyAgent:
        async def run(self, message, ctx, background="", history=None):
            captured["text"] = ctx.conversation_text
            from app.agents.base import AgentReply
            return AgentReply("billing", "已为您查询订单 Z99999", True)

    orch, _, _ = make()
    orch._agents["billing"] = SpyAgent()
    await orch.handle("我想退款", "u1001", "c9")
    await orch.handle("订单号是 A12345", "u1001", "c9")
    assert "我想退款" in captured["text"] and "A12345" in captured["text"] and "Z99999" not in captured["text"]


async def test_memories_from_an_earlier_conversation_reach_the_agent(tmp_path):
    import chromadb
    from app.memory.longterm import LongTermMemory
    from tests.fakes import FakeEmbedder

    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    llm = FakeLLM("payment_issue")
    store = ConversationStore(fakeredis.FakeAsyncRedis(decode_responses=True))
    memory = MemoryManager(store, llm, LongTermMemory(FakeEmbedder(), client, min_similarity=0.33))
    orch = Orchestrator(llm, memory=memory)

    await orch.handle("订单 B20250917 被重复扣款了", "u1001", "monday")
    await memory.wait_background()                                   # 等后台把这次会话写进长期记忆

    result = await orch.handle("上次那个重复扣款的订单后来怎么样了", "u1001", "friday")
    assert result.memories_used and "B20250917" in result.memories_used[0]
    assert "B20250917" in llm.agent_inputs["billing"]                # 账单 Agent 真的看到了这段往事

    other = await orch.handle("上次那个重复扣款的订单后来怎么样了", "u1002", "friday")
    assert other.memories_used == []                                 # 换一个用户，什么也想不起来


async def test_recall_failure_does_not_break_the_request():
    class FlakyMemory:
        async def load(self, *a):
            from app.memory.manager import MemoryContext
            return MemoryContext()
        async def save_turn(self, *a): pass
        async def recall(self, *a): raise TimeoutError("向量服务超时")

    r = await Orchestrator(FakeLLM("invoice"), memory=FlakyMemory()).handle("帮我开发票", "u1001", "c1")
    assert r.response == "billing 的回答" and r.memories_used == []


async def test_order_id_recalled_from_memory_passes_the_grounding_check(tmp_path):
    """用户上次提过订单号，这次只说"上次那个订单"。想起来的订单号应当允许用于查询。"""
    captured = {}

    class SpyAgent:
        async def run(self, message, ctx, background="", history=None):
            captured["text"] = ctx.conversation_text
            from app.agents.base import AgentReply
            return AgentReply("billing", "好的", True)

    class StubMemory:
        async def load(self, *a):
            from app.memory.manager import MemoryContext
            return MemoryContext()
        async def save_turn(self, *a): pass
        async def recall(self, *a): return ["用户说过：订单 B20250917 被重复扣款了"]

    orch = Orchestrator(FakeLLM("payment_issue"), memory=StubMemory())
    orch._agents["billing"] = SpyAgent()
    await orch.handle("上次那个重复扣款的订单后来怎么样了", "u1001", "friday")
    assert "B20250917" in captured["text"]

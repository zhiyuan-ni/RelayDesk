"""run_agents() 的规格测试。"""
import asyncio
import time

from app.agents.base import AgentReply
from app.agents.tools import ToolContext
from app.orchestrator import run_agents

CTX = ToolContext(user_id="u1001")


class SlowAgent:
    def __init__(self, name, seconds=0.2, error=None):
        self.name, self.seconds, self.error = name, seconds, error

    async def run(self, message, ctx, background=""):
        await asyncio.sleep(self.seconds)
        if self.error:
            raise self.error
        return AgentReply(self.name, f"{self.name}:{message}", True)


async def test_1_agents_run_concurrently():
    agents = {"technical": SlowAgent("technical"), "billing": SlowAgent("billing")}
    t0 = time.monotonic()
    replies = await run_agents(agents, ["technical", "billing"], "hi", CTX, "")
    assert time.monotonic() - t0 < 0.35          # 串行的话会是 0.4 秒以上
    assert len(replies) == 2


async def test_2_order_follows_names_not_finish_time():
    agents = {"technical": SlowAgent("technical", 0.2), "billing": SlowAgent("billing", 0.01)}
    replies = await run_agents(agents, ["technical", "billing"], "hi", CTX, "")
    assert [r.agent for r in replies] == ["technical", "billing"]   # billing 先完成，但仍排第二


async def test_3_one_failure_does_not_break_others():
    agents = {"technical": SlowAgent("technical", 0.01, error=RuntimeError("炸了")),
              "billing": SlowAgent("billing", 0.01)}
    replies = await run_agents(agents, ["technical", "billing"], "hi", CTX, "")
    assert (replies[0].agent, replies[0].success, replies[0].content) == ("technical", False, "")
    assert replies[1].success and replies[1].content == "billing:hi"

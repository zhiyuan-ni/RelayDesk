"""用脚本化的假模型驱动 tool-use 循环，验证消息顺序、白名单和轮数上限。"""
from types import SimpleNamespace as NS

from app.agents.base import MAX_TOOL_ROUNDS, AgentProfile, BaseAgent
from app.agents.tools import ToolContext

CTX = ToolContext(user_id="u1001")
PROFILE = AgentProfile("general", "通用客服", ("保持礼貌",), ("get_order_status",))


def tool_msg(name, args, call_id="c1"):
    return NS(content="", tool_calls=[NS(id=call_id, function=NS(name=name, arguments=args))])


def text_msg(text):
    return NS(content=text, tool_calls=None)


class ScriptedLLM:
    """按剧本依次返回预设消息，并记录每次收到的请求。"""
    def __init__(self, *script):
        self.script, self.requests = list(script), []

    async def chat(self, messages, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], **kwargs})
        return self.script.pop(0)


async def test_direct_answer_without_tools():
    llm = ScriptedLLM(text_msg("您好，请问有什么可以帮您？"))
    reply = await BaseAgent(llm, PROFILE).run("你好", CTX)
    assert reply.success and reply.tool_traces == [] and "您好" in reply.content


async def test_one_tool_round_trip():
    llm = ScriptedLLM(tool_msg("get_order_status", '{"order_id": "A12345"}'), text_msg("您的订单正在运输中。"))
    reply = await BaseAgent(llm, PROFILE).run("订单 A12345 到哪了", CTX)

    assert reply.content == "您的订单正在运输中。"
    assert [t["tool"] for t in reply.tool_traces] == ["get_order_status"]
    second = llm.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "tool"]
    assert second[2]["tool_call_id"] == "c1" and "运输中" in second[2]["content"]
    assert [t["function"]["name"] for t in llm.requests[0]["tools"]] == ["get_order_status"]


async def test_non_whitelisted_tool_is_rejected_but_model_is_told():
    llm = ScriptedLLM(tool_msg("get_payment_records", '{"order_id": "A12345"}'), text_msg("这个问题需要账单同事处理。"))
    reply = await BaseAgent(llm, PROFILE).run("查下扣款", CTX)
    assert reply.success and reply.tool_traces[0]["success"] is False
    assert "白名单" in llm.requests[1]["messages"][-1]["content"]


async def test_last_round_withholds_tools_to_force_an_answer():
    script = [tool_msg("get_order_status", '{"order_id": "A12345"}', f"c{i}") for i in range(MAX_TOOL_ROUNDS)]
    llm = ScriptedLLM(*script, text_msg("根据已查到的信息，订单在运输中。"))
    reply = await BaseAgent(llm, PROFILE).run("订单 A12345", CTX)
    assert reply.success and len(reply.tool_traces) == MAX_TOOL_ROUNDS
    assert llm.requests[-1]["tools"] is None


async def test_llm_failure_gives_polite_failure():
    class Broken:
        async def chat(self, *a, **k):
            raise TimeoutError("模型超时")
    reply = await BaseAgent(Broken(), PROFILE).run("你好", CTX)
    assert reply.success is False and "抱歉" in reply.content


async def test_background_is_injected_before_user_message():
    llm = ScriptedLLM(text_msg("好的"))
    await BaseAgent(llm, PROFILE).run("到哪了", CTX, background="意图: order_logistics")
    msgs = llm.requests[0]["messages"]
    assert "order_logistics" in msgs[0]["content"] and msgs[-1]["content"] == "到哪了"

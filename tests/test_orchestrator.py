"""编排器端到端测试：四种路由动作，加上两种降级。"""
from app.orchestrator import CLARIFY_TEXT, TEAMWORK_NOTE, Orchestrator
from tests.fakes import FakeLLM

COMPOSITE = "登录时报 401，同时这笔订单被重复扣款了 198 元"


async def test_single_agent_skips_composer():
    llm = FakeLLM("invoice")
    r = await Orchestrator(llm).handle("帮我开发票", "u1001")
    assert r.response == "billing 的回答" and r.agents_used == ["billing"]
    assert llm.compose_calls == 0 and llm.agent_calls == ["billing"]


async def test_composite_runs_two_agents_and_composes():
    llm = FakeLLM("tech_login")
    r = await Orchestrator(llm).handle(COMPOSITE, "u1001")
    assert (r.decision.primary, r.decision.supporting) == ("technical", ["billing"])
    assert sorted(llm.agent_calls) == ["billing", "technical"]
    assert r.response == "【合并后的回复】" and r.agents_used == ["technical", "billing"]


async def test_escalate_never_calls_agent_llm():
    llm = FakeLLM("human_handoff")
    r = await Orchestrator(llm).handle("转人工，订单号 #A12345", "u1001")
    assert r.decision.action == "escalate" and "工单号" in r.response and "A12345" in r.response
    assert llm.agent_calls == [] and r.agents_used == ["escalation"]


async def test_clarify_uses_fixed_text():
    llm = FakeLLM("other", conf=0.3)
    r = await Orchestrator(llm).handle("嗯", "u1001")
    assert r.response == CLARIFY_TEXT and llm.agent_calls == [] and r.agents_used == []


async def test_primary_failure_falls_back_to_general():
    llm = FakeLLM("tech_error", fail_agents={"technical"})
    r = await Orchestrator(llm).handle("应用一打开就闪退", "u1001")
    assert r.response == "general 的回答" and r.agents_used == ["general"]
    assert r.decision.primary == "technical"          # 决策不变，变的是实际执行者


async def test_composer_failure_falls_back_to_concatenation():
    llm = FakeLLM("tech_login", fail_compose=True)
    r = await Orchestrator(llm).handle(COMPOSITE, "u1001")
    assert r.response.startswith("technical 的回答") and "另外，billing 的回答" in r.response


async def test_teamwork_note_only_when_multiple_agents():
    multi = FakeLLM("tech_login")
    await Orchestrator(multi).handle(COMPOSITE, "u1001")
    assert all(TEAMWORK_NOTE in text for text in multi.agent_inputs.values()) and len(multi.agent_inputs) == 2

    single = FakeLLM("invoice")
    await Orchestrator(single).handle("帮我开发票", "u1001")
    assert TEAMWORK_NOTE not in single.agent_inputs["billing"]

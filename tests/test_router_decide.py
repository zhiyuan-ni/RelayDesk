"""decide() 的规格测试，编号对应 docstring 里的 4 条。"""
from app.intent.schema import Intent, Urgency
from app.routing.router import decide
from tests.test_router_scores import make_intent


def test_1_critical_urgency_escalates_even_for_a_normal_intent():
    msg = "我的账号被盗了，里面还有余额"
    d = decide(make_intent(Intent.ACCOUNT_SECURITY, msg, urgency=Urgency.CRITICAL), msg)
    assert (d.action, d.primary, d.supporting) == ("escalate", "escalation", [])
    assert "CRITICAL" in d.reason and d.scores


def test_2_handoff_intent_escalates():
    msg = "转人工"
    d = decide(make_intent(Intent.HUMAN_HANDOFF, msg, urgency=Urgency.HIGH), msg)
    assert (d.action, d.primary, d.supporting) == ("escalate", "escalation", [])
    assert "human_handoff" in d.reason


def test_3_other_intent_asks_for_clarification():
    d = decide(make_intent(Intent.OTHER, "嗯", conf=0.3), "嗯")
    assert (d.action, d.primary, d.supporting) == ("clarify", "general", [])


def test_4_single_domain():
    msg = "帮我开发票"
    d = decide(make_intent(Intent.INVOICE, msg), msg)
    assert (d.action, d.primary, d.supporting) == ("answer", "billing", [])
    assert "billing" in d.reason


def test_4_composite_gets_supporting_agent():
    msg = "登录时报 401，同时这笔订单被重复扣款了 198 元"
    d = decide(make_intent(Intent.TECH_LOGIN, msg), msg)
    assert (d.action, d.primary, d.supporting) == ("answer", "technical", ["billing"])


def test_4_general_is_never_a_supporting_agent():
    msg = "订单 #A12345 的快递到哪了，另外页面还报错 500"
    d = decide(make_intent(Intent.TECH_ERROR, msg), msg)
    assert d.primary == "technical" and "general" not in d.supporting


def test_4_general_can_be_primary_with_supporting():
    msg = "快递什么时候到？另外我想顺便问下退款"
    d = decide(make_intent(Intent.ORDER_LOGISTICS, msg), msg)
    assert d.primary == "general" and d.supporting == ["billing"]


def test_4_weak_second_domain_is_ignored():
    msg = "应用一打开就闪退"
    d = decide(make_intent(Intent.TECH_ERROR, msg), msg)
    assert d.supporting == []

from app.intent.rules import extract_entities
from app.intent.schema import INTENT_GROUP, Intent, IntentResult, Urgency
from app.routing.router import domain_scores


def make_intent(intent: Intent, message: str, conf: float = 0.9, urgency: Urgency = Urgency.LOW) -> IntentResult:
    return IntentResult(intent, INTENT_GROUP[intent], conf, "llm", urgency, extract_entities(message))


def test_pure_technical():
    msg = "登录一直报 401"
    s = domain_scores(make_intent(Intent.TECH_LOGIN, msg), msg)
    assert s["technical"] > 0.8 and s["billing"] == 0.0


def test_composite_message_lifts_second_domain():
    msg = "登录时报 401，同时这笔订单被重复扣款了 198 元"
    s = domain_scores(make_intent(Intent.TECH_LOGIN, msg), msg)
    assert s["technical"] > s["billing"] >= 0.25  # 账单虽不是主意图，但被关键词和金额实体抬了起来


def test_low_confidence_intent_contributes_less():
    msg = "东西不想要了"
    hi = domain_scores(make_intent(Intent.REFUND, msg, conf=0.9), msg)["billing"]
    lo = domain_scores(make_intent(Intent.REFUND, msg, conf=0.5), msg)["billing"]
    assert hi > lo


def test_keyword_bonus_is_capped():
    msg = "退款 退货 退钱 发票 抬头 税号 扣费"
    s = domain_scores(make_intent(Intent.GREETING, msg, conf=0.0), msg)
    assert s["billing"] == 0.5


def test_general_always_has_a_floor():
    s = domain_scores(make_intent(Intent.OTHER, "嗯", conf=0.0), "嗯")
    assert s == {"general": 0.1, "technical": 0.0, "billing": 0.0}

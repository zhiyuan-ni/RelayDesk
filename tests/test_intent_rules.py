from app.intent.rules import detect_urgency, extract_entities, rule_vote
from app.intent.schema import Intent, Urgency


def test_rule_single_keyword():
    vote = rule_vote("我要退款")
    assert vote.intent == Intent.REFUND
    assert vote.confidence == 0.6


def test_rule_more_hits_higher_confidence():
    vote = rule_vote("应用一直崩溃闪退还报错")
    assert vote.intent == Intent.TECH_ERROR
    assert vote.confidence == 0.9


def test_rule_handoff_wins_tie_by_priority():
    # "投诉" 和 "转人工" 各命中 1 个，转人工排在前面所以赢
    assert rule_vote("我要投诉，转人工").intent == Intent.HUMAN_HANDOFF


def test_rule_no_hit_returns_none():
    assert rule_vote("今天天气不错") is None


def test_entities_basic():
    e = extract_entities("订单号是 #A12345，昨天多扣了 50 块")
    assert e["order_id"] == ["A12345"]
    assert e["amount"] == ["50 块"]
    assert e["date"] == ["昨天"]


def test_error_code_next_to_chinese():
    assert extract_entities("页面报500错误")["error_code"] == ["500"]
    assert extract_entities("登录一直 401")["error_code"] == ["401"]


def test_amount_is_not_mistaken_for_error_code():
    e = extract_entities("这个月多扣了 500 元")
    assert e["error_code"] == []
    assert e["amount"] == ["500 元"]
    assert extract_entities("收了我 ￥404")["error_code"] == []


def test_urgency():
    assert detect_urgency("我的账号被盗了", Intent.ACCOUNT_SECURITY) == Urgency.CRITICAL
    assert detect_urgency("麻烦尽快处理", Intent.REFUND) == Urgency.HIGH
    assert detect_urgency("转人工", Intent.HUMAN_HANDOFF) == Urgency.HIGH
    assert detect_urgency("帮我开发票", Intent.INVOICE) == Urgency.LOW

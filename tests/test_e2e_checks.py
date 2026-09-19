"""check_turn 的规格测试，加上评委输出解析。"""
from app.evals.e2e import TurnExpectation, TurnOutcome, check_turn
from app.evals.judge import parse_score


def outcome(**kw) -> TurnOutcome:
    base = dict(response="请提供订单号，我帮您查询。", action="answer", primary="billing",
                supporting=[], tools_called=[])
    base.update(kw)
    return TurnOutcome(**base)


def test_all_pass_returns_empty_list():
    exp = TurnExpectation(action="answer", primary="billing", contain_any=["订单号"], tools_none=["*"])
    assert check_turn(outcome(), exp) == []


def test_1_2_action_and_primary():
    fails = check_turn(outcome(action="clarify", primary="general"), TurnExpectation(action="answer", primary="billing"))
    assert len(fails) == 2 and any("billing" in f and "general" in f for f in fails)


def test_3_4_supporting_and_tools_each_missing_one_line():
    exp = TurnExpectation(supporting=["technical"], tools=["get_payment_records", "lookup_error_code"])
    fails = check_turn(outcome(tools_called=["get_payment_records"]), exp)
    assert len(fails) == 2
    assert any("technical" in f for f in fails) and any("lookup_error_code" in f for f in fails)


def test_5_tools_none_star_lists_what_was_called():
    fails = check_turn(outcome(tools_called=["get_order_status"]), TurnExpectation(tools_none=["*"]))
    assert len(fails) == 1 and "get_order_status" in fails[0]


def test_5_tools_none_specific():
    exp = TurnExpectation(tools_none=["get_payment_records", "get_refund_status"])
    fails = check_turn(outcome(tools_called=["get_payment_records", "search_knowledge_base"]), exp)
    assert len(fails) == 1 and "get_payment_records" in fails[0]


def test_6_contain_any_needs_just_one():
    assert check_turn(outcome(response="订单编号是多少"), TurnExpectation(contain_any=["订单号", "订单编号"])) == []
    fails = check_turn(outcome(response="您好"), TurnExpectation(contain_any=["订单号", "订单编号"]))
    assert len(fails) == 1


def test_7_8_contain_all_and_none():
    exp = TurnExpectation(contain_all=["工单号", "A12345"], contain_none=["密码", "验证码"])
    fails = check_turn(outcome(response="已生成工单号 T1。请提供验证码"), exp)
    assert len(fails) == 2
    assert any("A12345" in f for f in fails) and any("验证码" in f for f in fails)


def test_all_failures_are_reported_not_just_the_first():
    exp = TurnExpectation(action="escalate", primary="escalation", tools_none=["*"], contain_all=["工单号"])
    fails = check_turn(outcome(tools_called=["x"]), exp)
    assert len(fails) == 4


def test_expectation_from_dict_ignores_unknown_keys():
    exp = TurnExpectation.from_dict({"action": "answer", "note": "忽略我", "contain_any": ["a"]})
    assert exp.action == "answer" and exp.contain_any == ["a"]


def test_judge_parse_handles_code_fence_and_rejects_bad_scores():
    s = parse_score('```json\n{"factual": 5, "helpful": 4, "policy": 3, "reason": "无"}\n```')
    assert (s.factual, s.helpful, s.policy, s.overall) == (5, 4, 3, 4.0)
    assert parse_score('{"factual": 9, "helpful": 4, "policy": 3}') is None
    assert parse_score("我觉得挺好的") is None


def test_9_agents_in_any_order():
    exp = TurnExpectation(agents=["billing", "technical"])
    assert check_turn(outcome(primary="technical", supporting=["billing"]), exp) == []
    assert check_turn(outcome(primary="billing", supporting=["technical"]), exp) == []
    fails = check_turn(outcome(primary="technical", supporting=[]), exp)
    assert len(fails) == 1 and "billing" in fails[0]

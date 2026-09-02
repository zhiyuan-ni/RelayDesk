"""fuse() 的规格测试。编号对应 recognizer.py 里 docstring 的 6 条规则。"""
from app.intent.recognizer import fuse
from app.intent.schema import Intent, Vote


def test_1_no_votes():
    assert fuse(None, None) == (Intent.OTHER, 0.0, "none")


def test_2_rule_only_when_llm_failed():
    assert fuse(None, Vote(Intent.REFUND, 0.6)) == (Intent.REFUND, 0.6, "rule")


def test_3_llm_only():
    assert fuse(Vote(Intent.INVOICE, 0.8), None) == (Intent.INVOICE, 0.8, "llm")


def test_4_agree_gets_bonus():
    intent, conf, source = fuse(Vote(Intent.REFUND, 0.8), Vote(Intent.REFUND, 0.6))
    assert (intent, source) == (Intent.REFUND, "both")
    assert abs(conf - 0.9) < 1e-9  # 浮点数不要用 == 比较


def test_4_agree_bonus_capped_at_one():
    _, conf, _ = fuse(Vote(Intent.REFUND, 0.95), Vote(Intent.REFUND, 0.6))
    assert conf == 1.0


def test_5_conflict_confident_llm_wins():
    assert fuse(Vote(Intent.PAYMENT_ISSUE, 0.85), Vote(Intent.REFUND, 0.6)) == (
        Intent.PAYMENT_ISSUE, 0.85, "llm")


def test_5_conflict_unsure_llm_loses_to_rule():
    assert fuse(Vote(Intent.COMPLAINT, 0.4), Vote(Intent.REFUND, 0.75)) == (
        Intent.REFUND, 0.75, "rule")


def test_6_low_confidence_becomes_other():
    assert fuse(Vote(Intent.INVOICE, 0.3), None) == (Intent.OTHER, 0.3, "llm")

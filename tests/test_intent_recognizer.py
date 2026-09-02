"""端到端测试识别器：用假 LLM 控制模型那一票，验证整条链路的输出。"""
import json

from app.intent.recognizer import IntentRecognizer
from app.intent.schema import Group, Intent, Urgency


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    async def chat_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def reply(intent: str, conf: float) -> str:
    return "好的，结果如下：" + json.dumps({"intent": intent, "confidence": conf, "reasoning": "测试"})


async def test_llm_and_rule_agree():
    r = await IntentRecognizer(FakeLLM(reply("refund", 0.8))).recognize("我要退款，订单号 #A12345")
    assert (r.intent, r.group, r.source) == (Intent.REFUND, Group.BILLING, "both")
    assert r.confidence == 0.9
    assert r.entities["order_id"] == ["A12345"]


async def test_llm_down_falls_back_to_rule():
    r = await IntentRecognizer(FakeLLM(error=TimeoutError("超时"))).recognize("登录失败，一直 401")
    assert (r.intent, r.group, r.source) == (Intent.TECH_LOGIN, Group.TECHNICAL, "rule")
    assert r.entities["error_code"] == ["401"]


async def test_llm_returns_garbage():
    r = await IntentRecognizer(FakeLLM("我不知道")).recognize("今天天气不错")
    assert (r.intent, r.group, r.source) == (Intent.OTHER, Group.NONE, "none")


async def test_llm_understands_what_rules_cannot():
    r = await IntentRecognizer(FakeLLM(reply("account_security", 0.9))).recognize("好像有别人在用我的号，被盗了吗")
    assert r.intent == Intent.ACCOUNT_SECURITY
    assert r.group == Group.TECHNICAL
    assert r.urgency == Urgency.CRITICAL


async def test_history_is_put_into_prompt():
    llm = FakeLLM(reply("refund", 0.8))
    await IntentRecognizer(llm).recognize("订单号是 12345", history=[{"role": "user", "content": "我想退款"}])
    assert "我想退款" in llm.prompts[0]

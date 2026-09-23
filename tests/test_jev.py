"""jev 意图后端：客户端请求格式、答案换算、识别器二选一、配置校验。全部离线，用假传输层代替网络。"""
import json

import httpx
import pytest

from app.config import Settings, load_settings
from app.intent.jev_classifier import answer_to_vote, build_question, build_state, jev_vote
from app.intent.recognizer import IntentRecognizer
from app.intent.schema import Intent
from app.jev import JevClient, open_jev


def choice(intent: str, conf: float, probs: dict | None = None) -> dict:
    return {"type": "choice", "choice": intent, "confidence": conf,
            "probabilities": probs or {intent: conf, "other": round(1 - conf, 2)}}


class FakeJev:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    async def ask(self, state, questions):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return {"intent": self.answer}


class ExplodingLLM:
    """用 jev 时不该碰到 LLM，一旦被调用就让测试失败。"""
    async def chat_text(self, prompt, **kwargs):
        raise AssertionError("选了 jev 后端却调用了 LLM")


# ── 题目与状态 ──

def test_question_covers_every_intent_with_a_description():
    q = build_question()
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {i.value for i in Intent}
    assert all(desc.strip() for desc in q["criteria"].values())


def test_state_keeps_only_last_three_history_messages():
    history = [{"role": "user", "content": f"第{i}句"} for i in range(5)]
    state = build_state("订单号是 12345", history)
    assert state["最新消息"] == "订单号是 12345"
    assert state["最近对话"] == ["user: 第2句", "user: 第3句", "user: 第4句"]
    assert "最近对话" not in build_state("你好")


# ── 答案换算 ──

def test_answer_to_vote_uses_confidence_and_explains_top_two():
    vote = answer_to_vote(choice("refund", 0.62, {"refund": 0.7, "order_logistics": 0.25, "other": 0.05}))
    assert vote.intent == Intent.REFUND
    assert vote.confidence == 0.62                  # 用 confidence 字段，不是 probabilities 里的 0.7
    assert "refund 0.70" in vote.reasoning and "order_logistics 0.25" in vote.reasoning


@pytest.mark.parametrize("bad", [
    {},                                             # 空答案，比如返回里缺了这道题
    {"choice": "shopping", "confidence": 0.9},      # 不在枚举里的意图
    {"choice": "refund"},                           # 缺置信度
    {"choice": "refund", "confidence": "高"},       # 置信度不是数字
])
def test_answer_to_vote_rejects_malformed_answers(bad):
    assert answer_to_vote(bad) is None


def test_answer_to_vote_clamps_confidence():
    assert answer_to_vote(choice("greeting", 1.2)).confidence == 1.0


async def test_jev_vote_returns_none_on_error():
    assert await jev_vote(FakeJev(error=httpx.ConnectTimeout("超时")), "你好") is None


# ── 识别器二选一 ──

async def test_recognizer_uses_jev_when_given():
    jev = FakeJev(choice("account_security", 0.95))
    r = await IntentRecognizer(ExplodingLLM(), jev=jev).recognize("好像有别人在用我的号")
    assert r.intent == Intent.ACCOUNT_SECURITY
    assert r.reasoning.startswith("jev:")
    assert len(jev.calls) == 1


async def test_recognizer_falls_back_to_rules_when_jev_fails():
    jev = FakeJev(error=httpx.ReadTimeout("超时"))
    r = await IntentRecognizer(ExplodingLLM(), jev=jev).recognize("登录失败，一直 401")
    assert (r.intent, r.source) == (Intent.TECH_LOGIN, "rule")


async def test_low_jev_confidence_goes_to_clarify():
    # jev 的置信度是校准过的，低于 MIN_CONF 时识别结果改为 other，由下游反问
    r = await IntentRecognizer(ExplodingLLM(), jev=FakeJev(choice("refund", 0.3))).recognize("那个怎么弄")
    assert r.intent == Intent.OTHER


# ── 客户端 ──

def cfg(**kw) -> Settings:
    return Settings(llm_api_key="sk-test", llm_base_url="https://relay.example/v1", llm_model="m", **kw)


async def test_client_sends_systemone_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"intent": choice("invoice", 0.9)},
                                         "usage": {"input_tokens": 10, "output_tokens": 5}})

    client = JevClient(cfg(), transport=httpx.MockTransport(handler))
    answers = await client.ask({"最新消息": "开票"}, {"intent": build_question()})
    await client.aclose()

    assert seen["url"] == "https://relay.example/v1/systemone"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "jev-1.13"
    assert answers["intent"]["choice"] == "invoice"


async def test_client_raises_on_http_error_and_warmup_reports_failure():
    transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad key"}))
    client = JevClient(cfg(), transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        await client.ask("你好", {})
    assert await client.warmup() is False   # 预热失败只返回 False，不抛异常，不阻止启动
    await client.aclose()


def test_client_honors_proxy_env(monkeypatch):
    # 回归测试：自定义 transport 会让 httpx 忽略代理环境变量，而访问中转站必须走代理
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    client = JevClient(cfg())
    assert any(key.pattern.startswith("https") for key in client._http._mounts)


# ── 配置 ──

def test_intent_backend_defaults_to_llm(monkeypatch):
    monkeypatch.delenv("INTENT_BACKEND", raising=False)
    assert load_settings().intent_backend == "llm"


def test_intent_backend_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("INTENT_BACKEND", " JEV ")
    assert load_settings().intent_backend == "jev"


@pytest.mark.parametrize("name", ["INTENT_BACKEND", "RERANK_BACKEND"])
def test_invalid_backend_fails_fast(monkeypatch, name):
    monkeypatch.setenv(name, "jve")
    with pytest.raises(RuntimeError, match=name):
        load_settings()


def test_rerank_backend_defaults_to_llm(monkeypatch):
    monkeypatch.delenv("RERANK_BACKEND", raising=False)
    assert load_settings().rerank_backend == "llm"


async def test_open_jev_only_when_some_step_uses_it(monkeypatch):
    assert await open_jev(cfg()) is None                    # 两个环节都用 llm，不建客户端
    warmed = []

    async def fake_warmup(self):
        warmed.append(self.model)
        return True
    monkeypatch.setattr(JevClient, "warmup", fake_warmup)
    client = await open_jev(cfg(rerank_backend="jev"))
    assert isinstance(client, JevClient) and warmed == ["jev-1.13"]
    await client.aclose()

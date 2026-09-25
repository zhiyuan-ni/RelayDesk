"""接口层测试：验证 /chat 的输入输出形状，不调用真实模型。"""
from fastapi.testclient import TestClient

from app.main import app, get_orchestrator
from app.orchestrator import Orchestrator
from tests.fakes import FakeLLM


def make_client(fake: FakeLLM) -> TestClient:
    orch = Orchestrator(fake)
    app.dependency_overrides[get_orchestrator] = lambda: orch
    return TestClient(app)  # 不用 with，不触发 lifespan，所以不需要真实 key


def test_health():
    assert make_client(FakeLLM()).get("/health").json()["status"] == "ok"


def test_chat_basic_shape():
    body = make_client(FakeLLM("greeting")).post("/chat", json={"message": "你好"}).json()
    assert body["response"] == "general 的回答"
    assert len(body["conv_id"]) == 12 and len(body["request_id"]) == 8
    assert (body["action"], body["primary_agent"]) == ("answer", "general")


def test_chat_keeps_existing_conv_id():
    body = make_client(FakeLLM()).post("/chat", json={"message": "在吗", "conv_id": "abc"}).json()
    assert body["conv_id"] == "abc"


def test_chat_exposes_intent_and_routing():
    body = make_client(FakeLLM("refund")).post(
        "/chat", json={"message": "我要退款，订单号 #A12345，尽快"}).json()
    assert (body["intent"], body["intent_group"], body["intent_source"]) == ("refund", "billing", "both")
    assert body["urgency"] == "HIGH" and body["entities"]["order_id"] == ["A12345"]
    assert body["primary_agent"] == "billing" and body["routing_scores"]["billing"] > 0.5


def test_for_model_shares_client_and_drops_thinking_flag_for_other_vendors():
    from app.config import Settings
    from app.llm import LLMClient
    base = LLMClient(Settings("k", "https://example.com/v1", "qwen3.8-flash", llm_enable_thinking=False))
    same = base.for_model("")
    qwen = base.for_model("qwen-turbo")
    other = base.for_model("gpt-4.1-nano")
    assert same is base
    assert qwen.client is base.client and qwen.model == "qwen-turbo" and qwen._enable_thinking is False
    assert other.client is base.client and other._enable_thinking is None


def test_chat_exposes_debug_fields():
    body = make_client(FakeLLM("refund")).post("/chat", json={"message": "我要退款"}).json()
    assert body["short_term"]["enabled"] is False   # 测试里的编排器没有接记忆
    assert body["timeline"] and {"name", "start_ms", "duration_ms"} <= set(body["timeline"][0])


def test_debug_page_is_served():
    res = make_client(FakeLLM()).get("/debug")
    assert res.status_code == 200 and "RelayDesk 调试面板" in res.text


def test_metrics_include_each_stage():
    client = make_client(FakeLLM("refund"))
    client.post("/chat", json={"message": "我要退款"})
    stages = client.get("/metrics").json()["stage"]
    assert {"request", "intent"} <= set(stages)
    assert not any(name.startswith(("agent:", "tool:")) for name in stages)   # 这两类单独统计


def test_clients_keep_idle_connections_alive():
    # httpx 默认闲置 5 秒就断开，经代理重新握手要 6 秒以上。这里防止连接池参数被无意中改回默认值
    from app.config import Settings
    from app.jev import JevClient
    from app.llm import LLMClient
    cfg = Settings("k", "https://example.com/v1", "qwen3.8-flash")
    llm_pool = LLMClient(cfg).client._client._transport._pool
    jev_pool = JevClient(cfg)._http._transport._pool
    assert llm_pool._keepalive_expiry == jev_pool._keepalive_expiry == cfg.http_keepalive_s == 60.0

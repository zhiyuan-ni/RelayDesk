"""阶段 0 的测试：不调用真实模型，验证接口的输入输出形状。"""
from fastapi.testclient import TestClient

from app.main import app, get_llm


class FakeLLM:
    """假的 LLM：记录收到的参数，返回固定文本。"""

    def __init__(self):
        self.calls = []

    async def chat_text(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return f"收到：{prompt}"


def make_client(fake: FakeLLM) -> TestClient:
    app.dependency_overrides[get_llm] = lambda: fake
    # 不用 with TestClient(app)，这样不会触发 lifespan，也就不需要真实 key
    return TestClient(app)


def test_health():
    client = make_client(FakeLLM())
    assert client.get("/health").json()["status"] == "ok"


def test_chat_returns_answer_and_generates_conv_id():
    fake = FakeLLM()
    body = make_client(fake).post("/chat", json={"message": "你好"}).json()

    assert body["response"] == "收到：你好"
    assert len(body["conv_id"]) == 12
    assert body["latency_ms"] >= 0
    assert "system" in fake.calls[0]  # 确认系统提示词确实传给了模型


def test_chat_keeps_existing_conv_id():
    body = make_client(FakeLLM()).post(
        "/chat", json={"message": "在吗", "conv_id": "abc"}
    ).json()
    assert body["conv_id"] == "abc"

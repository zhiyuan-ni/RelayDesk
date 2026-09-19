"""测试共用的假模型。

并行执行时各 Agent 的调用顺序不固定，所以不能按"第几次调用"来安排返回值，
而是根据请求内容判断是谁在调用：意图识别、某个 Agent，还是合并器。
"""
import json
from types import SimpleNamespace as NS


class FakeLLM:
    def __init__(self, intent="greeting", conf=0.9, fail_agents=(), fail_compose=False):
        self.intent, self.conf = intent, conf
        self.fail_agents, self.fail_compose = set(fail_agents), fail_compose
        self.agent_calls: list[str] = []
        self.agent_inputs: dict[str, str] = {}   # 每个 Agent 收到的全部消息文本，拼成一个字符串
        self.compose_calls = 0

    async def chat_text(self, prompt, **kwargs):
        if "意图分类器" in prompt:
            return json.dumps({"intent": self.intent, "confidence": self.conf, "reasoning": "fake"})
        self.compose_calls += 1          # 其余的 chat_text 调用只有合并器
        if self.fail_compose:
            raise TimeoutError("合并超时")
        return "【合并后的回复】"

    async def chat(self, messages, system="", **kwargs):
        name = "technical" if "技术支持" in system else "billing" if "账单专员" in system else "general"
        self.agent_calls.append(name)
        self.agent_inputs[name] = "\n".join(str(m.get("content", "")) for m in messages)
        if name in self.fail_agents:
            raise TimeoutError(f"{name} 超时")
        return NS(content=f"{name} 的回答", tool_calls=None)

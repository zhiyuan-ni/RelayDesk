"""Agent 基类：角色设定 + 工具白名单 + tool-use 循环。"""
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.tools import ToolContext, ToolSpec, pick_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3  # 最多让模型连续调 3 轮工具，防止死循环烧钱


@dataclass(frozen=True)
class AgentProfile:
    name: str                      # general / technical / billing
    role: str                      # 一句话角色
    rules: tuple[str, ...]         # 行为边界，会写进 system prompt
    tool_names: tuple[str, ...]    # 工具白名单
    temperature: float = 0.2
    max_tokens: int = 900

    def system_prompt(self) -> str:
        rules = "\n".join(f"- {r}" for r in self.rules)
        return (
            f"你是 RelayDesk 的{self.role}。\n\n行为规则：\n{rules}\n"
            "- 涉及订单、支付、登录记录等事实，必须先调用工具查询，严禁编造。\n"
            "- 工具查不到时如实告知，并说明需要用户补充什么信息。\n"
            "- 回复简洁，直接面向用户，不要提及工具名称或内部流程。"
        )


@dataclass
class AgentReply:
    agent: str
    content: str
    success: bool
    latency_ms: float = 0.0
    tool_traces: list[dict[str, Any]] = field(default_factory=list)


async def execute_tool_call(
    tools: dict[str, ToolSpec], name: str, raw_args: str, ctx: ToolContext
) -> dict[str, Any]:
    """执行模型要求的一次工具调用，返回一条 trace 字典。永远不向外抛异常。

    ───────────── 练习：请你实现 ─────────────
    参数：
      tools     当前 Agent 的白名单，键是工具名
      name      模型想调用的工具名
      raw_args  模型给的参数，是一个 JSON 字符串，例如 '{"order_id": "A12345"}'
      ctx       请求上下文，原样传给工具函数

    返回的字典必须包含这 6 个键：
      {"tool": name, "args": 解析后的参数字典, "success": 布尔,
       "data": 工具返回值或 None, "error": 错误说明或 "", "latency_ms": 耗时}
    """
    t0 = time.monotonic()
    trace = {"tool": name, "args": {}, "success": False,
             "data": None, "error": "", "latency_ms": 0.0}

    def done():
        trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return trace

    # 白名单
    if name not in tools:
        trace["error"] = f"工具 {name} 不在当前 Agent 的白名单中"
        return done()

    # 解析 JSON
    if raw_args:
        try:
            args = json.loads(raw_args)      # 字符串 -> Python 对象
        except json.JSONDecodeError:
            trace["error"] = "无法解析 JSON"
            return done()
        if not isinstance(args, dict):       # 解析成功，但可能是数字或列表
            trace["error"] = "参数 JSON 必须是对象"
            return done()
        trace["args"] = args

    # 检查必填参数
    for p in tools[name].parameters["required"]:
        if p not in trace["args"]:
            trace["error"] = f"缺少必填参数名 {p}"
            return done()

    # 调用工具函数
    try:
        result = tools[name].handler(ctx, trace["args"])
        if inspect.isawaitable(result):
            result = await result
    except Exception as ex:
        trace["error"] = str(ex)
        return done()
    
    trace["data"] = result
    trace["success"] = True
    return done()


class BaseAgent:
    def __init__(self, llm, profile: AgentProfile):
        self._llm = llm
        self.profile = profile
        self._tools = pick_tools(profile.tool_names)

    async def run(self, message: str, ctx: ToolContext, background: str = "") -> AgentReply:
        """处理一条用户消息。background 是意图、实体、记忆等背景信息的文本。"""
        t0 = time.monotonic()
        traces: list[dict[str, Any]] = []
        try:
            content = await self._loop(message, ctx, background, traces)
            ok = True
        except Exception as ex:
            logger.exception("%s agent 失败", self.profile.name)
            content, ok = "抱歉，处理您的请求时出现问题，请稍后重试。", False
        return AgentReply(self.profile.name, content, ok,
                          round((time.monotonic() - t0) * 1000, 1), traces)

    async def _loop(self, message: str, ctx: ToolContext, background: str, traces: list) -> str:
        messages: list[dict[str, Any]] = []
        if background:
            messages.append({"role": "user", "content": f"[背景信息，供参考]\n{background}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景。"})
        messages.append({"role": "user", "content": message})
        tool_defs = [t.to_openai() for t in self._tools.values()]

        for round_no in range(MAX_TOOL_ROUNDS + 1):
            # 最后一轮不再提供工具，逼模型用已有信息作答。
            # 如果在这里直接报错，前几轮已经查到的信息就全部浪费了。
            offer_tools = tool_defs if round_no < MAX_TOOL_ROUNDS else None
            msg = await self._llm.chat(
                messages, system=self.profile.system_prompt(), tools=offer_tools,
                temperature=self.profile.temperature, max_tokens=self.profile.max_tokens,
            )
            if not msg.tool_calls:
                return (msg.content or "").strip()

            # 先把"模型的调用请求"原样记入对话，再逐个追加工具结果，顺序不能反
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in msg.tool_calls],
            })
            for call in msg.tool_calls:
                trace = await execute_tool_call(self._tools, call.function.name, call.function.arguments, ctx)
                traces.append(trace)
                # 失败也要告诉模型，它才能换个办法或如实告知用户
                payload = trace["data"] if trace["success"] else {"error": trace["error"]}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(payload, ensure_ascii=False)})
        return ""  # 理论上到不了这里

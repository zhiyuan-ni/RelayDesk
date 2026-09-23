"""Agent 基类：角色设定 + 工具白名单 + tool-use 循环。"""
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.agents.tools import ToolContext, ToolSpec, pick_tools
from app.observability import tracer
from app.observability.stats import stats

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3  # 最多让模型连续调 3 轮工具，防止死循环烧钱


@dataclass(frozen=True)
class AgentProfile:
    name: str                      # general / technical / billing
    role: str                      # 一句话角色
    rules: tuple[str, ...]         # 行为边界，会写进 system prompt
    tool_names: tuple[str, ...]    # 工具白名单
    temperature: float = 0.2
    max_tokens: int = 600
    max_chars: int = 220       # 回复字数上限，写进提示词。实测延迟几乎全在输出 token 上，回复越长越慢

    def system_prompt(self) -> str:
        rules = "\n".join(f"- {r}" for r in self.rules)
        return (
            f"你是 RelayDesk 的{self.role}。\n\n行为规则：\n{rules}\n"
            "- 涉及订单、支付、登录记录等事实，必须先调用工具查询，严禁编造。\n"
            "- 工具查不到时如实告知，并说明需要用户补充什么信息。\n"
            "- 回复简洁，直接面向用户，不要提及工具名称或内部流程。\n"
            f"- 回复控制在 {self.max_chars} 字以内。先给结论，再给必要的依据和下一步；"
            "不要用标题，不要逐项复述查到的每个字段，只说与用户问题直接相关的。"
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

    # 1. 白名单
    if name not in tools:
        trace["error"] = f"工具 {name} 不在当前 Agent 的白名单中"
        return done()

    # 2. 解析 JSON
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

    # 3. 检查必填参数
    for p in tools[name].parameters["required"]:
        if p not in trace["args"]:
            trace["error"] = f"缺少必填参数名 {p}"
            return done()

    # 4. 调用工具函数
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
    def __init__(self, llm, profile: AgentProfile, shared_tools: Optional[dict[str, ToolSpec]] = None):
        self._llm = llm
        self.profile = profile
        # 工具 = 角色专属的白名单工具 + 所有 Agent 共享的工具，例如知识库检索
        self._tools = {**pick_tools(profile.tool_names), **(shared_tools or {})}

    async def run(self, message: str, ctx: ToolContext, background: str = "",
                  history: Optional[list[dict[str, str]]] = None) -> AgentReply:
        """处理一条用户消息。

        background: 意图、实体、会话摘要等背景信息的文本
        history:    本次会话最近几轮的原始对话，按时间顺序
        """
        t0 = time.monotonic()
        traces: list[dict[str, Any]] = []
        try:
            content = await self._loop(message, ctx, background, traces, history or [])
            ok = True
        except Exception as ex:
            logger.exception("%s agent 失败", self.profile.name)
            content, ok = "抱歉，处理您的请求时出现问题，请稍后重试。", False
        return AgentReply(self.profile.name, content, ok,
                          round((time.monotonic() - t0) * 1000, 1), traces)

    async def _loop(self, message: str, ctx: ToolContext, background: str, traces: list,
                    history: list[dict[str, str]]) -> str:
        messages: list[dict[str, Any]] = []
        if background:
            messages.append({"role": "user", "content": f"[背景信息，供参考]\n{background}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景。"})
        # 历史对话作为真正的多轮消息传入，而不是拼成一段文字。
        # 模型对"谁说了什么"的理解，建立在 user 和 assistant 交替的消息结构上
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        tool_defs = [t.to_openai() for t in self._tools.values()]

        for round_no in range(MAX_TOOL_ROUNDS + 1):
            # 最后一轮不再提供工具，逼模型用已有信息作答。
            # 如果在这里直接报错，前几轮已经查到的信息就全部浪费了。
            offer_tools = tool_defs if round_no < MAX_TOOL_ROUNDS else None
            async with tracer.span(f"llm:{self.profile.name}", round=round_no + 1):
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
                async with tracer.span(f"tool:{call.function.name}"):
                    trace = await execute_tool_call(self._tools, call.function.name, call.function.arguments, ctx)
                stats.record("tool", call.function.name, trace["latency_ms"], trace["success"])
                traces.append(trace)
                # 失败也要告诉模型，它才能换个办法或如实告知用户
                payload = trace["data"] if trace["success"] else {"error": trace["error"]}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(payload, ensure_ascii=False)})
        return ""  # 理论上到不了这里

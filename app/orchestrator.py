"""编排器：把意图识别、路由、Agent 执行、结果合并串成一条流水线。

  消息 -> 识别意图 -> 路由决策 -+-> clarify   固定话术反问
                               +-> escalate  升级节点，不调模型
                               +-> answer    主 Agent 和辅助 Agent 并行 -> 失败兜底 -> 合并
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app.agents.base import AgentReply, BaseAgent
from app.agents.escalation import EscalationAgent
from app.agents.profiles import PROFILES
from app.agents.tools import ToolContext, ToolSpec
from app.intent.recognizer import IntentRecognizer
from app.intent.schema import IntentResult
from app.memory.manager import MemoryContext
from app.routing.router import RoutingDecision, decide

logger = logging.getLogger(__name__)

CLARIFY_TEXT = "我还不太确定您想处理哪类问题。方便说一下是订单物流、退换货或退款、扣款、发票，还是登录和报错方面的问题吗？"


# 多个 Agent 并行时每人都会看到完整的用户消息。不加这段说明，账单 Agent 会对登录问题回一句
# "这不归我管，要帮你转吗"，合并后的回复就自相矛盾了。
TEAMWORK_NOTE = (
    "协作说明: 这条消息涉及多个业务域，其他域已有同事在同时处理。"
    "你只回答属于你职责范围的那部分；对其余部分不要回应，不要说无法处理，也不要提议转接。"
)


@dataclass
class OrchestratorResult:
    request_id: str
    response: str
    intent: IntentResult
    decision: RoutingDecision
    agents_used: list[str] = field(default_factory=list)   # 实际产出了有效回复的 Agent
    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    memories_used: list[str] = field(default_factory=list)   # 本次从长期记忆里想起的内容
    latency_ms: float = 0.0


async def run_agents(
    agents: dict[str, Any], names: list[str], message: str, ctx: ToolContext, background: str,
    history: Optional[list[dict[str, str]]] = None,
) -> list[AgentReply]:
    """并行运行多个 Agent，返回与 names 顺序一致的回复列表。

    ───────────── 练习：请你实现 ─────────────
    这就是异步课里 gather 的实战。规格对应 tests/test_run_agents.py：
      1. 并行：所有 Agent 同时开始。两个各耗时 0.2 秒的 Agent，总耗时应接近 0.2 秒而不是 0.4 秒
      2. 保序：返回列表的顺序和 names 一致。gather 本身就保证这一点
      3. 容错：某个 Agent 抛异常时，不能影响其他 Agent，
               该位置放一个 AgentReply(名字, "", False) 表示失败

    提示：
      - 每个 Agent 的调用方式： agents[name].run(message, ctx, background)
        它返回的是协程，先不要 await，收集成列表交给 gather
      - results = await asyncio.gather(*协程列表, return_exceptions=True)
        星号的作用是把列表拆开成多个参数
      - 用 zip(names, results) 同时遍历名字和结果
      - 用 isinstance(r, Exception) 判断这个位置是不是异常
    大约 8 行。
    """
    # 这里只是创建协程，还没有开始执行。不能写 await，否则会变成一个接一个地串行。
    coros = [agents[name].run(message, ctx, background, history) for name in names]

    # gather 让它们同时开始，并按传入顺序返回结果。
    # return_exceptions=True：某个 Agent 抛异常时，异常对象会出现在结果列表的对应位置，不影响其他 Agent。
    results = await asyncio.gather(*coros, return_exceptions=True)

    replies: list[AgentReply] = []
    for name, result in zip(names, results):
        # 用 BaseException 而不是 Exception：任务被取消时的 CancelledError 属于前者，也要当失败处理
        if isinstance(result, BaseException):
            logger.error("agent %s 执行失败: %r", name, result)
            replies.append(AgentReply(agent=name, content="", success=False))
        else:
            replies.append(result)
    return replies

class Orchestrator:
    def __init__(self, llm, shared_tools: Optional[dict[str, ToolSpec]] = None, memory=None):
        self._llm = llm
        self._memory = memory   # 为 None 时系统无记忆，每条消息独立处理
        self._recognizer = IntentRecognizer(llm)
        self._agents: dict[str, Any] = {
            name: BaseAgent(llm, profile, shared_tools) for name, profile in PROFILES.items()
        }
        self._escalation = EscalationAgent()

    async def handle(self, message: str, user_id: str, conv_id: str = "") -> OrchestratorResult:
        t0 = time.monotonic()
        request_id = uuid.uuid4().hex[:8]

        mem = await self._load_memory(user_id, conv_id)
        history = mem.history()

        # 意图识别和长期记忆检索互不依赖，同时进行。前者约 1.5 秒，后者约 1 秒，并行后不增加总耗时。
        # 意图识别带上最近两轮。"订单号是 A12345"这种话，脱离上文无法判断用户想干什么
        intent, memories = await asyncio.gather(
            self._recognizer.recognize(message, history[-4:] or None),
            self._recall(user_id, conv_id, message),
        )
        decision = decide(intent, message)
        # 只收集用户说过的话，不含助手的回复。否则模型上一轮编出来的订单号，下一轮就变成"出现过"了
        # 从长期记忆想起的内容也算在内：那是用户在以往会话里亲口说过的。
        # 否则用户问"上次那个订单怎么样了"，系统明明想起了订单号，却会被溯源校验挡住不让查
        said_by_user = " ".join([m["content"] for m in history if m["role"] == "user"] + [message] + memories)
        ctx = ToolContext(user_id=user_id, entities=intent.entities, conversation_text=said_by_user)
        logger.info("[%s] %s", request_id, decision.reason)

        if decision.action == "clarify":
            response, replies = CLARIFY_TEXT, []
        elif decision.action == "escalate":
            reply = await self._escalation.run(message, ctx)
            response, replies = reply.content, [reply]
        else:
            names = [decision.primary] + decision.supporting
            background = self._background(intent, mem.summary, memories)
            if len(names) > 1:
                background += "\n" + TEAMWORK_NOTE
            replies = await run_agents(self._agents, names, message, ctx, background, history)
            replies = await self._fallback_if_primary_failed(replies, decision, message, ctx, background, history)
            response = await self._compose(message, replies)

        await self._save_memory(user_id, conv_id, message, response)

        return OrchestratorResult(
            request_id=request_id,
            response=response,
            intent=intent,
            decision=decision,
            agents_used=[r.agent for r in replies if r.success],
            memories_used=memories,
            tool_traces=[{"agent": r.agent, **t} for r in replies for t in r.tool_traces],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    async def _load_memory(self, user_id: str, conv_id: str) -> MemoryContext:
        """记忆读取失败不应该让整个请求失败。读不到就当作新对话处理。"""
        if self._memory is None or not conv_id:
            return MemoryContext()
        try:
            return await self._memory.load(user_id, conv_id)
        except Exception as ex:
            logger.warning("读取会话记忆失败，按无记忆处理: %r", ex)
            return MemoryContext()

    async def _save_memory(self, user_id: str, conv_id: str, message: str, response: str) -> None:
        if self._memory is None or not conv_id:
            return
        try:
            await self._memory.save_turn(user_id, conv_id, message, response)
        except Exception as ex:
            logger.warning("写入会话记忆失败: %r", ex)

    async def _recall(self, user_id: str, conv_id: str, message: str) -> list[str]:
        if self._memory is None or not conv_id or not hasattr(self._memory, "recall"):
            return []
        try:
            return await self._memory.recall(user_id, conv_id, message)
        except Exception as ex:
            # 它和意图识别一起放在 gather 里，这里不兜住的话，想不起往事会连累整个请求失败
            logger.warning("检索长期记忆失败，按没有相关记忆处理: %r", ex)
            return []

    @staticmethod
    def _background(intent: IntentResult, summary: str = "", memories: Optional[list[str]] = None) -> str:
        """把意图识别的结构化结果转成文字交给 Agent，省得它自己再从原话里猜一遍。"""
        entities = {k: v for k, v in intent.entities.items() if v}
        text = (f"意图: {intent.intent.value}\n紧急度: {intent.urgency.name}\n"
                f"已从用户消息中抽取的信息: {entities or '无'}")
        if summary:
            text += f"\n本次会话更早内容的摘要: {summary}"
        if memories:
            listed = "\n".join(f"  - {m}" for m in memories)
            text += ("\n该用户以往会话中可能相关的记录，仅供参考。是否与当前问题有关由你判断，"
                     f"拿不准就向用户确认，不要直接当成事实:\n{listed}")
        return text

    async def _fallback_if_primary_failed(self, replies, decision, message, ctx, background, history) -> list[AgentReply]:
        """专业 Agent 挂了，让通用客服顶上，保证用户至少得到一个回应。"""
        if replies[0].success or decision.primary == "general":
            return replies
        logger.warning("%s agent 失败，降级到 general", decision.primary)
        fallback = await self._agents["general"].run(message, ctx, background, history)
        return [fallback] + replies[1:]

    async def _compose(self, message: str, replies: list[AgentReply]) -> str:
        """合并多个 Agent 的回复。只有一个有效回复时直接用，省一次模型调用。"""
        good = [r for r in replies if r.success and r.content.strip()]
        if not good:
            return "抱歉，系统暂时无法处理您的请求，请稍后再试，或回复「转人工」。"
        if len(good) == 1:
            return good[0].content

        drafts = "\n\n".join(f"【{r.agent} 的回复】\n{r.content}" for r in good)
        prompt = (
            "下面是几位客服同事针对同一条用户消息分别写的回复草稿，请合并成一条给用户的最终回复。\n"
            "要求：第一份草稿对应用户的主要问题，放在前面；去掉重复的寒暄和重复内容；"
            "不得新增草稿里没有的事实、金额或承诺；草稿之间有矛盾时，说明需要进一步核实；"
            "不要提及有多位同事或内部分工。\n\n"
            f"用户消息：{message}\n\n{drafts}"
        )
        try:
            merged = await self._llm.chat_text(prompt, temperature=0.1, max_tokens=1000)
            if merged:
                return merged
        except Exception as ex:
            logger.warning("合并失败，改用拼接: %s", ex)
        # 兜底：模型合并失败时按主次顺序直接拼接，宁可啰嗦也不丢信息
        return good[0].content + "".join(f"\n\n另外，{r.content}" for r in good[1:])

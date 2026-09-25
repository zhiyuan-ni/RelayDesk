"""记忆管理：读取上下文、写入对话、在对话过长时压缩。

压缩策略：
  消息数达到 COMPRESS_AT 时，把除了最近 KEEP_RECENT 条以外的旧消息交给模型，
  和已有摘要合并成一段新摘要，然后从 Redis 里删掉这些旧消息。
  效果是：无论聊多久，送给模型的内容都是 一段摘要 + 最近几条原文，长度有上限。

压缩放在后台执行。它要调一次模型，如果放在请求路径上，恰好触发压缩的那一轮用户会多等几秒。
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.memory.store import ConversationStore
from app.observability import tracer

logger = logging.getLogger(__name__)

COMPRESS_AT = 12        # 消息数达到多少条时触发压缩。一问一答算 2 条，即 6 轮
KEEP_RECENT = 4         # 压缩后保留最近几条原文，即 2 轮
CONTEXT_BUDGET = 1500   # 送给模型的最近对话，总字符数上限
SUMMARY_MAX_CHARS = 400


@dataclass
class MemoryContext:
    summary: str = ""
    recent: list[dict[str, Any]] = field(default_factory=list)   # 按时间顺序，形如 {"role": ..., "content": ...}
    total: int = 0   # 按预算挑选之前 Redis 里有多少条消息。和 len(recent) 的差就是这一轮没带上的条数

    def history(self) -> list[dict[str, str]]:
        """转成聊天接口需要的格式，去掉时间戳等多余字段。"""
        return [{"role": m["role"], "content": m["content"]} for m in self.recent]


def select_within_budget(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """从最新的消息往前取，取到总字符数快要超过预算为止，返回值仍按时间顺序。

    最新的一条无论多长都保留。遇到第一条放不下的就停止，不跳过它去拿更早的短消息，
    否则对话中间会缺一段。按字符预算而不是固定条数，是为了防止一条超长消息把上下文撑爆。
    """
    picked: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        size = len(message["content"])
        # picked 为空说明这是最新的一条，无条件保留；之后的每一条都要先看预算够不够。
        # 放不下就停止，不跳过去拿更早的短消息，否则对话中间会缺一段
        if picked and used + size > max_chars:
            break
        picked.append(message)
        used += size
    return picked[::-1]   # 收集时是从新到旧，翻转回时间顺序

EPISODE_MAX_CHARS = 500
RECALL_TIMEOUT_S = 3.0


def build_episode_text(summary: str, messages: list[dict[str, Any]]) -> str:
    """一个会话在长期记忆里的文本：摘要，加上用户最近说的原话。

    只取用户的话，不取助手的回复。要记住的是"用户关心什么、提到过哪些订单"，
    助手的长篇回复只会稀释语义，让检索变得不准。
    """
    said = "；".join(m["content"][:120] for m in messages if m["role"] == "user")
    parts = [p for p in (summary.strip(), f"用户说过：{said}" if said else "") if p]
    return "\n".join(parts)[:EPISODE_MAX_CHARS]


class MemoryManager:
    def __init__(self, store: ConversationStore, llm, longterm=None):
        self._store, self._llm, self._longterm = store, llm, longterm
        self._tasks: set[asyncio.Task] = set()   # 持有后台任务的引用，否则任务可能还没跑完就被垃圾回收

    def _in_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def load(self, user_id: str, conv_id: str) -> MemoryContext:
        messages, summary = await asyncio.gather(
            self._store.messages(user_id, conv_id),
            self._store.get_summary(user_id, conv_id),
        )
        return MemoryContext(summary=summary, recent=select_within_budget(messages, CONTEXT_BUDGET),
                             total=len(messages))

    async def save_turn(self, user_id: str, conv_id: str, user_msg: str, assistant_msg: str) -> None:
        await self._store.append(user_id, conv_id, "user", user_msg)
        n = await self._store.append(user_id, conv_id, "assistant", assistant_msg)
        # 两件后台工作，都不占用请求的响应时间
        self._in_background(self._after_turn(user_id, conv_id, needs_compression=n >= COMPRESS_AT))

    async def _after_turn(self, user_id: str, conv_id: str, needs_compression: bool) -> None:
        if needs_compression:
            await self.compress(user_id, conv_id)
        await self.remember(user_id, conv_id)   # 放在压缩之后，这样写入长期记忆的是最新的摘要

    async def remember(self, user_id: str, conv_id: str) -> None:
        """把当前会话的要点写入长期记忆。失败只记日志。"""
        if self._longterm is None:
            return
        try:
            messages, summary = await asyncio.gather(
                self._store.messages(user_id, conv_id), self._store.get_summary(user_id, conv_id))
            await self._longterm.remember(user_id, conv_id, build_episode_text(summary, messages))
        except Exception as ex:
            logger.warning("写入长期记忆失败: %r", ex)

    async def recall(self, user_id: str, conv_id: str, query: str) -> list[str]:
        """想起这个用户以往会话里和当前问题相关的内容。限时执行，超时或失败都返回空列表。

        结果记在时间线上：hit 想起了、miss 没有相关记忆、timeout 超时、error 出错。
        后三种返回的都是空列表，不记下来的话，调试时分不清是真没有还是没查成。
        """
        if self._longterm is None:
            return []
        async with tracer.span("memory_recall") as meta:
            try:
                hits = await asyncio.wait_for(
                    self._longterm.recall(user_id, query, exclude_conv_id=conv_id), timeout=RECALL_TIMEOUT_S)
            except TimeoutError:
                meta["status"] = "timeout"
                logger.warning("检索长期记忆超过 %.0f 秒，按没有相关记忆处理", RECALL_TIMEOUT_S)
                return []
            except Exception as ex:
                meta["status"] = "error"
                logger.warning("检索长期记忆失败，按没有相关记忆处理: %r", ex)
                return []
            meta["status"] = "hit" if hits else "miss"
            return [h["text"] for h in hits]

    async def compress(self, user_id: str, conv_id: str) -> bool:
        """把旧消息压缩进摘要。返回是否真的执行了压缩。任何失败都只记日志，原始消息保持不动。"""
        try:
            messages = await self._store.messages(user_id, conv_id)
            if len(messages) < COMPRESS_AT:
                return False
            old = messages[:-KEEP_RECENT]
            previous = await self._store.get_summary(user_id, conv_id)

            dialogue = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in old)
            prompt = (
                f"你在为客服系统维护会话摘要。请把已有摘要和新增对话合并成一段不超过 {SUMMARY_MAX_CHARS} 字的新摘要。\n"
                "必须保留：用户要解决的问题、订单号和金额等关键信息、已经查明的事实、还没解决的事项。\n"
                "去掉寒暄和重复内容。只输出摘要正文。\n\n"
                f"已有摘要：{previous or '无'}\n\n新增对话：\n{dialogue}"
            )
            summary = (await self._llm.chat_text(prompt, temperature=0.0, max_tokens=400)).strip()
            if not summary:
                return False

            # 顺序很重要：先保存摘要，再删旧消息。中途失败最多是摘要和原文同时存在，信息有重复但不会丢
            await self._store.set_summary(user_id, conv_id, summary[:SUMMARY_MAX_CHARS])
            await self._store.drop_oldest(user_id, conv_id, len(old))
            logger.info("会话 %s 压缩了 %d 条消息", conv_id, len(old))
            return True
        except Exception as ex:
            logger.warning("会话压缩失败，保留原始消息: %r", ex)
            return False

    async def wait_background(self) -> None:
        """等待后台任务结束。测试和服务关闭时使用。"""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

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

logger = logging.getLogger(__name__)

COMPRESS_AT = 12        # 消息数达到多少条时触发压缩。一问一答算 2 条，即 6 轮
KEEP_RECENT = 4         # 压缩后保留最近几条原文，即 2 轮
CONTEXT_BUDGET = 1500   # 送给模型的最近对话，总字符数上限
SUMMARY_MAX_CHARS = 400


@dataclass
class MemoryContext:
    summary: str = ""
    recent: list[dict[str, Any]] = field(default_factory=list)   # 按时间顺序，形如 {"role": ..., "content": ...}

    def history(self) -> list[dict[str, str]]:
        """转成聊天接口需要的格式，去掉时间戳等多余字段。"""
        return [{"role": m["role"], "content": m["content"]} for m in self.recent]


def select_within_budget(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """从最近的消息往前取，取到总字符数快要超过预算为止。返回值仍然按时间顺序。

    ───────────── 练习：请你实现 ─────────────
    背景：模型的上下文是要花钱的，也有长度上限。用户偶尔会贴一大段报错日志，
    如果固定取"最近 N 条"，一条超长消息就可能把预算撑爆。按字符预算来取更稳妥。

    规格，对应 tests/test_memory.py：
      1. messages 为空                         -> []
      2. 从最后一条开始往前累加每条消息 content 的长度，
         加上某一条会超过 max_chars 时就停下，那一条以及更早的都不要
      3. 最新的那一条无论多长都必须保留，哪怕它自己就超过了预算。
         否则用户刚说的话模型反而看不到
      4. 返回的列表保持原来的时间顺序，旧的在前，新的在后

    提示：
      - reversed(messages) 可以从后往前遍历
      - 用一个变量 used 记录已经用掉的字符数，用一个列表 picked 收集选中的消息
      - 规格 3 的判断：picked 为空时，说明现在处理的是最新的一条，无条件放进去
      - 你是从后往前收集的，所以 picked 的顺序是反的，返回前要翻转回来：picked[::-1]
    大约 8 行。
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

class MemoryManager:
    def __init__(self, store: ConversationStore, llm):
        self._store, self._llm = store, llm
        self._tasks: set[asyncio.Task] = set()   # 持有后台任务的引用，否则任务可能还没跑完就被垃圾回收

    async def load(self, user_id: str, conv_id: str) -> MemoryContext:
        messages, summary = await asyncio.gather(
            self._store.messages(user_id, conv_id),
            self._store.get_summary(user_id, conv_id),
        )
        return MemoryContext(summary=summary, recent=select_within_budget(messages, CONTEXT_BUDGET))

    async def save_turn(self, user_id: str, conv_id: str, user_msg: str, assistant_msg: str) -> None:
        await self._store.append(user_id, conv_id, "user", user_msg)
        n = await self._store.append(user_id, conv_id, "assistant", assistant_msg)
        if n >= COMPRESS_AT:
            task = asyncio.create_task(self.compress(user_id, conv_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

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

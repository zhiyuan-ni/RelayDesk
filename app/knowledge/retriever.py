"""检索器：在向量检索之上加三步，解决"找不全"和"排不准"。

  原始问题 ──────────────► 向量召回 ─┐
      │                              ├─► 合并去重 ─► LLM 重排 ─► Top-K
      └─► LLM 改写成 2 个变体 ─► 向量召回 ─┘

  改写解决"找不全"：用户说"钱被划走两回"，文档写的是"重复扣款"。换几种说法各搜一次，覆盖面更大。
  重排解决"排不准"：向量只能判断话题像不像。重排让模型把问题和候选放在一起读，判断哪段真的能回答问题。

  每一步失败都退回到上一步的结果：改写失败就只用原始问题，重排失败就用向量顺序。检索永远有结果。
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

RECALL_K = 5       # 每个查询召回多少候选。召回阶段宁多勿漏
N_VARIANTS = 2     # 改写出几个变体


@dataclass
class RetrievalResult:
    hits: list[dict[str, Any]]
    queries: list[str] = field(default_factory=list)   # 实际用来召回的全部查询，第一个是原始问题
    n_candidates: int = 0                              # 合并去重后的候选数
    reranked: bool = False                             # 重排是否成功。False 表示用的是向量顺序
    latency_ms: float = 0.0


def merge_hits(hit_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """合并多路召回结果：去重，同一片段保留最高分，按分数从高到低排序。

    ───────────── 练习：请你实现 ─────────────
    输入是"列表的列表"，每个内层列表是一次检索的结果，每条结果是一个字典：
        {"title": ..., "section": ..., "text": ..., "score": 0.71}
    同一个片段可能被多个查询同时召回，分数各不相同。

    规格，对应 tests/test_retriever.py：
      1. 输入为空，或内层全为空          -> []
      2. 用 (title, section, text) 三者组成的元组判断"是不是同一个片段"
      3. 同一个片段出现多次时，只保留 score 最高的那一条
      4. 返回的列表按 score 从高到低排序

    提示：
      - 用一个字典 best 来去重，键是那个三元组，值是目前见过的最高分的那条结果
            key = (hit["title"], hit["section"], hit["text"])
            if key not in best or hit["score"] > best[key]["score"]:
                best[key] = hit
      - 两层 for 循环：外层遍历 hit_lists，内层遍历每个列表里的 hit
      - 最后 sorted(best.values(), key=lambda h: h["score"], reverse=True)
        lambda h: h["score"] 是一个匿名小函数，意思是"给我一条结果 h，我返回它的分数"，sorted 按这个值排序
    大约 8 行。
    """
    best: dict[tuple, dict[str, Any]] = {}   # 键是片段的身份，值是目前见过的分数最高的那条
    for hits in hit_lists:
        for hit in hits:
            key = (hit["title"], hit["section"], hit["text"])
            if key not in best or hit["score"] > best[key]["score"]:
                best[key] = hit
    return sorted(best.values(), key=lambda h: h["score"], reverse=True)



def parse_json_list(raw: str) -> list:
    """从模型输出里截取第一个 JSON 数组。模型偶尔会在前后加说明文字。"""
    start, end = raw.index("["), raw.rindex("]") + 1
    data = json.loads(raw[start:end])
    if not isinstance(data, list):
        raise ValueError("不是数组")
    return data


def apply_order(candidates: list[dict], order: list, top_k: int) -> list[dict]:
    """按模型给的序号重新排列候选。对模型输出保持戒心：越界的、重复的、不是整数的序号一律忽略。"""
    picked, seen = [], set()
    for i in order:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(candidates) and i not in seen:
            seen.add(i)
            picked.append(candidates[i])
    # 模型漏掉的候选按原顺序补在后面，保证结果数量不因为模型偷懒而变少
    picked += [c for i, c in enumerate(candidates) if i not in seen]
    return picked[:top_k]


class Retriever:
    def __init__(self, kb, llm, rewrite: bool = True, rerank: bool = True, rerank_timeout_s: float = 4.0):
        self._kb, self._llm = kb, llm
        self._rewrite_on, self._rerank_on = rewrite, rerank
        # 分层超时：重排有自己的时限，比工具的总时限短。
        # 重排慢了就放弃重排、用向量顺序，召回的结果不会因此作废。只有一个总超时的话，重排一慢就什么都拿不到。
        self._rerank_timeout_s = rerank_timeout_s

    async def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        t0 = time.monotonic()

        # 原始问题的召回不需要等改写，两件事同时开始。改写要约 1 秒，这 1 秒不能白等。
        (original_hits, _), (variant_lists, variants) = await asyncio.gather(
            self._recall([query]),
            self._rewrite_then_recall(query),
        )
        candidates = merge_hits(original_hits + variant_lists)

        hits, reranked = candidates[:top_k], False
        if self._rerank_on and len(candidates) > 1:
            hits, reranked = await self._rerank(query, candidates, top_k)

        return RetrievalResult(hits, [query] + variants, len(candidates), reranked,
                               round((time.monotonic() - t0) * 1000, 1))

    async def _recall(self, queries: list[str]) -> tuple[list[list[dict]], list[str]]:
        if not queries:
            return [], []
        results = await asyncio.gather(*[self._kb.search(q, RECALL_K) for q in queries], return_exceptions=True)
        return [r for r in results if not isinstance(r, BaseException)], queries

    async def _rewrite_then_recall(self, query: str) -> tuple[list[list[dict]], list[str]]:
        variants = await self._rewrite(query) if self._rewrite_on else []
        return await self._recall(variants)

    async def _rewrite(self, query: str) -> list[str]:
        prompt = (
            f"用户在向客服提问。请把下面的问题改写成 {N_VARIANTS} 个用于检索知识库的查询。\n"
            "要求：把口语说法换成规范的业务术语；每个查询侧重问题的不同方面；不要添加原问题里没有的信息。\n"
            f"用户问题：{query}\n"
            '只返回 JSON 数组，例如 ["查询一", "查询二"]'
        )
        try:
            raw = await self._llm.chat_text(prompt, temperature=0.3, max_tokens=200)
            variants = [str(v).strip() for v in parse_json_list(raw) if str(v).strip()]
            return [v for v in dict.fromkeys(variants) if v != query][:N_VARIANTS]
        except Exception as ex:
            logger.warning("查询改写失败，只用原始问题: %s", ex)
            return []

    async def _rerank(self, query: str, candidates: list[dict], top_k: int) -> tuple[list[dict], bool]:
        listing = "\n".join(
            f"[{i}] {c['title']} / {c['section']}：{c['text'][:220]}" for i, c in enumerate(candidates))
        prompt = (
            "下面是用户的问题和若干知识库片段。请判断哪些片段能直接回答这个问题，按有用程度从高到低排列。\n"
            "注意区分话题相近和真正能回答：只是提到相同词语、但没有回答问题的片段要排在后面。\n"
            f"用户问题：{query}\n\n{listing}\n\n"
            "只返回由片段序号组成的 JSON 数组，例如 [2, 0, 3]"
        )
        try:
            raw = await asyncio.wait_for(
                self._llm.chat_text(prompt, temperature=0.0, max_tokens=100), timeout=self._rerank_timeout_s)
            return apply_order(candidates, parse_json_list(raw), top_k), True
        except Exception as ex:
            logger.warning("重排失败，使用向量顺序: %s", ex)
            return candidates[:top_k], False

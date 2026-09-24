"""检索器：在向量检索之上加三步，解决"找不全"和"排不准"。

  原始问题 ──────────────► 向量召回 ─┐
      │                              ├─► 合并去重 ─► LLM 重排 ─► Top-K
      └─► LLM 改写成 2 个变体 ─► 向量召回 ─┘

  改写解决"找不全"：用户说"钱被划走两回"，文档写的是"重复扣款"。换几种说法各搜一次，覆盖面更大。
  重排解决"排不准"：向量只能判断话题像不像。重排让模型把问题和候选放在一起读，判断哪段真的能回答问题。

  每一步失败都退回到上一步的结果：改写失败就只用原始问题，重排失败就用向量顺序。检索永远有结果。

重排有两种后端，由 RERANK_BACKEND 决定：
  llm  把全部候选列给模型，让它输出序号数组，再防御性地解析
  jev  每个候选一道是非题"这个片段能直接回答用户的问题吗"，按"是"的概率排序。
       15 条对比用例上与 LLM 重排的首位命中率相同，延迟约为其一半
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.observability import tracer

logger = logging.getLogger(__name__)

RECALL_K = 5       # 每个查询召回多少候选。召回阶段宁多勿漏
N_VARIANTS = 2     # 改写出几个变体
JEV_PASSAGE_CHARS = 400   # jev 每道题只看自己的片段，片段长不会挤占其他候选，可以比 LLM 路带得多

# jev 重排的题目。true/false 的说明和 LLM 路提示词里"区分话题相近和真正能回答"是同一条标准
JEV_RERANK_QUESTION = "这个片段能直接回答用户的问题吗"
JEV_RERANK_CRITERIA = {
    "true": "片段给出了用户所问事情的规则、条件、办理步骤或时效，读完就能回答用户",
    "false": "片段只是话题相近或提到相同的词，没有回答用户问的这件事",
}


@dataclass
class RetrievalResult:
    hits: list[dict[str, Any]]
    queries: list[str] = field(default_factory=list)   # 实际用来召回的全部查询，第一个是原始问题
    n_candidates: int = 0                              # 合并去重后的候选数
    reranked: bool = False                             # 重排是否成功。False 表示用的是向量顺序
    latency_ms: float = 0.0
    vector_ranks: list[int] = field(default_factory=list)   # hits 里每一条在向量排序中的名次，从 0 起。和下标不同说明重排改了顺序


def merge_hits(hit_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """合并多路召回结果：按 (标题, 小节, 正文) 去重，同一片段保留最高分，按分数从高到低排序。"""
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


def build_rerank_questions(candidates: list[dict]) -> dict[str, dict]:
    """每个候选一道 Noul 题，片段放在题目里，state 只放用户问题。

    一次请求问完全部候选：jev 对每道题分别作答，题与题之间互不可见，相当于逐对打分，
    但只有一次往返。实测比每个候选各发一个请求快 3 倍多，首位命中率也不低于后者。
    """
    return {
        f"c{i}": {
            "type": "noul",
            "instructions": {"片段": f"{c['title']} / {c['section']}：{c['text'][:JEV_PASSAGE_CHARS]}",
                             "问题": JEV_RERANK_QUESTION},
            "criteria": JEV_RERANK_CRITERIA,
        }
        for i, c in enumerate(candidates)
    }


def order_by_noul(answers: dict[str, dict], n: int) -> list[int]:
    """按"是"的概率从高到低给出候选序号。缺任何一个候选的答案都抛 KeyError，由调用方退回向量顺序。

    sorted 是稳定排序：概率相同的候选保持向量召回时的先后，相当于用向量分数打破平局。
    """
    scores = [float(answers[f"c{i}"]["noul"]) for i in range(n)]
    return sorted(range(n), key=lambda i: scores[i], reverse=True)


class Retriever:
    def __init__(self, kb, llm, rewrite: bool = True, rerank: bool = True, rerank_timeout_s: float = 4.0,
                 jev=None):
        """传了 jev 就用 jev 重排，否则用 LLM 重排。改写始终用 LLM，jev 不生成文字。"""
        self._kb, self._llm, self._jev = kb, llm, jev
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

        # merge_hits 已按 (标题, 小节, 正文) 去重，所以用内容相等来找名次不会找错
        vector_ranks = [candidates.index(h) for h in hits]
        return RetrievalResult(hits, [query] + variants, len(candidates), reranked,
                               round((time.monotonic() - t0) * 1000, 1), vector_ranks)

    async def _recall(self, queries: list[str]) -> tuple[list[list[dict]], list[str]]:
        if not queries:
            return [], []
        async with tracer.span("kb:recall", queries=len(queries)):
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
        backend = "jev" if self._jev is not None else "llm"
        order_fn = self._jev_order if self._jev is not None else self._llm_order
        try:
            async with tracer.span("kb:rerank", candidates=len(candidates), backend=backend):
                order = await asyncio.wait_for(order_fn(query, candidates), timeout=self._rerank_timeout_s)
            return apply_order(candidates, order, top_k), True
        except Exception as ex:
            logger.warning("重排失败（%s），使用向量顺序: %r", backend, ex)
            return candidates[:top_k], False

    async def _jev_order(self, query: str, candidates: list[dict]) -> list[int]:
        answers = await self._jev.ask({"用户问题": query}, build_rerank_questions(candidates))
        return order_by_noul(answers, len(candidates))

    async def _llm_order(self, query: str, candidates: list[dict]) -> list:
        listing = "\n".join(
            f"[{i}] {c['title']} / {c['section']}：{c['text'][:220]}" for i, c in enumerate(candidates))
        prompt = (
            "下面是用户的问题和若干知识库片段。请判断哪些片段能直接回答这个问题，按有用程度从高到低排列。\n"
            "注意区分话题相近和真正能回答：只是提到相同词语、但没有回答问题的片段要排在后面。\n"
            f"用户问题：{query}\n\n{listing}\n\n"
            "只返回由片段序号组成的 JSON 数组，例如 [2, 0, 3]"
        )
        raw = await self._llm.chat_text(prompt, temperature=0.0, max_tokens=100)
        return parse_json_list(raw)

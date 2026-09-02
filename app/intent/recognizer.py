"""意图识别器：把 LLM 路和规则路的两票融合成最终结论。"""
import time
from typing import Optional

from app.intent.llm_classifier import llm_vote
from app.intent.rules import detect_urgency, extract_entities, rule_vote
from app.intent.schema import INTENT_GROUP, Intent, IntentResult, Vote

# ── 融合参数：这三个数是你的设计决定，面试时要能解释为什么是这个值 ──
LLM_TRUST = 0.6    # 两路结论冲突时，LLM 置信度达到多少才采信 LLM
AGREE_BONUS = 0.1  # 两路结论一致时，置信度加多少
MIN_CONF = 0.5     # 最终置信度低于它就判为 OTHER，交给下游去反问澄清


def fuse(llm: Optional[Vote], rule: Optional[Vote]) -> tuple[Intent, float, str]:
    """融合两票，返回 (意图, 置信度, 来源)。来源取值 "both" / "llm" / "rule" / "none"。

    ───────────── 练习：请你实现 ─────────────
    规格，和 tests/test_intent_fuse.py 里的用例一一对应：
      1. 两票都没有            -> (OTHER, 0.0, "none")
      2. 只有规则票            -> 用规则票，来源 "rule"
      3. 只有 LLM 票           -> 用 LLM 票，来源 "llm"
      4. 两票意图一致          -> 用该意图，置信度 = LLM 置信度 + AGREE_BONUS，最大 1.0，来源 "both"
      5. 两票冲突：
           LLM 置信度 >= LLM_TRUST -> 用 LLM 票，来源 "llm"
           否则                    -> 用规则票，来源 "rule"
      6. 最后一步：如果选出的置信度 < MIN_CONF，把意图改成 OTHER，置信度和来源保持不变

    提示：
      - 判断"没有这一票"用  if llm is None
      - Vote 有 .intent 和 .confidence 两个属性
      - 可以先用 if/elif 选出 intent, conf, source 三个变量，最后统一做第 6 步再 return
    """

    if llm is None and rule is None:
        return (Intent.OTHER, 0.0, "none")
    elif llm is None and rule:
        intent, conf, source = rule.intent, rule.confidence, "rule"
    elif llm and rule is None:
        intent, conf, source = llm.intent, llm.confidence, "llm"
    else:
        # 意图一致
        if llm.intent == rule.intent:
            intent, conf, source = llm.intent, min(1.0, llm.confidence + AGREE_BONUS), "both"
        # 意图冲突
        elif llm.confidence >= LLM_TRUST:
            intent, conf, source = llm.intent, llm.confidence, "llm"
        else:
            intent, conf, source = rule.intent, rule.confidence, "rule"

    if conf < MIN_CONF:
        intent = Intent.OTHER
    return (intent, conf, source)


class IntentRecognizer:
    def __init__(self, llm):
        self._llm = llm

    async def recognize(
        self, message: str, history: Optional[list[dict[str, str]]] = None
    ) -> IntentResult:
        t0 = time.monotonic()

        rule = rule_vote(message)                          # 同步，微秒级
        llm = await llm_vote(self._llm, message, history)  # 异步，约 1 秒
        intent, conf, source = fuse(llm, rule)

        winner = llm if source in ("llm", "both") else rule
        return IntentResult(
            intent=intent,
            group=INTENT_GROUP[intent],
            confidence=round(conf, 3),
            source=source,
            urgency=detect_urgency(message, intent),
            entities=extract_entities(message),
            reasoning=winner.reasoning if winner else "",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

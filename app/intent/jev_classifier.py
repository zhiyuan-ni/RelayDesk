"""jev 路：把意图分类写成一道 Choice 题，和 LLM 路二选一，由 INTENT_BACKEND 决定。

和 LLM 路比，差别在三处：
  1. 不用解析文本。jev 只会返回选项里的值，LLM 路那套截取花括号、校验枚举的防御代码在这里用不上。
  2. 置信度有区分度。dev 集上 LLM 自报的置信度全在 0.9 以上，jev 的置信度 >=0.9 的一档准确率 99%，
     低于 0.9 的一档约 60%，融合函数里的阈值才真正起作用。
  3. jev 读题很字面，只按选项说明理解类别。边界规则必须写进每个选项的说明里，只靠意图名是不够的。
"""
import logging
from typing import Optional

from app.intent.llm_classifier import INTENT_NOTES
from app.intent.schema import Intent, Vote

logger = logging.getLogger(__name__)

QUESTION_ID = "intent"

INSTRUCTIONS = "用户最新消息的客服意图属于哪一类。最近对话只用于理解省略的内容，按最新消息的诉求分类"

# INTENT_NOTES 是给 LLM 的补充说明，没覆盖名字已经足够清楚的三类。jev 需要每个选项都有说明
EXTRA_NOTES: dict[Intent, str] = {
    Intent.GREETING: "打招呼、问在不在，没有提出具体诉求",
    Intent.COMPLAINT: "表达不满、抱怨服务或要投诉，没有提出具体业务诉求",
    Intent.INVOICE: "开发票、换开发票、发票抬头税号、报销凭证",
}


def build_question() -> dict:
    """选项说明和 LLM 路共用 INTENT_NOTES，标注指南里的边界规则只维护一份。"""
    return {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {i.value: INTENT_NOTES.get(i) or EXTRA_NOTES[i] for i in Intent},
    }


def build_state(message: str, history: Optional[list[dict[str, str]]] = None) -> dict:
    """只放判断需要的内容。jev 的准确率随无关内容增多而下降，所以历史和 LLM 路一样只带最近 3 条。"""
    state: dict = {"最新消息": message}
    if history:
        state["最近对话"] = [f"{m['role']}: {m['content'][:120]}" for m in history[-3:]]
    return state


def answer_to_vote(answer: dict) -> Optional[Vote]:
    """把 jev 的一个 Choice 答案换算成一票。结构不对时返回 None。

    答案形如 {"type": "choice", "choice": "refund", "confidence": 0.93,
             "probabilities": {"refund": 0.95, "order_logistics": 0.04, ...}}

    置信度用 confidence 字段，不用 probabilities[choice]：前者看整个分布，
    两个选项五五开时它会明显偏低，这正是融合函数需要的信号。
    reasoning 写上第一和第二候选及其概率，排查错例时能看出模型在哪两类之间犹豫。
    """
    try:
        intent = Intent(answer["choice"])
        conf = max(0.0, min(1.0, float(answer["confidence"])))
        ranked = sorted(answer.get("probabilities", {}).items(), key=lambda kv: kv[1], reverse=True)
        reasoning = "jev: " + "，".join(f"{k} {v:.2f}" for k, v in ranked[:2])
        return Vote(intent, conf, reasoning)
    except (ValueError, KeyError, TypeError):
        logger.warning("jev 意图答案结构异常: %r", answer)
        return None


async def jev_vote(jev, message: str, history: Optional[list[dict[str, str]]] = None) -> Optional[Vote]:
    """调用 jev 拿一票。任何失败都返回 None，和 LLM 路一样交给融合函数兜底。"""
    try:
        answers = await jev.ask(build_state(message, history), {QUESTION_ID: build_question()})
    except Exception as ex:  # 网络超时、限流、key 失效等
        logger.warning("jev 意图识别调用失败: %r", ex)
        return None
    return answer_to_vote(answers.get(QUESTION_ID, {}))

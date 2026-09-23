"""LLM-as-Judge：让一个模型给另一个模型的客服回复打分。

怎么防止"自己夸自己"，四条措施叠加：
  1. 评委和被评的是不同厂商的模型。同一个模型对自己的措辞风格有偏好，换一家能抵消一部分。
  2. 给评委一份"事实清单"。评委不是凭感觉判断回复好不好，而是逐条核对回复有没有和事实矛盾、
     有没有编造事实里没有的内容。这把主观评价变成了对照检查。
  3. 每个维度都有 1 到 5 分的锚点说明，而不是让评委自由发挥。
  4. 评委分数只是辅助指标。主指标是 e2e.py 里的确定性检查，那部分不经过任何模型。

评委仍然可能出错，所以分数低的回复要人工复核，评委的理由字段就是给人看的。
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

RUBRIC = """你是客服质量评审。请根据"事实清单"核对客服回复，按三个维度打分，每个维度 1 到 5 的整数。

事实一致性 factual：
  5 回复中的所有具体信息都能在事实清单中找到依据，没有编造
  3 有一处含糊或轻微偏差，但没有明显错误
  1 出现了事实清单里没有的具体信息，或与事实矛盾，例如编造了金额、时间、状态

有帮助 helpful：
  5 直接回应了用户的问题，用户看完知道该做什么
  3 回应了问题但缺少下一步，或者答非所问了一部分
  1 没有回应问题，或者让用户自己去查

边界 policy：
  5 没有承诺无法保证的结果，没有索要密码或验证码，需要人工的地方说明了需要人工
  3 措辞上有轻微的过度承诺，例如"一定会退款"
  1 承诺了退款成功或到账时间，或索要了密码、验证码、完整卡号

只返回 JSON：{"factual": 整数, "helpful": 整数, "policy": 整数, "reason": "一句话说明扣分原因，满分则写无"}"""


@dataclass
class JudgeScore:
    factual: int
    helpful: int
    policy: int
    reason: str = ""
    failed: bool = False   # 评委调用或解析失败时为 True，分数全为 0，不参与平均

    @property
    def overall(self) -> float:
        return round((self.factual + self.helpful + self.policy) / 3, 2)


def parse_score(raw: str) -> Optional[JudgeScore]:
    """从评委输出里解析分数。模型常把 JSON 包在 ```json 围栏里，先把围栏去掉。"""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        start, end = cleaned.index("{"), cleaned.rindex("}") + 1
        data = json.loads(cleaned[start:end])
        scores = {k: int(data[k]) for k in ("factual", "helpful", "policy")}
        if not all(1 <= v <= 5 for v in scores.values()):
            return None
        return JudgeScore(**scores, reason=str(data.get("reason", "")))
    except (ValueError, KeyError, TypeError):
        return None


class Judge:
    def __init__(self, llm):
        self._llm = llm

    async def score(self, conversation: list[dict[str, str]], reply: str, facts: str) -> JudgeScore:
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
        prompt = (f"{RUBRIC}\n\n事实清单：\n{facts}\n\n对话记录（最后一条是待评的客服回复）：\n{dialogue}\n"
                  f"assistant: {reply}")
        try:
            raw = await self._llm.chat_text(prompt, temperature=0.0, max_tokens=300)
        except Exception as ex:
            logger.warning("评委调用失败: %r", ex)
            return JudgeScore(0, 0, 0, f"评委调用失败: {ex}", failed=True)
        parsed = parse_score(raw)
        if parsed is None:
            logger.warning("评委输出无法解析: %r", raw[:200])
            return JudgeScore(0, 0, 0, "评委输出无法解析", failed=True)
        return parsed

"""LLM 路：few-shot 提示 + JSON 输出。负责理解规则认不出的说法和上下文。"""
import json
import logging
from typing import Optional

from app.intent.schema import Intent, Vote

logger = logging.getLogger(__name__)

# 每类一条示例。挑的都是"不含明显关键词"的说法，专门补规则路的短板。
FEW_SHOTS: list[tuple[str, Intent]] = [
    ("有人吗", Intent.GREETING),
    ("我上周买的东西怎么还没动静", Intent.ORDER_LOGISTICS),
    ("你们这服务我真是服了", Intent.COMPLAINT),
    ("东西不想要了，钱能还我吗", Intent.REFUND),
    ("公司报销需要开个票", Intent.INVOICE),
    ("卡里的钱被划走了两回", Intent.PAYMENT_ISSUE),
    ("密码明明是对的就是进不去", Intent.TECH_LOGIN),
    ("点开就自己关掉了", Intent.TECH_ERROR),
    ("好像有别人在用我的号", Intent.ACCOUNT_SECURITY),
    ("我不想跟机器人说话", Intent.HUMAN_HANDOFF),
    ("今天天气不错", Intent.OTHER),
]


def build_prompt(message: str, history: Optional[list[dict[str, str]]] = None) -> str:
    examples = "\n".join(f'  "{text}" -> {intent.value}' for text, intent in FEW_SHOTS)
    labels = ", ".join(i.value for i in Intent)
    ctx = ""
    if history:
        # 只带最近 3 条。多轮里用户常说"订单号是 12345"这种脱离上下文无法判断的话。
        lines = "\n".join(f"  {m['role']}: {m['content'][:120]}" for m in history[-3:])
        ctx = f"最近对话：\n{lines}\n\n"
    return (
        "你是客服意图分类器。判断用户最新消息的意图，只返回 JSON。\n\n"
        f"可选意图：{labels}\n\n示例：\n{examples}\n\n"
        f"{ctx}用户最新消息：\"{message}\"\n\n"
        '返回格式：{"intent": "<意图>", "confidence": <0到1的小数>, "reasoning": "<一句话理由>"}'
    )


def parse_vote(raw: str) -> Optional[Vote]:
    """从模型输出里解析出一票。模型偶尔会在 JSON 外面包一层说明文字，所以按花括号截取。"""
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        intent = Intent(data["intent"])  # 不在枚举里的值会抛 ValueError
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        return Vote(intent, conf, str(data.get("reasoning", "")))
    except (ValueError, KeyError, TypeError):
        logger.warning("LLM 意图输出无法解析: %r", raw[:200])
        return None


async def llm_vote(llm, message: str, history: Optional[list[dict[str, str]]] = None) -> Optional[Vote]:
    """调用模型拿一票。任何失败都返回 None，由融合函数决定怎么兜底。"""
    try:
        raw = await llm.chat_text(build_prompt(message, history), temperature=0.0, max_tokens=200)
    except Exception as ex:  # 网络超时、限流、key 失效等
        logger.warning("LLM 意图识别调用失败: %s", ex)
        return None
    return parse_vote(raw)

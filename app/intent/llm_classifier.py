"""LLM 路：few-shot 提示 + JSON 输出。负责理解规则认不出的说法和上下文。"""
import json
import logging
import re
from typing import Optional

from app.intent.schema import Intent, Vote

logger = logging.getLogger(__name__)

# 每类一条示例，refund 两条，因为它覆盖退和换修两种诉求。
# 挑的都是"不含明显关键词"的说法，专门补规则路的短板。
FEW_SHOTS: list[tuple[str, Intent]] = [
    ("有人吗", Intent.GREETING),
    ("我上周买的东西怎么还没动静", Intent.ORDER_LOGISTICS),
    ("你们这服务我真是服了", Intent.COMPLAINT),
    ("东西不想要了，钱能还我吗", Intent.REFUND),
    ("才用了两天就不转了，能给我换一台吗", Intent.REFUND),
    ("公司报销需要开个票", Intent.INVOICE),
    ("卡里的钱被划走了两回", Intent.PAYMENT_ISSUE),
    ("密码明明是对的就是进不去", Intent.TECH_LOGIN),
    ("点开就自己关掉了", Intent.TECH_ERROR),
    ("好像有别人在用我的号", Intent.ACCOUNT_SECURITY),
    ("我不想跟机器人说话", Intent.HUMAN_HANDOFF),
    ("今天天气不错", Intent.OTHER),
]

# 名字不足以说明范围的意图，在提示词里补一句说明。
# refund 的范围来自对一万段真实电商客服对话的统计：退换修加退款约占 23%，只接"退"会漏掉大半。
INTENT_NOTES: dict[Intent, str] = {
    Intent.ORDER_LOGISTICS: "订单状态、发货、物流、改地址，以及收货前的取消、拦截、拒收。取消或拒收之后问退款属于 refund",
    Intent.REFUND: "售后类，包括退货、换货、维修保修、少件错发要求补发、退款申请和退款进度",
    Intent.TECH_LOGIN: "登不上、忘记密码、收不到验证码。绑定手机已换或停用需要改绑的属于 account_security",
    Intent.ACCOUNT_SECURITY: "被盗、异常登录、改登录密码或支付密码、改绑或解绑手机邮箱、实名认证、注销",
    Intent.HUMAN_HANDOFF: "要转人工、问有没有真人、要客服电话。要快递员或站点电话的属于 order_logistics",
    Intent.TECH_ERROR: "闪退、白屏、报错、按钮点不了、提交不了、一直加载。因没货或限购等业务规则做不了的不算",
    Intent.PAYMENT_ISSUE: "支付环节的问题，包括重复扣款、多扣钱、支付失败、钱付了但充值或权益没到账。被多扣钱要求退回也属于这一类；退款没到账属于 refund",
    Intent.OTHER: "售前咨询（参数、库存、赠品、活动规则）、价保与优惠、安装预约与安装售后、产品使用方法、对客服本身的闲聊或评价、与平台业务无关的问题。"
                  "共同点是系统没有对应的业务数据可查，只能解释规则或转达"
}


def build_prompt(message: str, history: Optional[list[dict[str, str]]] = None,
                 with_notes: bool = True) -> str:
    """with_notes=False 只在评测消融时用，用来回答"意图说明到底有没有帮助"。线上始终为 True。"""
    examples = "\n".join(f'  "{text}" -> {intent.value}' for text, intent in FEW_SHOTS)
    labels = ", ".join(i.value for i in Intent)
    notes_block = ""
    if with_notes:
        notes = "\n".join(f"  {intent.value}: {text}" for intent, text in INTENT_NOTES.items())
        notes_block = f"意图说明：\n{notes}\n\n"
    ctx = ""
    if history:
        # 只带最近 3 条。多轮里用户常说"订单号是 12345"这种脱离上下文无法判断的话。
        lines = "\n".join(f"  {m['role']}: {m['content'][:120]}" for m in history[-3:])
        ctx = f"最近对话：\n{lines}\n\n"
    return (
        "你是客服意图分类器。判断用户最新消息的意图，只返回 JSON。\n\n"
        f"可选意图：{labels}\n\n{notes_block}示例：\n{examples}\n\n"
        f"{ctx}用户最新消息：\"{message}\"\n\n"
        '返回格式：{"intent": "<意图>", "confidence": <0到1的小数>, "reasoning": "<一句话理由>"}'
    )


def parse_vote(raw: str) -> Optional[Vote]:
    """从模型输出里解析出一票。模型偶尔会在 JSON 外面包一层说明文字，所以按花括号截取。"""
    try:
        raw = re.sub(r"```(?:json)?", "", raw)   # 有些模型会把 JSON 包在代码围栏里
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        intent = Intent(data["intent"])  # 不在枚举里的值会抛 ValueError
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        return Vote(intent, conf, str(data.get("reasoning", "")))
    except (ValueError, KeyError, TypeError):
        logger.warning("LLM 意图输出无法解析: %r", raw[:200])
        return None


async def llm_vote(llm, message: str, history: Optional[list[dict[str, str]]] = None,
                   with_notes: bool = True) -> Optional[Vote]:
    """调用模型拿一票。任何失败都返回 None，由融合函数决定怎么兜底。"""
    try:
        raw = await llm.chat_text(build_prompt(message, history, with_notes), temperature=0.0, max_tokens=200)
    except Exception as ex:  # 网络超时、限流、key 失效等
        logger.warning("LLM 意图识别调用失败: %s", ex)
        return None
    return parse_vote(raw)

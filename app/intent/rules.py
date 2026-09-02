"""规则路：关键词匹配、紧急度、正则实体抽取。

特点：零延迟、零成本、完全可解释。模型挂了它还能工作，所以是兜底。
缺点：只认字面，"钱被划走两回" 这种说法它认不出，所以需要 LLM 路配合。
"""
import re
from typing import Optional

from app.intent.schema import Intent, Urgency, Vote

# 字典顺序即优先级：命中数相同时，靠前的赢。转人工放最前，因为用户明确要求时必须尊重。
KEYWORDS: dict[Intent, list[str]] = {
    Intent.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工", "真人", "找你们经理"],
    Intent.PAYMENT_ISSUE: ["重复扣款", "扣了两次", "多扣", "乱扣", "支付失败", "付款失败", "扣费"],
    Intent.REFUND: ["退款", "退货", "退钱", "refund"],
    Intent.INVOICE: ["发票", "抬头", "税号", "invoice"],
    Intent.ACCOUNT_SECURITY: ["被盗", "异常登录", "改密码", "重置密码", "修改邮箱", "换绑", "注销账"],
    Intent.TECH_LOGIN: ["无法登录", "登录失败", "登不上", "登录不了", "验证码", "401"],
    Intent.TECH_ERROR: ["崩溃", "闪退", "报错", "白屏", "卡死", "500", "crash", "error"],
    Intent.ORDER_LOGISTICS: ["物流", "快递", "配送", "发货", "订单状态", "到哪了", "什么时候到", "运单"],
    Intent.COMPLAINT: ["投诉", "太差", "垃圾", "糟糕", "等了很久", "没人处理", "差评"],
    Intent.GREETING: ["你好", "您好", "在吗", "hello", "hi"],
}


def rule_vote(message: str) -> Optional[Vote]:
    """返回命中关键词最多的意图；一个都没命中则返回 None。"""
    msg = message.lower()
    best: Optional[Vote] = None
    best_hits = 0
    for intent, words in KEYWORDS.items():
        hits = [w for w in words if w in msg]
        if len(hits) > best_hits:  # 严格大于：平局时保留靠前的，即优先级高的
            best_hits = len(hits)
            # 命中 1 个词给 0.6，每多 1 个加 0.15，封顶 0.9。规则永远不给满分。
            conf = round(min(0.9, 0.6 + 0.15 * (len(hits) - 1)), 2)  # round 消除浮点误差
            best = Vote(intent, conf, reasoning=f"命中关键词: {', '.join(hits)}")
    return best


_CRITICAL_WORDS = ["被盗", "盗刷", "资金损失", "立刻处理", "十万火急"]
_HIGH_WORDS = ["紧急", "马上", "尽快", "赶紧", "今天必须", "急用"]


def detect_urgency(message: str, intent: Intent) -> Urgency:
    if any(w in message for w in _CRITICAL_WORDS):
        return Urgency.CRITICAL
    if any(w in message for w in _HIGH_WORDS) or intent in (Intent.HUMAN_HANDOFF, Intent.COMPLAINT):
        return Urgency.HIGH
    return Urgency.LOW


_MONEY_UNIT = r"(?:元|块|rmb|cny|usd|美元)"
_ORDER_RE = re.compile(r"(?:订单号?|order|#)\s*(?:是|为)?\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", re.I)
_AMOUNT_RE = re.compile(rf"((?:¥|￥)\s*\d+(?:\.\d{{1,2}})?|\d+(?:\.\d{{1,2}})?\s*{_MONEY_UNIT})", re.I)
_DATE_RE = re.compile(r"(今天|明天|昨天|前天|上周|本周|这周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)")
# 错误码：4xx 或 5xx 的三位数。
#   (?<![\d¥￥])  前面不能是数字或货币符号
#   (?!\d|\s*元…) 后面不能是数字，也不能跟着金额单位，否则 "多扣了 500 元" 会被误认成错误码
# 不用 \b 的原因：Python 里中文也算"单词字符"，"报500错误" 中 500 两侧没有单词边界，\b 匹配不上。
_ERROR_RE = re.compile(rf"(?<![\d¥￥.])([45]\d{{2}})(?!\d|\.\d|\s*{_MONEY_UNIT})", re.I)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v.strip()))  # 去重且保持顺序


def extract_entities(message: str) -> dict[str, list[str]]:
    """用正则抽取高价值实体。不调模型，所以每次请求零额外成本。"""
    return {
        "order_id": _unique(_ORDER_RE.findall(message)),
        "amount": _unique(_AMOUNT_RE.findall(message)),
        "date": _unique(_DATE_RE.findall(message)),
        "error_code": _unique(_ERROR_RE.findall(message)),
    }

"""意图识别的数据结构。先定义"输出长什么样"，再写"怎么算出来"。"""
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    # 继承 str 的好处：序列化成 JSON 时自动变成字符串，比较时也能直接和 "refund" 比
    GREETING = "greeting"
    ORDER_LOGISTICS = "order_logistics"
    COMPLAINT = "complaint"
    REFUND = "refund"  # 范围是整个售后：退货、换货、维修保修、补发、退款。名字沿用 refund
    INVOICE = "invoice"
    PAYMENT_ISSUE = "payment_issue"
    TECH_LOGIN = "tech_login"
    TECH_ERROR = "tech_error"
    ACCOUNT_SECURITY = "account_security"
    HUMAN_HANDOFF = "human_handoff"
    OTHER = "other"


class Group(str, Enum):
    """意图组 = 后面路由到哪类 Agent 的第一依据。"""
    GENERAL = "general"
    BILLING = "billing"
    TECHNICAL = "technical"
    ESCALATION = "escalation"
    NONE = "none"


INTENT_GROUP: dict[Intent, Group] = {
    Intent.GREETING: Group.GENERAL,
    Intent.ORDER_LOGISTICS: Group.GENERAL,
    Intent.COMPLAINT: Group.GENERAL,
    Intent.REFUND: Group.BILLING,
    Intent.INVOICE: Group.BILLING,
    Intent.PAYMENT_ISSUE: Group.BILLING,
    Intent.TECH_LOGIN: Group.TECHNICAL,
    Intent.TECH_ERROR: Group.TECHNICAL,
    # 决策：账户安全归技术组。处理动作是核验身份、排查异常登录、重置凭证，
    # 与登录故障排查重合，和资金核对关系不大。
    Intent.ACCOUNT_SECURITY: Group.TECHNICAL,
    Intent.HUMAN_HANDOFF: Group.ESCALATION,
    Intent.OTHER: Group.NONE,
}


class Urgency(int, Enum):
    LOW = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Vote:
    """某一路识别给出的一票。"""
    intent: Intent
    confidence: float  # 0 到 1
    reasoning: str = ""


@dataclass
class IntentResult:
    intent: Intent
    group: Group
    confidence: float
    source: str                      # "both" / "llm" / "rule" / "none"，说明结论来自哪一路
    urgency: Urgency
    entities: dict[str, list[str]] = field(default_factory=dict)
    reasoning: str = ""
    latency_ms: float = 0.0

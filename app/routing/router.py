"""路由：决定一条消息由谁主答、谁辅助，或者是否该澄清、该升级。

两步走：
  domain_scores()  给 general / technical / billing 三个业务域各打一个分
  decide()         根据分数和意图，产出结构化的 RoutingDecision

为什么不直接"意图组 -> Agent"一步映射？
  意图识别只输出一个标签，但用户一句话里可能有两件事，
  例如"登录报 401，而且这单还被扣了两次钱"。打分机制让第二件事也能被看见。
"""
from dataclasses import dataclass, field

from app.intent.rules import KEYWORDS
from app.intent.schema import INTENT_GROUP, Group, Intent, IntentResult, Urgency

# ── 打分参数：你的设计决定，阶段 5 可以用评测集调 ──
INTENT_WEIGHT = 0.7     # 意图所属的域直接加这么多，再乘以意图置信度
KEYWORD_WEIGHT = 0.25   # 每命中一个该域的关键词加这么多
KEYWORD_CAP = 0.5       # 关键词加分的上限，防止堆词刷分
GENERAL_BASE = 0.1      # 通用客服的保底分，保证任何时候都有人接
ENTITY_BONUS = {"error_code": ("technical", 0.2), "amount": ("billing", 0.15), "order_id": ("general", 0.1)}

# ── 决策参数 ──
SUPPORT_MIN = 0.25      # 辅助 Agent 的分数线。0.25 恰好等于命中一个关键词，
                        # 含义是：用户明确提到了另一个域的事，就值得让那个域的 Agent 也看一眼。
                        # 代价是每多一个辅助 Agent 就多几次模型调用，调高它可以省钱但会漏掉复合问题。


@dataclass
class RoutingDecision:
    action: str                       # "answer" 正常回答 / "clarify" 反问澄清 / "escalate" 转人工
    primary: str                      # general / technical / billing / escalation
    supporting: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""                  # 给人看的解释，调试和面试演示都靠它


def domain_scores(intent: IntentResult, message: str) -> dict[str, float]:
    scores = {"general": GENERAL_BASE, "technical": 0.0, "billing": 0.0}

    # 1. 意图：最强的信号。乘置信度，是为了让没把握的意图少说话。
    if intent.group.value in scores:
        scores[intent.group.value] += INTENT_WEIGHT * intent.confidence

    # 2. 关键词：复用规则路的词表，按意图组归到各域。这是发现"第二件事"的主要途径。
    msg = message.lower()
    hits = {"general": 0, "technical": 0, "billing": 0}
    for kw_intent, words in KEYWORDS.items():
        domain = INTENT_GROUP[kw_intent].value
        if domain in hits:
            hits[domain] += sum(1 for w in words if w in msg)
    for domain, n in hits.items():
        scores[domain] += min(KEYWORD_CAP, n * KEYWORD_WEIGHT)

    # 3. 实体：出现错误码偏技术，出现金额偏账单。
    for entity, (domain, bonus) in ENTITY_BONUS.items():
        if intent.entities.get(entity):
            scores[domain] += bonus

    return {k: round(v, 3) for k, v in scores.items()}


def decide(intent: IntentResult, message: str) -> RoutingDecision:
    """产出路由决策。

    按顺序判断：紧急度 CRITICAL 或用户要求转人工则升级；意图为 OTHER 则反问澄清；
    其余取分数最高的域为主 Agent，其他分数达到 SUPPORT_MIN 的专业域按分数排序作为辅助 Agent。
    通用客服只做主答或兜底，不做辅助。
    """
    scores = domain_scores(intent, message)
    tag = f"intent={intent.intent.value} conf={intent.confidence}"  # 每条 reason 的公共前缀

    # 1. 紧急情况优先级最高，即使意图本身很普通
    if intent.urgency == Urgency.CRITICAL:
        return RoutingDecision(
            action="escalate", primary="escalation", scores=scores,
            reason=f"{tag} | 紧急度 CRITICAL，直接转人工",
        )

    # 2. 用户明确要求转人工
    if intent.group == Group.ESCALATION:
        return RoutingDecision(
            action="escalate", primary="escalation", scores=scores,
            reason=f"{tag} | 用户要求人工服务",
        )

    # 3. 没看懂，先反问，避免把用户送错地方
    if intent.intent == Intent.OTHER:
        return RoutingDecision(
            action="clarify", primary="general", scores=scores,
            reason=f"{tag} | 意图不明，先反问澄清",
        )

    # 4. 常规路由：最高分主答，其余达标的专业域辅助
    primary = max(scores, key=scores.get)
    candidates = [
        domain for domain in scores
        if domain not in (primary, "general") and scores[domain] >= SUPPORT_MIN
    ]
    supporting = sorted(candidates, key=scores.get, reverse=True)
    return RoutingDecision(
        action="answer", primary=primary, supporting=supporting, scores=scores,
        reason=f"{tag} | primary={primary} supporting={supporting} scores={scores}",
    )

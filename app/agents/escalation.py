"""人工升级节点。刻意不调用模型。

理由：
  1. 用户要求转人工，或情况紧急时，最重要的是快和确定，模板回复零延迟。
  2. 这个节点一旦让模型自由发挥，它很容易"安慰性地"声称已经处理了某些事，造成事故。
生产环境里，这里是对接工单系统或人工坐席队列的位置。
"""
import json
import uuid

from app.agents.base import AgentReply
from app.agents.tools import ToolContext


class EscalationAgent:
    name = "escalation"

    async def run(self, message: str, ctx: ToolContext, background: str = "", history=None) -> AgentReply:
        ticket = "T" + uuid.uuid4().hex[:8].upper()
        known = {k: v for k, v in ctx.entities.items() if v}
        lines = [
            f"已为您转接人工客服，工单号 {ticket}。",
            f"已记录的信息：{json.dumps(known, ensure_ascii=False)}" if known else "目前还没有记录到订单号等关键信息，人工客服会向您确认。",
            "为了您的账户安全，请不要在对话中发送密码、短信验证码或完整银行卡号。",
        ]
        return AgentReply(self.name, "\n".join(lines), True, 0.0, [])

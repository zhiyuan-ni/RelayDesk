"""Agent 工具定义。

每个工具 = 一份给模型看的说明书（name / description / parameters）+ 一个真正干活的函数。
工具函数必须：只读、确定性、不伪造结果。查不到就明确返回"查不到"，绝不编造。
退款执行、改地址这类有副作用的操作不做成工具，统一走人工升级。
"""
from dataclasses import dataclass, field
from typing import Any, Callable

from app.business import store


@dataclass
class ToolContext:
    """工具执行时能看到的请求信息。模型给的参数不可全信，user_id 这类身份信息从这里取。"""
    user_id: str
    entities: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema，描述参数
    handler: Callable[[ToolContext, dict[str, Any]], Any]

    def to_openai(self) -> dict[str, Any]:
        """转成 OpenAI 协议要求的工具描述格式。"""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


def _params(required: list[str], **props: str) -> dict[str, Any]:
    """少写点样板：_params(["order_id"], order_id="订单号") 生成一份 JSON Schema。"""
    return {
        "type": "object",
        "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": required,
    }


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def get_order_status(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    order = store.get_order(args["order_id"])
    if order is None:
        return {"found": False, "message": f"未找到订单 {args['order_id']}，请用户核对订单号"}
    if order["user_id"] != ctx.user_id:
        # 越权保护：只能查自己的订单。不透露订单是否存在。
        return {"found": False, "message": "该订单不属于当前用户，无法查询"}
    return {"found": True, **{k: order[k] for k in ("item", "amount", "status", "paid_at", "shipped_at", "logistics")}}


def get_payment_records(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    order = store.get_order(args["order_id"])
    if order is None or order["user_id"] != ctx.user_id:
        return {"found": False, "message": "未找到该用户名下的这笔订单"}
    payments = store.get_payments(args["order_id"])
    success = [p for p in payments if p["status"] == "成功"]
    return {
        "found": True,
        "order_amount": order["amount"],
        "payments": payments,
        "successful_count": len(success),
        "total_paid": sum(p["amount"] for p in success),
        # 只陈述事实，不下"重复扣款"的结论，结论和退款由人工审核确认
        "note": "成功支付笔数大于 1 时可能存在重复扣款，需人工复核后处理",
    }


def get_refund_status(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    order = store.get_order(args["order_id"])
    if order is None or order["user_id"] != ctx.user_id:
        return {"found": False, "message": "未找到该用户名下的这笔订单"}
    refund = store.get_refund(args["order_id"])
    return {"found": True, "refund": refund} if refund else {"found": True, "refund": None, "message": "该订单暂无退款申请记录"}


def get_invoice_status(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    order = store.get_order(args["order_id"])
    if order is None or order["user_id"] != ctx.user_id:
        return {"found": False, "message": "未找到该用户名下的这笔订单"}
    invoice = store.get_invoice(args["order_id"])
    return {"found": True, "invoice": invoice} if invoice else {"found": True, "invoice": None, "message": "该订单尚未开具发票"}


_ERROR_CODES = {
    "401": ("认证失败", ["确认登录状态是否过期，重新登录", "检查系统时间是否准确", "清除缓存后重试"]),
    "403": ("权限不足", ["确认账号套餐是否包含该功能", "确认是否在受限网络环境"]),
    "404": ("资源不存在", ["确认访问的链接或订单号是否正确"]),
    "500": ("服务端异常", ["记录发生时间和操作步骤", "稍后重试", "持续出现需升级给技术团队查看服务端日志"]),
    "502": ("网关错误", ["通常是服务短暂不可用，稍后重试"]),
}


def lookup_error_code(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    code = str(args["error_code"]).strip()
    if code not in _ERROR_CODES:
        return {"known": False, "message": f"错误码 {code} 不在知识表中，请收集完整报错截图"}
    meaning, steps = _ERROR_CODES[code]
    return {"known": True, "error_code": code, "meaning": meaning, "suggested_steps": steps}


def get_login_events(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    # 注意：不接受模型传 user_id，永远用当前登录用户，防止被诱导查别人的记录
    events = store.get_login_events(ctx.user_id)
    failed = [e for e in events if e["result"] == "失败"]
    return {"events": events, "recent_failed_count": len(failed)}


# ── 工具注册表 ───────────────────────────────────────────────────────────────

ALL_TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("get_order_status", "按订单号查询订单状态和物流进度",
             _params(["order_id"], order_id="订单号，例如 A12345"), get_order_status),
    ToolSpec("get_payment_records", "按订单号查询该订单的全部支付流水，用于核对扣款",
             _params(["order_id"], order_id="订单号"), get_payment_records),
    ToolSpec("get_refund_status", "按订单号查询退款申请进度",
             _params(["order_id"], order_id="订单号"), get_refund_status),
    ToolSpec("get_invoice_status", "按订单号查询发票开具情况",
             _params(["order_id"], order_id="订单号"), get_invoice_status),
    ToolSpec("lookup_error_code", "查询 HTTP 错误码的含义和排查步骤",
             _params(["error_code"], error_code="三位错误码，例如 401"), lookup_error_code),
    ToolSpec("get_login_events", "查询当前用户最近的登录记录，用于排查登录故障和账号安全",
             _params([]), get_login_events),
]}


def pick_tools(names: tuple[str, ...]) -> dict[str, ToolSpec]:
    """按白名单取工具。Agent 只能拿到自己名单里的工具。"""
    return {n: ALL_TOOLS[n] for n in names}

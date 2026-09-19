"""模拟业务库。

真实系统里这里会是订单服务、支付服务的 API 客户端。我们用一个 JSON 文件代替，
目的是让工具"真的查得到东西"，同时保持整个项目不依赖外部系统。
对上层来说，换成真实服务只需要重写这个文件，工具和 Agent 都不用改。
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_DB_PATH = Path(__file__).parent / "mock_db.json"


@lru_cache(maxsize=1)  # 文件只读一次，之后直接用内存里的结果
def _db() -> dict[str, Any]:
    return json.loads(_DB_PATH.read_text(encoding="utf-8"))


def get_order(order_id: str) -> Optional[dict[str, Any]]:
    return _db()["orders"].get(order_id.strip().lstrip("#").upper())


def get_payments(order_id: str) -> list[dict[str, Any]]:
    oid = order_id.strip().lstrip("#").upper()
    return [p for p in _db()["payments"] if p["order_id"] == oid]


def get_refund(order_id: str) -> Optional[dict[str, Any]]:
    oid = order_id.strip().lstrip("#").upper()
    return next((r for r in _db()["refunds"] if r["order_id"] == oid), None)


def get_invoice(order_id: str) -> Optional[dict[str, Any]]:
    oid = order_id.strip().lstrip("#").upper()
    return next((i for i in _db()["invoices"] if i["order_id"] == oid), None)


def get_login_events(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    events = [e for e in _db()["login_events"] if e["user_id"] == user_id]
    return events[-limit:]

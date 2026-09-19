"""execute_tool_call() 的规格测试，编号对应 docstring 里的 5 条。"""
from app.agents.base import execute_tool_call
from app.agents.tools import ToolContext, ToolSpec, pick_tools

CTX = ToolContext(user_id="u1001")
TOOLS = pick_tools(("get_order_status", "get_login_events"))
KEYS = {"tool", "args", "success", "data", "error", "latency_ms"}


async def test_1_tool_not_in_whitelist():
    t = await execute_tool_call(TOOLS, "get_payment_records", '{"order_id": "A12345"}', CTX)
    assert set(t) == KEYS
    assert t["success"] is False and "白名单" in t["error"] and t["args"] == {} and t["data"] is None


async def test_2_bad_json():
    t = await execute_tool_call(TOOLS, "get_order_status", "{order_id: A12345", CTX)
    assert t["success"] is False and "JSON" in t["error"] and t["args"] == {}


async def test_2_empty_args_string_means_no_args():
    t = await execute_tool_call(TOOLS, "get_login_events", "", CTX)
    assert t["success"] is True and t["data"]["recent_failed_count"] == 2


async def test_3_missing_required_param():
    t = await execute_tool_call(TOOLS, "get_order_status", "{}", CTX)
    assert t["success"] is False and "order_id" in t["error"]


async def test_4_handler_raises():
    def boom(ctx, args):
        raise RuntimeError("数据库连接失败")
    tools = {"boom": ToolSpec("boom", "总是失败", {"type": "object", "properties": {}, "required": []}, boom)}
    t = await execute_tool_call(tools, "boom", "{}", CTX)
    assert t["success"] is False and t["error"] == "数据库连接失败" and t["data"] is None


async def test_5_success():
    t = await execute_tool_call(TOOLS, "get_order_status", '{"order_id": "A12345"}', CTX)
    assert set(t) == KEYS
    assert t["success"] is True and t["error"] == ""
    assert t["tool"] == "get_order_status" and t["args"] == {"order_id": "A12345"}
    assert t["data"]["status"] == "运输中" and t["latency_ms"] >= 0


async def test_5_async_handler_is_awaited():
    async def slow(ctx, args):
        return {"ok": True}
    tools = {"slow": ToolSpec("slow", "异步工具", {"type": "object", "properties": {}, "required": []}, slow)}
    t = await execute_tool_call(tools, "slow", "{}", CTX)
    assert t["success"] is True and t["data"] == {"ok": True}


async def test_2_json_but_not_an_object():
    # 合法 JSON 但不是对象。函数不能抛异常，必须返回失败 trace
    for raw in ("123", "[1, 2]", '"A12345"'):
        t = await execute_tool_call(TOOLS, "get_order_status", raw, CTX)
        assert t["success"] is False and "JSON" in t["error"] and t["args"] == {}

from app.agents.tools import ALL_TOOLS, ToolContext, pick_tools

U1 = ToolContext(user_id="u1001")
U2 = ToolContext(user_id="u1002")


def call(name, ctx, **args):
    return ALL_TOOLS[name].handler(ctx, args)


def test_order_status_found():
    r = call("get_order_status", U1, order_id="#a12345")  # 带井号和小写也应能查到
    assert r["found"] and r["status"] == "运输中"


def test_order_of_other_user_is_hidden():
    r = call("get_order_status", U2, order_id="A12345")
    assert r["found"] is False and "不属于" in r["message"]


def test_payment_records_show_two_successful_charges():
    r = call("get_payment_records", U1, order_id="B20250917")
    assert r["successful_count"] == 2 and r["total_paid"] == 396.0


def test_failed_payment_not_counted():
    r = call("get_payment_records", U2, order_id="D88002")
    assert r["successful_count"] == 1


def test_login_events_use_context_user_not_model_args():
    r = call("get_login_events", U1, user_id="u1002")  # 模型试图查别人，应被忽略
    assert r["recent_failed_count"] == 2
    assert all("境外" in e["location"] or e["result"] == "成功" for e in r["events"])


def test_unknown_error_code():
    assert call("lookup_error_code", U1, error_code="418")["known"] is False


def test_pick_tools_is_a_whitelist():
    tools = pick_tools(("lookup_error_code", "get_login_events"))
    assert set(tools) == {"lookup_error_code", "get_login_events"}
    assert all(t.to_openai()["type"] == "function" for t in tools.values())

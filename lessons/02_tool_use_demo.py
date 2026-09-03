"""看清 tool-use 的一来一回。

运行：uv run python lessons/02_tool_use_demo.py

tool-use 的本质
---------------
模型自己不能查数据库。我们做的是：
  1. 把"有哪些工具、各需要什么参数"用 JSON Schema 描述给模型。
  2. 模型如果觉得需要，就不直接回答，而是返回"我想调用 X 工具，参数是 Y"。
  3. 我们的代码真正去执行 X，把结果作为一条 role=tool 的消息追加进对话。
  4. 再次请求模型。它看到工具结果后给出最终回答，或者继续要求调用别的工具。

所以 Agent 本质上是一个循环，模型只负责"决定调什么"，执行权始终在我们的代码手里。
这就是为什么我们能做白名单、参数校验、超时和审计。
"""
import asyncio
import json
import sys
from pathlib import Path

# 本文件在 lessons/ 目录下，Python 默认只在这个目录里找模块。
# 把项目根目录加进搜索路径，才能 import app。
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.business import store  # noqa: E402
from app.config import settings
from app.llm import LLMClient

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "按订单号查询订单状态和物流信息",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号，例如 A12345"}},
            "required": ["order_id"],
        },
    },
}]


def show(title: str, obj) -> None:
    print(f"\n── {title} ──")
    print(json.dumps(obj, ensure_ascii=False, indent=2) if not isinstance(obj, str) else obj)


async def main() -> None:
    llm = LLMClient(settings)
    messages = [{"role": "user", "content": "帮我看看订单 A12345 到哪了"}]

    # 第 1 次请求：模型看到问题和工具清单
    msg = await llm.chat(messages, system="你是客服。需要订单信息时必须调用工具，不要编造。", tools=TOOLS)
    if not msg.tool_calls:
        show("模型没有调用工具，直接回答了", msg.content or "")
        return

    call = msg.tool_calls[0]
    show("第 1 次响应：模型要求调用工具", {"name": call.function.name, "arguments": call.function.arguments})

    # 我们的代码执行工具。注意 arguments 是 JSON 字符串，要先解析
    args = json.loads(call.function.arguments)
    result = store.get_order(args["order_id"]) or {"error": "订单不存在"}
    show("我们执行工具得到的结果", result)

    # 把"模型的调用请求"和"工具结果"都追加进对话，顺序不能错
    messages.append({
        "role": "assistant", "content": msg.content or "",
        "tool_calls": [{"id": call.id, "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments}}],
    })
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})

    # 第 2 次请求：模型看到工具结果，给出最终回答
    final = await llm.chat(messages, system="你是客服。需要订单信息时必须调用工具，不要编造。", tools=TOOLS)
    show("第 2 次响应：最终回答", final.content or "")


if __name__ == "__main__":
    asyncio.run(main())

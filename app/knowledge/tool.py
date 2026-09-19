"""把知识库检索包装成 Agent 可以调用的工具。

它和订单、支付那些工具的两点不同：
  1. 它是共享工具，三类业务 Agent 都能用，因为谁都可能被问到政策问题。
  2. 它的处理函数是 async 的，因为要调 Embedding 接口。
     execute_tool_call 里那两行 inspect.isawaitable 就是为它准备的：
     同步工具直接返回结果，异步工具返回的是协程，需要再 await 一次才能拿到结果。
"""
from typing import Any

from app.agents.tools import ToolContext, ToolSpec

TOOL_NAME = "search_knowledge_base"


def build_knowledge_tool(kb) -> ToolSpec:
    async def search_knowledge_base(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        hits = await kb.search(str(args["query"]), top_k=4)
        if not hits:
            return {"found": False, "message": "知识库中没有找到相关内容，请如实告知用户，不要编造政策"}
        return {"found": True, "results": hits}

    return ToolSpec(
        name=TOOL_NAME,
        description=("检索平台的政策与规则文档，包括退款退货、配送、发票、会员积分、账户安全、常见故障排查。"
                     "凡是涉及规则、时效、费用、条件的问题，必须先检索再回答。"),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索语句，用完整的一句话描述要查什么"}},
            "required": ["query"],
        },
        handler=search_knowledge_base,
    )

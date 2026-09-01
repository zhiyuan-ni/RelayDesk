"""异步入门：10 分钟看懂 async / await / gather。

运行：uv run python lessons/01_async_basics.py

核心直觉
--------
调一次大模型要等 1 到 3 秒，这段时间 CPU 其实在发呆，只是在等网络。
  - 同步写法：等的时候整个程序卡住，别的用户也得排队。
  - 异步写法：等的时候把 CPU 让出去，谁的结果先回来就先处理谁。

三个关键字
----------
  async def f():   定义一个"可以中途暂停"的函数，叫协程函数。
                   调用 f() 不会执行它，只得到一个协程对象。
  await f()        真正执行，并且在等待期间允许别的任务运行。
                   await 只能写在 async def 里面。
  asyncio.gather   同时启动多个协程，全部完成后按顺序返回结果列表。

这个项目里哪里会用到
--------------------
  - 意图识别：LLM 判断和向量匹配同时跑。
  - 多 Agent：技术 Agent 和账单 Agent 同时回答。
  - RAG：3 个改写后的子查询同时检索。
"""
import asyncio
import time


async def fake_llm_call(name: str, seconds: float) -> str:
    """用 sleep 模拟一次耗时的模型调用。"""
    await asyncio.sleep(seconds)  # 注意不是 time.sleep，后者会卡死整个程序
    return f"{name} 完成"


async def sequential() -> None:
    t0 = time.monotonic()
    a = await fake_llm_call("技术Agent", 1.0)
    b = await fake_llm_call("账单Agent", 1.5)
    print(f"[串行] {a}, {b}，耗时 {time.monotonic() - t0:.1f}s  <- 1.0 + 1.5")


async def parallel() -> None:
    t0 = time.monotonic()
    a, b = await asyncio.gather(
        fake_llm_call("技术Agent", 1.0),
        fake_llm_call("账单Agent", 1.5),
    )
    print(f"[并行] {a}, {b}，耗时 {time.monotonic() - t0:.1f}s  <- max(1.0, 1.5)")


async def one_fails() -> None:
    """一个任务出错时，不应该拖垮另一个。"""

    async def broken() -> str:
        await asyncio.sleep(0.2)
        raise RuntimeError("模型超时")

    results = await asyncio.gather(
        fake_llm_call("技术Agent", 0.5),
        broken(),
        return_exceptions=True,  # 出错的那个以异常对象的形式放进结果，不向外抛
    )
    for r in results:
        print("[容错]", "失败:" if isinstance(r, Exception) else "成功:", r)


# ───────────────────────── 练习（请你来写）─────────────────────────
# 目标：实现 call_with_timeout。
#   - 在 limit 秒内完成，返回 fake_llm_call 的结果
#   - 超时则返回字符串 f"{name} 超时，已降级"
# 提示：asyncio.wait_for(协程, timeout=秒数) 超时会抛 asyncio.TimeoutError
# 这正是阶段 3 工具层"超时控制 + fallback"的最小原型。
async def call_with_timeout(name: str, seconds: float, limit: float) -> str:
    try:
        return await asyncio.wait_for(fake_llm_call(name, seconds), timeout=limit)
    except asyncio.TimeoutError:
        return f"{name} 超时，已降级"


async def exercise() -> None:
    try:
        print("[练习]", await call_with_timeout("快的", 0.3, limit=1.0))  # 期望：快的 完成
        print("[练习]", await call_with_timeout("慢的", 2.0, limit=1.0))  # 期望：慢的 超时，已降级
    except NotImplementedError:
        print("[练习] 还没做，打开本文件找到 call_with_timeout")


async def main() -> None:
    await sequential()
    await parallel()
    await one_fails()
    await exercise()


if __name__ == "__main__":
    asyncio.run(main())  # 程序入口处用 asyncio.run 启动整个异步世界

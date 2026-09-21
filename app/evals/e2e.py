"""端到端评测的场景定义和确定性检查。

一条场景 = 一个用户 + 若干轮消息，每轮带一组期望。期望分两类：
    确定性检查：路由到了谁、调了哪些工具、回复里必须出现或绝不能出现什么。由代码判定，结果非对即错。
    质量评分：回复是否基于事实、是否有帮助、是否越界承诺。由评委模型打分，见 judge.py。

确定性检查是主指标。它检查的是系统行为，不受评委口味影响，每次运行结果可复现。
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnExpectation:
    action: str = ""                                   # answer / escalate / clarify，空表示不检查
    primary: str = ""                                  # 主 Agent，空表示不检查
    supporting: list[str] = field(default_factory=list)   # 必须包含的辅助 Agent
    tools: list[str] = field(default_factory=list)        # 必须调用过的工具
    tools_none: list[str] = field(default_factory=list)   # 绝不能调用的工具，["*"] 表示任何工具都不能调
    contain_any: list[str] = field(default_factory=list)  # 回复至少包含其中一个
    contain_all: list[str] = field(default_factory=list)  # 回复必须全部包含
    contain_none: list[str] = field(default_factory=list) # 回复一个都不能包含
    agents: list[str] = field(default_factory=list)       # 主辅不限，这些 Agent 都必须参与。复合问题用它，不写死谁主谁辅

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnExpectation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TurnOutcome:
    """一轮对话的实际结果，从 OrchestratorResult 提炼出检查需要的字段。"""
    response: str
    action: str
    primary: str
    supporting: list[str]
    tools_called: list[str]


def check_turn(outcome: TurnOutcome, expect: TurnExpectation) -> list[str]:
    """逐项核对一轮对话的实际结果与期望，返回所有未通过项的说明，空列表表示全部通过。

    不在第一个失败处返回，而是列出全部失败项，这样跑一次就能看到所有问题。
    核对的项目见 TurnExpectation 各字段的注释。
    """
    failures = []
    if expect.action and outcome.action != expect.action:
        failures.append(f"期望路由 {expect.action}，实际 {outcome.action}")
    if expect.primary and outcome.primary != expect.primary:
        failures.append(f"主 Agent 期望 {expect.primary}，实际 {outcome.primary}")
    for s in expect.supporting:
        if s not in outcome.supporting:
            failures.append(f"期望辅助 Agent 包含 {s}，实际 {outcome.supporting}")
    for t in expect.tools:
        if t not in outcome.tools_called:
            failures.append(f"期望调用工具包含 {t}，实际 {outcome.tools_called}")
    if expect.tools_none == ["*"]:
        if outcome.tools_called:
            failures.append(f"期望不调用任何工具，实际调用了 {outcome.tools_called}")
    else:
        for t in expect.tools_none:
            if t in outcome.tools_called:
                failures.append(f"期望不调用工具 {t}，实际调用了")
    if expect.contain_any and not any(s in outcome.response for s in expect.contain_any):
        failures.append(f"期望回复至少包含其中一个 {expect.contain_any}，实际回复没有")
    for s in expect.contain_all:
        if s not in outcome.response:
            failures.append(f"期望回复包含 {s}，实际回复没有")
    involved = [outcome.primary] + outcome.supporting
    for agent in expect.agents:
        if agent not in involved:
            failures.append(f"Agent {agent} 应当参与，实际参与的是 {involved}")
    for s in expect.contain_none:
        if s in outcome.response:
            failures.append(f"期望回复不包含 {s}，实际回复包含了")
    return failures

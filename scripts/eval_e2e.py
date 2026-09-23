"""端到端评测：把真实场景灌进完整的编排链路，核对行为，再让评委打分。

运行：uv run python scripts/eval_e2e.py [--no-judge] [--only 场景id]

链路里除了记忆用内存版 Redis 替代之外，其余全部是真的：真实模型、真实知识库、真实工具。
每个场景用独立的会话号，场景之间互不影响；同一场景内的多轮共享会话，用来测记忆。
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import fakeredis  # noqa: E402

from app.config import Settings, settings  # noqa: E402
from app.evals.e2e import TurnExpectation, TurnOutcome, check_turn  # noqa: E402
from app.evals.judge import Judge  # noqa: E402
from app.knowledge.embedder import Embedder  # noqa: E402
from app.knowledge.kb import KnowledgeBase  # noqa: E402
from app.knowledge.retriever import Retriever  # noqa: E402
from app.knowledge.tool import build_knowledge_tool  # noqa: E402
from app.jev import JevClient, open_jev  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.memory.manager import MemoryManager  # noqa: E402
from app.memory.store import ConversationStore  # noqa: E402
from app.orchestrator import Orchestrator  # noqa: E402

SCENARIOS_PATH = ROOT / "evals" / "e2e_scenarios.json"
REPORT_PATH = ROOT / "evals" / "reports" / "e2e_latest.json"


def build_orchestrator(llm: LLMClient, jev: Optional[JevClient] = None) -> Orchestrator:
    kb = KnowledgeBase(Embedder(settings, client=llm.client), settings.kb_path)
    retriever = Retriever(kb, llm.for_model(settings.rerank_model), rewrite=settings.retrieval_rewrite,
                          rerank=settings.retrieval_rerank, jev=jev if settings.rerank_backend == "jev" else None)
    tool = build_knowledge_tool(retriever)
    memory = MemoryManager(ConversationStore(fakeredis.FakeAsyncRedis(decode_responses=True)), llm)
    return Orchestrator(llm, shared_tools={tool.name: tool}, memory=memory,
                        intent_llm=llm.for_model(settings.intent_model),
                        intent_jev=jev if settings.intent_backend == "jev" else None)


async def run_scenario(orch: Orchestrator, judge, scenario: dict, run_id: str) -> dict:
    conv_id = f"e2e-{scenario['id']}-{run_id}"
    conversation: list[dict[str, str]] = []
    turns_out = []
    for i, turn in enumerate(scenario["turns"]):
        t0 = time.monotonic()
        result = await orch.handle(turn["message"], scenario["user_id"], conv_id)
        latency = round((time.monotonic() - t0) * 1000)
        outcome = TurnOutcome(
            response=result.response, action=result.decision.action, primary=result.decision.primary,
            supporting=result.decision.supporting, tools_called=[t["tool"] for t in result.tool_traces],
        )
        failures = check_turn(outcome, TurnExpectation.from_dict(turn["expect"]))
        score = await judge.score(conversation, result.response, scenario["facts"]) if judge else None
        conversation += [{"role": "user", "content": turn["message"]}, {"role": "assistant", "content": result.response}]
        turns_out.append({
            "turn": i + 1, "message": turn["message"], "response": result.response, "latency_ms": latency,
            "action": outcome.action, "primary": outcome.primary, "supporting": outcome.supporting,
            "tools_called": outcome.tools_called, "failures": failures,
            "judge": None if score is None else {"factual": score.factual, "helpful": score.helpful,
                                                  "policy": score.policy, "overall": score.overall,
                                                  "reason": score.reason, "failed": score.failed},
        })
    return {"id": scenario["id"], "title": scenario["title"], "turns": turns_out}


async def main(use_judge: bool, only: str) -> None:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if only:
        scenarios = [s for s in scenarios if s["id"] == only]
    llm = LLMClient(settings)
    jev = await open_jev(settings)   # 和服务启动一致地预热，否则第一轮的延迟里多出约 5 秒建连时间
    orch = build_orchestrator(llm, jev)
    print(f"意图后端 {settings.intent_backend}，重排后端 {settings.rerank_backend}\n")
    judge = None
    if use_judge:
        judge_cfg = Settings(**{**settings.__dict__, "llm_model": settings.judge_model, "llm_enable_thinking": None})
        judge = Judge(LLMClient(judge_cfg))
        print(f"被评模型 {settings.llm_model}，评委模型 {settings.judge_model}\n")

    run_id = str(int(time.time()))
    results = [await run_scenario(orch, judge, s, run_id) for s in scenarios]

    total_turns = passed_turns = 0
    scores = []
    for r in results:
        turn_ok = all(not t["failures"] for t in r["turns"])
        print(f"{'✓' if turn_ok else '✗'} {r['id']}  {r['title']}")
        for t in r["turns"]:
            total_turns += 1
            passed_turns += not t["failures"]
            j = t["judge"]
            jtxt = "" if not j else (" 评委失败" if j["failed"] else f" 评委 {j['factual']}/{j['helpful']}/{j['policy']}")
            print(f"    第 {t['turn']} 轮 {'通过' if not t['failures'] else '失败'}  {t['latency_ms']}ms  "
                  f"路由 {t['action']}/{t['primary']}{'+' + ','.join(t['supporting']) if t['supporting'] else ''}  "
                  f"工具 {t['tools_called'] or '无'}{jtxt}")
            for f in t["failures"]:
                print(f"        ✗ {f}")
            if j and not j["failed"]:
                scores.append(j)
                if j["overall"] < 4:
                    print(f"        评委: {j['reason']}")

    print(f"\n确定性检查：{passed_turns}/{total_turns} 轮通过，{sum(all(not t['failures'] for t in r['turns']) for r in results)}/{len(results)} 个场景全部通过")
    if scores:
        avg = {k: round(sum(s[k] for s in scores) / len(scores), 2) for k in ("factual", "helpful", "policy", "overall")}
        print(f"评委均分（1 到 5）：事实 {avg['factual']}  有帮助 {avg['helpful']}  边界 {avg['policy']}  综合 {avg['overall']}")

    # 只跑单个场景、或有环节不用默认的 llm 后端时另存一份，不覆盖全量报告
    report_path = REPORT_PATH.with_name(f"e2e_{only}.json") if only else REPORT_PATH
    backends = {"intent": settings.intent_backend, "rerank": settings.rerank_backend}
    suffix = "".join(f"_{step}-{b}" for step, b in backends.items() if b != "llm")
    report_path = report_path.with_name(f"{report_path.stem}{suffix}.json")
    report_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": settings.llm_model,
        "judge_model": settings.judge_model if use_judge else None, "backends": backends,
        "turns_total": total_turns, "turns_passed": passed_turns, "scenarios": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-judge", action="store_true", help="只做确定性检查，不调评委")
    p.add_argument("--only", default="", help="只跑一个场景，传场景 id")
    a = p.parse_args()
    asyncio.run(main(not a.no_judge, a.only))

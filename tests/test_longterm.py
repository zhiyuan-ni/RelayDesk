"""长期记忆测试：rank_memories 的规格，加上存取、用户隔离和模型迁移。"""
import chromadb
import pytest

from app.memory.longterm import SECONDS_PER_DAY, LongTermMemory, rank_memories
from app.memory.manager import build_episode_text
from tests.fakes import FakeEmbedder

NOW = 1_800_000_000.0


def mem(conv_id: str, similarity: float, days_ago: float = 0.0) -> dict:
    return {"conv_id": conv_id, "text": f"{conv_id} 的内容", "similarity": similarity,
            "ts": NOW - days_ago * SECONDS_PER_DAY}


# ── rank_memories 规格 ──

def test_1_current_conversation_is_excluded():
    ranked = rank_memories([mem("c1", 0.9), mem("c2", 0.8)], NOW, exclude_conv_id="c1")
    assert [m["conv_id"] for m in ranked] == ["c2"]


def test_2_weak_matches_are_dropped():
    ranked = rank_memories([mem("c1", 0.49), mem("c2", 0.50)], NOW, min_similarity=0.5)
    assert [m["conv_id"] for m in ranked] == ["c2"]


def test_3_half_life_decay():
    [fresh] = rank_memories([mem("c1", 0.8, days_ago=0)], NOW, half_life_days=30)
    [one] = rank_memories([mem("c1", 0.8, days_ago=30)], NOW, half_life_days=30)
    [two] = rank_memories([mem("c1", 0.8, days_ago=60)], NOW, half_life_days=30)
    assert (fresh["final_score"], one["final_score"], two["final_score"]) == (0.8, 0.4, 0.2)


def test_4_recent_memory_can_outrank_a_more_similar_old_one():
    old_but_similar = mem("old", 0.90, days_ago=90)      # 0.90 * 0.125 = 0.1125
    recent = mem("recent", 0.60, days_ago=1)             # 约 0.586
    ranked = rank_memories([old_but_similar, recent], NOW, half_life_days=30)
    assert [m["conv_id"] for m in ranked] == ["recent", "old"]


def test_4_top_k():
    hits = [mem(f"c{i}", 0.5 + i * 0.1) for i in range(4)]
    assert [m["conv_id"] for m in rank_memories(hits, NOW, top_k=2)] == ["c3", "c2"]


def test_5_input_is_not_mutated_and_empty_is_fine():
    original = mem("c1", 0.8)
    rank_memories([original], NOW)
    assert "final_score" not in original and rank_memories([], NOW) == []


# ── build_episode_text ──

def test_episode_text_keeps_only_what_the_user_said():
    text = build_episode_text("用户在处理退款。", [
        {"role": "user", "content": "订单 B20250917 扣了两次"},
        {"role": "assistant", "content": "这是一段很长的客服回复" * 20},
    ])
    assert "用户在处理退款。" in text and "B20250917" in text and "很长的客服回复" not in text


# ── LongTermMemory ──

class Clock:
    def __init__(self): self.t = NOW
    def __call__(self): return self.t


@pytest.fixture
def ltm(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    # 假向量模型给出的相似度整体偏低，相关内容约 0.4、无关内容约 0.28，所以门槛取两者之间
    return LongTermMemory(FakeEmbedder(), client, now=Clock(), min_similarity=0.33)


async def test_remember_then_recall_from_another_conversation(ltm):
    await ltm.remember("u1", "c1", "用户咨询订单 B20250917 重复扣款，已转人工复核")
    await ltm.remember("u1", "c2", "用户询问发票抬头怎么修改")
    hits = await ltm.recall("u1", "上次那个重复扣款的订单怎么样了", exclude_conv_id="c3")
    assert [h["conv_id"] for h in hits] == ["c1"]          # 发票那条相似度不够，被门槛挡掉


async def test_same_conversation_updates_one_record(ltm):
    await ltm.remember("u1", "c1", "第一版")
    await ltm.remember("u1", "c1", "用户咨询订单重复扣款")
    assert await ltm.count() == 1


async def test_users_never_see_each_others_memories(ltm):
    await ltm.remember("u1", "c1", "用户咨询订单 B20250917 重复扣款")
    assert await ltm.recall("u2", "重复扣款的订单", exclude_conv_id="") == []
    assert await ltm.recall("u1", "重复扣款的订单", exclude_conv_id="")


async def test_current_conversation_is_not_recalled(ltm):
    await ltm.remember("u1", "c1", "用户咨询订单重复扣款")
    assert await ltm.recall("u1", "重复扣款", exclude_conv_id="c1") == []


async def test_changing_embedding_model_migrates_existing_memories(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    first = LongTermMemory(FakeEmbedder("fake-embed-a"), client, now=Clock())
    assert await first.ensure_ready() == "reused"
    await first.remember("u1", "c1", "用户咨询订单重复扣款")

    second = LongTermMemory(FakeEmbedder("fake-embed-b"), client, now=Clock(), min_similarity=0.33)
    assert await second.ensure_ready() == "migrated" and await second.count() == 1
    assert (await second.recall("u1", "重复扣款", exclude_conv_id=""))[0]["text"] == "用户咨询订单重复扣款"

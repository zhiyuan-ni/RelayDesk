"""pack_sentences() 的规格测试，加上 chunk_markdown 的集成测试。"""
from app.knowledge.chunker import chunk_markdown, pack_sentences, split_sentences

S = ["第一句。", "第二句。", "第三句。", "第四句。"]   # 每句 4 个字符


def test_split_sentences_keeps_punctuation():
    assert split_sentences("你好。在吗？好的！") == ["你好。", "在吗？", "好的！"]


def test_1_empty():
    assert pack_sentences([], 10) == []


def test_2_everything_fits_in_one_chunk():
    assert pack_sentences(S, 100) == ["第一句。第二句。第三句。第四句。"]


def test_2_greedy_packing_without_overlap():
    assert pack_sentences(S, 8, overlap=0) == ["第一句。第二句。", "第三句。第四句。"]


def test_3_overlap_repeats_last_sentence():
    # 上限 8 个字符即两句。每个新片段以上一片段的最后一句开头
    assert pack_sentences(S, 8, overlap=1) == ["第一句。第二句。", "第二句。第三句。", "第三句。第四句。"]


def test_4_oversized_sentence_stands_alone():
    long = "这是一个非常非常长的句子超过了上限。"
    assert pack_sentences(["短句。", long, "尾句。"], 8, overlap=0) == ["短句。", long, "尾句。"]


def test_5_last_chunk_is_not_lost():
    assert pack_sentences(["甲。", "乙。", "丙。"], 4, overlap=0) == ["甲。乙。", "丙。"]


def test_chunk_markdown_uses_headings():
    md = "# 退款政策\n\n## 时效\n审核要三天。到账要七天。\n\n## 运费\n用户承担。\n"
    chunks = chunk_markdown(md, max_chars=100)
    assert [(c.title, c.section) for c in chunks] == [("退款政策", "时效"), ("退款政策", "运费")]
    assert chunks[0].text == "审核要三天。到账要七天。"
    assert chunks[0].for_embedding().startswith("退款政策 - 时效\n")


def test_real_docs_produce_reasonable_chunks():
    from pathlib import Path
    docs = sorted(Path("app/knowledge/docs").glob("*.md"))
    assert len(docs) == 6
    for path in docs:
        chunks = chunk_markdown(path.read_text(encoding="utf-8"))
        assert len(chunks) >= 3 and all(c.title and c.section and c.text for c in chunks)

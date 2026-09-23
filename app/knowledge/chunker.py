"""文档切片。

为什么要切片：
  1. 整篇文档做一个向量，语义会被平均掉。"退款时效"那一段的含义会被其他段落稀释。
  2. 检索结果要塞进模型的上下文，片段越精准，噪音越少、token 越省。

两级切分：
  先按 Markdown 二级标题切成小节，因为作者分节时已经按主题分好了，这是最可靠的语义边界。
  小节仍然太长时，再按句子贪心打包，并让相邻片段重叠一句，防止关键信息恰好被切在边界上。
"""
import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[。！？!?\n])")  # 在句末标点之后切，标点留在句子里


@dataclass
class Chunk:
    title: str      # 文档标题，来自一级标题
    section: str    # 小节标题，来自二级标题
    text: str

    def for_embedding(self) -> str:
        """向量化时带上标题。"1 到 3 个工作日到账"单独看不知道在说什么，加上"退款时效"语义就完整了。"""
        return f"{self.title} - {self.section}\n{self.text}"


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def pack_sentences(sentences: list[str], max_chars: int, overlap: int = 1) -> list[str]:
    """把句子贪心地打包成片段，每个片段不超过 max_chars 个字符。

    新片段以上一个片段的最后 overlap 个句子开头，防止关键信息恰好被切在边界上。
    单个句子本身超过上限时不再细分，独自成为一个片段。
    """
    current: list[str] = []   # 当前片段里的句子
    chunks: list[str] = []    # 已完成的片段
    for sentence in sentences:
        if current and len("".join(current)) + len(sentence) > max_chars:
            chunks.append("".join(current))
            # 重叠只决定新片段怎么开头。current[-0:] 会返回整个列表，所以 0 要单独处理
            current = current[-overlap:] if overlap > 0 else []
        current.append(sentence)
    if current:
        # 最后一个片段原样收尾。它的开头已经带着上一个片段的重叠句，这里不能再截取，否则会丢内容
        chunks.append("".join(current))
    return chunks


def chunk_markdown(markdown: str, max_chars: int = 300, overlap: int = 1) -> list[Chunk]:
    """把一篇 Markdown 文档切成若干 Chunk。"""
    title, section, buffer = "", "", []
    sections: list[tuple[str, str]] = []   # (小节标题, 小节正文)

    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            if "".join(buffer).strip():
                sections.append((section, "\n".join(buffer).strip()))
            section, buffer = line[3:].strip(), []
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append((section, "\n".join(buffer).strip()))

    return [
        Chunk(title, sec, piece)
        for sec, body in sections
        for piece in pack_sentences(split_sentences(body), max_chars, overlap)
    ]

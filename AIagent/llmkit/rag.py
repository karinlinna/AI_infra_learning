"""极简 RAG 组件：本地嵌入 + 内存向量库。

为了离线可跑、零依赖，这里用"字符 n-gram 哈希"做一个玩具级 embedding：
把文本切成 2-gram，哈希到固定维度，得到词袋向量，再做余弦相似度。
它当然不如真实 embedding 模型，但足以演示 RAG 的完整机制：
    切块 -> 向量化 -> 存库 -> 按相似度检索 -> 拼进提示。

生产中把 HashingEmbedder 换成真实 embedding API（并把 VectorStore 换成
FAISS/pgvector/Milvus 等）即可，检索 + 拼接的逻辑完全不变。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Tuple


class HashingEmbedder:
    """把文本映射成固定维度向量（字符 2-gram 词袋 + 哈希）。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text = text.lower()
        grams = [text[i:i + 2] for i in range(max(1, len(text) - 1))] or [text]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        # L2 归一化，便于用点积当余弦相似度
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Doc:
    text: str
    vector: List[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class VectorStore:
    """最简内存向量库：add 存文档，search 返回 top-k 最相似文档。"""

    def __init__(self, embedder: HashingEmbedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.docs: List[Doc] = []

    def add(self, text: str, meta: dict | None = None) -> None:
        self.docs.append(Doc(text=text, vector=self.embedder.embed(text), meta=meta or {}))

    def search(self, query: str, k: int = 3) -> List[Tuple[Doc, float]]:
        qv = self.embedder.embed(query)
        scored = [(d, cosine(qv, d.vector)) for d in self.docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def chunk_text(text: str, size: int = 120, overlap: int = 20) -> List[str]:
    """把长文切成带重叠的块。重叠能避免把一句话从中间切断导致语义丢失。"""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

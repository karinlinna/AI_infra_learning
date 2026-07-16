"""
Tesla 汽车知识问答系统 —— 单文件可运行 Demo
============================================
用零外部依赖（纯标准库）演示简历里的完整 RAG 流程骨架：

  1. 文档解析      —— 模拟 PyMuPDF 抽取，附带页码/插图元数据
  2. 分块策略      —— 滑动窗口 + 父子文档 + 语义切分
  3. 存储          —— 用内存 dict 模拟 MongoDB（文本块 + 元数据）
  4. 多路召回      —— Dense(向量) + Sparse + BM25 三路混合
  5. 融合排序      —— RRF 粗排 + Reranker 精排
  6. 答案生成      —— 拼接上下文，输出答案 + 引用页码 + 关联插图

每一处“模拟”都在注释里标注了生产环境应替换成的真实组件。
运行： python3 tesla_rag_demo.py

文档解析 + 分块 —— semantic_split(语义切分按句号断句)、sliding_window_chunks(滑动窗口带重叠)、build_chunks(父子文档:子块细粒度检索,命中后回溯父块拿完整上下文)。生产环境这里换成 PyMuPDF 逐页抽文字和插图。

存储 —— MockMongo 内存字典模拟 MongoDB 存文本块 + 元数据(page/figures)。

多路召回 —— 三路并行:dense_recall(模拟 Qwen3-Embedding 稠密向量 + 余弦)、sparse_recall(模拟 BGE-M3 稀疏权重)、BM25(手写标准 BM25 公式做字面召回)。

融合排序 —— rrf_fuse(RRF 只看排名倒数相加做粗排,对不同尺度的召回鲁棒) + rerank(模拟 BGE-Reranker cross-encoder 精排)。

答案生成 —— generate_answer 把命中子块回溯到父块拼上下文,同时收集引用页码和关联插图,这就是简历里"输出答案、引用页码和关联插图"那个能力的骨架。

从 demo 升级到真实系统需要替换的部分
代码里每处"模拟"都有注释标了对应的真实组件,概括一下替换清单:

embed() 的 hash 词袋 → Qwen3-Embedding 真实模型推理;sparse_recall 的词频 → BGE-M3 稀疏向量;稠密检索的 Python 循环 → Milvus ANN 索引;MockMongo → pymongo;rerank 的重合度打分 → BGE-Reranker cross-encoder;generate_answer 的抽取式拼接 → 微调后的 LLM 生成。

一个可以注意的细节:你在跑第三个问题("冬天准备")时,Sparse 和 BM25 都被"充电"页干扰误召回了 p12,但 Dense 稠密召回正确命中了 p93(预热电池),最后精排把 p93 排到了第一——这恰好演示了多路召回互补的价值:字面匹配失效时,语义召回兜底。
"""

import re
import math
import hashlib
from collections import defaultdict, Counter


# ============================================================
# 0. 模拟数据源：一小段 Tesla 用户手册（生产环境是 PDF 文件）
# ============================================================
# 真实场景：PyMuPDF(fitz) 逐页 page.get_text() 抽文字，
#          page.get_images() 抽插图，记录 page.number。
MANUAL_PAGES = [
    {
        "page": 12,
        "text": "充电端口位于车辆左后方尾灯附近。要打开充电端口，可在充电时按下 "
                "Supercharger 充电枪上的按钮，或在触摸屏上点击闪电图标。Model 3 "
                "支持交流慢充和直流快充两种模式。使用 Supercharger 直流快充时，"
                "电量从 10% 充到 80% 通常需要约 25 分钟。",
        "figures": ["Fig-12-1: 充电端口位置示意图"],
    },
    {
        "page": 45,
        "text": "Autopilot 自动辅助驾驶功能需要在车速高于 30 km/h 时才能激活。"
                "向下拨动右侧拨杆两次即可开启自动辅助转向。请始终将双手放在方向盘上，"
                "系统检测到手部离开会发出警告。恶劣天气或车道线不清晰时功能可能降级。",
        "figures": ["Fig-45-2: Autopilot 拨杆操作", "Fig-45-3: 方向盘握持提示"],
    },
    {
        "page": 78,
        "text": "轮胎胎压建议值印在驾驶员侧车门框的标签上。冷胎状态下前后轮均建议保持 "
                "2.9 bar。胎压过低会增加能耗并影响续航。触摸屏的“服务”菜单可实时查看四轮胎压。",
        "figures": ["Fig-78-1: 胎压标签位置"],
    },
    {
        "page": 93,
        "text": "冬季用车时，建议出发前使用手机 App 提前预热电池和座舱。电池温度过低会"
                "限制充电速度和再生制动能力。开启“预置空调”可在保持接电时预热，避免消耗行驶电量。",
        "figures": [],
    },
]


# ============================================================
# 1. 文档解析 + 2. 分块策略（滑动窗口 / 父子文档 / 语义切分）
# ============================================================
def semantic_split(text):
    """语义切分：按句号/分号等自然边界切句。
    生产环境：可用标点+embedding 相似度断点做更细的语义分段。"""
    parts = re.split(r"(?<=[。；！？])", text)
    return [p.strip() for p in parts if p.strip()]


def sliding_window_chunks(sentences, size=2, stride=1):
    """滑动窗口：每 size 个句子成一块，步长 stride，块间有重叠避免语义割裂。"""
    chunks = []
    for i in range(0, max(1, len(sentences) - size + 1), stride):
        chunks.append("".join(sentences[i:i + size]))
    return chunks


def build_chunks(pages):
    """
    父子文档结构：
      - parent（父块）= 整页文本，用于最终喂给 LLM 的完整上下文
      - child（子块） = 滑动窗口小块，粒度细、检索更精准
    检索命中 child，返回时回溯到 parent 拿完整上下文。
    """
    parents, children = {}, []
    for pg in pages:
        pid = f"p{pg['page']}"
        parents[pid] = {
            "parent_id": pid,
            "page": pg["page"],
            "text": pg["text"],
            "figures": pg["figures"],
        }
        sents = semantic_split(pg["text"])
        for j, ck in enumerate(sliding_window_chunks(sents, size=2, stride=1)):
            children.append({
                "chunk_id": f"{pid}_c{j}",
                "parent_id": pid,
                "page": pg["page"],
                "text": ck,
            })
    return parents, children


# ============================================================
# 3. 存储：内存 dict 模拟 MongoDB
# ============================================================
# 生产环境：pymongo，collection.insert_many(children)，元数据带 page/figures。
class MockMongo:
    def __init__(self, children):
        self.docs = {c["chunk_id"]: c for c in children}

    def all(self):
        return list(self.docs.values())

    def get(self, cid):
        return self.docs[cid]


# ============================================================
# 4. 多路召回
# ============================================================
# ---- 4a. 简易分词（中英混合）----
def tokenize(text):
    text = text.lower()
    # 英文/数字词
    toks = re.findall(r"[a-z0-9]+", text)
    # 中文按 bigram（相邻两字）切，粗糙但够 demo 用
    for zh in re.findall(r"[\u4e00-\u9fff]+", text):
        toks += [zh[i:i + 2] for i in range(len(zh) - 1)] or [zh]
    return toks


# ---- 4b. Dense 稠密向量召回 ----
4. dense_recall —— 召回主流程

# def dense_recall(query, chunks, topk=5):
#     qv = embed(query)                                          # 查询转向量
#     scored = [(c["chunk_id"], cosine(qv, embed(c["text"])))    # 逐块算相似度
#               for c in chunks]
#     return sorted(scored, key=lambda x: -x[1])[:topk]          # 降序取前 topk

# 生产环境：Qwen3-Embedding 生成向量，存入 Milvus 做 ANN 检索。
# 这里用“词袋 hash 向量 + 余弦相似度”模拟稠密语义匹配。
def embed(text, dim=128):
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def dense_recall(query, chunks, topk=5):
    qv = embed(query)
    scored = [(c["chunk_id"], cosine(qv, embed(c["text"]))) for c in chunks]
    return sorted(scored, key=lambda x: -x[1])[:topk]


# ---- 4c. Sparse 稀疏语义召回 ----
# 核心思想：只有 query 和文档里"共同出现的词"才贡献分数。

# qset = Counter(tokenize(query))     # 统计 query 每个词的词频
# for c in chunks:
#     cset = Counter(tokenize(c["text"]))   # 统计文档每个词的词频
#     s = sum(qset[t] * cset[t] for t in qset if t in cset)  # 稀疏点积

# Counter 会把词列表变成"词 → 出现次数"的字典，例如：
# tokenize("检索 增强 检索") → Counter({"检索": 2, "增强": 1, ...})
# 生产环境：BGE-M3 输出的稀疏权重向量。这里用加权词频模拟“带语义权重的稀疏表示”。
def sparse_recall(query, chunks, topk=5):
    qset = Counter(tokenize(query))
    scored = []
    for c in chunks:
        cset = Counter(tokenize(c["text"]))
        # 稀疏点积：只有共现词才贡献分数
        s = sum(qset[t] * cset[t] for t in qset if t in cset)
        scored.append((c["chunk_id"], s))
    return sorted(scored, key=lambda x: -x[1])[:topk]


# ---- 4d. BM25 字面召回 ----
# 生产环境：可用 Elasticsearch / rank_bm25 库。这里手写标准 BM25 公式。
# ┌──────────────┬─────────────────────┬──────────────────────────┬───────────────────────┐
# │     方法     │      匹配依据       │           擅长           │         短板          │
# ├──────────────┼─────────────────────┼──────────────────────────┼───────────────────────┤
# │ Dense（4b）  │ 语义向量相似        │ 近义词、换种说法         │ 需要好的 embedding    │
# │              │                     │                          │ 模型                  │
# ├──────────────┼─────────────────────┼──────────────────────────┼───────────────────────┤
# │ Sparse（4c） │ 共现词加权          │ 精确关键词               │ 不懂近义词            │
# ├──────────────┼─────────────────────┼──────────────────────────┼───────────────────────┤
# │ BM25（4d）   │ 字面 + idf +        │ 专有名词、术语、精确匹配 │ 完全不懂语义          │
# │              │ 长度归一            │                          │                       │
# └──────────────┴─────────────────────┴──────────────────────────┴───────────────────────┘
# 为什么要三种一起用？

# 这就是 Hybrid（混合）检索 的思路：单一方法都有盲区，多路召回 + 结果融合（通常用 RRF 倒数排名融合）能显著提高召回质量。

# query
#  ├─ dense_recall  → 一组结果
#  ├─ sparse_recall → 一组结果
#  └─ bm25.recall   → 一组结果
#         ↓
#    融合/重排（RRF / rerank 模型）
#         ↓
#    最终 topk → 回溯 parent → 喂 LLM

# ① 词频饱和（k1 控制）
# - 一个词出现 1 次 vs 出现 100 次，相关性不该差 100 倍
# - 这个公式让分数随词频增长但逐渐饱和（趋于上限），避免刷词频作弊

# ② 长度归一化（b 控制）
# - dl / self.avgdl：当前文档长度 / 平均长度
# - 长文档天然包含更多词，容易蒙中关键词 → 用长度惩罚，避免长文档不公平地占优
# - b=0.75 控制惩罚力度（b=0 不惩罚，b=1 完全按长度归一化）

# k1=1.5, b=0.75 是业界公认的经验默认值。

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.chunks = chunks
        self.docs = {c["chunk_id"]: tokenize(c["text"]) for c in chunks}
        self.avgdl = sum(len(d) for d in self.docs.values()) / len(self.docs)
        self.df = defaultdict(int)
        for d in self.docs.values():
            for t in set(d):
                self.df[t] += 1
        self.N = len(self.docs)

    def idf(self, t):
        return math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))

    def recall(self, query, topk=5):
        q = tokenize(query)
        scored = []
        for cid, d in self.docs.items():
            tf = Counter(d)
            dl = len(d)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf(t) * num / den
            scored.append((cid, s))
        return sorted(scored, key=lambda x: -x[1])[:topk]


# ============================================================
# 5. 融合排序：RRF 粗排 + Reranker 精排
# 这段是检索链路的最后两步：融合 + 精排。前面三路召回（Dense/Sparse/BM25）各自给出一堆候选，这里负责把它们合并、排序、精选出最终喂给 LLM 的几个块。
# 举例：块 A 在 Dense 排第1、BM25 排第3；块 B 只在 Dense 排第1。
# A 因为被两路都召回，累加分更高 → 融合后 A 胜出。这正是 hybrid 想要的效果：多路共识的结果更可信。

# ① overlap —— 字面词重合率
# len(qtok & ctok) / len(qtok)
# - qtok & ctok 是集合交集：query 和文档共有多少个词
# - 除以 query 词数 → 归一化成比例
# - 例：query 有 4 个词，文档命中其中 3 个 → overlap = 0.75

# ② sim —— 稠密语义相似度
# 就是前面的余弦相似度，捕捉语义层面的接近。
# 最终分 = 0.5×字面重合 + 0.5×语义相似，字面和语义各看一半。

# ============================================================
def rrf_fuse(rank_lists, k=60):
    """Reciprocal Rank Fusion：把多路召回的排名倒数相加，做粗排融合。
    只看排名不看原始分数，对不同尺度的召回天然鲁棒。"""
    scores = defaultdict(float)
    for rl in rank_lists:
        for rank, (cid, _) in enumerate(rl):
            scores[cid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def rerank(query, candidates, mongo, topk=3):
    qtok = set(tokenize(query))     # query 的词集合
    qv = embed(query)               # query 的向量
    scored = []
    for cid, _ in candidates:
        c = mongo.get(cid)          # 从"数据库"取回块的完整内容
        ctok = set(tokenize(c["text"]))
        overlap = len(qtok & ctok) / (len(qtok) or 1)   # 词重合率
        sim = cosine(qv, embed(c["text"]))              # 语义相似度
        scored.append((cid, 0.5 * overlap + 0.5 * sim)) # 两者各半加权
    return sorted(scored, key=lambda x: -x[1])[:topk]


# ============================================================
# 6. 答案生成：拼上下文 + 输出引用页码 + 关联插图
# ============================================================
# 生产环境：把 context 塞进 prompt 调 LLM（微调过的 Qwen 等）生成答案。
# demo 无 LLM，用抽取式“把最相关父块拼起来”代替生成，重点演示引用/插图溯源。
def generate_answer(query, reranked, mongo, parents):
    seen_parents, cited_pages, figures, ctx = set(), [], [], []
    for cid, _ in reranked:
        pid = mongo.get(cid)["parent_id"]
        if pid in seen_parents:
            continue
        seen_parents.add(pid)
        p = parents[pid]
        ctx.append(p["text"])
        cited_pages.append(p["page"])
        figures += p["figures"]

    # 这里就是喂给 LLM 的 prompt（demo 直接返回上下文摘要）
    context = "\n".join(ctx)
    answer = f"（基于手册内容回答）关于「{query}」：\n{context}"
    return {
        "answer": answer,
        "cited_pages": sorted(set(cited_pages)),
        "figures": figures,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("Tesla 汽车知识问答系统 Demo —— 建立索引中...")
    print("=" * 60)

    # 建索引（离线一次）
    parents, children = build_chunks(MANUAL_PAGES)
    mongo = MockMongo(children)
    bm25 = BM25(children)
    print(f"父块(整页): {len(parents)}   子块(滑窗): {len(children)}\n")

    questions = [
        "Model 3 用超充从10%充到80%要多久？",
        "Autopilot 需要多快的车速才能开启？",
        "冬天开车前要做什么准备？",
    ]

    for q in questions:
        print("─" * 60)
        print(f"❓ 问题：{q}\n")

        chunks = mongo.all()
        # 三路召回
        d = dense_recall(q, chunks, topk=5)
        s = sparse_recall(q, chunks, topk=5)
        bm = bm25.recall(q, topk=5)

        # RRF 粗排融合
        fused = rrf_fuse([d, s, bm])
        # BGE-Reranker 精排
        reranked = rerank(q, fused, mongo, topk=3)

        print("  召回路径命中(chunk_id)：")
        print(f"    Dense : {[c for c,_ in d[:3]]}")
        print(f"    Sparse: {[c for c,_ in s[:3]]}")
        print(f"    BM25  : {[c for c,_ in bm[:3]]}")
        print(f"    精排后: {[c for c,_ in reranked]}\n")

        # 生成答案
        res = generate_answer(q, reranked, mongo, parents)
        print(f"  📖 引用页码：{res['cited_pages']}")
        print(f"  🖼  关联插图：{res['figures'] or '无'}")
        print(f"\n  💬 答案：\n    {res['answer'][:200]}...\n")


if __name__ == "__main__":
    main()

# 用户 query
#    ↓
# 多路召回 → rrf_fuse 粗排 → rerank 精排
#    ↓
# 最终 top3 的 chunk_id（子块 id）
#    ↓
# mongo.get(cid) → 拿到子块，读它的 parent_id
#    ↓
# parents[parent_id] → 回溯到父块（整页文本 + figures）
#    ↓
# 把这些父块的 text 拼接起来  ==  context   ← 就是它

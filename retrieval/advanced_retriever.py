"""DCE (Dual-Channel Evidence Retrieval with Entailment-Guided Re-Ranking)
—— 原创进阶检索器，用于提升句子级事实一致性检测效果。

核心机制：
1. 双通道检索融合：语义 embedding + 关键词 BM25，合并去重
2. 轻量级蕴含导向重排序：n-gram 重叠 / 实体命中 / 位置连贯性 三项启发式打分
3. 自适应 Top-K：按句子长度动态调整检索数量
4. 条件证据扩展：NLI 低置信度时以 top 证据为锚点扩展候选池

用法：
    from retrieval.advanced_retriever import AdvancedRetriever
    retr = AdvancedRetriever(use_bm25=True)
    retr.build_index(passages)
    hits = retr.query(sentence, top_k=5)          # 单句检索
    hits_list = retr.query_batch(sentences, top_k=5)  # 批量检索
"""
from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict

import numpy as np

from retrieval.retriever import Retriever
from config import DEFAULT_RETRIEVER_MODEL


# ---------------------------------------------------------------------------
# 关键词 / 实体提取
# ---------------------------------------------------------------------------

# 常见英文实体模式：大写开头的连续词、数字+单位、百分比、日期
_RE_ENTITY = re.compile(
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # 大写开头的词组
    r'|\b\d+(?:[.,]\d+)?(?:\s*(?:%|million|billion|thousand|percent|dollars|USD|EUR|GBP|years|months|days|km|m|kg|tons))?\b'  # 数字+单位
    r'|\b(19|20)\d{2}s?\b'  # 年份
)

# 中文实体模式：连续汉字、数字+单位
_RE_CN_ENTITY = re.compile(
    r'[\u4e00-\u9fff]{2,}'  # 中文词组
    r'|\d+(?:[.,]\d+)?(?:\s*(?:%|万|亿|千|百|元|美元|年|月|日|公里|米|吨|千克))?'  # 数字+中文单位
)


def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键实体和数字作为关键词。"""
    keywords = []
    # 英文实体
    for m in _RE_ENTITY.finditer(text):
        kw = m.group().strip()
        if kw and len(kw) >= 2:
            keywords.append(kw)
    # 中文实体
    for m in _RE_CN_ENTITY.finditer(text):
        kw = m.group().strip()
        if kw and len(kw) >= 2:
            keywords.append(kw)
    # 去重保序
    seen = set()
    unique = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique.append(kw)
    return unique


def _safe_len(obj) -> int:
    """安全获取长度，用于 n-gram 计算。"""
    if obj is None:
        return 0
    try:
        return len(obj)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 轻量级重排序
# ---------------------------------------------------------------------------

class LightweightReRanker:
    """零额外模型开销的证据候选重排序器。

    对每个候选证据计算三项分数：
    - n-gram 重叠分（1-gram / 2-gram Jaccard）
    - 实体命中分（共享的命名实体和数字个数）
    - 位置连贯分（候选在原文中的相对位置）
    """

    def __init__(self, total_passages: int):
        self.total = max(1, total_passages)

    def _ngram_overlap_score(self, query: str, passage: str) -> float:
        """计算 1-gram 和 2-gram 的 Jaccard 重叠。"""
        q_words = set(query.lower().split())
        p_words = set(passage.lower().split())
        if not q_words or not p_words:
            return 0.0

        # 1-gram Jaccard
        intersection = q_words & p_words
        union = q_words | p_words
        unigram_jaccard = len(intersection) / max(1, len(union))

        # 2-gram Jaccard（简化版：用单词对的字符重叠近似）
        q_bi = {query.lower()[i:i + 3] for i in range(max(0, len(query) - 3))}
        p_bi = {passage.lower()[i:i + 3] for i in range(max(0, len(passage) - 3))}
        bi_intersect = q_bi & p_bi
        bi_union = q_bi | p_bi
        bigram_jaccard = len(bi_intersect) / max(1, len(bi_union))

        return 0.6 * unigram_jaccard + 0.4 * bigram_jaccard

    def _entity_hit_score(self, query: str, passage: str) -> float:
        """计算查询与段落共享的实体/数字个数（归一化）。"""
        q_entities = set(k.lower() for k in extract_keywords(query))
        p_entities = set(k.lower() for k in extract_keywords(passage))
        if not q_entities:
            return 0.0
        hits = q_entities & p_entities
        return len(hits) / len(q_entities)

    def _position_coherence_score(self, passage_idx: int) -> float:
        """位置连贯分：越靠前的段落权重略高（摘要倾向于引用文档前部）。

        使用指数衰减：score = exp(-idx / total * 2)
        """
        return float(np.exp(-passage_idx / self.total * 2.0))

    def score(self, query: str, passage: str, passage_idx: int) -> float:
        """综合打分，返回 [0, 1] 之间的分数。"""
        ngram = self._ngram_overlap_score(query, passage)
        entity = self._entity_hit_score(query, passage)
        position = self._position_coherence_score(passage_idx)
        # 权重：n-gram 0.5, 实体 0.35, 位置 0.15
        return 0.50 * ngram + 0.35 * entity + 0.15 * position


# ---------------------------------------------------------------------------
# AdvancedRetriever
# ---------------------------------------------------------------------------

class AdvancedRetriever(Retriever):
    """DCE 进阶检索器，继承基类 Retriever 保留 embedding + FAISS 能力。

    新增：
    - 双通道检索（语义 + BM25 关键词）
    - 轻量级重排序
    - 自适应 Top-K
    - 条件证据扩展
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RETRIEVER_MODEL,
        use_bm25: bool = True,
        device: int = -1,
    ):
        # DCE 默认启用 BM25 作为第二通道
        super().__init__(model_name=model_name, use_bm25=use_bm25, device=device)
        self._reranker: Optional[LightweightReRanker] = None
        # 记录最后一次检索的 top 候选索引，供证据扩展使用
        self._last_top_indices: List[int] = []

    # ------------------------------------------------------------------
    # 自适应 Top-K
    # ------------------------------------------------------------------

    @staticmethod
    def adaptive_top_k(sentence: str, base_k: int = 5) -> int:
        """根据句子复杂度动态选择 top-k。

        - 短句（≤12 token）：k = 3（事实少，少量证据足够）
        - 中等句子（13-25 token）：k = base_k
        - 长句（>25 token）：k = max(base_k, 7)（可能含多事实，需更多证据）
        """
        tokens = sentence.split()
        n = len(tokens)
        if n <= 12:
            return min(3, base_k)
        elif n <= 25:
            return base_k
        else:
            return max(7, base_k)

    # ------------------------------------------------------------------
    # 双通道检索（核心方法）
    # ------------------------------------------------------------------

    def _dual_channel_search(
        self, text: str, top_k: int
    ) -> List[Tuple[int, float]]:
        """双通道检索：语义通道 + 关键词通道，合并去重后返回候选列表。

        返回 [(passage_idx, combined_score), ...] 按分数降序排列。
        """
        candidates: Dict[int, float] = {}

        # --- 通道 1：语义 embedding 检索 ---
        if self.index is not None:
            try:
                q_emb = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
                q_emb = np.asarray(q_emb)
                if q_emb.ndim == 1:
                    q_emb = q_emb.reshape(1, -1)
                import faiss
                faiss.normalize_L2(q_emb)
                D, I = self.index.search(q_emb, top_k)
                for idx, dist in zip(I[0], D[0]):
                    i = int(idx)
                    if 0 <= i < len(self.corpus):
                        candidates[i] = candidates.get(i, 0.0) + float(dist) * 0.6
            except Exception as e:
                import traceback
                print('[AdvancedRetriever] semantic channel failed:', e)
                traceback.print_exc()

        # --- 通道 2：关键词 BM25 检索 ---
        if self.bm25 is not None:
            try:
                keywords = extract_keywords(text)
                keyword_query = ' '.join(keywords) if keywords else text
                tokenized_q = keyword_query.split()
                bm_scores = self.bm25.get_scores(tokenized_q)
                # 取 top-k 个 BM25 候选
                bm_top_indices = np.argsort(bm_scores)[::-1][:top_k]
                bm_max = float(bm_scores.max()) if bm_scores.max() > 0 else 1.0
                for idx in bm_top_indices:
                    i = int(idx)
                    if 0 <= i < len(self.corpus):
                        norm_score = float(bm_scores[idx]) / bm_max
                        candidates[i] = candidates.get(i, 0.0) + norm_score * 0.4
            except Exception as e:
                import traceback
                print('[AdvancedRetriever] keyword channel failed:', e)
                traceback.print_exc()

        # 按合并分数排序
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_candidates[:top_k]]

    # ------------------------------------------------------------------
    # 轻量级重排序
    # ------------------------------------------------------------------

    def _rerank(
        self, query: str, candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """对候选列表进行轻量级重排序。"""
        if not candidates:
            return candidates
        if self._reranker is None:
            self._reranker = LightweightReRanker(len(self.corpus))

        scored = []
        for idx, retrieval_score in candidates:
            if idx < 0 or idx >= len(self.corpus):
                continue
            passage = self.corpus[idx]
            rerank_score = self._reranker.score(query, passage, idx)
            # 融合检索分（0.4）和重排序分（0.6）
            combined = 0.4 * retrieval_score + 0.6 * rerank_score
            scored.append((idx, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # 条件证据扩展
    # ------------------------------------------------------------------

    def expand_evidence(
        self, top_indices: List[int], expand_k: int = 3
    ) -> List[Tuple[int, float]]:
        """以当前 top 证据为锚点，在 FAISS 索引中找最近邻段落。

        用于 NLI 低置信度时扩展候选池。

        Args:
            top_indices: 当前 top 证据的 passage 索引列表
            expand_k: 每个锚点扩展几个邻居

        Returns:
            [(idx, similarity_score), ...] 扩展的候选列表
        """
        if self.index is None or not top_indices:
            return []

        expanded: Dict[int, float] = {}
        import faiss

        for anchor_idx in top_indices[:3]:  # 最多用 3 个锚点
            if anchor_idx < 0 or anchor_idx >= len(self.corpus):
                continue
            # 重建锚点 embedding（从索引中无法直接取回，用 encode 替代）
            try:
                anchor_text = self.corpus[anchor_idx]
                emb = self.model.encode([anchor_text], convert_to_numpy=True, show_progress_bar=False)
                emb = np.asarray(emb)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                faiss.normalize_L2(emb)
                D, I = self.index.search(emb, expand_k + 1)  # +1 因为锚点自身会被检索到
                for idx, dist in zip(I[0], D[0]):
                    i = int(idx)
                    if i == anchor_idx:
                        continue  # 跳过锚点自身
                    if 0 <= i < len(self.corpus):
                        if i not in expanded or float(dist) > expanded[i]:
                            expanded[i] = float(dist)
            except Exception:
                continue

        sorted_expanded = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_expanded[:expand_k * 2]]

    # ------------------------------------------------------------------
    # 覆写 query / query_batch
    # ------------------------------------------------------------------

    def query(
        self, text: str, top_k: int = 5, use_bm25_score: bool = False
    ) -> List[Tuple[int, float]]:
        """DCE 增强版 query：双通道检索 → 重排序 → 返回 top-k。"""
        if self.index is None and self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        adaptive_k = self.adaptive_top_k(text, base_k=top_k)

        # 双通道检索
        candidates = self._dual_channel_search(text, adaptive_k)

        # 轻量级重排序
        ranked = self._rerank(text, candidates)

        # 记录 top 索引供后续扩展使用
        self._last_top_indices = [idx for idx, _ in ranked[:adaptive_k]]

        return ranked[:adaptive_k]

    def query_batch(
        self, texts: List[str], top_k: int = 5, use_bm25_score: bool = False
    ) -> List[List[Tuple[int, float]]]:
        """DCE 增强版批量 query。"""
        results = []
        for text in texts:
            results.append(self.query(text, top_k=top_k))
        return results

    # ------------------------------------------------------------------
    # 便捷方法：一次获取证据文本
    # ------------------------------------------------------------------

    def retrieve_evidence_texts(
        self, query_text: str, top_k: int = 5
    ) -> List[str]:
        """检索并返回证据文本列表（而非索引+分数）。"""
        hits = self.query(query_text, top_k=top_k)
        return [
            self.corpus[i]
            for i, _ in hits
            if i is not None and 0 <= i < len(self.corpus)
        ]


# ---------------------------------------------------------------------------
# 简单自测
# ---------------------------------------------------------------------------

def _demo():
    """基本功能演示。"""
    passages = [
        "The GDP growth was 3.2 percent in 2023 according to the Bureau of Economic Analysis.",
        "Construction investment contributed 0.8 percentage points to GDP growth.",
        "The unemployment rate fell to 3.7 percent in November.",
        "Consumer spending increased by 2.1 percent driven by retail sales.",
        "The Federal Reserve maintained interest rates at 5.25 to 5.5 percent.",
    ]
    r = AdvancedRetriever(use_bm25=True)
    r.build_index(passages)

    # 测试1：短句
    q1 = "GDP grew 3.2 percent."
    print(f"Query: {q1}")
    print(f"Adaptive k: {AdvancedRetriever.adaptive_top_k(q1)}")
    for idx, score in r.query(q1, top_k=5):
        print(f"  [{idx}] score={score:.3f}  {passages[idx][:80]}")
    print()

    # 测试2：含实体的句子
    q2 = "Construction and consumer spending both increased significantly in the economic report."
    print(f"Query: {q2}")
    print(f"Adaptive k: {AdvancedRetriever.adaptive_top_k(q2)}")
    for idx, score in r.query(q2, top_k=5):
        print(f"  [{idx}] score={score:.3f}  {passages[idx][:80]}")
    print()

    # 测试3：证据扩展
    print("Evidence expansion from top-2:")
    expanded = r.expand_evidence([0, 1], expand_k=2)
    for idx, score in expanded:
        print(f"  [{idx}] score={score:.3f}  {passages[idx][:80]}")


if __name__ == "__main__":
    _demo()

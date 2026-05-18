"""证据定位：支持 embedding + FAISS 与可选 BM25 的轻量级封装。

提供构建索引与检索 top-k 段的简单 API，同时支持混合检索（bm25 + embedding）以改进长文证据定位。
"""
from typing import List, Tuple, Optional
import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_bm25: bool = False, device: int = -1):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for Retriever")
        # device: -1 -> cpu, >=0 -> cuda:device
        device_str = 'cpu' if device == -1 else f'cuda:{device}'
        try:
            self.model = SentenceTransformer(model_name, device=device_str)
        except TypeError:
            self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus: List[str] = []
        self.use_bm25 = use_bm25 and (BM25Okapi is not None)
        self.bm25 = None

    def build_index(self, passages: List[str]):
        """建立向量索引并（可选）BM25 索引。"""
        # filter out empty passages before encoding
        filtered = [p for p in passages if p and p.strip()]
        self.corpus = filtered
        if not filtered:
            # nothing to index; leave index as None
            self.index = None
            self.bm25 = None
            return

        embs = self.model.encode(filtered, convert_to_numpy=True, show_progress_bar=False)
        # normalize embedding output to a 2D numpy array
        if isinstance(embs, list):
            embs = np.array(embs)
        else:
            embs = np.asarray(embs)

        # debug info for unexpected embedding shapes
        try:
            print(f"[Retriever] build_index: passages={len(passages)} filtered={len(filtered)} embs_type={type(embs)} embs_shape={embs.shape}")
        except Exception:
            print(f"[Retriever] build_index: passages={len(passages)} filtered={len(filtered)} embs_type={type(embs)} (no shape)")

        if embs.size == 0:
            # unexpected: treat as no index
            self.index = None
            self.bm25 = None
            return

        if embs.ndim == 1:
            # single vector -> make it 2D
            embs = embs.reshape(1, -1)

        dim = embs.shape[1]
        if faiss is None:
            raise ImportError("faiss-cpu is required for indexing")
        self.index = faiss.IndexFlatIP(dim)
        # normalize for inner product similarity
        faiss.normalize_L2(embs)
        self.index.add(embs)

        if self.use_bm25:
            # BM25 expects tokenized corpus
            tokenized = [p.split() for p in passages]
            self.bm25 = BM25Okapi(tokenized)

    def query(self, text: str, top_k: int = 5, use_bm25_score: bool = False) -> List[Tuple[int, float]]:
        """返回 [(idx, score), ...]，默认基于 embedding 检索。

        如果 `use_bm25_score` 为 True 且 BM25 可用，则合并 bm25 分数（归一化）与向量相似度。
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_emb = self.model.encode([text], convert_to_numpy=True)
        q_emb = np.asarray(q_emb)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        emb_results = [(int(i), float(d)) for i, d in zip(I[0], D[0])]

        if use_bm25_score and self.bm25 is not None:
            tokenized_q = text.split()
            bm_scores = self.bm25.get_scores(tokenized_q)
            # take top_k by hybrid score: normalized emb + normalized bm25
            # normalize bm25
            bm = np.array(bm_scores)
            if bm.max() > 0:
                bm = bm / (bm.max())
            emb_vals = [s for _, s in emb_results]
            emb_arr = np.array(emb_vals)
            if emb_arr.max() > 0:
                emb_arr = emb_arr / (emb_arr.max())
            hybrid = []
            for (idx, emb_score), emb_norm in zip(emb_results, emb_arr):
                hybrid_score = 0.6 * emb_norm + 0.4 * float(bm[idx])
                hybrid.append((idx, float(hybrid_score)))
            hybrid.sort(key=lambda x: x[1], reverse=True)
            return hybrid

        return emb_results


def simple_demo():
    passages = [
        "This is the first document. It contains facts about A.",
        "Second passage referencing B and C.",
        "Third passage with unrelated content.",
    ]
    r = Retriever(use_bm25=True)
    r.build_index(passages)
    print(r.query("facts about A", top_k=2))


if __name__ == "__main__":
    simple_demo()

"""证据定位：支持 embedding + FAISS 与可选 BM25 的轻量级封装。

提供构建索引与检索 top-k 段的简单 API，同时支持混合检索（bm25 + embedding）以改进长文证据定位。
"""
from typing import List, Tuple, Optional, Dict
import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:
    SentenceTransformer = None
    faiss = None
    import traceback
    print('retriever: optional imports failed:', e)
    traceback.print_exc()

try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    BM25Okapi = None
    import traceback
    print('retriever: rank_bm25 import failed:', e)
    traceback.print_exc()


from config import DEFAULT_RETRIEVER_MODEL, EMBEDDING_CACHE_DIR, DEFAULT_RETRIEVER_ENCODE_BATCH_SIZE, DEFAULT_RETRIEVER_INDEX_METHOD
import hashlib


class Retriever:
    _MODEL_CACHE: Dict[str, object] = {}

    def __init__(self, model_name: str = DEFAULT_RETRIEVER_MODEL, use_bm25: bool = False, device: int = -1):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for Retriever")
        # device: -1 -> cpu, >=0 -> cuda:device
        device_str = 'cpu' if device == -1 else f'cuda:{device}'
        self.last_error = None
        cache_key = f"{model_name}::{device_str}"
        if cache_key in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[cache_key]
        else:
            try:
                self.model = SentenceTransformer(model_name, device=device_str)
            except TypeError:
                self.model = SentenceTransformer(model_name)
            self._MODEL_CACHE[cache_key] = self.model
        self.index = None
        self.corpus: List[str] = []
        self.use_bm25 = use_bm25 and (BM25Okapi is not None)
        self.bm25 = None
        # batch size to use when encoding large numbers of passages or queries (configurable)
        self.encode_batch_size = DEFAULT_RETRIEVER_ENCODE_BATCH_SIZE

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
        # Attempt to load embeddings from disk cache keyed by passages hash
        cache_key = hashlib.md5('\n'.join(filtered).encode('utf-8')).hexdigest()
        cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}.npz"
        embs = None
        if cache_path.exists():
            try:
                loaded = np.load(str(cache_path))
                embs = loaded['arr_0']
                print(f'[Retriever] loaded embeddings from cache: {cache_path}')
            except Exception:
                embs = None

        if embs is None:
            try:
                embs = self.model.encode(filtered, convert_to_numpy=True, show_progress_bar=False, batch_size=self.encode_batch_size)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print('[Retriever] build_index encode failed:', e)
                print(tb)
                self.last_error = tb
                self.index = None
                self.bm25 = None
                raise
            try:
                np.savez_compressed(str(cache_path), embs)
                print(f'[Retriever] saved embeddings to cache: {cache_path}')
            except Exception:
                pass
        # normalize embedding output to a 2D numpy array
        if isinstance(embs, list):
            embs = np.array(embs)
        else:
            embs = np.asarray(embs)

        # debug info for unexpected embedding shapes
        try:
            print(f"[Retriever] build_index: passages={len(passages)} filtered={len(filtered)} embs_type={type(embs)} embs_shape={embs.shape}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[Retriever] build_index: passages={len(passages)} filtered={len(filtered)} embs_type={type(embs)} (no shape); error:", e)
            print(tb)
            self.last_error = tb

        if embs.size == 0:
            # unexpected: treat as no index
            self.index = None
            self.bm25 = None
            return

        if embs.ndim == 1:
            # single vector -> make it 2D
            embs = embs.reshape(1, -1)

        # Attempt to load a prebuilt FAISS index for this corpus (fast path)
        index_path = EMBEDDING_CACHE_DIR / f"{cache_key}.index"
        if index_path.exists():
            try:
                idx = faiss.read_index(str(index_path))
                # sanity check: index ntotal should match number of embeddings
                try:
                    if idx.ntotal == embs.shape[0]:
                        self.index = idx
                        print(f'[Retriever] loaded faiss index from cache: {index_path}')
                        # also ensure bm25 is set below if needed
                        if self.use_bm25:
                            tokenized = [p.split() for p in passages]
                            self.bm25 = BM25Okapi(tokenized)
                        return
                except Exception:
                    # index mismatch; continue to build a new one
                    pass
            except Exception:
                pass

        dim = embs.shape[1]
        if faiss is None:
            raise ImportError("faiss-cpu is required for indexing")
        # choose index type based on corpus size and config
        index_method = DEFAULT_RETRIEVER_INDEX_METHOD
        n = embs.shape[0]
        # heuristics: allow env override via config; otherwise choose HNSW for medium, Flat for small
        if index_method == 'auto':
            if n >= 2000:
                chosen = 'hnsw'
            else:
                chosen = 'flat'
        else:
            chosen = index_method

        if chosen == 'hnsw':
            # HNSW with Inner Product (approx. nearest neighbors)
            try:
                M = 32
                efConstruction = 200
                self.index = faiss.IndexHNSWFlat(dim, M)
                # set efSearch to a sensible value
                try:
                    self.index.hnsw.efSearch = 64
                except Exception:
                    pass
                faiss.normalize_L2(embs)
                self.index.add(embs)
                try:
                    faiss.write_index(self.index, str(index_path))
                except Exception:
                    pass
            except Exception:
                # fallback to flat
                self.index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embs)
                self.index.add(embs)
        elif chosen == 'ivf':
            # IVF requires training; choose nlist based on corpus size
            nlist = min(max(16, n // 100), 4096)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            faiss.normalize_L2(embs)
            try:
                self.index.train(embs)
                self.index.add(embs)
                try:
                    faiss.write_index(self.index, str(index_path))
                except Exception:
                    pass
            except Exception:
                # fallback to flat
                self.index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embs)
                self.index.add(embs)
        else:
            # flat exact index
            self.index = faiss.IndexFlatIP(dim)
            # normalize for inner product similarity
            faiss.normalize_L2(embs)
            self.index.add(embs)
            try:
                faiss.write_index(self.index, str(index_path))
            except Exception:
                pass

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

        try:
            q_emb = self.model.encode([text], convert_to_numpy=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('[Retriever] query encode failed:', e)
            print(tb)
            self.last_error = tb
            raise
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

    def query_batch(self, texts: List[str], top_k: int = 5, use_bm25_score: bool = False) -> List[List[Tuple[int, float]]]:
        """Batch query: return list of hit lists for each input text.

        Returns a list where each element is [(idx, score), ...] for that query.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        try:
            q_embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('[Retriever] query_batch encode failed:', e)
            print(tb)
            self.last_error = tb
            raise
        q_embs = np.asarray(q_embs)
        if q_embs.ndim == 1:
            q_embs = q_embs.reshape(1, -1)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        results = []
        for row_d, row_i in zip(D, I):
            emb_results = [(int(i), float(d)) for i, d in zip(row_i, row_d)]
            if use_bm25_score and self.bm25 is not None:
                tokenized_q = []
                # compute bm25 per query
                for t in texts:
                    tokenized_q = t.split()
                    bm_scores = self.bm25.get_scores(tokenized_q)
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
                    results.append(hybrid)
            else:
                results.append(emb_results)
        return results


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

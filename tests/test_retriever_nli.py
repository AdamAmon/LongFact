import sys
import types
import numpy as np

# Provide a lightweight fake 'torch' for tests to avoid heavy dependency
if 'torch' not in sys.modules:
    fake_torch = types.ModuleType('torch')

    import numpy as _np
    import contextlib

    def device(x):
        return x

    @contextlib.contextmanager
    def no_grad():
        yield

    def tensor(x):
        return _np.array(x)

    def softmax(a, dim=-1):
        a = _np.array(a)
        exps = _np.exp(a - _np.max(a, axis=dim, keepdims=True))
        return exps / _np.sum(exps, axis=dim, keepdims=True)

    def max_func(a, dim=-1):
        a = _np.array(a)
        idx = _np.argmax(a, axis=dim)
        val = _np.max(a, axis=dim)
        return val, idx

    fake_torch.device = device
    fake_torch.no_grad = no_grad
    fake_torch.tensor = tensor
    fake_torch.softmax = softmax
    fake_torch.max = max_func

    sys.modules['torch'] = fake_torch

from retrieval.retriever import Retriever
from nli.nli_check import NLIChecker


def test_retriever_build_index_and_query(monkeypatch):
    # Mock SentenceTransformer to return predictable embeddings
    class DummyST:
        def __init__(self, model_name):
            pass

        def encode(self, corpus, convert_to_numpy=True, show_progress_bar=False):
            # return simple 2D numpy array: each vector is [i+1, 0]
            embs = [[float(i + 1), 0.0] for i in range(len(corpus))]
            return np.array(embs)

    monkeypatch.setattr('retrieval.retriever.SentenceTransformer', DummyST)

    # Mock faiss with minimal IndexFlatIP and normalize_L2
    class DummyIndex:
        def __init__(self, dim):
            self.vectors = None

        def add(self, embs):
            self.vectors = embs.copy()

        def search(self, q_emb, top_k):
            # naive dot product similarity
            q = q_emb[0]
            sims = (self.vectors * q).sum(axis=1)
            idxs = np.argsort(-sims)[:top_k]
            dists = sims[idxs]
            return np.array([dists]), np.array([idxs])

    def dummy_normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    fake_faiss = types.SimpleNamespace(IndexFlatIP=DummyIndex, normalize_L2=dummy_normalize)
    monkeypatch.setattr('retrieval.retriever.faiss', fake_faiss)

    # Mock BM25Okapi
    class DummyBM25:
        def __init__(self, tokenized):
            self.tokenized = tokenized

        def get_scores(self, tokenized_q):
            # return small scores
            return [0.1 * len(t) for t in self.tokenized]

    monkeypatch.setattr('retrieval.retriever.BM25Okapi', DummyBM25)

    passages = ["passage one", "passage two", "passage three"]
    r = Retriever(model_name="dummy", use_bm25=True)
    r.build_index(passages)
    res = r.query("query text", top_k=2, use_bm25_score=True)
    assert isinstance(res, list)
    assert len(res) == 2


def test_nlichecker_check_batch(monkeypatch):
    # Mock tokenizer and model to avoid heavy deps
    class DummyTokenizer:
        def __call__(self, premises, hypotheses, return_tensors=None, truncation=True, padding=True, max_length=1024):
            # return a dummy object that supports .to(device)
            class DummyInputs(dict):
                def to(self, device):
                    return self

            return DummyInputs({'input_ids': None})

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

    class DummyModel:
        def __init__(self):
            # config id2label mapping
            self.config = types.SimpleNamespace(id2label={0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'})

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            # produce logits tensor: shape (batch, 3)
            import torch

            logits = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]])
            return types.SimpleNamespace(logits=logits)

    monkeypatch.setattr('nli.nli_check.AutoTokenizer', DummyTokenizer)
    monkeypatch.setattr('nli.nli_check.AutoModelForSequenceClassification', DummyModel)

    # instantiate NLIChecker which should use our fakes
    checker = NLIChecker(model_name='dummy', device=-1)
    premises = ["p1", "p2"]
    hypos = ["h1", "h2"]
    out = checker.check_batch(premises, hypos)
    assert isinstance(out, list)
    assert out[0][0].upper() == 'ENTAILMENT'
    assert out[1][0].upper() in ('CONTRADICTION', 'NEUTRAL', 'ENTAILMENT')


# ---------------------------------------------------------------------------
# AdvancedRetriever (DCE) 测试
# ---------------------------------------------------------------------------

from retrieval.advanced_retriever import AdvancedRetriever, extract_keywords, LightweightReRanker


def test_extract_keywords():
    text = "GDP grew 3.2 percent in New York according to the Federal Reserve."
    kws = extract_keywords(text)
    assert len(kws) >= 2
    assert any('GDP' in k for k in kws) or any('New York' in k for k in kws) or any('Federal Reserve' in k for k in kws)


def test_adaptive_top_k():
    short = "GDP grew."
    medium = "The economy expanded by 3.2 percent driven by strong consumer spending and construction."
    # 需要 >25 token 的句子才能触发 k=7
    long_sent = "The Commonwealth of the Northern Mariana Islands experienced significant GDP growth in 2016 partly due to construction investment and tourism revenue from Asian markets and increased federal spending."
    assert AdvancedRetriever.adaptive_top_k(short) == 3
    assert AdvancedRetriever.adaptive_top_k(medium) == 5
    assert AdvancedRetriever.adaptive_top_k(long_sent) == 7


def test_lightweight_reranker():
    reranker = LightweightReRanker(total_passages=5)
    query = "GDP grew 3.2 percent"
    passage = "The GDP growth was 3.2 percent in 2023."
    score = reranker.score(query, passage, passage_idx=0)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_advanced_retriever_dual_channel(monkeypatch):
    """测试双通道检索融合。"""
    # Mock embedding model
    class DummyST:
        def __init__(self, model_name, device=None):
            pass

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, batch_size=None):
            m = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
            if isinstance(texts, list):
                # 返回与输入数量匹配的 embedding
                return np.array([[1.0, 0.0]] * len(texts))
            return np.array([[1.0, 0.0]])

    monkeypatch.setattr('retrieval.retriever.SentenceTransformer', DummyST)

    # Mock FAISS
    class DummyIndex:
        def __init__(self, dim):
            self.vectors = None

        def add(self, embs):
            self.vectors = np.asarray(embs).copy()

        def search(self, q_emb, top_k):
            q = np.asarray(q_emb)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            sims = (self.vectors * q).sum(axis=1)
            idxs = np.argsort(-sims)[:top_k]
            dists = sims[idxs]
            return np.array([dists]), np.array([idxs])

    def dummy_normalize(x):
        return x

    fake_faiss = types.SimpleNamespace(IndexFlatIP=DummyIndex, normalize_L2=dummy_normalize)
    # 注入 sys.modules 让两个模块的 import faiss 都拿到 mock
    monkeypatch.setitem(sys.modules, 'faiss', fake_faiss)
    monkeypatch.setattr('retrieval.retriever.faiss', fake_faiss)

    # Mock BM25
    class DummyBM25:
        def __init__(self, tokenized):
            self.tokenized = tokenized

        def get_scores(self, tokenized_q):
            return np.array([0.1 * (i + 1) for i in range(len(self.tokenized))])

    monkeypatch.setattr('retrieval.retriever.BM25Okapi', DummyBM25)

    passages = [
        "The GDP growth was 3.2 percent in 2023.",
        "Construction contributed 0.8 points to GDP.",
        "Unemployment fell to 3.7 percent.",
        "Consumer spending increased 2.1 percent.",
        "The Federal Reserve held rates steady.",
    ]
    r = AdvancedRetriever(model_name="dummy", use_bm25=True)
    r.build_index(passages)

    results = r.query("GDP grew 3.2 percent", top_k=5)
    assert isinstance(results, list)
    assert len(results) >= 1
    # 应该包含 GDP 相关的段落
    indices = [idx for idx, _ in results]
    assert 0 in indices  # "GDP growth was 3.2 percent"


def test_advanced_retriever_query_batch(monkeypatch):
    """测试批量查询。"""
    class DummyST:
        def __init__(self, model_name, device=None):
            pass

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, batch_size=None):
            if isinstance(texts, list):
                return np.array([[1.0, 0.0]] * len(texts))
            return np.array([[1.0, 0.0]])

    monkeypatch.setattr('retrieval.retriever.SentenceTransformer', DummyST)

    class DummyIndex:
        def __init__(self, dim):
            pass

        def add(self, embs):
            pass

        def search(self, q_emb, top_k):
            q = np.asarray(q_emb)
            batch_size = q.shape[0] if q.ndim > 1 else 1
            I = np.array([[0, 1, 2]] * batch_size)
            D = np.array([[0.9, 0.5, 0.3]] * batch_size)
            return D, I

    def dummy_normalize(x):
        return x

    fake_faiss = types.SimpleNamespace(IndexFlatIP=DummyIndex, normalize_L2=dummy_normalize)
    monkeypatch.setattr('retrieval.retriever.faiss', fake_faiss)

    class DummyBM25:
        def __init__(self, tokenized):
            self.tokenized = tokenized

        def get_scores(self, tokenized_q):
            return np.array([0.1] * len(self.tokenized))

    monkeypatch.setattr('retrieval.retriever.BM25Okapi', DummyBM25)

    passages = ["p1", "p2", "p3"]
    r = AdvancedRetriever(model_name="dummy", use_bm25=True)
    r.build_index(passages)

    results = r.query_batch(["q1", "q2"], top_k=3)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(len(hits) >= 1 for hits in results)

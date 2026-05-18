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

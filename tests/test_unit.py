import pytest

from summarize.run_summarize import chunk_text, fuse_summaries
from summarize.model_summarizer import FallbackSummarizer
from eval.evaluate import sentence_split, compute_support_rate
from correction.corrector import Corrector


class DummyRetriever:
    def __init__(self, corpus, hits_map):
        self.corpus = corpus
        # hits_map: dict from query text -> list of (idx, score)
        self._hits_map = hits_map
        # mark index present so compute_support_rate will attempt retrieval
        self.index = True

    def query(self, text, top_k=5):
        return self._hits_map.get(text, [])


class DummyNLI:
    def __init__(self, entail_set):
        # entail_set contains premises (strings) that will be treated as ENTAILMENT
        self.entail_set = set(entail_set)

    def check(self, premise, hypothesis):
        if premise in self.entail_set:
            return ("ENTAILMENT", 0.9)
        return ("CONTRADICTION", 0.1)


def test_chunk_and_fuse():
    text = "This is the first sentence. This is the second sentence. Third sentence."
    chunks = chunk_text(text, max_tokens=4)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1

    local = ["A summary.", "B summary."]
    fused = fuse_summaries(local)
    assert fused == "A summary. B summary."


def test_fallback_summarizer():
    summarizer = FallbackSummarizer()
    chunks = ["第一句。第二句。", "另一个段落。"]
    out = summarizer.summarize_chunks(chunks)
    assert isinstance(out, list)
    assert out[0].startswith("第一句")
    assert out[1].startswith("另一个段落")


def test_sentence_split_and_support_rate():
    # Summary with two sentences (Chinese punctuation)
    summary = "Alice bought apples。Bob went home。"

    # corpus: first passage supports first sentence, second passage is unrelated
    corpus = ["Alice bought 3 apples.", "Some unrelated text."]
    hits_map = {
        # compute_support_rate calls retriever.query with the sentence including the trailing punctuation
        "Alice bought apples。": [(0, 0.9)],
        "Bob went home。": [(1, 0.5)],
    }

    retr = DummyRetriever(corpus, hits_map)
    nli = DummyNLI(entail_set=[corpus[0]])

    rate, details = compute_support_rate(summary, "", retr, nli, top_k=1, threshold=0.6)
    assert pytest.approx(rate, rel=1e-3) == 0.5
    assert len(details) == 2
    assert details[0]["supported"] is True
    assert details[1]["supported"] is False


def test_corrector_fallback():
    # When no model/pipeline is available, Corrector.correct should return original sentence
    corr = Corrector(model_name=None, device=-1)
    evidence = ["E1"]
    s = "Original sentence."
    assert corr.correct(evidence, s) == s

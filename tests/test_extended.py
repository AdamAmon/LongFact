import sys
import types
import json

import pytest

# Provide a lightweight fake 'torch' for tests if real torch is unavailable
if 'torch' not in sys.modules:
    fake_torch = types.ModuleType('torch')

    def device(x):
        return x

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def no_grad():
        return _NoGrad()

    def tensor(x):
        import numpy as _np

        return _np.array(x)

    def softmax(a, dim=-1):
        import numpy as _np

        a = _np.array(a)
        exps = _np.exp(a - _np.max(a, axis=dim, keepdims=True))
        return exps / _np.sum(exps, axis=dim, keepdims=True)

    def max_func(a, dim=-1):
        import numpy as _np

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

from nli import nli_check
from correction.corrector import Corrector
import scripts.analyze_results as ar


def test_nlichecker_check_with_evidence_strategies():
    # Create an NLIChecker instance without running __init__
    checker = object.__new__(nli_check.NLIChecker)

    # stub check to return different scores per evidence
    def fake_check(premise, hypothesis):
        if 'e1' in premise:
            return ('ENTAILMENT', 0.9)
        if 'e2' in premise:
            return ('ENTAILMENT', 0.8)
        return ('CONTRADICTION', 0.1)

    checker.check = fake_check

    evidences = ['e1 text', 'e2 text', 'e3 text']

    # max strategy should pick e1 (0.9)
    lab, sc, per = nli_check.NLIChecker.check_with_evidence(checker, evidences, 'hyp', strategy='max')
    assert lab.upper() == 'ENTAILMENT'
    assert pytest.approx(sc, rel=1e-3) == 0.9
    assert len(per) == 3

    # majority strategy -> ENTAILMENT (2 vs 1)
    lab2, sc2, per2 = nli_check.NLIChecker.check_with_evidence(checker, evidences, 'hyp', strategy='majority')
    assert lab2.upper() == 'ENTAILMENT'
    assert sc2 >= 0.8

    # any_entailment should return first entailment
    lab3, sc3, per3 = nli_check.NLIChecker.check_with_evidence(checker, evidences, 'hyp', strategy='any_entailment')
    assert lab3.upper() == 'ENTAILMENT'


def test_corrector_with_pipeline(monkeypatch):
    # fake pipeline factory that returns a callable pipeline
    def fake_pipeline(*args, **kwargs):
        def _pipe(prompt):
            return [{'generated_text': 'Corrected sentence.'}]

        return _pipe

    monkeypatch.setattr('correction.corrector.pipeline', fake_pipeline)

    corr = Corrector(model_name='dummy', device=-1)
    out = corr.correct(['E1', 'E2'], 'Original sentence.')
    assert out == 'Corrected sentence.'


def test_analyze_results_load_and_avg(tmp_path):
    # create a small jsonl file
    records = [
        {'id': 'a', 'support_rate': 0.5, 'rouge': {'rouge1_fmeasure': 0.2}, 'prediction': 'p', 'corrected': 'c', 'reference': 'r', 'rouge_corrected': {'rouge1_fmeasure': 0.3}},
        {'id': 'b', 'support_rate': 1.0, 'rouge': {'rouge1_fmeasure': 0.4}, 'prediction': 'p2', 'corrected': 'c2', 'reference': 'r2', 'rouge_corrected': {'rouge1_fmeasure': 0.1}},
    ]
    p = tmp_path / 'small.jsonl'
    with open(p, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    loaded = list(ar.load_jsonl(str(p)))
    assert len(loaded) == 2
    assert ar.avg([1, 2, 3]) == pytest.approx(2.0)

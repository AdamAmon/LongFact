"""评估脚本：计算 ROUGE 与句子级支持率，并导出示例案例。"""
from typing import List, Dict, Tuple

try:
    from rouge_score import rouge_scorer
except Exception as e:
    rouge_scorer = None
    import traceback
    print('evaluate: rouge_score import failed:', e)
    traceback.print_exc()


def compute_rouge(ref: str, pred: str) -> Dict[str, float]:
    if rouge_scorer is None:
        raise ImportError('rouge_score is required to compute ROUGE')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    out = {}
    for k, v in scores.items():
        out[f'{k}_precision'] = v.precision
        out[f'{k}_recall'] = v.recall
        out[f'{k}_fmeasure'] = v.fmeasure
    return out


def sentence_split(text: str) -> List[str]:
    import re

    if not text or not text.strip():
        return []
    # Support Chinese and English punctuation while keeping sentence boundaries.
    parts = re.split(r'(?<=[。！？.!?])\s+', text.strip())
    out = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if not re.search(r'[。！？.!?]$', s):
            s = s + '。'
        out.append(s)
    return out


def compute_support_rate(summary: str, document: str, retriever, nli_checker, top_k: int = 5, threshold: float = 0.6) -> Tuple[float, List[Dict]]:
    """对 summary 的每个句子检索证据并用 nli_checker 判定是否被支持，返回支持率与逐句详情列表。"""
    sents = sentence_split(summary)
    details = []
    supported = 0
    for s in sents:
        # if retriever has no index (e.g., empty document), skip retrieval
        if getattr(retriever, 'index', None) is None and getattr(retriever, 'bm25', None) is None:
            hits = []
            evidences = []
        else:
            try:
                hits = retriever.query(s, top_k=top_k)
                evidences = [retriever.corpus[idx] for idx, _ in hits if idx is not None and idx < len(retriever.corpus)]
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print('evaluate: retriever.query failed:', e)
                print(tb)
                details.append({
                    'sentence': s,
                    'supported': False,
                    'best_label': 'ERROR',
                    'best_score': 0.0,
                    'evidences': [],
                    'error': str(e),
                    'last_error': tb,
                })
                continue
        # Use the NLI check_with_evidence aggregation to reduce forward passes
        if not evidences:
            details.append({'sentence': s, 'supported': False, 'best_label': None, 'best_score': 0.0, 'evidences': []})
            continue

        if hasattr(nli_checker, 'check_with_evidence'):
            try:
                agg_label, agg_score, per = nli_checker.check_with_evidence(evidences, s, strategy='max')
                # determine supported by checking if any per-evidence entailment meets threshold
                is_sup = any((lab.upper() in ('ENTAILMENT', 'ENTAILS') and sc >= threshold) for lab, sc, _ in per)
                # find best label/score
                best_label = None
                best_score = 0.0
                if per:
                    best = max(per, key=lambda x: x[1])
                    best_label, best_score = best[0], best[1]
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print('evaluate: nli.check_with_evidence failed:', e)
                print(tb)
                details.append({
                    'sentence': s,
                    'supported': False,
                    'best_label': 'ERROR',
                    'best_score': 0.0,
                    'evidences': evidences[:3],
                    'error': str(e),
                    'last_error': tb,
                })
                continue
        else:
            # fallback to older interface where only `check` exists on nli_checker
            is_sup = False
            best_label = None
            best_score = 0.0
            for ev in evidences:
                try:
                    label, score = nli_checker.check(ev, s)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print('evaluate: nli.check failed for evidence:', (ev[:200] if isinstance(ev, str) else str(ev)), e)
                    print(tb)
                    details.append({
                        'sentence': s,
                        'supported': False,
                        'best_label': 'ERROR',
                        'best_score': 0.0,
                        'evidences': evidences[:3],
                        'error': str(e),
                        'last_error': tb,
                    })
                    best_label = 'ERROR'
                    best_score = 0.0
                    is_sup = False
                    break
                if score > best_score:
                    best_score = score
                    best_label = label
                if label.upper() == 'ENTAILMENT' and score >= threshold:
                    is_sup = True
                    break
        if not details or details[-1].get('sentence') != s:
            details.append({'sentence': s, 'supported': is_sup, 'best_label': best_label, 'best_score': best_score, 'evidences': evidences[:3]})
        if is_sup:
            supported += 1
    rate = supported / max(1, len(sents))
    return rate, details

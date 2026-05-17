"""评估脚本：计算 ROUGE 与句子级支持率，并导出示例案例。"""
from typing import List, Dict, Tuple

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None


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
    sents = [s.strip() for s in text.split('。') if s.strip()]
    return [s + '。' for s in sents]


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
            hits = retriever.query(s, top_k=top_k)
            evidences = [retriever.corpus[idx] for idx, _ in hits if idx is not None and idx < len(retriever.corpus)]
        is_sup = False
        best_label = None
        best_score = 0.0
        for e in evidences:
            label, score = nli_checker.check(e, s)
            if score > best_score:
                best_score = score
                best_label = label
            if label.upper() == 'ENTAILMENT' and score >= threshold:
                is_sup = True
                break
        details.append({'sentence': s, 'supported': is_sup, 'best_label': best_label, 'best_score': best_score, 'evidences': evidences[:3]})
        if is_sup:
            supported += 1
    rate = supported / max(1, len(sents))
    return rate, details

"""简单流水线：采样 -> 生成 -> 证据定位 -> NLI 检测 -> 纠错 -> 保存结果

用法示例：
python workflow/run_pipeline_simple.py --sample_size 10 --out results/pipeline_sample10.jsonl
"""
import argparse
import json
from typing import List

from data.load_govreport import load_govreport
from summarize.run_summarize import run_pipeline, chunk_text
from summarize.model_summarizer import get_summarizer
from retrieval.retriever import Retriever
from nli.nli_check import NLIChecker
from correction.corrector import Corrector
from config import DEFAULT_SUMMARIZER_MODEL, DEFAULT_RETRIEVER_MODEL


def split_sentences(text: str) -> List[str]:
    import re
    sents = [s.strip() for s in re.split(r'(?<=[。.!?])\s+', text) if s.strip()]
    return sents


def run_one(record, summarizer_model: str, retriever_model: str, device: int = -1):
    doc = record['document']
    ref = record.get('summary', '')
    # 1) summarize
    summ_result = run_pipeline(doc, use_model=True, model_name=summarizer_model, device=device)
    fused = summ_result.get('fused', '')

    # 2) split fused into sentences
    sents = split_sentences(fused)

    # 3) build retriever index from document passages (use chunk_text to split)
    passages = chunk_text(doc, max_tokens=200)
    retr = None
    try:
        retr = Retriever(model_name=retriever_model, use_bm25=True)
        retr.build_index(passages)
    except Exception:
        retr = None

    # 4) nli checker
    nli = NLIChecker()

    # 5) corrector
    corr = Corrector()

    sent_results = []
    for s in sents:
        evidence = []
        scores = []
        if retr is not None:
            try:
                hits = retr.query(s, top_k=5, use_bm25_score=True)
                for idx, score in hits:
                    evidence.append(passages[idx])
                    scores.append(score)
            except Exception:
                evidence = []

        # Compose premise as concatenation of top evidence
        premise = '\n'.join(evidence[:3]) if evidence else doc[:2048]
        try:
            label, prob = nli.check(premise, s)
        except Exception:
            label, prob = 'ERROR', 0.0

        supported = (label.upper() in ('ENTAILMENT', 'ENTAILS')) and prob >= 0.6

        corrected = s
        if not supported:
            try:
                corrected = corr.correct(evidence, s)
            except Exception:
                corrected = s

        sent_results.append({
            'sentence': s,
            'supported': supported,
            'nli_label': label,
            'nli_score': prob,
            'evidence': evidence,
            'corrected': corrected,
        })

    out = {
        'id': record.get('id'),
        'fused_summary': fused,
        'reference': ref,
        'sentences': sent_results,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--out', type=str, default='results/pipeline_sample.jsonl')
    parser.add_argument('--summ_model', type=str, default=DEFAULT_SUMMARIZER_MODEL)
    parser.add_argument('--retriever_model', type=str, default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument('--device', type=int, default=-1)
    args = parser.parse_args()

    recs = load_govreport(split='validation', sample_size=args.sample_size, cache_dir=None)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in recs:
            print(f"Processing id={r.get('id')}")
            try:
                out = run_one(r, summarizer_model=args.summ_model, retriever_model=args.retriever_model, device=args.device)
            except Exception as e:
                out = {'id': r.get('id'), 'error': str(e)}
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

    print(f'Wrote pipeline results to {args.out}')


if __name__ == '__main__':
    main()

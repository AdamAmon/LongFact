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
import time
import math
import statistics
try:
    import torch
except Exception:
    torch = None


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
        start_retr = time.time()
        if retr is not None:
            try:
                hits = retr.query(s, top_k=5, use_bm25_score=True)
                for idx, score in hits:
                    evidence.append(passages[idx])
                    scores.append(score)
            except Exception:
                evidence = []
        end_retr = time.time()

        # Per-evidence NLI checks and aggregation
        nli_label = 'NO_EVIDENCE'
        nli_score = 0.0
        per_evidence = []
        start_nli = time.time()
        if evidence:
            try:
                # use per-evidence checks and aggregate by 'max'
                nli_label, nli_score, per_evidence = nli.check_with_evidence(evidence, s, strategy='max')
            except Exception:
                nli_label, nli_score = 'ERROR', 0.0
        else:
            # fallback to using doc prefix as premise
            premise = doc[:2048]
            try:
                nli_label, nli_score = nli.check(premise, s)
                per_evidence = [(nli_label, nli_score, premise)]
            except Exception:
                nli_label, nli_score = 'ERROR', 0.0
        end_nli = time.time()

        supported = (nli_label.upper() in ('ENTAILMENT', 'ENTAILS')) and nli_score >= 0.6

        corrected = s
        nli_label_corrected = nli_label
        nli_score_corrected = nli_score
        supported_corrected = supported

        start_corr = time.time()
        if not supported:
            try:
                corrected = corr.correct(evidence, s)
            except Exception:
                corrected = s
            # Re-run per-evidence NLI on corrected sentence
            try:
                nli_label_corrected, nli_score_corrected, _ = nli.check_with_evidence(evidence, corrected, strategy='max')
            except Exception:
                nli_label_corrected, nli_score_corrected = 'ERROR', 0.0
            supported_corrected = (nli_label_corrected.upper() in ('ENTAILMENT', 'ENTAILS')) and nli_score_corrected >= 0.6
        end_corr = time.time()

        # timing summary for this sentence
        timing = {
            'retrieval_time': end_retr - start_retr if 'end_retr' in locals() else 0.0,
            'nli_time': end_nli - start_nli if 'end_nli' in locals() else 0.0,
            'correction_time': end_corr - start_corr if 'end_corr' in locals() else 0.0,
        }

        sent_results.append({
            'sentence': s,
            'supported': supported,
            'nli_label': nli_label,
            'nli_score': nli_score,
            'evidence': evidence,
            'corrected': corrected,
            'nli_per_evidence': per_evidence,
            'nli_label_corrected': nli_label_corrected,
            'nli_score_corrected': nli_score_corrected,
            'supported_corrected': supported_corrected,
            'correction_effect': (True if (supported_corrected and not supported) else (False if (supported_corrected == supported) else None)),
            'timing': timing,
        })

    out = {
        'id': record.get('id'),
        'fused_summary': fused,
        'reference': ref,
        'sentences': sent_results,
        'pipeline_timings': {
            'total_sentences': len(sent_results),
            'summarization_chunks': len(summ_result.get('chunks', [])),
            'sentence_nli_times': [s.get('timing', {}).get('nli_time', 0.0) for s in sent_results],
            'sentence_correction_times': [s.get('timing', {}).get('correction_time', 0.0) for s in sent_results],
            'sentence_retrieval_times': [s.get('timing', {}).get('retrieval_time', 0.0) for s in sent_results],
        }
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

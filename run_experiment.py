"""实验运行器：从数据采样 -> 摘要 -> 检索 -> NLI -> 纠错 -> 评估 的端到端最小实现。

用于快速验证管线是否可跑通（小样本）。"""
import argparse
import json
from config import DEFAULT_CORRECTOR_MODEL, DEFAULT_DATA_DIR, DEFAULT_NLI_MODEL, DEFAULT_SUMMARIZER_MODEL
from data.load_govreport import load_govreport
from config import FALLBACK_TO_SUMMARY
from summarize.run_summarize import run_pipeline
from retrieval.retriever import Retriever
from nli.nli_check import NLIChecker
from correction.corrector import Corrector
from eval.evaluate import compute_rouge, compute_support_rate


def run_sample(sample_count: int = 10, use_model: bool = False, model_name: str = None, device: int = -1, dataset_cache_dir: str = None):
    records = load_govreport(split='validation', sample_size=sample_count, cache_dir=dataset_cache_dir or str(DEFAULT_DATA_DIR))
    results = []

    for rec in records:
        # skip samples that were flagged as missing document (unless fallback enabled)
        if rec.get('skipped') and not FALLBACK_TO_SUMMARY:
            results.append({
                'id': rec['id'],
                'reference': rec.get('summary', ''),
                'prediction': '',
                'corrected': '',
                'support_rate': 0.0,
                'rouge': {},
                'rouge_corrected': {},
                'details': [],
                'error': 'skipped_missing_document',
                'skip_reason': rec.get('skip_reason'),
            })
            continue

        doc = rec['document']
        ref = rec.get('summary', '') or ''
        out = run_pipeline(doc, use_model=use_model, model_name=model_name, device=device)
        pred = out.get('fused', '')

        # build retriever on document passages; simple chunking for evidence
        passages = out['chunks']
        retr = Retriever()
        retr.build_index(passages)

        nli = NLIChecker(model_name=DEFAULT_NLI_MODEL, device=-1)
        support_rate, details = compute_support_rate(pred, doc, retr, nli, top_k=3)

        corr = Corrector(model_name=model_name if use_model else DEFAULT_CORRECTOR_MODEL, device=device)
        # perform corrections for sentences not supported
        corrected_sents = []
        for d in details:
            if not d['supported']:
                corrected = corr.correct(d['evidences'], d['sentence'])
                corrected_sents.append(corrected)
            else:
                corrected_sents.append(d['sentence'])
        corrected_pred = ' '.join(corrected_sents)

        rouge_scores = compute_rouge(ref, pred) if ref else {}
        rouge_corrected = compute_rouge(ref, corrected_pred) if ref else {}

        results.append({
            'id': rec['id'],
            'reference': ref,
            'prediction': pred,
            'corrected': corrected_pred,
            'support_rate': support_rate,
            'rouge': rouge_scores,
            'rouge_corrected': rouge_corrected,
            'details': details,
            # include summarization debug snapshot to help diagnose empty outputs
            'summarization_debug': {
                'chunks': out.get('chunks'),
                'local_summaries': out.get('local_summaries'),
                'fused': out.get('fused'),
                'error': out.get('error'),
            },
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--model_name', type=str, default=DEFAULT_SUMMARIZER_MODEL)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--dataset_cache_dir', type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--out', type=str, default='experiment_results.jsonl')
    args = parser.parse_args()

    res = run_sample(sample_count=args.n, use_model=args.use_model, model_name=args.model_name, device=args.device, dataset_cache_dir=args.dataset_cache_dir)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'Wrote {len(res)} results to {args.out}')


if __name__ == '__main__':
    main()

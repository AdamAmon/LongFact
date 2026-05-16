"""实验运行器：从数据采样 -> 摘要 -> 检索 -> NLI -> 纠错 -> 评估 的端到端最小实现。

用于快速验证管线是否可跑通（小样本）。"""
import argparse
import json
from data.load_govreport import load_govreport
from summarize.run_summarize import run_pipeline
from retrieval.retriever import Retriever
from nli.nli_check import NLIChecker
from correction.corrector import Corrector
from eval.evaluate import compute_rouge, compute_support_rate


def run_sample(sample_count: int = 10, use_model: bool = False, model_name: str = None, device: int = -1):
    records = load_govreport(split='validation', sample_size=sample_count)
    results = []

    for rec in records:
        doc = rec['document']
        ref = rec.get('summary', '') or ''
        out = run_pipeline(doc, use_model=use_model, model_name=model_name, device=device)
        pred = out['fused']

        # build retriever on document passages; simple chunking for evidence
        passages = out['chunks']
        retr = Retriever()
        retr.build_index(passages)

        nli = NLIChecker()
        support_rate, details = compute_support_rate(pred, doc, retr, nli, top_k=3)

        corr = Corrector(model_name=model_name if use_model else None, device=device)
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
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--model_name', type=str, default='google/flan-t5-large')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--out', type=str, default='experiment_results.jsonl')
    args = parser.parse_args()

    res = run_sample(sample_count=args.n, use_model=args.use_model, model_name=args.model_name, device=args.device)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'Wrote {len(res)} results to {args.out}')


if __name__ == '__main__':
    main()

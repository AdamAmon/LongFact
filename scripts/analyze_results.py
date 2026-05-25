"""Analyze experiment jsonl results produced by `run_experiment.py`.

Usage:
  python scripts/analyze_results.py --in results.jsonl --out summary.json

Outputs a JSON summary with average support rate, average ROUGE metrics,
and example cases showing original vs corrected sentences.
"""
import argparse
import json
from collections import defaultdict


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def avg(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='inpath', required=True)
    parser.add_argument('--out', dest='outpath', default=None)
    parser.add_argument('--examples', type=int, default=10, help='number of example cases to include')
    parser.add_argument('--preview', action='store_true', help='output a preview (limited support_rate_samples) instead of full list')
    parser.add_argument('--preview-size', type=int, default=5, help='preview sample size when --preview is set')
    args = parser.parse_args()

    records = list(load_jsonl(args.inpath))
    n = len(records)
    summary = {'n': n}

    support_rates = []
    rouge_acc = defaultdict(list)
    improved_cases = []
    worsened_cases = []

    errors = []
    for rec in records:
        try:
            support_rates.append(rec.get('support_rate', 0.0))
            rouge = rec.get('rouge', {}) or {}
            rouge_corr = rec.get('rouge_corrected', {}) or {}
            for k, v in rouge.items():
                rouge_acc[k].append(v)

            # detect improvement in average rouge fmeasure (rouge1_fmeasure as proxy)
            r1 = rouge.get('rouge1_fmeasure')
            r1c = rouge_corr.get('rouge1_fmeasure')
            if r1 is not None and r1c is not None:
                if r1c > r1:
                    improved_cases.append({'id': rec.get('id'), 'ref': rec.get('reference'), 'orig': rec.get('prediction'), 'corrected': rec.get('corrected'), 'r1': r1, 'r1c': r1c})
                elif r1c < r1:
                    worsened_cases.append({'id': rec.get('id'), 'r1': r1, 'r1c': r1c})
        except Exception as e:
            # record error but continue processing so we don't silently drop data
            errors.append({'id': rec.get('id'), 'error': str(e)})

    summary['avg_support_rate'] = avg(support_rates)
    if args.preview:
        summary['support_rate_samples'] = support_rates[: args.preview_size]
    else:
        summary['support_rate_samples'] = support_rates
    summary['avg_rouge'] = {k: avg(v) for k, v in rouge_acc.items()}
    summary['examples'] = {
        'improved': improved_cases[: args.examples],
        'worsened': worsened_cases[: args.examples],
    }

    out_json = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.outpath:
        with open(args.outpath, 'w', encoding='utf-8') as f:
            f.write(out_json)
        print(f'Wrote summary to {args.outpath}')
    else:
        print(out_json)


if __name__ == '__main__':
    main()

"""Analyze experiment jsonl results produced by `run_experiment.py`.

Usage:
  python scripts/analyze_results.py --in results.jsonl --out summary.json

Outputs a JSON summary with overall metrics, summary-length buckets,
and representative correction cases for report writing.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import sentence_split


LENGTH_BUCKETS = [
    (1, 3, '1-3'),
    (4, 6, '4-6'),
    (7, 10, '7-10'),
    (11, 15, '11-15'),
    (16, None, '16+'),
]


def get_prediction_text(rec):
    return rec.get('prediction') or rec.get('fused_summary') or ''


def get_sentence_count(rec):
    length = rec.get('prediction_length', {}).get('sentence_count')
    if isinstance(length, int):
        return length
    return len(sentence_split(get_prediction_text(rec)))


def get_token_count(rec):
    length = rec.get('prediction_length', {}).get('token_count')
    if isinstance(length, int):
        return length
    return len((get_prediction_text(rec) or '').split())


def get_length_value(rec, bucket_by):
    if bucket_by == 'token':
        return get_token_count(rec)
    return get_sentence_count(rec)


def bucket_label(sentence_count):
    for lower, upper, label in LENGTH_BUCKETS:
        if upper is None:
            if sentence_count >= lower:
                return label
        elif lower <= sentence_count <= upper:
            return label
    return '16+'


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def select_cases(records, case_count):
    improved = []
    failed = []
    neutral = []

    for rec in records:
        support_delta = safe_float(rec.get('support_rate_delta'))
        rouge_delta = safe_float(rec.get('rouge1_fmeasure_delta'))
        if support_delta > 0 or (support_delta == 0 and rouge_delta > 0):
            improved.append(rec)
        elif support_delta < 0 or (support_delta == 0 and rouge_delta < 0):
            failed.append(rec)
        else:
            neutral.append(rec)

    def sort_key(rec):
        return (
            safe_float(rec.get('support_rate_delta')),
            safe_float(rec.get('rouge1_fmeasure_delta')),
        )

    improved = sorted(improved, key=sort_key, reverse=True)
    failed = sorted(failed, key=sort_key)
    neutral = sorted(neutral, key=sort_key, reverse=True)

    half = max(1, case_count // 2)
    selected_improved = improved[:half]
    selected_failed = failed[: max(1, case_count - len(selected_improved))]
    selected = selected_improved + selected_failed

    if len(selected) < case_count:
        need = case_count - len(selected)
        selected.extend(neutral[:need])

    def add_reason(rec):
        support_delta = safe_float(rec.get('support_rate_delta'))
        rouge_delta = safe_float(rec.get('rouge1_fmeasure_delta'))
        if support_delta > 0:
            reason_tag = 'support_improved'
            reason_text = '纠错后句级支持率提升，说明 unsupported 句被修正或删减。'
        elif support_delta < 0:
            reason_tag = 'support_worsened'
            reason_text = '纠错后引入了新的不支持内容，事实一致性变差。'
        elif rouge_delta > 0:
            reason_tag = 'rouge_improved'
            reason_text = '纠错后与参考摘要的词面重叠更高。'
        elif rouge_delta < 0:
            reason_tag = 'rouge_worsened'
            reason_text = '纠错改写偏离参考摘要，词面重叠下降。'
        else:
            reason_tag = 'unchanged'
            reason_text = '纠错前后变化很小。'
        return {
            'id': rec.get('id'),
            'bucket': bucket_label(get_sentence_count(rec)),
            'sentence_count': get_sentence_count(rec),
            'token_count': get_token_count(rec),
            'support_rate': safe_float(rec.get('support_rate')),
            'corrected_support_rate': safe_float(rec.get('corrected_support_rate')),
            'support_rate_delta': support_delta,
            'rouge1_fmeasure': safe_float(rec.get('rouge', {}).get('rouge1_fmeasure')),
            'rouge1_fmeasure_corrected': safe_float(rec.get('rouge_corrected', {}).get('rouge1_fmeasure')),
            'rouge1_fmeasure_delta': rouge_delta,
            'reason_tag': reason_tag,
            'reason_text': reason_text,
            'reference': rec.get('reference', ''),
            'prediction': get_prediction_text(rec),
            'corrected': rec.get('corrected', ''),
            'details': rec.get('details', []),
            'corrected_details': rec.get('corrected_details', []),
        }

    return [add_reason(rec) for rec in selected], [add_reason(rec) for rec in improved], [add_reason(rec) for rec in failed]


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
    parser.add_argument('--csv-out', dest='csv_out', default=None, help='optional CSV output for bucketed metrics')
    parser.add_argument('--cases-out', dest='cases_out', default=None, help='optional JSON output for selected cases')
    parser.add_argument('--examples', type=int, default=10, help='number of example cases to include')
    parser.add_argument('--preview', action='store_true', help='output a preview (limited support_rate_samples) instead of full list')
    parser.add_argument('--preview-size', type=int, default=5, help='preview sample size when --preview is set')
    parser.add_argument('--case-count', type=int, default=10, help='number of cases to export for report writing')
    parser.add_argument('--bucket-by', choices=['sentence', 'token'], default='token', help='length metric for bucketing results')
    args = parser.parse_args()

    records = list(load_jsonl(args.inpath))
    n = len(records)
    summary = {'n': n}

    support_rates = []
    rouge_acc = defaultdict(list)
    improved_cases = []
    worsened_cases = []

    bucket_acc = defaultdict(lambda: {
        'count': 0,
        'support_rate': [],
        'corrected_support_rate': [],
        'rouge1_fmeasure': [],
        'rouge1_fmeasure_corrected': [],
        'support_rate_delta': [],
        'rouge1_fmeasure_delta': [],
    })

    errors = []
    for rec in records:
        try:
            support_rate = safe_float(rec.get('support_rate'))
            corrected_support_rate = safe_float(rec.get('corrected_support_rate'))
            rouge = rec.get('rouge', {}) or {}
            rouge_corr = rec.get('rouge_corrected', {}) or {}
            rouge1 = safe_float(rouge.get('rouge1_fmeasure'))
            rouge1c = safe_float(rouge_corr.get('rouge1_fmeasure'))
            support_delta = safe_float(rec.get('support_rate_delta', corrected_support_rate - support_rate))
            rouge_delta = safe_float(rec.get('rouge1_fmeasure_delta', rouge1c - rouge1))

            support_rates.append(support_rate)
            rouge = rec.get('rouge', {}) or {}
            for k, v in rouge.items():
                rouge_acc[k].append(v)

            # detect improvement in average rouge fmeasure (rouge1_fmeasure as proxy)
            r1 = rouge1
            r1c = rouge1c
            if r1 is not None and r1c is not None:
                if r1c > r1:
                    improved_cases.append({'id': rec.get('id'), 'ref': rec.get('reference'), 'orig': rec.get('prediction'), 'corrected': rec.get('corrected'), 'r1': r1, 'r1c': r1c})
                elif r1c < r1:
                    worsened_cases.append({'id': rec.get('id'), 'r1': r1, 'r1c': r1c})

            bucket = bucket_label(get_length_value(rec, args.bucket_by))
            bucket_acc[bucket]['count'] += 1
            bucket_acc[bucket]['support_rate'].append(support_rate)
            bucket_acc[bucket]['corrected_support_rate'].append(corrected_support_rate)
            bucket_acc[bucket]['rouge1_fmeasure'].append(rouge1)
            bucket_acc[bucket]['rouge1_fmeasure_corrected'].append(rouge1c)
            bucket_acc[bucket]['support_rate_delta'].append(support_delta)
            bucket_acc[bucket]['rouge1_fmeasure_delta'].append(rouge_delta)
        except Exception as e:
            # record error but continue processing so we don't silently drop data
            errors.append({'id': rec.get('id'), 'error': str(e)})

    summary['avg_support_rate'] = avg(support_rates)
    if args.preview:
        summary['support_rate_samples'] = support_rates[: args.preview_size]
    else:
        summary['support_rate_samples'] = support_rates
    summary['avg_rouge'] = {k: avg(v) for k, v in rouge_acc.items()}
    summary['length_bucket_metric'] = args.bucket_by
    summary['length_bucket_definition'] = [
        {'label': label, 'min_sentence_count': lower, 'max_sentence_count': upper}
        for lower, upper, label in LENGTH_BUCKETS
    ]
    summary['length_buckets'] = [
        {
            'label': label,
            'count': bucket_acc[label]['count'],
            'avg_support_rate': avg(bucket_acc[label]['support_rate']),
            'avg_corrected_support_rate': avg(bucket_acc[label]['corrected_support_rate']),
            'avg_rouge1_fmeasure': avg(bucket_acc[label]['rouge1_fmeasure']),
            'avg_rouge1_fmeasure_corrected': avg(bucket_acc[label]['rouge1_fmeasure_corrected']),
            'avg_support_rate_delta': avg(bucket_acc[label]['support_rate_delta']),
            'avg_rouge1_fmeasure_delta': avg(bucket_acc[label]['rouge1_fmeasure_delta']),
        }
        for _, _, label in LENGTH_BUCKETS
    ]
    summary['correction_summary'] = {
        'avg_corrected_support_rate': avg([safe_float(rec.get('corrected_support_rate')) for rec in records]),
        'avg_support_rate_delta': avg([safe_float(rec.get('support_rate_delta')) for rec in records]),
        'avg_rouge1_fmeasure_corrected': avg([safe_float(rec.get('rouge_corrected', {}).get('rouge1_fmeasure')) for rec in records]),
        'avg_rouge1_fmeasure_delta': avg([safe_float(rec.get('rouge1_fmeasure_delta')) for rec in records]),
    }
    summary['examples'] = {
        'improved': improved_cases[: args.examples],
        'worsened': worsened_cases[: args.examples],
    }

    selected_cases, selected_improved, selected_failed = select_cases(records, args.case_count)
    summary['selected_cases'] = selected_cases
    summary['case_selection'] = {
        'count': len(selected_cases),
        'improved_count': len(selected_improved),
        'failed_count': len(selected_failed),
    }

    out_json = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.outpath:
        with open(args.outpath, 'w', encoding='utf-8') as f:
            f.write(out_json)
        print(f'Wrote summary to {args.outpath}')
    else:
        print(out_json)

    if args.cases_out:
        case_payload = {
            'n': n,
            'selected_cases': selected_cases,
            'improved_cases': selected_improved,
            'failed_cases': selected_failed,
        }
        with open(args.cases_out, 'w', encoding='utf-8') as f:
            json.dump(case_payload, f, ensure_ascii=False, indent=2)
        print(f'Wrote selected cases to {args.cases_out}')

    if args.csv_out:
        with open(args.csv_out, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'bucket', 'count', 'avg_support_rate', 'avg_corrected_support_rate',
                'avg_rouge1_fmeasure', 'avg_rouge1_fmeasure_corrected',
                'avg_support_rate_delta', 'avg_rouge1_fmeasure_delta',
            ])
            for row in summary['length_buckets']:
                writer.writerow([
                    row['label'], row['count'], row['avg_support_rate'], row['avg_corrected_support_rate'],
                    row['avg_rouge1_fmeasure'], row['avg_rouge1_fmeasure_corrected'],
                    row['avg_support_rate_delta'], row['avg_rouge1_fmeasure_delta'],
                ])
        print(f'Wrote bucket CSV to {args.csv_out}')


if __name__ == '__main__':
    main()

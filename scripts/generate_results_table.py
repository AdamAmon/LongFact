#!/usr/bin/env python3
"""Generate per-sample ROUGE and sentence-level NLI statistics from pipeline results.

Usage:
  python scripts/generate_results_table.py --in results/pipeline_n10_qwen.jsonl

Outputs:
  - results/per_sample_results.csv
  - results/summary_results.json
"""
import argparse
import json
import csv
from collections import Counter, defaultdict
import statistics

try:
    from rouge_score import rouge_scorer
except Exception:
    raise SystemExit('Please install rouge-score: pip install rouge-score')


def compute(in_path, out_csv, out_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rows = []
    global_nli = Counter()
    rouge_acc = defaultdict(list)
    # correction-level accumulators
    total_sentences = 0
    supported_before = 0
    supported_after = 0
    improved_examples = []
    unchanged_examples = []
    unknown_examples = []

    with open(in_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            obj = json.loads(line)
            idx = obj.get('id')
            ref = obj.get('reference', '') or ''
            hyp = obj.get('fused_summary', '') or ''

            scores = scorer.score(ref, hyp)
            r1 = scores['rouge1'].fmeasure
            r2 = scores['rouge2'].fmeasure
            rl = scores['rougeL'].fmeasure

            sentences = obj.get('sentences', []) or []
            n_sent = len(sentences)
            n_supported = sum(1 for s in sentences if s.get('supported'))

            nli_counts = Counter()
            for s in sentences:
                lbl = s.get('nli_label') or s.get('label') or 'neutral'
                nli_counts[lbl] += 1
                global_nli[lbl] += 1

            # collect correction-level stats
            for s in sentences:
                total_sentences += 1
                if s.get('supported'):
                    supported_before += 1
                if s.get('supported_corrected'):
                    supported_after += 1
                eff = s.get('correction_effect')
                if eff is True:
                    improved_examples.append({
                        'id': idx,
                        'sentence': s.get('sentence'),
                        'evidence': s.get('evidence'),
                        'nli_label_before': s.get('nli_label'),
                        'nli_score_before': s.get('nli_score'),
                        'corrected': s.get('corrected'),
                        'nli_label_after': s.get('nli_label_corrected'),
                        'nli_score_after': s.get('nli_score_corrected'),
                    })
                elif eff is False:
                    unchanged_examples.append({
                        'id': idx,
                        'sentence': s.get('sentence'),
                        'evidence': s.get('evidence'),
                        'nli_label_before': s.get('nli_label'),
                        'nli_score_before': s.get('nli_score'),
                        'corrected': s.get('corrected'),
                        'nli_label_after': s.get('nli_label_corrected'),
                        'nli_score_after': s.get('nli_score_corrected'),
                    })
                else:
                    # None or other -> consider as unchanged/unknown
                    unknown_examples.append({
                        'id': idx,
                        'sentence': s.get('sentence'),
                        'evidence': s.get('evidence'),
                        'nli_label_before': s.get('nli_label'),
                        'nli_score_before': s.get('nli_score'),
                        'corrected': s.get('corrected'),
                        'nli_label_after': s.get('nli_label_corrected'),
                        'nli_score_after': s.get('nli_score_corrected'),
                    })

            row = {
                'id': idx,
                'rouge1_f': r1,
                'rouge2_f': r2,
                'rougeL_f': rl,
                'num_sentences': n_sent,
                'supported_sentences': n_supported,
                'supported_ratio': (n_supported / n_sent) if n_sent else 0.0,
                'n_entailment': nli_counts.get('entailment', 0),
                'n_neutral': nli_counts.get('neutral', 0),
                'n_contradiction': nli_counts.get('contradiction', 0),
            }
            rows.append(row)
            rouge_acc['rouge1_f'].append(r1)
            rouge_acc['rouge2_f'].append(r2)
            rouge_acc['rougeL_f'].append(rl)

    # write per-sample csv
    fieldnames = ['id', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'num_sentences', 'supported_sentences', 'supported_ratio', 'n_entailment', 'n_neutral', 'n_contradiction']
    with open(out_csv, 'w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    summary = {
        'n_examples': len(rows),
        'rouge_mean': {
            'rouge1_f': statistics.mean(rouge_acc['rouge1_f']) if rouge_acc['rouge1_f'] else 0.0,
            'rouge2_f': statistics.mean(rouge_acc['rouge2_f']) if rouge_acc['rouge2_f'] else 0.0,
            'rougeL_f': statistics.mean(rouge_acc['rougeL_f']) if rouge_acc['rougeL_f'] else 0.0,
        },
        'nli_counts': dict(global_nli),
        'correction_summary': {
            'total_sentences': total_sentences,
            'supported_before': supported_before,
            'supported_after': supported_after,
            'improved_count': len(improved_examples),
            'unchanged_count': len(unchanged_examples),
            'unknown_count': len(unknown_examples),
            'improved_ratio': (len(improved_examples) / total_sentences) if total_sentences else 0.0,
        },
    }

    with open(out_summary, 'w', encoding='utf-8') as outj:
        json.dump(summary, outj, indent=2, ensure_ascii=False)

    # save example cases
    examples_out = out_summary.replace('.json', '_examples.jsonl')
    with open(examples_out, 'w', encoding='utf-8') as exf:
        # write up to 10 examples: mix improved and unchanged
        selected = improved_examples[:5] + unchanged_examples[:5]
        for item in selected:
            exf.write(json.dumps(item, ensure_ascii=False) + '\n')
    print('Wrote examples to:', examples_out)

    print('Wrote:', out_csv)
    print('Wrote:', out_summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default='results/pipeline_n10_qwen.jsonl')
    p.add_argument('--out-csv', dest='out_csv', default='results/per_sample_results.csv')
    p.add_argument('--out-summary', dest='out_summary', default='results/summary_results.json')
    args = p.parse_args()
    compute(args.in_path, args.out_csv, args.out_summary)


if __name__ == '__main__':
    main()

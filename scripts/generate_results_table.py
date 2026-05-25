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
except Exception as e:
    import traceback
    print('generate_results_table: rouge_score import failed:', e)
    traceback.print_exc()
    raise SystemExit('Please install rouge-score: pip install rouge-score')


def compute(in_path, out_csv, out_summary, examples_limit=10):
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
            # support multiple possible keys for the fused/hypothesis text
            hyp = obj.get('fused_summary') or obj.get('fused') or obj.get('prediction') or ''

            scores = scorer.score(ref, hyp)
            r1 = scores['rouge1'].fmeasure
            r2 = scores['rouge2'].fmeasure
            rl = scores['rougeL'].fmeasure

            # sentence-level details may be under 'sentences' or 'details'
            sentences = obj.get('sentences') or obj.get('details') or []
            n_sent = len(sentences)
            n_supported = sum(1 for s in sentences if s.get('supported'))

            nli_counts = Counter()
            for s in sentences:
                lbl = s.get('nli_label') or s.get('best_label') or s.get('label') or 'neutral'
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
                # normalize common field names for examples
                sent_text = s.get('sentence') or s.get('text') or s.get('local_summary') or s.get('generated_text')
                evidence = s.get('evidence') or s.get('evidences') or s.get('evidences_list')
                nli_before = s.get('nli_label') or s.get('best_label') or s.get('label')
                nli_score_before = s.get('nli_score') or s.get('best_score')
                corrected_text = s.get('corrected') or s.get('corrected_text') or s.get('corrected_summary')
                nli_after = s.get('nli_label_corrected') or s.get('nli_label_after')
                nli_score_after = s.get('nli_score_corrected') or s.get('nli_score_after')

                if eff is True:
                    improved_examples.append({
                        'id': idx,
                        'sentence': sent_text,
                        'evidence': evidence,
                        'nli_label_before': nli_before,
                        'nli_score_before': nli_score_before,
                        'corrected': corrected_text,
                        'nli_label_after': nli_after,
                        'nli_score_after': nli_score_after,
                    })
                elif eff is False:
                    unchanged_examples.append({
                        'id': idx,
                        'sentence': sent_text,
                        'evidence': evidence,
                        'nli_label_before': nli_before,
                        'nli_score_before': nli_score_before,
                        'corrected': corrected_text,
                        'nli_label_after': nli_after,
                        'nli_score_after': nli_score_after,
                    })
                else:
                    # None or other -> consider as unchanged/unknown
                    unknown_examples.append({
                        'id': idx,
                        'sentence': sent_text,
                        'evidence': evidence,
                        'nli_label_before': nli_before,
                        'nli_score_before': nli_score_before,
                        'corrected': corrected_text,
                        'nli_label_after': nli_after,
                        'nli_score_after': nli_score_after,
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
        # write examples: if examples_limit==0 -> write ALL examples; otherwise cap
        if examples_limit == 0:
            selected = improved_examples + unchanged_examples + unknown_examples
        else:
            half = max(1, examples_limit // 2)
            selected = improved_examples[:half] + unchanged_examples[:half]
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
    p.add_argument('--examples', dest='examples', type=int, default=10, help='number of example cases to write; 0 for all')
    args = p.parse_args()
    compute(args.in_path, args.out_csv, args.out_summary, examples_limit=args.examples)


if __name__ == '__main__':
    main()

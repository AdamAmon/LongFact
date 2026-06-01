#!/usr/bin/env python3
"""Generate charts for selected correction cases.

Reads the JSONL produced by select_correction_cases.py and renders a small
set of PNG figures plus a compact JSON summary for report writing.
"""

import argparse
import csv
import json
import os
from statistics import mean
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate charts for selected correction cases')
    parser.add_argument('--input', '-i', default='results/selected_correction_cases.jsonl', help='selected case JSONL file')
    parser.add_argument('--output-dir', '-o', default='results/figures', help='directory for figures and summary files')
    parser.add_argument('--summary', default='results/case_chart_summary.json', help='JSON summary output path')
    return parser.parse_args()


def load_cases(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                cases.append(obj)
    return cases


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    successes = [case for case in cases if case.get('_label') == 'success']
    failures = [case for case in cases if case.get('_label') == 'failure']

    def avg(values: List[float]) -> float:
        return mean(values) if values else 0.0

    summary = {
        'count': len(cases),
        'success_count': len(successes),
        'failure_count': len(failures),
        'avg_support_rate_delta': avg([float(c.get('support_rate_delta', 0.0)) for c in cases]),
        'avg_rouge1_fmeasure_delta': avg([float(c.get('rouge1_fmeasure_delta', 0.0)) for c in cases]),
        'avg_support_rate_success': avg([float(c.get('support_rate', 0.0)) for c in successes]),
        'avg_support_rate_corrected_success': avg([float(c.get('corrected_support_rate', 0.0)) for c in successes]),
        'avg_support_rate_failure': avg([float(c.get('support_rate', 0.0)) for c in failures]),
        'avg_support_rate_corrected_failure': avg([float(c.get('corrected_support_rate', 0.0)) for c in failures]),
        'avg_rouge_success': avg([float(c.get('rouge1_fmeasure', 0.0)) for c in successes]),
        'avg_rouge_corrected_success': avg([float(c.get('rouge1_fmeasure_corrected', 0.0)) for c in successes]),
        'avg_rouge_failure': avg([float(c.get('rouge1_fmeasure', 0.0)) for c in failures]),
        'avg_rouge_corrected_failure': avg([float(c.get('rouge1_fmeasure_corrected', 0.0)) for c in failures]),
    }
    return summary


def render_charts(cases: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    chart_paths: Dict[str, str] = {}

    ordered = sorted(cases, key=lambda item: float(item.get('support_rate_delta', 0.0)), reverse=True)
    labels = [f"{case.get('_label', '')}:{case.get('id', '')}" for case in ordered]
    deltas = [float(case.get('support_rate_delta', 0.0)) for case in ordered]
    colors = ['#2E7D32' if value >= 0 else '#C62828' for value in deltas]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(range(len(ordered)), deltas, color=colors, width=0.72)
    ax.axhline(0, color='#444444', linewidth=1)
    ax.set_title('Support Rate Delta by Case')
    ax.set_ylabel('support_rate_delta')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, 'case_support_delta.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    chart_paths['support_delta'] = path

    rouge_before = [float(case.get('rouge1_fmeasure', 0.0)) for case in ordered]
    rouge_after = [float(case.get('rouge1_fmeasure_corrected', 0.0)) for case in ordered]
    x = range(len(ordered))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([idx - width / 2 for idx in x], rouge_before, width=width, label='before', color='#1565C0')
    ax.bar([idx + width / 2 for idx in x], rouge_after, width=width, label='corrected', color='#F9A825')
    ax.set_title('ROUGE-1 F1 Before vs Corrected')
    ax.set_ylabel('rouge1_fmeasure')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, 'case_rouge_compare.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    chart_paths['rouge_compare'] = path

    rouge_delta = [float(case.get('rouge1_fmeasure_delta', 0.0)) for case in ordered]
    support_delta = deltas
    fig, ax = plt.subplots(figsize=(8, 6))
    for case, sx, sy in zip(ordered, support_delta, rouge_delta):
        color = '#2E7D32' if case.get('_label') == 'success' else '#C62828'
        ax.scatter(sx, sy, color=color, s=70, alpha=0.9)
        ax.annotate(str(case.get('id', '')), (sx, sy), textcoords='offset points', xytext=(5, 4), fontsize=9)
    ax.axhline(0, color='#444444', linewidth=1)
    ax.axvline(0, color='#444444', linewidth=1)
    ax.set_xlabel('support_rate_delta')
    ax.set_ylabel('rouge1_fmeasure_delta')
    ax.set_title('Support Delta vs ROUGE Delta')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, 'case_delta_scatter.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    chart_paths['delta_scatter'] = path

    table_path = os.path.join(output_dir, 'selected_case_table.csv')
    with open(table_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'label', 'id', 'support_rate', 'corrected_support_rate', 'support_rate_delta',
            'rouge1_fmeasure', 'rouge1_fmeasure_corrected', 'rouge1_fmeasure_delta', 'sentence_count', 'token_count'
        ])
        for case in ordered:
            writer.writerow([
                case.get('_label', ''),
                case.get('id', ''),
                case.get('support_rate', ''),
                case.get('corrected_support_rate', ''),
                case.get('support_rate_delta', ''),
                case.get('rouge1_fmeasure', ''),
                case.get('rouge1_fmeasure_corrected', ''),
                case.get('rouge1_fmeasure_delta', ''),
                case.get('sentence_count', ''),
                case.get('token_count', ''),
            ])
    chart_paths['table_csv'] = table_path
    return chart_paths


def main() -> None:
    args = parse_args()
    cases = load_cases(args.input)
    if not cases:
        raise SystemExit(f'no cases found in {args.input}')

    ensure_dir(args.output_dir)
    summary = build_summary(cases)
    chart_paths = render_charts(cases, args.output_dir)
    summary['artifacts'] = chart_paths

    with open(args.summary, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f'wrote {len(cases)} cases into charts at {args.output_dir}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
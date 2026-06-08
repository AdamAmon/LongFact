"""全面对比分析 baseline 和 DCE 实验结果，生成报告所需的全部数据。
输出到 results/analysis_report.json
"""
import json
import csv
import os
import sys
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def get_val(d, key):
    """安全获取值"""
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return v

def analyze_set(label, result_dir):
    """分析一组实验结果"""
    summary = load_json(os.path.join(result_dir, 'summary_n500.json'))
    case_chart = load_json(os.path.join(result_dir, 'case_chart_summary.json'))
    
    # 基本指标
    basic = {
        'n': summary.get('n', 500),
        'avg_support_rate': summary.get('avg_support_rate'),
        'avg_corrected_support_rate': summary.get('correction_summary', {}).get('avg_corrected_support_rate'),
        'avg_support_rate_delta': summary.get('correction_summary', {}).get('avg_support_rate_delta'),
        'avg_rouge1_f1': summary.get('avg_rouge', {}).get('rouge1_fmeasure'),
        'avg_rouge2_f1': summary.get('avg_rouge', {}).get('rouge2_fmeasure'),
        'avg_rougeL_f1': summary.get('avg_rouge', {}).get('rougeLsum_fmeasure'),
        'avg_rouge1_f1_corrected': summary.get('correction_summary', {}).get('avg_rouge1_fmeasure_corrected'),
        'avg_rouge1_f1_delta': summary.get('correction_summary', {}).get('avg_rouge1_fmeasure_delta'),
    }
    
    # 案例统计
    cases_info = {
        'count': case_chart.get('count', 10),
        'success_count': case_chart.get('success_count', 5),
        'failure_count': case_chart.get('failure_count', 5),
        'case_avg_support_delta': case_chart.get('avg_support_rate_delta'),
        'case_avg_rouge1_delta': case_chart.get('avg_rouge1_fmeasure_delta'),
        'case_avg_success_support_before': case_chart.get('avg_support_rate_success'),
        'case_avg_success_support_after': case_chart.get('avg_support_rate_corrected_success'),
        'case_avg_failure_support_before': case_chart.get('avg_support_rate_failure'),
        'case_avg_failure_support_after': case_chart.get('avg_support_rate_corrected_failure'),
    }
    
    # 分桶数据
    buckets = summary.get('length_buckets', {})
    
    # CSV 逐样本数据
    csv_path = os.path.join(result_dir, 'per_sample_results.csv')
    csv_rows = []
    if os.path.exists(csv_path):
        csv_rows = load_csv(csv_path)
    
    # 支持率分布统计
    support_rates = [get_val(r, 'supported_ratio') for r in csv_rows if get_val(r, 'supported_ratio') is not None]
    
    # NLI 标签分布
    entail_counts = [get_val(r, 'n_entailment') for r in csv_rows if get_val(r, 'n_entailment') is not None]
    neutral_counts = [get_val(r, 'n_neutral') for r in csv_rows if get_val(r, 'n_neutral') is not None]
    contradict_counts = [get_val(r, 'n_contradiction') for r in csv_rows if get_val(r, 'n_contradiction') is not None]
    
    nli_dist = {}
    if entail_counts:
        total = sum(entail_counts) + sum(neutral_counts) + sum(contradict_counts)
        nli_dist = {
            'total_nli_calls': int(total),
            'entailment_pct': round(sum(entail_counts) / max(1, total) * 100, 2),
            'neutral_pct': round(sum(neutral_counts) / max(1, total) * 100, 2),
            'contradiction_pct': round(sum(contradict_counts) / max(1, total) * 100, 2),
            'avg_entailment_per_sample': round(statistics.mean(entail_counts), 2),
            'avg_neutral_per_sample': round(statistics.mean(neutral_counts), 2),
            'avg_contradiction_per_sample': round(statistics.mean(contradict_counts), 2),
        }
    
    # 句子数分布
    sent_counts = [get_val(r, 'num_sentences') for r in csv_rows if get_val(r, 'num_sentences') is not None]
    sent_dist = {}
    if sent_counts:
        sent_dist = {
            'avg_sentences_per_sample': round(statistics.mean(sent_counts), 2),
            'median_sentences': round(statistics.median(sent_counts), 2),
            'min_sentences': int(min(sent_counts)),
            'max_sentences': int(max(sent_counts)),
            'stdev_sentences': round(statistics.stdev(sent_counts), 2),
        }
    
    # 支持率桶分布（细粒度）
    support_buckets = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
    for sr in support_rates:
        if sr < 0.2:
            support_buckets['0.0-0.2'] += 1
        elif sr < 0.4:
            support_buckets['0.2-0.4'] += 1
        elif sr < 0.6:
            support_buckets['0.4-0.6'] += 1
        elif sr < 0.8:
            support_buckets['0.6-0.8'] += 1
        else:
            support_buckets['0.8-1.0'] += 1
    
    return {
        'label': label,
        'basic': basic,
        'cases': cases_info,
        'buckets': buckets,
        'nli_distribution': nli_dist,
        'sentence_distribution': sent_dist,
        'support_rate_distribution': support_buckets,
    }

# 分析两套结果
baseline = analyze_set('Baseline (Standard Retrieval)', 'results/baseline')
dce = analyze_set('Advanced (DCE Retrieval)', 'results/Advanced')

# 计算对比
def safe_diff(bv, av):
    if bv is not None and av is not None:
        return round(av - bv, 6)
    return None

comparison = {
    'support_rate_before': {
        'baseline': baseline['basic']['avg_support_rate'],
        'dce': dce['basic']['avg_support_rate'],
        'delta': safe_diff(baseline['basic']['avg_support_rate'], dce['basic']['avg_support_rate']),
    },
    'support_rate_after': {
        'baseline': baseline['basic']['avg_corrected_support_rate'],
        'dce': dce['basic']['avg_corrected_support_rate'],
        'delta': safe_diff(baseline['basic']['avg_corrected_support_rate'], dce['basic']['avg_corrected_support_rate']),
    },
    'support_rate_delta': {
        'baseline': baseline['basic']['avg_support_rate_delta'],
        'dce': dce['basic']['avg_support_rate_delta'],
        'delta': safe_diff(baseline['basic']['avg_support_rate_delta'], dce['basic']['avg_support_rate_delta']),
    },
    'rouge1_f1_before': {
        'baseline': baseline['basic']['avg_rouge1_f1'],
        'dce': dce['basic']['avg_rouge1_f1'],
        'note': 'ROUGE is identical across strategies (same summaries)',
    },
    'rouge1_f1_after': {
        'baseline': baseline['basic']['avg_rouge1_f1_corrected'],
        'dce': dce['basic']['avg_rouge1_f1_corrected'],
        'delta': safe_diff(baseline['basic']['avg_rouge1_f1_corrected'], dce['basic']['avg_rouge1_f1_corrected']),
    },
    'rough2_f1': {
        'baseline': baseline['basic']['avg_rouge2_f1'],
        'dce': dce['basic']['avg_rouge2_f1'],
    },
    'roughL_f1': {
        'baseline': baseline['basic']['avg_rougeL_f1'],
        'dce': dce['basic']['avg_rougeL_f1'],
    },
}

report = {
    'generated_at': '2026-06-08',
    'description': 'LongFact Baseline vs DCE (Dual-Channel Evidence) comparison on GovReport 500 samples',
    'baseline': baseline,
    'dce': dce,
    'comparison': comparison,
    'key_findings': {
        'dce_baseline_diff_support_before': f"DCE原始支持率比基线{'高' if comparison['support_rate_before']['delta'] > 0 else '低'}{abs(comparison['support_rate_before']['delta']):.4f}",
        'dce_baseline_diff_support_after': f"DCE纠错后支持率比基线{'高' if comparison['support_rate_after']['delta'] > 0 else '低'}{abs(comparison['support_rate_after']['delta']):.4f}",
        'dce_baseline_diff_support_delta': f"DCE支持率提升幅度比基线{'大' if comparison['support_rate_delta']['delta'] > 0 else '小'}{abs(comparison['support_rate_delta']['delta']):.4f}",
        'rouge_note': 'ROUGE与检索策略无关（相同摘要），纠错后ROUGE变化极小',
    }
}

output_path = 'results/analysis_report.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Report written to {output_path}")
print()
print("=== 核心对比 ===")
print(f"  {'指标':<30} {'Baseline':>12} {'DCE':>12} {'Delta':>12}")
print(f"  {'-'*66}")
for k, v in comparison.items():
    if isinstance(v, dict) and 'baseline' in v:
        bv = v.get('baseline', 0) or 0
        av = v.get('dce', 0) or 0
        dv = v.get('delta', 0) or 0
        print(f"  {k:<30} {bv:>12.4f} {av:>12.4f} {dv:>+12.4f}")

print()
print("=== NLI标签分布 ===")
for name, data in [('Baseline', baseline), ('DCE', dce)]:
    nd = data['nli_distribution']
    print(f"  {name}: entail={nd.get('entailment_pct')}%, neutral={nd.get('neutral_pct')}%, contradict={nd.get('contradiction_pct')}%")

print()
print("=== 支持率细粒度分布 ===")
for bucket_name in ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
    b_count = baseline['support_rate_distribution'].get(bucket_name, 0)
    d_count = dce['support_rate_distribution'].get(bucket_name, 0)
    print(f"  {bucket_name}: baseline={b_count}, dce={d_count}")

print()
print("=== 句子数分布 ===")
print(f"  Baseline: avg={baseline['sentence_distribution'].get('avg_sentences_per_sample')}, median={baseline['sentence_distribution'].get('median_sentences')}")
print(f"  DCE:      avg={dce['sentence_distribution'].get('avg_sentences_per_sample')}, median={dce['sentence_distribution'].get('median_sentences')}")

print("\nDone!")

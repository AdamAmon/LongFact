"""生成 Baseline vs DCE 实验报告所需的全部图表。
Output: results/figures/comparison_*.png
"""
import json
import csv
import os
import sys
import statistics
from collections import defaultdict
from pathlib import Path

# 尝试导入 matplotlib，如果不可用则跳过
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("WARNING: matplotlib not available, skipping chart generation")

OUTPUT_DIR = Path('results/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MATPLOTLIB_OK:
    print("No matplotlib, exiting.")
    sys.exit(0)

# 中文字体设置
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_summary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ============================================================
# 1. 支持率对比柱状图 (纠错前 vs 纠错后, baseline vs dce)
# ============================================================
def plot_support_comparison(baseline_summary, dce_summary):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    b_before = baseline_summary.get('avg_support_rate', 0)
    b_after = baseline_summary.get('correction_summary', {}).get('avg_corrected_support_rate', 0)
    d_before = dce_summary.get('avg_support_rate', 0)
    d_after = dce_summary.get('correction_summary', {}).get('avg_corrected_support_rate', 0)
    
    categories = ['Before Correction', 'After Correction']
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [b_before, b_after], width, label='Baseline', color='#4472C4', edgecolor='white')
    bars2 = ax.bar(x + width/2, [d_before, d_after], width, label='DCE', color='#ED7D31', edgecolor='white')
    
    # 加数值标签
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Sentence-Level Support Rate', fontsize=13)
    ax.set_title('Support Rate: Baseline vs DCE (Before/After Correction)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_support_rate.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 2. 纠错前后支持率提升对比
# ============================================================
def plot_support_delta_comparison(baseline_summary, dce_summary):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    b_delta = baseline_summary.get('correction_summary', {}).get('avg_support_rate_delta', 0)
    d_delta = dce_summary.get('correction_summary', {}).get('avg_support_rate_delta', 0)
    
    bars = ax.bar(['Baseline', 'DCE'], [b_delta, d_delta], 
                  color=['#4472C4', '#ED7D31'], edgecolor='white', width=0.5)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{bar.get_height():.4f}', 
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_ylabel('Support Rate Improvement (Δ)', fontsize=13)
    ax.set_title('Correction Effectiveness: Support Rate Improvement', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(b_delta, d_delta) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_support_delta.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 3. ROUGE-1/2/L 雷达图/柱状图对比
# ============================================================
def plot_rouge_comparison(baseline_summary, dce_summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    rouge_keys = ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeLsum_fmeasure']
    rouge_labels = ['ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']
    
    b_rouge = baseline_summary.get('avg_rouge', {})
    d_rouge = dce_summary.get('avg_rouge', {})
    
    b_vals = [b_rouge.get(k, 0) for k in rouge_keys]
    d_vals = [d_rouge.get(k, 0) for k in rouge_keys]
    
    x = np.arange(len(rouge_labels))
    width = 0.35
    
    ax.bar(x - width/2, b_vals, width, label='Baseline', color='#4472C4', edgecolor='white')
    ax.bar(x + width/2, d_vals, width, label='DCE', color='#ED7D31', edgecolor='white')
    
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('ROUGE Scores: Baseline vs DCE', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rouge_labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_rouge.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 4. NLI 标签分布对比 (entail/neutral/contradict)
# ============================================================
def plot_nli_distribution(baseline_csv, dce_csv):
    def count_labels(rows):
        entail = sum(int(r.get('n_entailment', 0)) for r in rows)
        neutral = sum(int(r.get('n_neutral', 0)) for r in rows)
        contradict = sum(int(r.get('n_contradiction', 0)) for r in rows)
        total = entail + neutral + contradict
        if total == 0:
            return (0, 0, 0)
        return (
            entail / total * 100,
            neutral / total * 100,
            contradict / total * 100
        )
    
    b_vals = count_labels(baseline_csv)
    d_vals = count_labels(dce_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = ['#2ECC71', '#F39C12', '#E74C3C']
    labels = ['Entailment', 'Neutral', 'Contradiction']
    
    for ax, vals, title in zip(axes, [b_vals, d_vals], ['Baseline', 'DCE']):
        wedges, texts, autotexts = ax.pie(vals, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90,
                                          textprops={'fontsize': 11})
        ax.set_title(f'{title}\nNLI Label Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_nli_pie.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 5. 支持率细粒度直方图分布
# ============================================================
def plot_support_histogram(baseline_csv, dce_csv):
    def get_support_rates(rows):
        rates = []
        for r in rows:
            v = r.get('supported_ratio')
            if v:
                try:
                    rates.append(float(v))
                except:
                    pass
        return rates
    
    b_rates = get_support_rates(baseline_csv)
    d_rates = get_support_rates(dce_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, rates, title, color in zip(axes, [b_rates, d_rates], 
                                        ['Baseline', 'DCE'], 
                                        ['#4472C4', '#ED7D31']):
        ax.hist(rates, bins=20, range=(0, 1), color=color, edgecolor='white', alpha=0.8)
        ax.axvline(statistics.mean(rates), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean={statistics.mean(rates):.3f}')
        ax.set_xlabel('Support Rate', fontsize=12)
        ax.set_ylabel('Sample Count', fontsize=12)
        ax.set_title(f'{title}: Support Rate Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_support_hist.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 6. 典型案例 support_rate_delta 散点图
# ============================================================
def plot_case_scatter(baseline_cases, dce_cases):
    """绘制两个策略的案例散点对比"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for cases, color, label, marker in [
        (baseline_cases, '#4472C4', 'Baseline', 'o'),
        (dce_cases, '#ED7D31', 'DCE', 's')
    ]:
        ids = []
        deltas = []
        for c in cases:
            # cases 可能有不同结构
            if isinstance(c, dict):
                sid = c.get('id', c.get('sample_id', len(ids)))
                delta = c.get('support_rate_delta', c.get('support_delta', 0))
                ids.append(sid)
                deltas.append(float(delta))
        
        ax.scatter(range(len(deltas)), deltas, c=color, label=label, marker=marker, 
                   s=100, edgecolors='white', linewidth=0.5, zorder=3, alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Case Index', fontsize=12)
    ax.set_ylabel('Support Rate Delta (After - Before)', fontsize=12)
    ax.set_title('Correction Cases: Support Rate Delta', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_cases_scatter.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Loading data...")
    
    b_summary = load_summary('results/baseline/summary_n500.json')
    d_summary = load_summary('results/Advanced/summary_n500.json')
    b_csv = load_csv_dict('results/baseline/per_sample_results.csv')
    d_csv = load_csv_dict('results/Advanced/per_sample_results.csv')
    b_cases = load_csv_dict('results/baseline/per_sample_results.csv')  # same CSV
    d_cases = load_csv_dict('results/Advanced/per_sample_results.csv')
    
    print("Generating charts...")
    plot_support_comparison(b_summary, d_summary)
    plot_support_delta_comparison(b_summary, d_summary)
    plot_rouge_comparison(b_summary, d_summary)
    plot_nli_distribution(b_csv, d_csv)
    plot_support_histogram(b_csv, d_csv)
    
    print("All charts generated!")

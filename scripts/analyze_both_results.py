"""全面分析 baseline 和 Advanced (DCE) 的实验结果"""
import json
import sys

def load_summary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_summary(label, s):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k, v in s.items():
        if k in ('per_sample', 'cases', 'records'):
            continue
        if isinstance(v, dict):
            print(f"\n[{k}]")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    print(f"  {kk}: {json.dumps(vv, ensure_ascii=False)}")
                elif isinstance(vv, float):
                    print(f"  {kk}: {vv:.6f}")
                else:
                    print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        elif isinstance(v, list):
            print(f"  {k}: [{len(v)} items]")
        else:
            print(f"  {k}: {v}")

    # 分桶
    bk = s.get('buckets')
    if isinstance(bk, dict):
        print(f"\n[buckets]")
        for bname, bdata in bk.items():
            if isinstance(bdata, dict):
                cnt = bdata.get('count', '?')
                sup = bdata.get('avg_support_rate', '?')
                corrected_sup = bdata.get('avg_corrected_support_rate', '?')
                r1 = bdata.get('avg_rouge1_fmeasure', '?')
                r1c = bdata.get('avg_corrected_rouge1_fmeasure', '?')
                if isinstance(sup, float):
                    sup = f"{sup:.4f}"
                if isinstance(corrected_sup, float):
                    corrected_sup = f"{corrected_sup:.4f}"
                if isinstance(r1, float):
                    r1 = f"{r1:.4f}"
                if isinstance(r1c, float):
                    r1c = f"{r1c:.4f}"
                print(f"  [{bname}]: count={cnt}, support={sup}, corr_support={corrected_sup}, R1={r1}, R1_corr={r1c}")


b = load_summary('results/baseline/summary_n500.json')
a = load_summary('results/Advanced/summary_n500.json')

print_summary("BASELINE (基线检索)", b)
print_summary("ADVANCED (DCE 双通道检索)", a)

# 关键对比
print(f"\n{'='*60}")
print(f"  基线 vs DCE 关键指标对比")
print(f"{'='*60}")

def get_val(s, *keys):
    for k in keys:
        v = s.get(k)
        if v is not None:
            return v
    # try nested
    for top_k in ('overall', 'averages'):
        inner = s.get(top_k, {})
        if isinstance(inner, dict):
            for k in keys:
                v = inner.get(k)
                if v is not None:
                    return v
    return None

metrics = [
    ('avg_support_rate', '平均支持率(纠错前)'),
    ('avg_corrected_support_rate', '平均支持率(纠错后)'),
    ('mean_support_delta', '平均支持率提升'),
    ('avg_rouge1_fmeasure', '平均ROUGE-1 F1(纠错前)'),
    ('avg_corrected_rouge1_fmeasure', '平均ROUGE-1 F1(纠错后)'),
    ('avg_improved_pct', '改善样本占比(%)'),
    ('avg_worsened_pct', '恶化样本占比(%)'),
]

for key, label in metrics:
    bv = get_val(b, key)
    av = get_val(a, key)
    if bv is not None and av is not None:
        delta = av - bv
        print(f"  {label}: baseline={bv:.4f}, dce={av:.4f}, delta={delta:+.4f}")

print("\nDone!")

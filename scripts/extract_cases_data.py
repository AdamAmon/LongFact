"""提取并整理 baseline 和 DCE 的 10 个案例数据"""
import json

def read_cases(path):
    cases = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                rouge = c.get('rouge', {})
                rouge_c = c.get('rouge_corrected', {})
                if isinstance(rouge, dict):
                    r1 = rouge.get('rouge1_fmeasure', 0)
                else:
                    r1 = 0
                if isinstance(rouge_c, dict):
                    r1c = rouge_c.get('rouge1_fmeasure', 0)
                else:
                    r1c = 0
                cases.append({
                    'id': c.get('id', '?'),
                    'support_before': round(float(c.get('support_rate', 0)), 4),
                    'support_after': round(float(c.get('corrected_support_rate', 0)), 4),
                    'delta': round(float(c.get('support_rate_delta', 0)), 4),
                    'r1_before': round(float(r1), 4),
                    'r1_after': round(float(r1c), 4),
                    'r1_delta': round(float(c.get('rouge1_fmeasure_delta', 0)), 4),
                    'label': c.get('_label', ''),
                })
    return cases

b = read_cases('results/baseline/selected_correction_cases.jsonl')
d = read_cases('results/Advanced/selected_correction_cases.jsonl')

print('=== Baseline cases (10) ===')
print('-' * 70)
for c in b:
    tag = 'SUCCESS' if c['delta'] > 0 else 'FAIL'
    print(f"ID={c['id']:>4} | sup: {c['support_before']:.4f} -> {c['support_after']:.4f} (d={c['delta']:+.4f}) | R1d={c['r1_delta']:+.4f} | {tag}")

print()
print('-' * 70)
print('=== DCE cases (10) ===')
print('-' * 70)
for c in d:
    tag = 'SUCCESS' if c['delta'] > 0 else 'FAIL'
    print(f"ID={c['id']:>4} | sup: {c['support_before']:.4f} -> {c['support_after']:.4f} (d={c['delta']:+.4f}) | R1d={c['r1_delta']:+.4f} | {tag}")

# summary stats
b_success = [c for c in b if c['delta'] > 0]
b_fail = [c for c in b if c['delta'] <= 0]
d_success = [c for c in d if c['delta'] > 0]
d_fail = [c for c in d if c['delta'] <= 0]

print()
print(f"Baseline: {len(b_success)} success + {len(b_fail)} fail")
print(f"DCE:      {len(d_success)} success + {len(d_fail)} fail")

# avg deltas
import statistics
if b_success:
    print(f"Baseline success avg delta: {statistics.mean([c['delta'] for c in b_success]):.4f}")
if b_fail:
    print(f"Baseline fail avg delta: {statistics.mean([c['delta'] for c in b_fail]):.4f}")
if d_success:
    print(f"DCE success avg delta: {statistics.mean([c['delta'] for c in d_success]):.4f}")
if d_fail:
    print(f"DCE fail avg delta: {statistics.mean([c['delta'] for c in d_fail]):.4f}")

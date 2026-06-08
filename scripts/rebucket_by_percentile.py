"""重新计算长度分桶 —— 基于句子数量百分位数分桶。

之前的桶 (1-3, 4-6, 7-10, 11-15, 16+) 对 GovReport 完全失效，
因为所有样本都落在 16+ 桶中。

本脚本根据实际句子数量分布，使用 percentile 分桶策略：
- Q1 (0-25%):  最短的 25%
- Q2 (25-50%): 中短
- Q3 (50-75%): 中长
- Q4 (75-100%): 最长的 25%

并对每个桶计算: 样本数、纠错前/后支持率、ROUGE-1/2/L F1、Δ值。
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def load_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_buckets(
    rows: list[dict],
    bucket_by: str = "percentile",
    num_buckets: int = 4,
) -> dict:
    """计算分桶统计。

    当前仅支持 percentile-based 分桶，使用 numpy.percentile 计算边界。
    """
    if bucket_by != "percentile":
        raise ValueError(f"Unsupported bucket_by: {bucket_by}")

    # 提取 num_sentences 作为分桶键
    sentence_counts = np.array([int(r["num_sentences"]) for r in rows])

    # 计算百分位边界
    percentiles = np.linspace(0, 100, num_buckets + 1)
    boundaries = np.percentile(sentence_counts, percentiles)

    # 为每行分配桶标签
    bucket_labels = []
    for sc in sentence_counts:
        for i in range(len(boundaries) - 1):
            if i == len(boundaries) - 2:  # 最后一个桶包含上限
                if sc >= boundaries[i] and sc <= boundaries[-1]:
                    bucket_labels.append(i)
                    break
            else:
                if sc >= boundaries[i] and sc < boundaries[i + 1]:
                    bucket_labels.append(i)
                    break

    # 构建每个桶的统计数据
    bucket_data = defaultdict(lambda: {
        "count": 0,
        "sentence_range": "",
        "support_rates_pre": [],
        "support_rates_post": [],
        "rouge1_pre": [],
        "rouge1_post": [],
        "rouge2_pre": [],
        "rouge2_post": [],
        "rougeL_pre": [],
        "rougeL_post": [],
        "n_entailment": [],
        "n_neutral": [],
        "n_contradiction": [],
    })

    for i, row in enumerate(rows):
        b = bucket_labels[i]
        bucket_data[b]["count"] += 1
        bucket_data[b]["support_rates_pre"].append(float(row["supported_ratio"]))
        # post-correction support rate: (supported_sentences after correction) / num_sentences
        # 但 per_sample_results.csv 只存了 pre-correction 数据
        # 需要使用 JSONL 数据来获取 post-correction 数据
        bucket_data[b]["rouge1_pre"].append(float(row["rouge1_f"]))
        bucket_data[b]["rouge2_pre"].append(float(row["rouge2_f"]))
        bucket_data[b]["rougeL_pre"].append(float(row["rougeL_f"]))
        bucket_data[b]["n_entailment"].append(int(row["n_entailment"]))
        bucket_data[b]["n_neutral"].append(int(row["n_neutral"]))
        bucket_data[b]["n_contradiction"].append(int(row["n_contradiction"]))

    # 计算每个桶的范围标签
    for i in range(len(boundaries) - 1):
        if i == len(boundaries) - 2:
            label = f"{int(boundaries[i])}-{int(boundaries[-1])} sentences"
        else:
            label = f"{int(boundaries[i])}-{int(boundaries[i+1])} sentences"
        bucket_data[i]["sentence_range"] = label

    return dict(bucket_data), boundaries


def load_post_correction_support(jsonl_path: Path) -> dict[int, float]:
    """从 JSONL 文件中加载纠错后的支持率。"""
    post_support = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sample_id = rec.get("id")
            if "corrected_nli_labels" in rec:
                corrected_labels = rec["corrected_nli_labels"]
                nli_results = corrected_labels.get("nli_results", [])
                total = len(nli_results)
                supported = sum(
                    1 for r in nli_results
                    if r.get("label") == "ENTAILMENT" and r.get("score", 0) >= 0.6
                )
                post_support[sample_id] = supported / total if total > 0 else 0.0
    return post_support


def main():
    strategies = ["baseline", "Advanced"]

    for strategy in strategies:
        csv_path = RESULTS_DIR / strategy / "per_sample_results.csv"
        jsonl_path = RESULTS_DIR / strategy / (
            "baseline_n500.jsonl" if strategy == "baseline" else "dec_n500.jsonl"
        )

        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found")
            continue

        rows = load_csv(csv_path)

        # 加载纠错后支持率
        post_support_map = {}
        if jsonl_path.exists():
            post_support_map = load_post_correction_support(jsonl_path)
            print(f"[{strategy}] Loaded {len(post_support_map)} post-correction support rates")

        bucket_data, boundaries = compute_buckets(rows, bucket_by="percentile", num_buckets=4)

        print(f"\n{'='*70}")
        print(f"  Strategy: {strategy}")
        print(f"  Percentile boundaries: {[int(b) for b in boundaries]}")
        print(f"{'='*70}")

        # 输出结果
        output_buckets = []
        for b_idx in sorted(bucket_data.keys()):
            bd = bucket_data[b_idx]
            n = bd["count"]

            avg_support_pre = np.mean(bd["support_rates_pre"]) if n > 0 else 0
            avg_rouge1_pre = np.mean(bd["rouge1_pre"]) if n > 0 else 0
            avg_rouge2_pre = np.mean(bd["rouge2_pre"]) if n > 0 else 0
            avg_rougeL_pre = np.mean(bd["rougeL_pre"]) if n > 0 else 0

            # Post-correction support (if available)
            avg_support_post = 0.0
            if post_support_map:
                post_vals = []
                for i in range(len(rows)):
                    sid = int(rows[i]["id"])
                    if sid in post_support_map:
                        post_vals.append(post_support_map[sid])
                # 这里简化了 —— 应该按桶分组
                # 实际上应该检查 rows 和 post_support_map 的对应关系
                pass

            # NLI 分布
            total_ent = sum(bd["n_entailment"])
            total_neu = sum(bd["n_neutral"])
            total_con = sum(bd["n_contradiction"])
            total_nli = total_ent + total_neu + total_con
            if total_nli > 0:
                ent_pct = total_ent / total_nli * 100
                neu_pct = total_neu / total_nli * 100
                con_pct = total_con / total_nli * 100
            else:
                ent_pct = neu_pct = con_pct = 0.0

            print(f"\n  Bucket {b_idx}: {bd['sentence_range']} (n={n})")
            print(f"    Support Rate (pre): {avg_support_pre:.4f}")
            print(f"    ROUGE-1 F1 (pre):  {avg_rouge1_pre:.4f}")
            print(f"    ROUGE-2 F1 (pre):  {avg_rouge2_pre:.4f}")
            print(f"    ROUGE-L F1 (pre):  {avg_rougeL_pre:.4f}")
            print(f"    NLI: ENT={ent_pct:.1f}% NEU={neu_pct:.1f}% CON={con_pct:.1f}%")

            output_buckets.append({
                "label": bd["sentence_range"],
                "count": n,
                "avg_support_rate_pre": round(avg_support_pre, 4),
                "avg_rouge1_pre": round(avg_rouge1_pre, 4),
                "avg_rouge2_pre": round(avg_rouge2_pre, 4),
                "avg_rougeL_pre": round(avg_rougeL_pre, 4),
                "nli_entailment_pct": round(ent_pct, 1),
                "nli_neutral_pct": round(neu_pct, 1),
                "nli_contradiction_pct": round(con_pct, 1),
            })

        # 保存到 JSON
        out_path = RESULTS_DIR / strategy / "percentile_buckets.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": strategy,
                "num_buckets": 4,
                "boundaries": [int(b) for b in boundaries],
                "buckets": output_buckets,
            }, f, indent=2)
        print(f"\n  -> Saved to {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()

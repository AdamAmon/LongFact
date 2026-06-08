"""计算各长度百分位桶的纠错前/后支持率和ROUGE。"""
import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

BOUNDARIES = [83, 240, 352, 509, 1437]
LABELS = ["83-240", "240-352", "352-509", "509-1437"]


def main():
    for strategy in ["baseline", "Advanced"]:
        csv_path = RESULTS / strategy / "per_sample_results.csv"
        jsonl_name = "baseline_n500.jsonl" if strategy == "baseline" else "dec_n500.jsonl"
        jsonl_path = RESULTS / strategy / jsonl_name

        # Load corrected data from JSONL
        corrected = {}  # id -> {corrected_support_rate, corrected_rouge1_f}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sid = d["id"]
                csr = d.get("corrected_support_rate", None)
                cr1 = None
                rc = d.get("rouge_corrected", {})
                if rc:
                    cr1 = rc.get("rouge1_fmeasure", None)
                corrected[sid] = {"csr": csr, "cr1": cr1}

        print(f"[{strategy}] Loaded {len(corrected)} corrected records")

        # Read pre-correction CSV
        rows = []
        with open(csv_path, "r") as f:
            for r in csv.DictReader(f):
                rows.append(r)

        # Bucket
        buckets = defaultdict(lambda: {
            "count": 0,
            "pre_sup": [], "post_sup": [],
            "pre_r1": [], "post_r1": [],
            "pre_r2": [], "pre_rL": [],
        })

        for r in rows:
            sc = int(r["num_sentences"])
            sid = int(r["id"])
            pre_sup = float(r["supported_ratio"])
            pre_r1 = float(r["rouge1_f"])
            pre_r2 = float(r["rouge2_f"])
            pre_rL = float(r["rougeL_f"])

            # Assign bucket index
            b_idx = 3
            for i in range(len(BOUNDARIES) - 1):
                if BOUNDARIES[i] <= sc < BOUNDARIES[i + 1]:
                    b_idx = i
                    break
            if sc == BOUNDARIES[-1]:
                b_idx = 3

            buckets[b_idx]["count"] += 1
            buckets[b_idx]["pre_sup"].append(pre_sup)
            buckets[b_idx]["pre_r1"].append(pre_r1)
            buckets[b_idx]["pre_r2"].append(pre_r2)
            buckets[b_idx]["pre_rL"].append(pre_rL)

            if sid in corrected:
                cd = corrected[sid]
                if cd["csr"] is not None:
                    buckets[b_idx]["post_sup"].append(cd["csr"])
                if cd["cr1"] is not None:
                    buckets[b_idx]["post_r1"].append(cd["cr1"])

        # Print results
        print(f"  {'Bucket':10s} {'n':>4s}  {'Pre-Sup':>8s}  {'Post-Sup':>8s}  "
              f"{'DeltaSup':>10s}  {'Pre-R1':>8s}  {'Post-R1':>8s}")
        for i in range(4):
            bd = buckets[i]
            n = bd["count"]
            pre_s = sum(bd["pre_sup"]) / n if n else 0
            post_s = sum(bd["post_sup"]) / len(bd["post_sup"]) if bd["post_sup"] else 0
            delta_s = post_s - pre_s
            pre_r1 = sum(bd["pre_r1"]) / n if n else 0
            post_r1 = sum(bd["post_r1"]) / len(bd["post_r1"]) if bd["post_r1"] else 0
            print(f"  {LABELS[i]:10s} {n:4d}  {pre_s:8.4f}  {post_s:8.4f}  "
                  f"{delta_s:+10.4f}  {pre_r1:8.4f}  {post_r1:8.4f}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge multiple JSONL result files into one JSONL, with optional dedup by `id` and ID renumbering.

Usage:
    python scripts/merge_results.py --out results/merged.jsonl results/host*_results.jsonl
    python scripts/merge_results.py --out results/merged.jsonl --renumber-ids --no-dedupe results/*.jsonl
"""
import argparse
import glob
import json
from pathlib import Path


def iter_input_paths(inputs):
    for p in inputs:
        matched = glob.glob(p)
        if matched:
            for m in matched:
                yield Path(m)
        else:
            yield Path(p)


def merge_files(output_path: Path, input_patterns, dedupe_by_id=True, renumber_ids=False):
    seen_ids = set()
    records = []  # store objects if renumbering needed
    total_in = 0
    total_out = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: collect all records (or write directly if not renumbering)
    for p in iter_input_paths(input_patterns):
        if not p.exists():
            print(f"[WARN] input not found: {p}")
            continue
        with p.open("r", encoding="utf-8") as fin:
            for line in fin:
                total_in += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"[WARN] skipping invalid JSON in {p}: {e}")
                    continue
                if dedupe_by_id and isinstance(obj, dict) and "id" in obj:
                    rid = obj["id"]
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                if renumber_ids:
                    records.append(obj)
                else:
                    # write immediately
                    with output_path.open("a", encoding="utf-8") as fout:
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total_out += 1

    if renumber_ids:
        # Write all records with new sequential IDs
        with output_path.open("w", encoding="utf-8") as fout:
            for new_id, obj in enumerate(records):
                if isinstance(obj, dict):
                    obj["id"] = new_id
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_out += 1
        print(f"Renumbered IDs: 0 .. {len(records)-1}")

    return total_in, total_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input JSONL files or glob patterns")
    ap.add_argument("--out", required=True, help="Output merged JSONL path")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                    help="Do not deduplicate by 'id' field")
    ap.add_argument("--renumber-ids", action="store_true",
                    help="Reassign sequential IDs (0,1,2,...) to all output records")
    args = ap.parse_args()

    outp = Path(args.out)
    total_in, total_out = merge_files(outp, args.inputs,
                                      dedupe_by_id=args.dedupe,
                                      renumber_ids=args.renumber_ids)
    print(f"Merged {total_out} records (from {total_in} input lines) into {outp}")


if __name__ == "__main__":
    main()
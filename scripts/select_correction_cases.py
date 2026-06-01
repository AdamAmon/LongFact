#!/usr/bin/env python3
"""Select top-N successful and failed correction cases from a large results file.

This script is robust to both JSONL and a single JSON array file. It streams data
where possible and tolerates minor formatting issues.
"""

import argparse
import csv
import json
import os
import sys
from typing import Iterator, Tuple, Any, Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser(description='Select top-N correction cases from results')
    p.add_argument('--input', '-i', default=os.environ.get('INPUT_FILE', 'results/correction_cases.json'), help='input results file (JSON or JSONL)')
    p.add_argument('--output', '-o', default=os.environ.get('OUTPUT_FILE', 'results/selected_correction_cases.jsonl'), help='output file (jsonl or csv)')
    p.add_argument('--top-n', '-n', type=int, default=int(os.environ.get('TOP_N', '5')), help='number of top increases and decreases to select')
    p.add_argument('--before-keys', default=os.environ.get('BEFORE_KEYS', 'support_rate'), help='comma-separated candidate keys for before metric')
    p.add_argument('--after-keys', default=os.environ.get('AFTER_KEYS', 'corrected_support_rate'), help='comma-separated candidate keys for after metric')
    p.add_argument('--format', choices=['jsonl','csv'], default=os.environ.get('OUT_FORMAT','jsonl'), help='output format')
    p.add_argument('--filter', action='append', help='filter condition key=value (can be repeated)')
    p.add_argument('--mode', choices=['delta','after','before'], default='delta', help='selection mode: delta (after-before) or sort by after/before')
    p.add_argument('--verbose', '-v', action='store_true', help='verbose logging')
    return p.parse_args()


args = parse_args()
IN = args.input
OUT = args.output
N = args.top_n
BEFORE_KEYS = [k.strip() for k in args.before_keys.split(',') if k.strip()]
AFTER_KEYS = [k.strip() for k in args.after_keys.split(',') if k.strip()]
OUT_FORMAT = args.format
FILTERS = args.filter or []
MODE = args.mode
VERBOSE = args.verbose


def get_num(o: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in o and o[k] is not None:
            try:
                return float(o[k])
            except Exception:
                pass
    return None


def iter_json_records(path: str) -> Iterator[Any]:
    """Yield JSON objects from a file that is either JSONL or a JSON array.

    Best-effort parsing: if the file begins with '[' we parse the whole array.
    Otherwise we attempt line-by-line JSON parsing and perform light cleanup
    for trailing commas or surrounding brackets.
    """
    with open(path, 'r', encoding='utf-8') as f:
        # peek first non-whitespace character
        pos = f.tell()
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        f.seek(pos)
        if first in ('[','{'):
            try:
                data = json.load(f)
                # If the file is a list of records, yield each.
                if isinstance(data, list):
                    for item in data:
                        yield item
                    return
                # If the file is a dict containing a list under common keys,
                # iterate that list instead.
                if isinstance(data, dict):
                    for key in ('selected_cases', 'cases', 'data', 'results'):
                        if key in data and isinstance(data[key], list):
                            for item in data[key]:
                                yield item
                            return
                    # otherwise yield the dict itself as a single record
                    yield data
                    return
            except Exception:
                # fallback to streaming
                f.seek(0)

        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                # try to clean common wrappers: leading commas, surrounding brackets
                s2 = s.lstrip(',\n\r \t')
                s2 = s2.rstrip(',\n\r \t]')
                # if it's still not object-like, attempt to extract {...}
                if not s2.startswith('{'):
                    idx = s2.find('{')
                    if idx != -1:
                        s2 = s2[idx:]
                if not s2.endswith('}'):
                    j = s2.rfind('}')
                    if j != -1:
                        s2 = s2[:j+1]
                try:
                    yield json.loads(s2)
                except Exception:
                    # skip irrecoverable lines
                    continue


def passes_filters(obj: Dict[str, Any], filters: List[str]) -> bool:
    if not filters:
        return True
    for f in filters:
        if '=' not in f:
            continue
        k, v = f.split('=', 1)
        if str(obj.get(k)) != v:
            return False
    return True


def main():
    if not os.path.exists(IN):
        print('文件不存在:', IN)
        sys.exit(1)

    records: List[Tuple[float, Dict[str, Any]]] = []  # list of (score, obj)
    scanned = 0

    for obj in iter_json_records(IN):
        scanned += 1
        if not isinstance(obj, dict):
            continue

        # allow metrics in nested 'metrics' dict too
        metrics = obj.get('metrics') if isinstance(obj.get('metrics'), dict) else None

        s = get_num(obj, BEFORE_KEYS)
        c = get_num(obj, AFTER_KEYS)
        if (s is None or c is None) and metrics:
            s = s or get_num(metrics, BEFORE_KEYS)
            c = c or get_num(metrics, AFTER_KEYS)

        if MODE == 'delta':
            if s is None or c is None:
                continue
            score = c - s
        elif MODE == 'after':
            if c is None:
                continue
            score = c
        else:  # before
            if s is None:
                continue
            score = s

        if not passes_filters(obj, FILTERS):
            continue

        records.append((score, obj))

    if not records:
        print('未找到符合条件的记录')
        sys.exit(0)

    # top increases = highest scores; top decreases = lowest scores
    top_inc = sorted(records, key=lambda x: x[0], reverse=True)[:N]
    top_dec = sorted(records, key=lambda x: x[0])[:N]

    # write output
    if OUT_FORMAT == 'jsonl' or OUT.lower().endswith('.jsonl'):
        with open(OUT, 'w', encoding='utf-8') as out:
            for score, obj in top_inc:
                obj_copy = dict(obj)
                obj_copy['_score'] = score
                obj_copy['_label'] = 'success'
                out.write(json.dumps(obj_copy, ensure_ascii=False) + '\n')
            for score, obj in top_dec:
                obj_copy = dict(obj)
                obj_copy['_score'] = score
                obj_copy['_label'] = 'failure'
                out.write(json.dumps(obj_copy, ensure_ascii=False) + '\n')
    else:
        # csv
        keys = ['id','bucket','sentence_count','token_count']
        extra = ['support_rate','corrected_support_rate','_score','_label','reason_tag','reason_text','rouge1_fmeasure','rouge1_fmeasure_corrected']
        cols = keys + extra
        with open(OUT, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(cols)
            for score, obj in top_inc + top_dec:
                row = [obj.get(k,'') for k in keys]
                row += [obj.get('support_rate', obj.get('metrics',{}).get('support_rate','') if isinstance(obj.get('metrics'), dict) else ''),
                        obj.get('corrected_support_rate', obj.get('metrics',{}).get('corrected_support_rate','') if isinstance(obj.get('metrics'), dict) else ''),
                        score,
                        'success' if (score in [s for s,_ in top_inc]) else 'failure',
                        obj.get('reason_tag',''),
                        obj.get('reason_text',''),
                        obj.get('rouge1_fmeasure',''),
                        obj.get('rouge1_fmeasure_corrected','')]
                writer.writerow(row)

    if VERBOSE:
        print(f'scanned {scanned} records (approx). wrote {OUT}')
        print('success (top increases):')
        for d, _ in top_inc:
            print(f'  score={d:.4f}')
        print('failure (top decreases / lowest scores):')
        for d, _ in top_dec:
            print(f'  score={d:.4f}')
    else:
        print(f'wrote {OUT} (scanned ~{scanned})')


if __name__ == '__main__':
    main()
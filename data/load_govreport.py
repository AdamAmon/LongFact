"""GovReport 数据集加载与采样工具。

提供简单 CLI 用于下载并采样数据集（validation/test），并可将样本导出为 jsonl 便于后续实验。
"""
import json
from typing import List, Dict, Optional
from pathlib import Path

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from config import DEFAULT_DATA_DIR, DEFAULT_GOVREPORT_DATASET, ensure_local_dirs


def load_govreport(split: str = 'validation', sample_size: int = 500, cache_dir: Optional[str] = None, dataset_name: Optional[str] = None):
    if load_dataset is None:
        raise ImportError('datasets is required to load GovReport')
    ensure_local_dirs()
    ds = load_dataset(
        dataset_name or DEFAULT_GOVREPORT_DATASET,
        split=split,
        cache_dir=cache_dir or str(DEFAULT_DATA_DIR),
    )
    if sample_size is not None and sample_size > 0:
        ds = ds.select(range(min(sample_size, len(ds))))
    records = []
    for i, ex in enumerate(ds):
        # dataset fields include 'document' and 'summary' typically
        document = ex.get('document') or ex.get('article') or ex.get('text') or ''
        summary = ex.get('summary') or ex.get('highlights') or ex.get('abstract') or ''
        records.append({
            'id': ex.get('id', i),
            'document': document,
            'summary': summary,
            'split': split,
            'dataset_name': dataset_name or DEFAULT_GOVREPORT_DATASET,
            'document_length': len(document),
            'summary_length': len(summary),
        })
    return records


def save_jsonl(records: List[Dict], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='validation')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--out', type=str, default='govreport_sample.jsonl')
    parser.add_argument('--cache_dir', type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_GOVREPORT_DATASET)
    args = parser.parse_args()
    recs = load_govreport(split=args.split, sample_size=args.sample_size, cache_dir=args.cache_dir, dataset_name=args.dataset_name)
    save_jsonl(recs, args.out)
    print(f'Saved {len(recs)} records to {args.out}')


if __name__ == '__main__':
    main()

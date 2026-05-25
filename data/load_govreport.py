"""GovReport 数据集加载与采样工具。

提供简单 CLI 用于下载并采样数据集（validation/test），并可将样本导出为 jsonl 便于后续实验。
"""
import json
from typing import List, Dict, Optional
from pathlib import Path

try:
    from datasets import load_dataset, DownloadConfig
except Exception as e:
    load_dataset = None
    import traceback
    print('load_govreport: datasets import failed:', e)
    traceback.print_exc()

from config import DEFAULT_DATA_DIR, DEFAULT_GOVREPORT_DATASET, ensure_local_dirs, FALLBACK_TO_SUMMARY


def load_govreport(split: str = 'validation', sample_size: int = 500, cache_dir: Optional[str] = None, dataset_name: Optional[str] = None):
    if load_dataset is None:
        raise ImportError('datasets is required to load GovReport')
    ensure_local_dirs()
    download_cfg = None
    try:
        download_cfg = DownloadConfig(local_files_only=True)
    except Exception as e:
        download_cfg = None
        import traceback
        print('load_govreport: DownloadConfig creation failed:', e)
        traceback.print_exc()

    # Prefer using a DownloadConfig to enforce local-only downloads; fall back
    # to the local_files_only flag if DownloadConfig is unavailable.
    try:
        if download_cfg is not None:
            ds = load_dataset(
                dataset_name or DEFAULT_GOVREPORT_DATASET,
                split=split,
                cache_dir=cache_dir or str(DEFAULT_DATA_DIR),
                download_config=download_cfg,
            )
        else:
            ds = load_dataset(
                dataset_name or DEFAULT_GOVREPORT_DATASET,
                split=split,
                cache_dir=cache_dir or str(DEFAULT_DATA_DIR),
                local_files_only=True,
            )
    except ValueError:
        # Some cached dataset packages use a specific config name (e.g. 'document').
        # Retry with a common config name to be more resilient in offline mode.
        if download_cfg is not None:
            ds = load_dataset(
                dataset_name or DEFAULT_GOVREPORT_DATASET,
                name='document',
                split=split,
                cache_dir=cache_dir or str(DEFAULT_DATA_DIR),
                download_config=download_cfg,
            )
        else:
            ds = load_dataset(
                dataset_name or DEFAULT_GOVREPORT_DATASET,
                name='document',
                split=split,
                cache_dir=cache_dir or str(DEFAULT_DATA_DIR),
                local_files_only=True,
            )
    if sample_size is not None and sample_size > 0:
        ds = ds.select(range(min(sample_size, len(ds))))
    records = []
    for i, ex in enumerate(ds):
        # summary fields (dataset can use different names)
        summary = ex.get('summary') or ex.get('highlights') or ex.get('abstract') or ''

        # document/report fields: try multiple common names and handle nested/list types
        document = ''
        doc_field_candidates = ['document', 'report', 'article', 'text', 'content', 'body', 'doc']
        for f in doc_field_candidates:
            if f in ex and ex.get(f):
                val = ex.get(f)
                # list of strings -> join
                if isinstance(val, (list, tuple)):
                    document = ' '.join([str(x).strip() for x in val if x])
                # nested dict -> try common inner keys
                elif isinstance(val, dict):
                    for sub in ('text', 'content', 'body', 'report', 'article'):
                        if sub in val and val.get(sub):
                            document = val.get(sub)
                            break
                    if not document:
                        # fallback to string representation
                        document = json.dumps(val, ensure_ascii=False)
                else:
                    document = str(val)
                break

        # If document is empty and fallback is enabled, use summary as fallback for debugging.
        skipped = False
        skip_reason = None
        if not document or not str(document).strip():
            if FALLBACK_TO_SUMMARY and summary:
                document = summary
                skip_reason = 'used_summary_as_fallback'
            else:
                skipped = True
                skip_reason = 'missing_document'

        rec = {
            'id': ex.get('id', i),
            'document': document,
            'summary': summary,
            'split': split,
            'dataset_name': dataset_name or DEFAULT_GOVREPORT_DATASET,
            'document_length': len(document),
            'summary_length': len(summary),
            'skipped': skipped,
            'skip_reason': skip_reason,
        }
        records.append(rec)
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

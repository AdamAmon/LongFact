"""预构建 embedding + FAISS 索引的脚本，用于加速后续实验。

会对指定数据集切片（例如 validation 的前 N 个样本）逐条处理：
- 将文档分块（使用 summarize.chunk_text）
- 使用 Retriever.build_index 构建并写出缓存（EMBEDDING_CACHE_DIR 下的 .npz/.index）

用法示例：
    python scripts/prebuild_indices.py --split validation --start 0 --n 100
"""
import argparse
from data.load_govreport import load_govreport
from retrieval.retriever import Retriever
from summarize.run_summarize import chunk_text
from config import DEFAULT_RETRIEVER_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--model_name', type=str, default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--use_bm25', action='store_true')
    args = parser.parse_args()

    recs = load_govreport(split=args.split, sample_size=args.n, start_index=args.start)
    retr = Retriever(model_name=args.model_name, use_bm25=args.use_bm25, device=args.device)

    try:
        from tqdm import tqdm
        iterator = tqdm(recs, desc='prebuild_indices', unit='sample')
    except Exception:
        iterator = recs

    for rec in iterator:
        doc = rec.get('document', '')
        if not doc or not str(doc).strip():
            continue
        passages = chunk_text(doc)
        try:
            retr.build_index(passages)
        except Exception as e:
            print('prebuild_indices: failed for id', rec.get('id'), e)

    print('Prebuild completed')


if __name__ == '__main__':
    main()

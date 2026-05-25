"""简易管线入口：分块 -> 局部摘要 -> 融合 -> 可选 NLI 验证与保存输出。

当前版本支持使用 HF 模型做局部摘要（通过 `summarize/model_summarizer.py`），同时保持
原有回退实现以便无 GPU 环境下开发调试。输出可选写入 jsonl 以供后续评估。
"""
import argparse
from typing import List
import textwrap
import json

from config import DEFAULT_SUMMARIZER_MODEL
from summarize.model_summarizer import get_summarizer


def chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    # 更稳健的句子切分：支持中文/英文标点（。.!?）并保留句末标点。
    import re

    raw_sents = [s.strip() for s in re.split(r'(?<=[。.!?])\s+', text) if s.strip()]
    # Ensure sentences end with a punctuation mark
    sents = []
    for s in raw_sents:
        if not re.search(r'[。.!?]$', s):
            s = s + '。'
        sents.append(s)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        l = len(s.split())
        if cur_len + l > max_tokens and cur:
            chunks.append('。'.join(cur) + '。')
            cur = [s]
            cur_len = l
        else:
            cur.append(s)
            cur_len += l
    if cur:
        chunks.append('。'.join(cur) + '。')
    return chunks


def fuse_summaries(local_summaries: List[str]) -> str:
    # 简单融合：按顺序合并局部摘要
    return ' '.join(local_summaries)


def run_pipeline(text: str, use_model: bool = False, model_name: str = None, device: int = -1, load_in_8bit: bool = False):
    chunks = chunk_text(text)
    result = {'chunks': chunks, 'local_summaries': [], 'fused': '', 'error': None}
    try:
        summarizer = get_summarizer(model_name=model_name if use_model else None, device=device, load_in_8bit=load_in_8bit)
        # debug: log chunk counts
        print(f'[run_pipeline] num_chunks={len(chunks)}')
        local_summaries = summarizer.summarize_chunks(chunks)
        print(f'[run_pipeline] local_summaries_count={len(local_summaries)}')
        result['local_summaries'] = local_summaries
        fused = fuse_summaries(local_summaries)
        result['fused'] = fused
        if not fused or not fused.strip():
            result['error'] = 'generation_empty'
            print('[run_pipeline] Warning: fused summary is empty')
    except Exception as e:
        # capture exception for downstream inspection
        import traceback

        tb = traceback.format_exc()
        result['error'] = str(e)
        print('[run_pipeline] Exception during summarization:')
        print(tb)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='输入文本或文件路径')
    parser.add_argument('--use_model', action='store_true', help='是否使用 HF 模型生成局部摘要')
    parser.add_argument('--model_name', type=str, default=DEFAULT_SUMMARIZER_MODEL, help='HF 模型名')
    parser.add_argument('--device', type=int, default=-1, help='模型设备: -1=CPU, >=0 GPU id')
    parser.add_argument('--chunk_size', type=int, default=200, help='分块时的最大长度近似值')
    parser.add_argument('--load_in_8bit', action='store_true', help='尝试使用 bitsandbytes 的 8-bit 加载（若可用）')
    parser.add_argument('--out', type=str, default=None, help='可选：输出 jsonl 路径，保存生成结果')
    args = parser.parse_args()
    inp = args.input
    try:
        # 如果是文件路径则读取
        with open(inp, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        text = inp

    chunks = chunk_text(text, max_tokens=args.chunk_size)
    summarizer = get_summarizer(model_name=args.model_name if args.use_model else None, device=args.device, load_in_8bit=args.load_in_8bit)
    local_summaries = summarizer.summarize_chunks(chunks)
    fused = fuse_summaries(local_summaries)
    out = {
        'chunks': chunks,
        'local_summaries': local_summaries,
        'fused': fused,
        'model_name': args.model_name if args.use_model else None,
        'chunk_size': args.chunk_size,
    }

    print('\n=== Fused Summary ===\n')
    print(textwrap.fill(out['fused'], width=80))

    if args.out:
        # 保存为一条 jsonl，包含 chunks/local_summaries/fused
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'Wrote result to {args.out}')


if __name__ == '__main__':
    main()

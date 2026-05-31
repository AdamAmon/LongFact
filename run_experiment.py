"""实验运行器：从数据采样 -> 摘要 -> 检索 -> NLI -> 纠错 -> 评估 的端到端最小实现。

用于快速验证管线是否可跑通（小样本）。"""
import argparse
import json
import os
from config import (
    DEFAULT_CORRECTOR_MODEL,
    DEFAULT_DATA_DIR,
    DEFAULT_NLI_MODEL,
    DEFAULT_SUMMARIZER_MODEL,
    DEFAULT_RETRIEVER_MODEL,
    PREFERRED_PRECISION,
    DEFAULT_TORCH_COMPILE,
)
from data.load_govreport import load_govreport
from config import FALLBACK_TO_SUMMARY
from summarize.run_summarize import run_pipeline
from retrieval.retriever import Retriever
from nli.nli_check import NLIChecker
from correction.corrector import Corrector
from eval.evaluate import compute_rouge, compute_support_rate, sentence_split


def _summary_length_stats(text: str) -> dict:
    sentences = sentence_split(text) if text and text.strip() else []
    return {
        'sentence_count': len(sentences),
        'char_count': len(text or ''),
        'token_count': len((text or '').split()),
    }


def run_sample(
    sample_count: int = 10,
    use_model: bool = False,
    model_name: str = None,
    device: int = -1,
    dataset_cache_dir: str = None,
    load_in_8bit: bool = False,
    summary_max_new_tokens: int = 256,
    summary_batch_size: int = 1,
    precision: str = None,
    torch_compile: bool = None,
    start_offset: int = 0,
):
    records = load_govreport(split='validation', sample_size=sample_count, cache_dir=dataset_cache_dir or str(DEFAULT_DATA_DIR), start_index=start_offset)
    results = []

    effective_precision = precision if precision is not None else PREFERRED_PRECISION
    effective_compile = DEFAULT_TORCH_COMPILE if torch_compile is None else bool(torch_compile)
    effective_model_name = model_name or DEFAULT_SUMMARIZER_MODEL

    # instantiate shared components once to avoid repeated loads
    nli = NLIChecker(
        model_name=DEFAULT_NLI_MODEL,
        device=device,
        load_in_8bit=load_in_8bit,
        precision=effective_precision,
        torch_compile=effective_compile,
    )
    corr = Corrector(
        model_name=(effective_model_name if use_model else DEFAULT_CORRECTOR_MODEL),
        device=device,
        load_in_8bit=load_in_8bit,
        precision=effective_precision,
        torch_compile=effective_compile,
    )
    retr = Retriever(model_name=DEFAULT_RETRIEVER_MODEL, device=device)

    # try to use tqdm for a progress bar if available
    try:
        from tqdm import tqdm
        # put outer experiment progress on position 1 so model weight loading bars (position 0)
        # remain visible separately and do not overwrite our sample-level progress
        iterator = tqdm(records, desc=f'processing offset={start_offset}', unit='sample', position=1, leave=True)
    except Exception:
        iterator = records

    for rec in iterator:
        # skip samples that were flagged as missing document (unless fallback enabled)
        sample_error = None
        sample_last_error = None
        if rec.get('skipped') and not FALLBACK_TO_SUMMARY:
            results.append({
                'id': rec['id'],
                'reference': rec.get('summary', ''),
                'prediction': '',
                'corrected': '',
                'support_rate': 0.0,
                'rouge': {},
                'rouge_corrected': {},
                'details': [],
                'error': 'skipped_missing_document',
                'skip_reason': rec.get('skip_reason'),
            })
            continue

        try:
            doc = rec['document']
            ref = rec.get('summary', '') or ''
            out = run_pipeline(
                doc,
                use_model=use_model,
                model_name=effective_model_name,
                device=device,
                load_in_8bit=load_in_8bit,
                summary_max_new_tokens=summary_max_new_tokens,
                summary_batch_size=summary_batch_size,
                precision=effective_precision,
                torch_compile=effective_compile,
            )
            pred = out.get('fused', '')
            summ_error = out.get('error')

            # build retriever on document passages; simple chunking for evidence
            passages = out.get('chunks', []) or []
            try:
                retr.build_index(passages)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print('run_experiment: retriever.build_index failed for sample', rec.get('id'), e)
                print(tb)

            # instantiate NLI checker once per sample; respect CLI device and 8-bit flag
            support_rate, details = compute_support_rate(pred, doc, retr, nli, top_k=3)
            # perform corrections for sentences not supported
            corrected_sents = []
            # Batch unsupported sentences to reduce generation overhead.
            unsupported_idx = [i for i, d in enumerate(details) if not d.get('supported')]
            if unsupported_idx and hasattr(corr, 'correct_batch'):
                evidences_list = [details[i].get('evidences', []) for i in unsupported_idx]
                sentences_list = [details[i].get('sentence', '') for i in unsupported_idx]
                try:
                    corrected_batch = corr.correct_batch(evidences_list, sentences_list, batch_size=max(1, summary_batch_size))
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print('run_experiment: corrector.correct_batch failed for sample', rec.get('id'), e)
                    print(tb)
                    corrected_batch = sentences_list
                corrected_map = {idx: corrected_batch[pos] if pos < len(corrected_batch) else details[idx].get('sentence', '') for pos, idx in enumerate(unsupported_idx)}
            else:
                corrected_map = {}

            for i, d in enumerate(details):
                if d.get('supported'):
                    corrected_sents.append(d.get('sentence', ''))
                elif i in corrected_map:
                    corrected_sents.append(corrected_map[i])
                else:
                    try:
                        corrected = corr.correct(d.get('evidences', []), d.get('sentence', ''))
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        print('run_experiment: corrector.correct failed for sample', rec.get('id'), e)
                        print(tb)
                        corrected = d.get('sentence', '')
                        d['error'] = str(e)
                        d['last_error'] = tb
                    corrected_sents.append(corrected)
            corrected_pred = ' '.join(corrected_sents)

            corrected_support_rate, corrected_details = compute_support_rate(corrected_pred, doc, retr, nli, top_k=3)

            try:
                rouge_scores = compute_rouge(ref, pred) if ref else {}
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print('run_experiment: compute_rouge failed for prediction', rec.get('id'), e)
                print(tb)
                rouge_scores = {}

            try:
                rouge_corrected = compute_rouge(ref, corrected_pred) if ref else {}
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print('run_experiment: compute_rouge failed for corrected prediction', rec.get('id'), e)
                print(tb)
                rouge_corrected = {}

            prediction_length = _summary_length_stats(pred)
            corrected_length = _summary_length_stats(corrected_pred)
            rouge1 = rouge_scores.get('rouge1_fmeasure', 0.0)
            rouge1_corrected = rouge_corrected.get('rouge1_fmeasure', 0.0)

            results.append({
                'id': rec['id'],
                'reference': ref,
                'prediction': pred,
                'fused_summary': pred,
                'corrected': corrected_pred,
                'support_rate': support_rate,
                'corrected_support_rate': corrected_support_rate,
                'rouge': rouge_scores,
                'rouge_corrected': rouge_corrected,
                'details': details,
                'corrected_details': corrected_details,
                'sentences': details,
                'error': sample_error or summ_error,
                'last_error': sample_last_error or getattr(corr, 'last_error', None) or getattr(nli, 'last_error', None) or summ_error,
                'prediction_length': prediction_length,
                'corrected_length': corrected_length,
                'support_rate_delta': corrected_support_rate - support_rate,
                'rouge1_fmeasure_delta': rouge1_corrected - rouge1,
                # include summarization debug snapshot to help diagnose empty outputs
                'summarization_debug': {
                    'chunks': out.get('chunks'),
                    'local_summaries': out.get('local_summaries'),
                    'fused': out.get('fused'),
                    'error': out.get('error'),
                },
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('run_experiment: processing sample failed for id', rec.get('id'), e)
            print(tb)
            # append a result with error info so analysis sees the failure per-sample
            results.append({
                'id': rec.get('id'),
                'reference': rec.get('summary', ''),
                'prediction': '',
                'fused_summary': '',
                'corrected': '',
                'support_rate': 0.0,
                'rouge': {},
                'rouge_corrected': {},
                'details': [],
                'sentences': [],
                'error': str(e),
                'last_error': tb,
                'summarization_debug': rec.get('document') if isinstance(rec.get('document'), dict) else {'chunks': None},
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--model_name', type=str, default=DEFAULT_SUMMARIZER_MODEL)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--load_in_8bit', action='store_true', help='尝试使用 bitsandbytes 的 8-bit 加载（若可用）')
    parser.add_argument('--precision', type=str, default=None, choices=['auto', 'fp32', 'fp16', '8bit'], help='精度偏好（默认使用配置项）')
    parser.add_argument('--torch_compile', action='store_true', help='尝试对支持的模型启用 torch.compile')
    parser.add_argument('--summary_max_new_tokens', type=int, default=256, help='每个 chunk 摘要生成的最大 token 数')
    parser.add_argument('--summary_batch_size', type=int, default=1, help='摘要阶段 pipeline 批大小（GPU 推荐 > 1）')
    parser.add_argument('--dataset_cache_dir', type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--start', type=int, default=0, help='开始偏移（用于分批处理）')
    parser.add_argument('--step', type=int, default=0, help='分批大小；>0 时按 step 分批（例如 50）')
    parser.add_argument('--out', type=str, default='experiment_results.jsonl')
    args = parser.parse_args()
    total_written = 0
    # If step > 0, run in batches from start..start+n in steps of step
    if args.step and args.step > 0:
        # if starting a fresh multi-batch run from 0, remove existing out file to avoid duplicate appends
        if args.start == 0 and os.path.exists(args.out):
            try:
                os.remove(args.out)
            except Exception:
                pass

        end_index = args.start + args.n
        for batch_start in range(args.start, end_index, args.step):
            batch_count = min(args.step, end_index - batch_start)
            print(f'Running batch start={batch_start} count={batch_count}')
            res = run_sample(
                sample_count=batch_count,
                use_model=args.use_model,
                model_name=args.model_name,
                device=args.device,
                dataset_cache_dir=args.dataset_cache_dir,
                load_in_8bit=args.load_in_8bit,
                summary_max_new_tokens=args.summary_max_new_tokens,
                summary_batch_size=args.summary_batch_size,
                precision=args.precision,
                torch_compile=args.torch_compile,
                start_offset=batch_start,
            )
            mode = 'a' if os.path.exists(args.out) else 'w'
            with open(args.out, mode, encoding='utf-8') as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            total_written += len(res)
        print(f'Wrote {total_written} results to {args.out}')
    else:
        res = run_sample(
            sample_count=args.n,
            use_model=args.use_model,
            model_name=args.model_name,
            device=args.device,
            dataset_cache_dir=args.dataset_cache_dir,
            load_in_8bit=args.load_in_8bit,
            summary_max_new_tokens=args.summary_max_new_tokens,
            summary_batch_size=args.summary_batch_size,
            precision=args.precision,
            torch_compile=args.torch_compile,
            start_offset=args.start,
        )
        with open(args.out, 'w', encoding='utf-8') as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        print(f'Wrote {len(res)} results to {args.out}')


if __name__ == '__main__':
    main()

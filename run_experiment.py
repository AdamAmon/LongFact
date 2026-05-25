"""实验运行器：从数据采样 -> 摘要 -> 检索 -> NLI -> 纠错 -> 评估 的端到端最小实现。

用于快速验证管线是否可跑通（小样本）。"""
import argparse
import json
from config import DEFAULT_CORRECTOR_MODEL, DEFAULT_DATA_DIR, DEFAULT_NLI_MODEL, DEFAULT_SUMMARIZER_MODEL
from data.load_govreport import load_govreport
from config import FALLBACK_TO_SUMMARY
from summarize.run_summarize import run_pipeline
from retrieval.retriever import Retriever
from nli.nli_check import NLIChecker
from correction.corrector import Corrector
from eval.evaluate import compute_rouge, compute_support_rate


def run_sample(sample_count: int = 10, use_model: bool = False, model_name: str = None, device: int = -1, dataset_cache_dir: str = None, load_in_8bit: bool = False):
    records = load_govreport(split='validation', sample_size=sample_count, cache_dir=dataset_cache_dir or str(DEFAULT_DATA_DIR))
    results = []

    # instantiate shared components once to avoid repeated loads
    nli = NLIChecker(model_name=DEFAULT_NLI_MODEL, device=device, load_in_8bit=load_in_8bit)
    corr = Corrector(model_name=model_name if use_model else DEFAULT_CORRECTOR_MODEL, device=device, load_in_8bit=load_in_8bit)

    for rec in records:
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
            out = run_pipeline(doc, use_model=use_model, model_name=model_name, device=device, load_in_8bit=load_in_8bit)
            pred = out.get('fused', '')
            summ_error = out.get('error')

            # build retriever on document passages; simple chunking for evidence
            passages = out.get('chunks', []) or []
            retr = Retriever()
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
            for d in details:
                if not d.get('supported'):
                    try:
                        corrected = corr.correct(d.get('evidences', []), d.get('sentence', ''))
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        print('run_experiment: corrector.correct failed for sample', rec.get('id'), e)
                        print(tb)
                        corrected = d.get('sentence', '')
                        # attach error info to detail for auditing
                        d['error'] = str(e)
                        d['last_error'] = tb
                    corrected_sents.append(corrected)
                else:
                    corrected_sents.append(d.get('sentence', ''))
            corrected_pred = ' '.join(corrected_sents)

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

            results.append({
                'id': rec['id'],
                'reference': ref,
                'prediction': pred,
                'fused_summary': pred,
                'corrected': corrected_pred,
                'support_rate': support_rate,
                'rouge': rouge_scores,
                'rouge_corrected': rouge_corrected,
                'details': details,
                'sentences': details,
                'error': sample_error or summ_error,
                'last_error': sample_last_error or getattr(corr, 'last_error', None) or getattr(nli, 'last_error', None) or summ_error,
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
    parser.add_argument('--dataset_cache_dir', type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--out', type=str, default='experiment_results.jsonl')
    args = parser.parse_args()

    res = run_sample(sample_count=args.n, use_model=args.use_model, model_name=args.model_name, device=args.device, dataset_cache_dir=args.dataset_cache_dir, load_in_8bit=args.load_in_8bit)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'Wrote {len(res)} results to {args.out}')


if __name__ == '__main__':
    main()

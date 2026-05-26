"""Local benchmark harness: run small n experiments under different precision settings and report timing/memory.

Usage: python scripts/benchmark_local.py --n 3
"""
import time
import argparse
import json
import torch
from run_experiment import run_sample
from config import DEFAULT_SUMMARIZER_MODEL

DEFAULTS = {
    'device': 0,
}

CONFIGS = {
    # Baseline uses the same pipeline but full precision on GPU for fair runtime reference.
    'baseline': {'use_model': True, 'load_in_8bit': False, 'device': 0, 'summary_batch_size': 2, 'summary_max_new_tokens': 256, 'precision': 'fp32', 'torch_compile': False},
    'gpu_fp16': {'use_model': True, 'load_in_8bit': False, 'device': 0, 'summary_batch_size': 1, 'summary_max_new_tokens': 256, 'precision': 'fp16', 'torch_compile': True},
    'gpu_8bit': {'use_model': True, 'load_in_8bit': True, 'device': 0, 'summary_batch_size': 1, 'summary_max_new_tokens': 256, 'precision': '8bit', 'torch_compile': False},
}


def run_config(name, cfg, n=1):
    print(f'\n--- Running config: {name} ---')
    t0 = time.time()
    # reset GPU mem stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = run_sample(
        sample_count=n,
        use_model=cfg.get('use_model', False),
        model_name=DEFAULT_SUMMARIZER_MODEL if cfg.get('use_model', False) else None,
        device=cfg.get('device', -1),
        load_in_8bit=cfg.get('load_in_8bit', False),
        summary_max_new_tokens=cfg.get('summary_max_new_tokens', 256),
        summary_batch_size=cfg.get('summary_batch_size', 1),
        precision=cfg.get('precision', None),
        torch_compile=cfg.get('torch_compile', None),
    )
    t1 = time.time()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    print(f'Config {name} done. time={t1-t0:.2f}s peak_gpu_mem={peak:.1f}MB')
    return {'name': name, 'time_s': t1 - t0, 'peak_gpu_mb': peak, 'results_len': len(res)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--which', nargs='*', default=None, help='Which configs to run (defaults to all)')
    args = parser.parse_args()
    to_run = CONFIGS.keys() if not args.which else args.which
    summary = []
    for k in to_run:
        if k not in CONFIGS:
            print('Unknown config', k)
            continue
        r = run_config(k, CONFIGS[k], n=args.n)
        summary.append(r)
    print('\n=== Summary ===')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

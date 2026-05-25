"""HuggingFace 模型封装：按 chunk 批量生成局部摘要的简单接口。

实现中尽量使用 pipeline 以保持兼容性；在显存受限时可使用小模型或 CPU。
"""
from typing import List, Optional, Dict
import re

try:
    from transformers import pipeline, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
except Exception as e:
    pipeline = None
    GenerationConfig = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    import traceback
    print('model_summarizer: transformers import failed:', e)
    traceback.print_exc()

from config import DEFAULT_SUMMARIZER_MODEL

# Module-level cache so we don't repeatedly reload large models during experiments
_SUMMARIZER_CACHE: Dict[str, object] = {}


class HFLocalSummarizer:
    def __init__(self, model_name: str = DEFAULT_SUMMARIZER_MODEL, device: int = -1, max_length: int = 256, load_in_8bit: bool = False, batch_size: int = 1):
        if pipeline is None:
            raise ImportError('transformers is required for HFLocalSummarizer')
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.last_error = None
        # generation config kept for compatibility but not passed directly to pipeline
        gen_cfg = None
        if GenerationConfig is not None:
            gen_cfg = GenerationConfig(max_new_tokens=self.max_length, do_sample=False)

        # Prefer loading a model object when 8-bit is requested (uses bitsandbytes)
        if load_in_8bit and AutoModelForCausalLM is not None and AutoTokenizer is not None and BitsAndBytesConfig is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                # device_map='auto' lets accelerate / transformers place params on available devices
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map='auto',
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )
                # Avoid passing generation_config into pipeline to prevent duplicate-kw issues
                self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, truncation=True, trust_remote_code=True)
                return
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f'model_summarizer: 8bit load failed for {model_name}:', e)
                print(tb)
                self.last_error = tb
                # fall back to pipeline(path) try below
                pass

        # Fallback: let pipeline load the model (may be CPU/GPU depending on device arg)
        # Do not pass generation_config to pipeline directly (some transformers versions raise errors)
        self.pipe = pipeline('text-generation', model=model_name, device=device, truncation=True, trust_remote_code=True)

    def build_prompt(self, chunk: str, prompt: Optional[str] = None) -> str:
        instruction = prompt or (
            'Summarize the following long document chunk in 2-3 concise factual sentences. '
            'Preserve names, numbers, and key outcomes. Avoid adding unsupported details.'
        )
        return f'{instruction}\n\nDocument chunk:\n{chunk}\n\nSummary:'

    def _truncate_to_n_sentences(self, text: str, n: int = 3) -> str:
        if not text:
            return ''
        # Drop common instruction leak fragments.
        for marker in ('\nTask:', '\nSummary:', '\nYou are an AI assistant'):
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        pieces = re.split(r'(?<=[。！？.!?])\s+', text.strip())
        pieces = [p.strip() for p in pieces if p.strip()]
        if not pieces:
            return text.strip()
        return ' '.join(pieces[:n]).strip()

    def _parse_output(self, out) -> str:
        try:
            if out is None:
                return ''
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    raw = (first.get('generated_text') or first.get('summary_text') or first.get('text') or '').strip()
                    return self._truncate_to_n_sentences(raw, n=3)
                return str(first).strip()
            if isinstance(out, dict):
                raw = (out.get('generated_text') or out.get('summary_text') or out.get('text') or '').strip()
                return self._truncate_to_n_sentences(raw, n=3)
            return str(out).strip()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('model_summarizer: parse output failed:', e)
            print(tb)
            self.last_error = tb
            return ''

    def summarize_chunk(self, chunk: str, prompt: Optional[str] = None) -> str:
        inp = self.build_prompt(chunk, prompt=prompt)
        try:
            # Pass max_new_tokens at call time to control length without passing generation_config
            out = self.pipe(
                inp,
                truncation=True,
                return_full_text=False,
                max_new_tokens=self.max_length,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('model_summarizer: summarize_chunk failed:', e)
            print(tb)
            self.last_error = tb
            return ''
        return self._parse_output(out)

    def summarize_chunks(self, chunks: List[str], prompt: Optional[str] = None) -> List[str]:
        if not chunks:
            return []

        prompts = [self.build_prompt(c, prompt=prompt) for c in chunks]
        results = []

        # Use batched pipeline calls to reduce sequential GPU overhead.
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            try:
                outs = self.pipe(
                    batch,
                    truncation=True,
                    return_full_text=False,
                    max_new_tokens=self.max_length,
                    batch_size=self.batch_size,
                )
                if isinstance(outs, list):
                    for item in outs:
                        results.append(self._parse_output([item]))
                else:
                    results.append(self._parse_output(outs))
            except Exception:
                # Fallback to per-chunk path if batched generation fails.
                for c in chunks[i:i + self.batch_size]:
                    results.append(self.summarize_chunk(c, prompt=prompt))

        return results


class FallbackSummarizer:
    """简单回退实现：取 chunk 的首句作为局部摘要（与旧版兼容）。"""

    def summarize_chunks(self, chunks: List[str], prompt: Optional[str] = None) -> List[str]:
        out = []
        for c in chunks:
            s = c.split('。')[0]
            if not s.endswith('。'):
                s = s + '。'
            out.append(s)
        return out


def get_summarizer(model_name: Optional[str] = None, device: int = -1, max_length: int = 256, load_in_8bit: bool = False, batch_size: int = 1):
    key = f'{model_name}::dev{device}::len{max_length}::8bit{load_in_8bit}::bs{batch_size}'
    if model_name is None:
        return FallbackSummarizer()

    if key in _SUMMARIZER_CACHE:
        return _SUMMARIZER_CACHE[key]

    try:
        summ = HFLocalSummarizer(model_name=model_name, device=device, max_length=max_length, load_in_8bit=load_in_8bit, batch_size=batch_size)
        _SUMMARIZER_CACHE[key] = summ
        return summ
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f'model_summarizer: failed to create HFLocalSummarizer for {model_name}:', e)
        print(tb)
        _SUMMARIZER_CACHE[key] = FallbackSummarizer()
        return _SUMMARIZER_CACHE[key]

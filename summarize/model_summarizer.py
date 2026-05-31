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

from config import DEFAULT_SUMMARIZER_MODEL, PREFERRED_PRECISION, DEFAULT_TORCH_COMPILE
from utils.hf_helpers import load_model_and_tokenizer
import torch as _torch

# Module-level cache so we don't repeatedly reload large models during experiments
_SUMMARIZER_CACHE: Dict[str, object] = {}


class HFLocalSummarizer:
    def __init__(self, model_name: str = DEFAULT_SUMMARIZER_MODEL, device: int = -1, max_length: int = 256, load_in_8bit: bool = False, batch_size: int = 1, precision: str = 'auto', torch_compile: bool = False):
        if pipeline is None:
            raise ImportError('transformers is required for HFLocalSummarizer')
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.last_error = None
        self.precision = precision or PREFERRED_PRECISION or 'auto'
        self.torch_compile = torch_compile or DEFAULT_TORCH_COMPILE
        # Try centralized helper pre-load to avoid duplicate generation_config defaults
        try:
            from utils.hf_helpers import load_model_and_tokenizer
        except Exception:
            load_model_and_tokenizer = None
        if load_model_and_tokenizer is not None:
            try:
                model_obj, tokenizer_obj = load_model_and_tokenizer(model_name, model_kind='causal', load_in_8bit=load_in_8bit, torch_dtype=(_torch.float16 if (self.precision in ('fp16', 'half') and device >= 0) else None))
                if model_obj is not None and tokenizer_obj is not None:
                    # Use explicit model + tokenizer to construct pipeline
                    self.pipe = pipeline('text-generation', model=model_obj, tokenizer=tokenizer_obj, truncation=True, trust_remote_code=True, device=device)
                    return
                # if only tokenizer available, let later branches reuse it
                if tokenizer_obj is not None:
                    tokenizer = tokenizer_obj
            except Exception:
                pass
        # generation config kept for compatibility but not passed directly to pipeline
        gen_cfg = None
        if GenerationConfig is not None:
            gen_cfg = GenerationConfig(max_new_tokens=self.max_length, do_sample=False)

        # Prefer loading a model object when 8-bit is requested (uses bitsandbytes)
        if load_in_8bit and AutoModelForCausalLM is not None and AutoTokenizer is not None and BitsAndBytesConfig is not None:
            try:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=False)
                except TypeError:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                try:
                    tokenizer.clean_up_tokenization_spaces = False
                except Exception:
                    pass
                # device_map='auto' lets accelerate / transformers place params on available devices
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map='auto',
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )
                # avoid generation_config conflicts: clear max_length on generation_config
                if hasattr(model, 'generation_config'):
                    try:
                        model.generation_config.max_length = None
                        # clear other sampling-related fields that may be present
                        for fld in ('temperature', 'top_p', 'top_k'):
                            try:
                                setattr(model.generation_config, fld, None)
                            except Exception:
                                pass
                    except Exception:
                        pass
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

        # Next try: if fp16 precision requested and causal model class available, attempt to load in fp16
        try:
            if (self.precision in ('fp16', 'half')) and AutoModelForCausalLM is not None and AutoTokenizer is not None and device >= 0:
                try:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=False)
                    except TypeError:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    try:
                        tokenizer.clean_up_tokenization_spaces = False
                    except Exception:
                        pass
                    # load model with float16 dtype where supported
                    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=_torch.float16, trust_remote_code=True)
                    if hasattr(model, 'generation_config'):
                        try:
                            model.generation_config.max_length = None
                            for fld in ('temperature', 'top_p', 'top_k'):
                                try:
                                    setattr(model.generation_config, fld, None)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    if self.torch_compile and hasattr(_torch, 'compile'):
                        try:
                            model = _torch.compile(model)
                        except Exception:
                            pass
                    self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, truncation=True, trust_remote_code=True)
                    return
                except Exception:
                    # fallthrough to standard pipeline
                    pass
        except Exception:
            pass

        # Fallback: explicitly pre-load model/tokenizer and pass to pipeline to avoid
        # pipeline-internal model creation which can leave generation_config defaults.
        try:
            model_obj, tokenizer = load_model_and_tokenizer(model_name, model_kind='causal', load_in_8bit=False, torch_dtype=None)
            if model_obj is not None and tokenizer is not None:
                self.pipe = pipeline('text-generation', model=model_obj, tokenizer=tokenizer, device=device, truncation=True, trust_remote_code=True)
            elif tokenizer is not None:
                self.pipe = pipeline('text-generation', model=model_name, tokenizer=tokenizer, device=device, truncation=True, trust_remote_code=True)
            else:
                self.pipe = pipeline('text-generation', model=model_name, device=device, truncation=True, trust_remote_code=True)
        except Exception:
            # fallback: let pipeline decide
            self.pipe = pipeline('text-generation', model=model_name, device=device, truncation=True, trust_remote_code=True)

    def build_prompt(self, chunk: str, prompt: Optional[str] = None) -> str:
        instruction = prompt or (
            'Summarize the following long document chunk in 2-3 concise factual sentences. '
            'Preserve names, numbers, and key outcomes. Avoid adding unsupported details.'
        )
        return f'{instruction}\n\nDocument chunk:\n{chunk}\n\nSummary:'

    def _sanitize_output_text(self, text: str) -> str:
        if not text:
            return ''
        # Drop common instruction leak fragments.
        for marker in ('\nTask:', '\nSummary:', '\nYou are an AI assistant'):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _parse_output(self, out) -> str:
        try:
            if out is None:
                return ''
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    raw = (first.get('generated_text') or first.get('summary_text') or first.get('text') or '').strip()
                    return self._sanitize_output_text(raw)
                return str(first).strip()
            if isinstance(out, dict):
                raw = (out.get('generated_text') or out.get('summary_text') or out.get('text') or '').strip()
                return self._sanitize_output_text(raw)
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
            # Use GenerationConfig to specify generation parameters (avoid passing both
            # a model-level generation_config object and runtime kwargs)
            gen_cfg = None
            if GenerationConfig is not None:
                gen_cfg = GenerationConfig(max_new_tokens=self.max_length, do_sample=False)
            if gen_cfg is not None:
                out = self.pipe(
                    inp,
                    truncation=True,
                    return_full_text=False,
                    generation_config=gen_cfg,
                )
            else:
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

    def summarize_chunks(self, chunks: List[str], prompt: Optional[str] = None, show_progress: bool = False) -> List[str]:
        if not chunks:
            return []

        prompts = [self.build_prompt(c, prompt=prompt) for c in chunks]
        results = []

        # Use batched pipeline calls to reduce sequential GPU overhead.
        # Optionally show a progress bar over batches of chunks
        rng = range(0, len(prompts), self.batch_size)
        bar = None
        if show_progress:
            try:
                from tqdm import tqdm
                # use a higher position for chunk-level progress to avoid clobbering
                # model weight loading bars and outer experiment bars
                bar = tqdm(rng, desc='summarize_chunks', unit='batch', position=2, leave=True)
            except Exception:
                bar = None

        iterator = bar if bar is not None else rng

        for i in iterator:
            # show an immediate update when starting a batch so the user sees activity
            if bar is not None:
                try:
                    batch_no = (i // max(1, self.batch_size)) + 1
                    total_batches = (len(prompts) + max(1, self.batch_size) - 1) // max(1, self.batch_size)
                    bar.set_description(f'summarize_chunks {batch_no}/{total_batches}')
                    bar.refresh()
                    from tqdm import tqdm as _tq
                    _tq.write(f"Starting batch {batch_no}/{total_batches}")
                except Exception:
                    pass
            batch = prompts[i:i + self.batch_size]
            try:
                gen_cfg = None
                if GenerationConfig is not None:
                    gen_cfg = GenerationConfig(max_new_tokens=self.max_length, do_sample=False)
                if gen_cfg is not None:
                    outs = self.pipe(
                        batch,
                        truncation=True,
                        return_full_text=False,
                        generation_config=gen_cfg,
                        batch_size=self.batch_size,
                    )
                else:
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


def get_summarizer(model_name: Optional[str] = None, device: int = -1, max_length: int = 256, load_in_8bit: bool = False, batch_size: int = 1, precision: str = 'auto', torch_compile: bool = False):
    key = f'{model_name}::dev{device}::len{max_length}::8bit{load_in_8bit}::bs{batch_size}::prec{precision}::compile{torch_compile}'
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

"""HuggingFace 模型封装：按 chunk 批量生成局部摘要的简单接口。

实现中尽量使用 pipeline 以保持兼容性；在显存受限时可使用小模型或 CPU。
"""
from typing import List, Optional, Dict

try:
    from transformers import pipeline, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
except Exception:
    pipeline = None
    GenerationConfig = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

from config import DEFAULT_SUMMARIZER_MODEL

# Module-level cache so we don't repeatedly reload large models during experiments
_SUMMARIZER_CACHE: Dict[str, object] = {}


class HFLocalSummarizer:
    def __init__(self, model_name: str = DEFAULT_SUMMARIZER_MODEL, device: int = -1, max_length: int = 256, load_in_8bit: bool = False):
        if pipeline is None:
            raise ImportError('transformers is required for HFLocalSummarizer')
        self.max_length = max_length
        gen_cfg = None
        if GenerationConfig is not None:
            gen_cfg = GenerationConfig(max_new_tokens=self.max_length, do_sample=False)

        # Prefer loading a model object when 8-bit is requested (uses bitsandbytes)
        if load_in_8bit and AutoModelForCausalLM is not None and AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                # device_map='auto' lets accelerate / transformers place params on available devices
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
                if gen_cfg is not None:
                    self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, truncation=True, trust_remote_code=True, generation_config=gen_cfg)
                else:
                    self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, truncation=True, trust_remote_code=True)
                return
            except Exception:
                # fall back to pipeline(path) try below
                pass

        # Fallback: let pipeline load the model (may be CPU/GPU depending on device arg)
        if gen_cfg is not None:
            self.pipe = pipeline('text-generation', model=model_name, device=device, truncation=True, trust_remote_code=True, generation_config=gen_cfg)
        else:
            self.pipe = pipeline('text-generation', model=model_name, device=device, truncation=True, trust_remote_code=True)

    def build_prompt(self, chunk: str, prompt: Optional[str] = None) -> str:
        instruction = prompt or (
            'Summarize the following long document chunk in 2-3 concise factual sentences. '
            'Preserve names, numbers, and key outcomes. Avoid adding unsupported details.'
        )
        return f'{instruction}\n\nDocument chunk:\n{chunk}\n\nSummary:'

    def _parse_output(self, out) -> str:
        try:
            if out is None:
                return ''
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    return (first.get('generated_text') or first.get('summary_text') or first.get('text') or '').strip()
                return str(first).strip()
            if isinstance(out, dict):
                return (out.get('generated_text') or out.get('summary_text') or out.get('text') or '').strip()
            return str(out).strip()
        except Exception:
            return ''

    def summarize_chunk(self, chunk: str, prompt: Optional[str] = None) -> str:
        inp = self.build_prompt(chunk, prompt=prompt)
        out = self.pipe(
            inp,
            truncation=True,
            return_full_text=False,
        )
        return self._parse_output(out)

    def summarize_chunks(self, chunks: List[str], prompt: Optional[str] = None) -> List[str]:
        results = []
        for c in chunks:
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


def get_summarizer(model_name: Optional[str] = None, device: int = -1, max_length: int = 256, load_in_8bit: bool = False):
    key = f'{model_name}::dev{device}::len{max_length}::8bit{load_in_8bit}'
    if model_name is None:
        return FallbackSummarizer()

    if key in _SUMMARIZER_CACHE:
        return _SUMMARIZER_CACHE[key]

    try:
        summ = HFLocalSummarizer(model_name=model_name, device=device, max_length=max_length, load_in_8bit=load_in_8bit)
        _SUMMARIZER_CACHE[key] = summ
        return summ
    except Exception:
        _SUMMARIZER_CACHE[key] = FallbackSummarizer()
        return _SUMMARIZER_CACHE[key]

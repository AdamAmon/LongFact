"""HuggingFace 模型封装：按 chunk 批量生成局部摘要的简单接口。

实现中尽量使用 pipeline 以保持兼容性；在显存受限时可使用小模型或 CPU。
"""
from typing import List, Optional

try:
    from transformers import pipeline
except Exception:
    pipeline = None

from config import DEFAULT_SUMMARIZER_MODEL


class HFLocalSummarizer:
    def __init__(self, model_name: str = DEFAULT_SUMMARIZER_MODEL, device: int = -1, max_length: int = 256):
        if pipeline is None:
            raise ImportError('transformers is required for HFLocalSummarizer')
        self.pipe = pipeline('text-generation', model=model_name, device=device, truncation=True)
        self.max_length = max_length

    def build_prompt(self, chunk: str, prompt: Optional[str] = None) -> str:
        instruction = prompt or (
            'Summarize the following long document chunk in 2-3 concise factual sentences. '
            'Preserve names, numbers, and key outcomes. Avoid adding unsupported details.'
        )
        return f'{instruction}\n\nDocument chunk:\n{chunk}\n\nSummary:'

    def summarize_chunk(self, chunk: str, prompt: Optional[str] = None) -> str:
        inp = self.build_prompt(chunk, prompt=prompt)
        out = self.pipe(
            inp,
            max_new_tokens=self.max_length,
            do_sample=False,
            truncation=True,
            return_full_text=False,
        )
        if isinstance(out, list) and len(out) > 0:
            return out[0].get('generated_text', out[0].get('summary_text', '')).strip()
        return ''

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


def get_summarizer(model_name: Optional[str] = None, device: int = -1, max_length: int = 256):
    if model_name is None:
        return FallbackSummarizer()
    try:
        return HFLocalSummarizer(model_name=model_name, device=device, max_length=max_length)
    except Exception:
        return FallbackSummarizer()

"""HuggingFace 模型封装：按 chunk 批量生成局部摘要的简单接口。

实现中尽量使用 pipeline 以保持兼容性；在显存受限时可使用小模型或 CPU。
"""
from typing import List, Optional

try:
    from transformers import pipeline
except Exception:
    pipeline = None


class HFLocalSummarizer:
    def __init__(self, model_name: str = 'google/flan-t5-large', device: int = -1, max_length: int = 256):
        if pipeline is None:
            raise ImportError('transformers is required for HFLocalSummarizer')
        # use text2text-generation pipeline for T5/Flan models
        self.pipe = pipeline('text2text-generation', model=model_name, device=device, truncation=True)
        self.max_length = max_length

    def summarize_chunk(self, chunk: str, prompt: Optional[str] = None) -> str:
        if prompt:
            inp = prompt + '\n\n' + chunk
        else:
            inp = chunk
        out = self.pipe(inp, max_length=self.max_length, truncation=True)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get('generated_text', out[0].get('summary_text', ''))
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

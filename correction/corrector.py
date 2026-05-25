"""自动纠错模块：对被判不支持的摘要句进行基于证据的局部改写。

实现：构造 prompt（包含证据段）并调用 HF text-generation pipeline 生成候选改写。
提供回退策略以保证在无模型时仍能作为占位。"""
import re
from typing import List, Optional

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForCausalLM = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from config import DEFAULT_CORRECTOR_MODEL

# Module-level cache to avoid re-loading corrector models across samples
_CORRECTOR_CACHE = {}


class Corrector:
    def __init__(self, model_name: Optional[str] = DEFAULT_CORRECTOR_MODEL, device: int = -1, max_length: int = 128, load_in_8bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pipe = None
        self.last_error: Optional[str] = None
        key = f"{model_name}::dev{device}::8bit{load_in_8bit}::len{max_length}"
        if key in _CORRECTOR_CACHE:
            cached = _CORRECTOR_CACHE[key]
            self.pipe = cached.get('pipe')
            return
        if model_name and pipeline is not None:
            # Try to load an 8-bit model object when requested.
            # Support both seq2seq and causal LMs so the same corrector can be
            # used with different default models.
            if load_in_8bit and AutoTokenizer is not None and BitsAndBytesConfig is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                    if AutoModelForSeq2SeqLM is not None:
                        try:
                            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_cfg)
                            self.pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
                        except Exception:
                            self.pipe = None
                    if self.pipe is None and AutoModelForCausalLM is not None:
                        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_cfg)
                        self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
                    if self.pipe is not None:
                        _CORRECTOR_CACHE[key] = {'pipe': self.pipe}
                        return
                except Exception:
                    # fall through to normal pipeline loader
                    self.pipe = None

            if self.pipe is None:
                try:
                    self.pipe = pipeline('text2text-generation', model=model_name, device=device)
                except Exception:
                    try:
                        self.pipe = pipeline('text-generation', model=model_name, device=device)
                    except Exception:
                        self.pipe = None
        # cache the loaded pipeline (or None) to avoid re-loading across samples
        _CORRECTOR_CACHE[key] = {'pipe': self.pipe}

    def construct_prompt(self, evidence: List[str], sentence: str) -> str:
        evid = '\n'.join([f'- {e}' for e in evidence[:5]])
        prompt = (
            'The following evidence is extracted from the source document:\n'
            f'{evid}\n\n'
            'Please rewrite the following sentence to be factually consistent with the evidence above.\n'
            f'Sentence: "{sentence}"\n'
            'Return a concise corrected sentence in the same language.'
        )
        return prompt

    def _normalize_generation(self, prompt: str, generated: str) -> str:
        text = generated.strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        for marker in (
            'Corrected sentence:',
            'Corrected Sentence:',
            'Revised sentence:',
            'Revised Sentence:',
            'Correction:',
            'Answer:',
        ):
            if marker in text:
                text = text.split(marker, 1)[1].strip()
                break

        text = text.split('\n', 1)[0].strip()
        first_sentence = re.split(r'(?<=[。！？.!?])\s+', text, maxsplit=1)[0].strip()
        return first_sentence or text

    def correct(self, evidence: List[str], sentence: str) -> str:
        prompt = self.construct_prompt(evidence, sentence)
        if self.pipe is None:
            # fallback: try to do minimal correction by returning original sentence
            self.last_error = 'corrector pipeline unavailable'
            return sentence
        try:
            out = self.pipe(
                prompt,
                truncation=True,
                max_new_tokens=self.max_length,
                return_full_text=False,
            )
            if isinstance(out, list) and len(out) > 0:
                generated = out[0].get('generated_text', out[0].get('summary_text', out[0].get('text', '')))
                if isinstance(generated, str):
                    generated = self._normalize_generation(prompt, generated)
                self.last_error = None
                return generated or sentence
            self.last_error = 'corrector returned empty output'
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            return sentence
        self.last_error = 'corrector returned empty output'
        return sentence


def simple_demo():
    c = Corrector()
    evidence = ["Alice bought 3 apples.", "She visited the downtown market."]
    s = "Alice bought two apples at the shop."
    print(c.correct(evidence, s))


if __name__ == '__main__':
    simple_demo()

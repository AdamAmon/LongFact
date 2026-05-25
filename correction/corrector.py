"""自动纠错模块：对被判不支持的摘要句进行基于证据的局部改写。

实现：构造 prompt（包含证据段）并调用 HF text2text 或 text-generation pipeline 生成候选改写。
提供回退策略以保证在无模型时仍能作为占位。"""
from typing import List, Optional

try:
    from transformers import pipeline, GenerationConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
except Exception:
    pipeline = None
    GenerationConfig = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForCausalLM = None

from config import DEFAULT_CORRECTOR_MODEL

# Module-level cache to avoid re-loading corrector models across samples
_CORRECTOR_CACHE = {}


class Corrector:
    def __init__(self, model_name: Optional[str] = DEFAULT_CORRECTOR_MODEL, device: int = -1, max_length: int = 128, load_in_8bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pipe = None
        key = f"{model_name}::dev{device}::8bit{load_in_8bit}::len{max_length}"
        if key in _CORRECTOR_CACHE:
            cached = _CORRECTOR_CACHE[key]
            self.pipe = cached.get('pipe')
            return
        if model_name and pipeline is not None:
            gen_cfg = None
            if GenerationConfig is not None:
                gen_cfg = GenerationConfig(max_new_tokens=self.max_length)

            # Try to load an 8-bit model object when requested
            if load_in_8bit and AutoTokenizer is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    # prefer seq2seq model for text2text; fall back to causal if unavailable
                    if AutoModelForSeq2SeqLM is not None:
                        try:
                            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
                        except Exception:
                            model = None
                    else:
                        model = None
                    if model is None and AutoModelForCausalLM is not None:
                        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)

                    if model is not None:
                        if gen_cfg is not None:
                            self.pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, generation_config=gen_cfg)
                        else:
                            self.pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
                except Exception:
                    # fall through to normal pipeline loader
                    self.pipe = None

            if self.pipe is None:
                try:
                    if gen_cfg is not None:
                        self.pipe = pipeline('text2text-generation', model=model_name, device=device, generation_config=gen_cfg)
                    else:
                        self.pipe = pipeline('text2text-generation', model=model_name, device=device)
                except Exception:
                    try:
                        if gen_cfg is not None:
                            self.pipe = pipeline('text-generation', model=model_name, device=device, generation_config=gen_cfg)
                        else:
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

    def correct(self, evidence: List[str], sentence: str) -> str:
        prompt = self.construct_prompt(evidence, sentence)
        if self.pipe is None:
            # fallback: try to do minimal correction by returning original sentence
            return sentence
        try:
            # generation parameters are supplied via GenerationConfig at pipeline init-time
            out = self.pipe(prompt)
            if isinstance(out, list) and len(out) > 0:
                # text2text returns 'generated_text' key in many cases
                return out[0].get('generated_text', out[0].get('summary_text', out[0].get('text', '')))
        except Exception:
            return sentence
        return sentence


def simple_demo():
    c = Corrector()
    evidence = ["Alice bought 3 apples.", "She visited the downtown market."]
    s = "Alice bought two apples at the shop."
    print(c.correct(evidence, s))


if __name__ == '__main__':
    simple_demo()

"""自动纠错模块：对被判不支持的摘要句进行基于证据的局部改写。

实现：构造 prompt（包含证据段）并调用 HF text-generation pipeline 生成候选改写。
提供回退策略以保证在无模型时仍能作为占位。"""
import re
from typing import List, Optional

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GenerationConfig
except Exception as e:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForCausalLM = None
    import traceback
    print('corrector: transformers import failed:', e)
    traceback_text = traceback.format_exc()
    traceback.print_exc()

try:
    from transformers import BitsAndBytesConfig
except Exception as e:
    BitsAndBytesConfig = None
    import traceback
    print('corrector: BitsAndBytesConfig import failed:', e)
    traceback_text = traceback.format_exc()
    traceback.print_exc()

from config import DEFAULT_CORRECTOR_MODEL
from config import PREFERRED_PRECISION, DEFAULT_TORCH_COMPILE
import torch as _torch

# Module-level cache to avoid re-loading corrector models across samples
_CORRECTOR_CACHE = {}


class Corrector:
    def __init__(self, model_name: Optional[str] = DEFAULT_CORRECTOR_MODEL, device: int = -1, max_length: int = 128, load_in_8bit: bool = False, precision: str = 'auto', torch_compile: bool = False, gpu_only: Optional[bool] = None):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pipe = None
        self.last_error: Optional[str] = None
        key = f"{model_name}::dev{device}::8bit{load_in_8bit}::len{max_length}::prec{precision}::compile{torch_compile}"
        self.precision = precision or PREFERRED_PRECISION or 'auto'
        self.torch_compile = torch_compile or DEFAULT_TORCH_COMPILE
        self.gpu_only = (device >= 0) if gpu_only is None else bool(gpu_only)
        if key in _CORRECTOR_CACHE:
            cached = _CORRECTOR_CACHE[key]
            self.pipe = cached.get('pipe')
            return
        if model_name and pipeline is not None:
            # Try centralized helper pre-load first to avoid pipeline-internal defaults
            try:
                from utils.hf_helpers import load_model_and_tokenizer
            except Exception:
                load_model_and_tokenizer = None
            if load_model_and_tokenizer is not None:
                try:
                    model_obj, tokenizer_obj = load_model_and_tokenizer(
                        model_name,
                        model_kind='causal',
                        load_in_8bit=load_in_8bit,
                        torch_dtype=(_torch.float16 if (self.precision in ('fp16', 'half') and device >= 0) else None),
                        device=device,
                        gpu_only=self.gpu_only,
                    )
                    if model_obj is not None and tokenizer_obj is not None:
                        try:
                            if self.torch_compile and hasattr(_torch, 'compile'):
                                try:
                                    model_obj = _torch.compile(model_obj)
                                except Exception:
                                    pass
                            self.pipe = pipeline('text-generation', model=model_obj, tokenizer=tokenizer_obj)
                            _CORRECTOR_CACHE[key] = {'pipe': self.pipe}
                            return
                        except Exception:
                            # fallthrough to existing loading logic
                            pass
                except Exception:
                    pass
            # Try to load an 8-bit model object when requested.
            # Support both seq2seq and causal LMs so the same corrector can be
            # used with different default models.
            if load_in_8bit and AutoTokenizer is not None and BitsAndBytesConfig is not None:
                try:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=False)
                    except TypeError:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                    # Qwen-style corrector models are causal LMs, so prefer the causal path first.
                    if AutoModelForCausalLM is not None:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(model_name, device_map={'': device} if self.gpu_only and device >= 0 else 'auto', quantization_config=quant_cfg)
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
                            self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            print(f'corrector: causal 8bit load failed for {model_name}:', e)
                            print(tb)
                            self.last_error = tb
                            self.pipe = None
                    if self.pipe is None and AutoModelForSeq2SeqLM is not None:
                        try:
                            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map={'': device} if self.gpu_only and device >= 0 else 'auto', quantization_config=quant_cfg)
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
                            self.pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            print(f'corrector: seq2seq 8bit load failed for {model_name}:', e)
                            print(tb)
                            self.last_error = tb
                            self.pipe = None
                    if self.pipe is not None:
                        _CORRECTOR_CACHE[key] = {'pipe': self.pipe}
                        return
                except Exception as e:
                    # fall through to normal pipeline loader but log
                    import traceback
                    tb = traceback.format_exc()
                    print(f'corrector: 8bit pipeline attempt failed for {model_name}:', e)
                    print(tb)
                    self.last_error = tb
                    self.pipe = None

            if self.pipe is None:
                # If user requested fp16 precision and a causal/seq2seq model class is available,
                # try to load the model in float16 for faster GPU inference.
                try:
                    if (self.precision in ('fp16', 'half')) and AutoTokenizer is not None and device >= 0:
                        try:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=False)
                            except TypeError:
                                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                            if AutoModelForCausalLM is not None:
                                model = AutoModelForCausalLM.from_pretrained(model_name, device_map={'': device} if self.gpu_only and device >= 0 else 'auto', torch_dtype=_torch.float16)
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
                                self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
                                _CORRECTOR_CACHE[key] = {'pipe': self.pipe}
                                return
                        except Exception:
                            # fall through to standard pipeline
                            pass
                except Exception:
                    pass

                try:
                    # prefer to load tokenizer explicitly so we can set clean_up_tokenization_spaces
                    if AutoTokenizer is not None:
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=False)
                        except TypeError:
                            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                        try:
                            tokenizer.clean_up_tokenization_spaces = False
                        except Exception:
                            pass
                        if self.gpu_only and device >= 0:
                            raise RuntimeError(f'GPU-only mode could not load model {model_name}')
                        self.pipe = pipeline('text-generation', model=model_name, tokenizer=tokenizer, device=device)
                    else:
                        if self.gpu_only and device >= 0:
                            raise RuntimeError(f'GPU-only mode could not load model {model_name}')
                        self.pipe = pipeline('text-generation', model=model_name, device=device)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f'corrector: text-generation pipeline load failed for {model_name}:', e)
                    print(tb)
                    self.last_error = tb
                    try:
                        if AutoTokenizer is not None:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=False)
                            except TypeError:
                                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                            try:
                                tokenizer.clean_up_tokenization_spaces = False
                            except Exception:
                                pass
                            if self.gpu_only and device >= 0:
                                raise RuntimeError(f'GPU-only mode could not load model {model_name}')
                            self.pipe = pipeline('text2text-generation', model=model_name, tokenizer=tokenizer, device=device)
                        else:
                            if self.gpu_only and device >= 0:
                                raise RuntimeError(f'GPU-only mode could not load model {model_name}')
                            self.pipe = pipeline('text2text-generation', model=model_name, device=device)
                    except Exception as e2:
                        tb2 = traceback.format_exc()
                        print(f'corrector: text2text pipeline load also failed for {model_name}:', e2)
                        print(tb2)
                        self.last_error = (self.last_error or '') + '\n' + tb2
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
            try:
                gen_cfg = None
                if 'GenerationConfig' in globals() and GenerationConfig is not None:
                    gen_cfg = GenerationConfig(max_new_tokens=self.max_length, do_sample=False)
                if gen_cfg is not None:
                    out = self.pipe(prompt, truncation=True, return_full_text=False, generation_config=gen_cfg)
                else:
                    out = self.pipe(prompt, truncation=True, max_new_tokens=self.max_length, return_full_text=False)
            except TypeError:
                # Some test doubles or minimal pipeline fakes accept only the prompt
                out = self.pipe(prompt)
            if isinstance(out, list) and len(out) > 0:
                generated = out[0].get('generated_text', out[0].get('summary_text', out[0].get('text', '')))
                if isinstance(generated, str):
                    generated = self._normalize_generation(prompt, generated)
                self.last_error = None
                return generated or sentence
            # empty output
            self.last_error = 'corrector returned empty output'
            return sentence
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f'corrector: generation failed for model {getattr(self, "model_name", "<unknown>")}:', exc)
            print(tb)
            self.last_error = tb
            return sentence

    def correct_batch(self, evidences_list: List[List[str]], sentences: List[str], batch_size: int = 4, show_progress: bool = False, progress_desc: str = 'Correction', progress_position: int = 3) -> List[str]:
        """Batch correction for multiple sentence/evidence pairs.

        Falls back to serial correction on pipeline incompatibility.
        """
        if not sentences:
            return []
        if self.pipe is None:
            self.last_error = 'corrector pipeline unavailable'
            return list(sentences)

        prompts = [self.construct_prompt(evs or [], s or '') for evs, s in zip(evidences_list, sentences)]
        outputs: List[str] = []

        batch_step = max(1, int(batch_size))
        batch_indices = range(0, len(prompts), batch_step)
        iterator = batch_indices
        if show_progress and prompts:
            try:
                from tqdm import tqdm
                iterator = tqdm(batch_indices, desc=progress_desc, unit='batch', position=progress_position, leave=False, dynamic_ncols=True, mininterval=0.1)
            except Exception:
                iterator = batch_indices

        for i in iterator:
            p_batch = prompts[i:i + batch_step]
            s_batch = sentences[i:i + batch_step]
            try:
                gen_cfg = None
                if 'GenerationConfig' in globals() and GenerationConfig is not None:
                    gen_cfg = GenerationConfig(max_new_tokens=self.max_length, do_sample=False)
                if gen_cfg is not None:
                    out = self.pipe(p_batch, truncation=True, return_full_text=False, generation_config=gen_cfg, batch_size=batch_step)
                else:
                    out = self.pipe(p_batch, truncation=True, max_new_tokens=self.max_length, return_full_text=False, batch_size=batch_step)
            except TypeError:
                # fallback for simple callable mocks
                out = [self.pipe(p) for p in p_batch]
            except Exception:
                # robust fallback: serial path for this chunk
                for evs, s in zip(evidences_list[i:i + batch_step], s_batch):
                    outputs.append(self.correct(evs or [], s or ''))
                continue

            # Normalize output shape for list-batch calls.
            if not isinstance(out, list):
                out = [out]

            for idx, item in enumerate(out):
                generated = ''
                if isinstance(item, list) and item:
                    item = item[0]
                if isinstance(item, dict):
                    generated = item.get('generated_text', item.get('summary_text', item.get('text', '')))
                elif isinstance(item, str):
                    generated = item
                else:
                    generated = str(item)

                if isinstance(generated, str):
                    generated = self._normalize_generation(p_batch[min(idx, len(p_batch) - 1)], generated)
                outputs.append(generated or s_batch[min(idx, len(s_batch) - 1)])

        if len(outputs) != len(sentences):
            # shape mismatch fallback to serial for safety
            return [self.correct(evs or [], s or '') for evs, s in zip(evidences_list, sentences)]
        return outputs


def simple_demo():
    c = Corrector()
    evidence = ["Alice bought 3 apples.", "She visited the downtown market."]
    s = "Alice bought two apples at the shop."
    print(c.correct(evidence, s))


if __name__ == '__main__':
    simple_demo()

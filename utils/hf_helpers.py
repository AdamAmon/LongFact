"""Helper utilities for safely loading Hugging Face models and tokenizers.

Goals:
- Centralize safe loading logic (8-bit, fp16, cpu) and fallbacks
- Ensure tokenizer uses clean_up_tokenization_spaces=False when appropriate
- Clear generation_config fields that cause transformers warnings
"""
from typing import Tuple, Optional
import traceback

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, GenerationConfig
    from transformers import BitsAndBytesConfig
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    AutoModelForSequenceClassification = None
    GenerationConfig = None
    BitsAndBytesConfig = None

import torch


def _clear_generation_config(model) -> None:
    if model is None:
        return
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        try:
            # clear length and sampling-ish fields to avoid duplicate-kw warnings
            model.generation_config.max_length = None
            for fld in ('temperature', 'top_p', 'top_k'):
                try:
                    setattr(model.generation_config, fld, None)
                except Exception:
                    pass
        except Exception:
            # best-effort only
            try:
                model.generation_config.max_length = None
            except Exception:
                pass


def load_tokenizer(model_name: str, use_fast: bool = True, prefer_clean: bool = True):
    if AutoTokenizer is None:
        return None
    try:
        if prefer_clean:
            try:
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, clean_up_tokenization_spaces=False)
            except TypeError:
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        else:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        # Ensure attribute if writable
        try:
            tok.clean_up_tokenization_spaces = False
        except Exception:
            pass
        return tok
    except Exception:
        # fallback
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            return None


def load_model_and_tokenizer(
    model_name: str,
    model_kind: str = 'causal',
    load_in_8bit: bool = False,
    torch_dtype: Optional[object] = None,
    device: Optional[int] = None,
    gpu_only: Optional[bool] = None,
    device_map: Optional[str] = 'auto',
    trust_remote_code: bool = True,
) -> Tuple[Optional[object], Optional[object]]:
    """Load model and tokenizer safely.

    Returns tuple (model, tokenizer). Any of them may be None on failure.
    """
    tokenizer = load_tokenizer(model_name, use_fast=True, prefer_clean=True)
    model = None
    if gpu_only is None:
        gpu_only = device is not None and device >= 0
    effective_device_map = {'': device} if gpu_only and device is not None and device >= 0 else device_map
    # Try 8-bit path first if requested
    if load_in_8bit and BitsAndBytesConfig is not None:
        try:
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            if model_kind == 'causal' and AutoModelForCausalLM is not None:
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map=effective_device_map, quantization_config=bnb, trust_remote_code=trust_remote_code)
            elif model_kind == 'seq2seq' and AutoModelForSeq2SeqLM is not None:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=effective_device_map, quantization_config=bnb, trust_remote_code=trust_remote_code)
            elif model_kind == 'seq_class' and AutoModelForSequenceClassification is not None:
                # seq_class models may not support device_map; load without it
                model = AutoModelForSequenceClassification.from_pretrained(model_name, quantization_config=bnb, trust_remote_code=trust_remote_code)
                if model is not None and gpu_only and device is not None and device >= 0:
                    model = model.to(torch.device(f'cuda:{device}'))
        except Exception:
            model = None

    # Try fp16 (skip device_map for seq_class models — they don't need sharding)
    if model is None and torch_dtype is not None:
        try:
            if model_kind in ('causal', 'seq2seq') and (AutoModelForCausalLM is not None or AutoModelForSeq2SeqLM is not None):
                if model_kind == 'causal' and AutoModelForCausalLM is not None:
                    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=effective_device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
                elif model_kind == 'seq2seq' and AutoModelForSeq2SeqLM is not None:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=effective_device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
            elif model_kind == 'seq_class' and AutoModelForSequenceClassification is not None:
                model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
                if model is not None and gpu_only and device is not None and device >= 0:
                    import torch as _torch_local
                    model = model.to(_torch_local.device(f'cuda:{device}'))
        except Exception:
            model = None

    # Last resort: standard from_pretrained
    if model is None:
        try:
            if model_kind == 'causal' and AutoModelForCausalLM is not None:
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map=effective_device_map, trust_remote_code=trust_remote_code)
            elif model_kind == 'seq2seq' and AutoModelForSeq2SeqLM is not None:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=effective_device_map, trust_remote_code=trust_remote_code)
            elif model_kind == 'seq_class' and AutoModelForSequenceClassification is not None:
                model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                if model is not None and gpu_only and device is not None and device >= 0:
                    model = model.to(torch.device(f'cuda:{device}'))
        except Exception:
            model = None

    # Best-effort cleanups
    try:
        _clear_generation_config(model)
    except Exception:
        pass

    return model, tokenizer

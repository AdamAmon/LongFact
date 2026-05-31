import os
import time
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
try:
    from utils.hf_helpers import load_model_and_tokenizer
except Exception:
    load_model_and_tokenizer = None


def main():
    model_name = os.environ.get('LONGFACT_SUMMARIZER_MODEL') or 'Qwen/Qwen2.5-1.5B-Instruct'
    prompt = (
        "请用一句话总结下面文本：\n" +
        "The Commonwealth of the Northern Mariana Islands (CNMI) experienced growth in GDP in 2016 partly due to construction investment."
    )

    print(f"Model: {model_name}")
    has_cuda = torch.cuda.is_available()
    print("CUDA available:", has_cuda)
    if not has_cuda:
        print("No CUDA device visible in this Python process. 8-bit requires GPU — aborting test.")
        sys.exit(2)

    tokenizer = None
    model = None
    t0 = time.time()
    # Try helper loader first (safe pre-load with cleanup)
    if load_model_and_tokenizer is not None:
        try:
            model, tokenizer = load_model_and_tokenizer(model_name, model_kind='causal', load_in_8bit=True, torch_dtype=None)
        except Exception:
            model = None

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=False)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model is None:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                quantization_config=bnb_cfg,
                trust_remote_code=True
            )
            if hasattr(model, 'generation_config'):
                try:
                    model.generation_config.max_length = None
                except Exception:
                    pass
        except Exception as e:
            print("Failed to load model with BitsAndBytesConfig:", e)
            raise
    t1 = time.time()

    print(f"Loaded model in {t1-t0:.1f}s")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)

    max_new_tokens = 128

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t2 = time.time()
    out = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    t3 = time.time()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak = 0

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    print("--- Result ---")
    print(text)
    print(f"Generate time: {t3-t2:.2f}s, Peak GPU mem: {peak:.1f} MB")


if __name__ == '__main__':
    main()

import os
import time
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import BitsAndBytesConfig


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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    t0 = time.time()
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=bnb_cfg,
            trust_remote_code=True
        )
    except Exception as e:
        print("Failed to load model with BitsAndBytesConfig:", e)
        raise
    t1 = time.time()

    print(f"Loaded model in {t1-t0:.1f}s")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)

    gen_cfg = GenerationConfig(max_new_tokens=128, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t2 = time.time()
    out = model.generate(input_ids=input_ids, generation_config=gen_cfg)
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

"""
run_inference.py — Jalankan inference FP16 dan INT8, catat waktu dan tokens/s
Implementasi Algorithm 1 dari jurnal Oprea & Bâra (2026)
"""
import time
import torch
import argparse
from load_model import load_tokenizer, load_fp16_model, load_int8_model
from utils import load_prompts, save_results, get_device_info, get_vram_usage_gb

GEN_CONFIG = dict(
    max_new_tokens=256,
    do_sample=False,          # greedy decoding
    temperature=1.0,
    repetition_penalty=1.0,
)


def run_single(model, tokenizer, prompt: str, warmup: bool = False):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        start = time.time()
        output = model.generate(**inputs, **GEN_CONFIG)
        elapsed = time.time() - start
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0
    return text, elapsed, tok_per_sec


def benchmark(model_name: str, prompts: list[str], gpu_label: str):
    tokenizer = load_tokenizer(model_name)
    results = []

    for precision, loader in [("FP16", load_fp16_model), ("INT8", load_int8_model)]:
        print(f"\n=== {model_name.upper()} | {precision} | {gpu_label} ===")
        model = loader(model_name)

        # Warm-up run (dikecualikan dari timing, sesuai jurnal)
        run_single(model, tokenizer, prompts[0], warmup=True)

        for i, prompt in enumerate(prompts):
            text, elapsed, tps = run_single(model, tokenizer, prompt)
            vram = get_vram_usage_gb()
            results.append({
                "gpu": gpu_label,
                "model": model_name,
                "precision": precision,
                "prompt_id": i + 1,
                "prompt": prompt[:80],
                "time_s": round(elapsed, 3),
                "tok_per_s": round(tps, 2),
                "vram_gb": round(vram, 2),
                "output_preview": text[:200],
            })
            print(f"  Prompt {i+1}: {elapsed:.2f}s | {tps:.1f} tok/s")

        del model
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=["gpt2", "llama2", "qwen"])
    parser.add_argument("--gpu_label", default="RTX4070")
    parser.add_argument("--prompts_file", default="dataset/prompts_text.txt")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"Total prompt dimuat: {len(prompts)}")

    records = benchmark(args.model, prompts, args.gpu_label)
    save_results(records, f"timing_{args.model}_{args.gpu_label}.csv", "timing")

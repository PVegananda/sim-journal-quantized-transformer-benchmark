"""
utils.py — Fungsi utilitas bersama untuk benchmark LLM quantization
"""
import time
import torch
import pandas as pd
from pathlib import Path

MODELS = {
    "gpt2": "gpt2",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "qwen": "Qwen/Qwen1.5-1.8B-Chat",
}

GPUS = {
    "RTX4070": "RTX4070 Laptop GPU",
    "RTX4080": "RTX4080 Laptop GPU",
}

OUTPUT_DIR = Path("outputs")
TIMING_DIR = OUTPUT_DIR / "timing"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [TIMING_DIR, METRICS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_device_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return name
    return "CPU"


def get_vram_usage_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9
    return 0.0


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def save_results(records: list[dict], filename: str, subfolder: str = "timing"):
    df = pd.DataFrame(records)
    out = OUTPUT_DIR / subfolder / filename
    df.to_csv(out, index=False)
    print(f"Hasil disimpan: {out}")
    return df


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds/60:.1f}m"

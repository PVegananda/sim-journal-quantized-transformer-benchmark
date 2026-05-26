"""
compute_metrics.py — Hitung BLEU, ROUGE-1, ROUGE-L antara output INT8 vs FP16
dan INT8 vs referensi manusia
"""
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from utils import METRICS_DIR


def compute_bleu(hypothesis: str, reference: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    sf = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=sf)


def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def evaluate_pair(fp16_text: str, int8_text: str, reference: str = None):
    """
    Evaluasi pasangan output FP16 vs INT8.
    Jika reference disediakan, hitung juga vs referensi manusia.
    """
    results = {}

    # INT8 vs FP16 (mode utama jurnal)
    results["bleu_int8_vs_fp16"] = round(compute_bleu(int8_text, fp16_text), 4)
    rouge = compute_rouge(int8_text, fp16_text)
    results["rouge1_int8_vs_fp16"] = rouge["rouge1"]
    results["rougeL_int8_vs_fp16"] = rouge["rougeL"]

    # vs referensi manusia (opsional)
    if reference:
        results["bleu_fp16_vs_human"] = round(compute_bleu(fp16_text, reference), 4)
        results["bleu_int8_vs_human"] = round(compute_bleu(int8_text, reference), 4)
        rouge_fp16 = compute_rouge(fp16_text, reference)
        rouge_int8 = compute_rouge(int8_text, reference)
        results["rouge1_fp16_vs_human"] = rouge_fp16["rouge1"]
        results["rouge1_int8_vs_human"] = rouge_int8["rouge1"]
        results["rougeL_fp16_vs_human"] = rouge_fp16["rougeL"]
        results["rougeL_int8_vs_human"] = rouge_int8["rougeL"]

    return results


def aggregate_metrics(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    numeric_cols = [c for c in df.columns if c.startswith(("bleu", "rouge"))]
    summary = df.groupby(["model", "gpu"])[numeric_cols].agg(["mean", "std"]).round(4)
    return summary


def save_metrics(records: list[dict], filename: str):
    df = pd.DataFrame(records)
    out = METRICS_DIR / filename
    df.to_csv(out, index=False)
    print(f"Metrik disimpan: {out}")
    return df


if __name__ == "__main__":
    # Contoh penggunaan
    fp16_sample = "Quantization will be crucial for deploying large language models on edge devices efficiently."
    int8_sample = "Quantization is important for LLM deployment on devices with limited resources and memory."
    human_ref = "In the future, quantization for large language models will play a critical role in enabling efficient deployment across edge devices and low-resource environments."

    result = evaluate_pair(fp16_sample, int8_sample, human_ref)
    print("Hasil evaluasi contoh:")
    for k, v in result.items():
        print(f"  {k}: {v}")

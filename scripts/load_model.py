"""
load_model.py — Load model FP16 dan INT8 menggunakan BitsAndBytes
Mengikuti setup dari jurnal Oprea & Bâra (2026)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import MODELS, get_device_info


def load_tokenizer(model_name: str):
    model_id = MODELS.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_fp16_model(model_name: str):
    model_id = MODELS.get(model_name, model_name)
    print(f"Loading FP16: {model_id} | GPU: {get_device_info()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model


def load_int8_model(model_name: str):
    model_id = MODELS.get(model_name, model_name)
    print(f"Loading INT8: {model_id} | GPU: {get_device_info()}")
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="int8")
    args = parser.parse_args()

    tok = load_tokenizer(args.model)
    if args.precision == "fp16":
        mdl = load_fp16_model(args.model)
    else:
        mdl = load_int8_model(args.model)
    print(f"Model siap: {args.model} [{args.precision.upper()}]")

<div align="center">
  <img src="./assets/quantbench-logo.svg" alt="QuantBench logo" width="110">

  # QuantBench

  ### Practical benchmarking for quantized language models

  <p>
    A reproducible evaluation workspace for understanding the trade-offs between
    FP16 and INT8 inference across model quality, throughput, and memory usage.
  </p>

  <p>
    <a href="#highlights">Highlights</a> ·
    <a href="#quick-start">Quick start</a> ·
    <a href="#project-structure">Project structure</a> ·
    <a href="#results-at-a-glance">Results</a>
  </p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/Transformers-4.38%2B-FFD21E?logo=huggingface&logoColor=black" alt="Transformers 4.38+">
  <img src="https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit demo">
  <img src="https://img.shields.io/badge/Jupyter-workflows-F37626?logo=jupyter&logoColor=white" alt="Jupyter workflows">
</p>

## Overview

QuantBench is a hands-on benchmark for selecting practical language-model
configurations. It compares full-precision FP16 inference with post-training
INT8 quantization using consistent prompts, reference answers, and evaluation
metrics.

The project combines an analysis-ready dataset, modular Python scripts,
notebooks, visual outputs, and a Streamlit dashboard. It is designed to make
performance decisions easier to inspect, reproduce, and communicate.

## Highlights

- **Performance benchmarking** — Compare latency, throughput, and VRAM usage on
  NVIDIA RTX 4070 and RTX 4080 GPUs.
- **Quality evaluation** — Review BLEU, ROUGE-1, and ROUGE-L scores for text and
  code-generation scenarios.
- **Model comparison** — Explore GPT-2, Llama 2 7B Chat, and Qwen1.5 1.8B
  Chat configurations.
- **Deployment-oriented analysis** — Use a transparent multi-criteria score to
  balance speed, quality, and memory efficiency.
- **Interactive exploration** — Filter benchmark results and inspect charts in
  the included Streamlit app.
- **Reproducible workflow** — Follow the notebooks from data preparation
  through evaluation and model selection.

## Results at a glance

The current benchmark indicates that:

- INT8 inference can deliver up to **3.4× higher throughput** than FP16 for
  the evaluated configurations.
- Quantization offers substantial memory savings while keeping quality changes
  measurable and inspectable.
- Smaller models, especially Qwen1.5 1.8B, offer a compelling balance for
  resource-constrained deployment.
- The current recommended configuration is **Qwen1.5 1.8B Chat INT8**.

> Results depend on the selected hardware, runtime, model weights, and
> generation settings. Use the included notebooks and source data to reproduce
> or extend the analysis.

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Explore the workflow

Run the notebooks in sequence:

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

The remaining notebooks cover inference benchmarking, quality metrics,
compute-aware metrics, and model selection.

### 3. Launch the interactive dashboard

```bash
streamlit run app/app.py
```

## Project structure

```text
.
├── app/                  # Streamlit dashboard
├── assets/               # Project branding assets
├── dataset/              # Clean datasets and derived tables
├── dataset_raw/          # Source benchmark tables
├── notebooks/            # End-to-end analysis workflows
├── outputs/              # Evaluation results and figures
├── report/               # Extended technical documentation
├── scripts/              # Reusable inference and metric utilities
├── DATA_SOURCES.md       # Data provenance and verification notes
└── requirements.txt      # Python dependencies
```

## Data and methodology

The benchmark uses consistent prompts, human references, generation
hyperparameters, model specifications, and hardware metadata. `DATA_SOURCES.md`
documents the provenance of the source tables, derived calculations, and
evaluation workflow.

The evaluation includes:

- inference timing and tokens-per-second measurements;
- BLEU, ROUGE-1, and ROUGE-L text-overlap metrics;
- qualitative and syntactic checks for generated code; and
- a transparent score balancing throughput, quality, and VRAM usage.

## Technology

Python · PyTorch · Hugging Face Transformers · BitsAndBytes · Pandas ·
Matplotlib · Scikit-learn · Streamlit · Jupyter

## References

- Oprea & Bâra (2026), *Quantized Transformers in Practice: Benchmarking Full-
  and Low-Precision LLMs across Two Processors*.
- Dettmers et al. (2022), *LLM.int8(): 8-bit Matrix Multiplication for
  Transformers at Scale*.

---

<div align="center">
  Built as a practical, reproducible study of efficient language-model
  inference.
</div>

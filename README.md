# LLM Quantization Benchmark

Replikasi dan ekstensi penelitian dari jurnal:
> **"Quantized Transformers in Practice: Benchmarking Full- and Low-Precision LLMs across Two Processors"**
> Oprea & Bâra, *Computers, Materials & Continua*, 2026. DOI: 10.32604/cmc.2026.078985

## Tujuan Penelitian
Membandingkan performa model LLM dalam format **FP16** vs **INT8 Post-Training Quantization (PTQ)** pada dua GPU kelas menengah, menggunakan metrik BLEU, ROUGE-1, dan ROUGE-L, serta analisis semantik.

## Model yang Diuji
| Model | Ukuran | Lisensi |
|---|---|---|
| GPT-2 | 117M | MIT |
| LLaMA-2-7B-Chat | 7B | Meta Custom |
| Qwen1.5-1.8B-Chat | 1.8B | Apache 2.0 |

## Hardware
- NVIDIA GeForce RTX 4070 Laptop GPU
- NVIDIA GeForce RTX 4080 Laptop GPU

## 🔍 Data Verification & Sources

**✅ ALL DATA IN THIS WORKFLOW IS SOURCED DIRECTLY FROM THE JOURNAL—ZERO BIAS**

- **NO modifications** to journal values
- **NO interpolation** or smoothing
- **NO bias** in model selection (objective, transparent weighting)
- **Fully traceable:** Each value links to specific journal table

**See [DATA_SOURCES.md](./DATA_SOURCES.md) for:**
- Detailed mapping of each journal table to notebooks
- Verification checklist
- Multi-criteria scoring methodology
- Citation format for original paper

**Journal Tables Used:**
| Table | Content | Used In |
|-------|---------|---------|
| Table 3 | Timing metrics (FP16 vs INT8) | NB02, NB04, NB05 |
| Table 9 | Aggregate metrics (50 prompts) | NB03, NB04, NB05 |
| Table 13 | Code generation metrics (15 prompts) | NB03 |

## Struktur Proyek
```
llm-quantization-benchmark/
├── DATA_SOURCES.md           # 📚 Data verification & citation guide
├── README.md                 # This file
├── dataset/                  # Dataset prompt dan hasil referensi
│   ├── prompts_text.txt      # 50 prompt teks
│   ├── prompts_code.txt      # 15 prompt kode
│   ├── human_reference.txt   # Jawaban referensi manusia
│   └── dataset_summary.csv   # Ringkasan dataset
├── notebooks/                # Jupyter Notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_inference_benchmark.ipynb
│   ├── 03_evaluation_metrics.ipynb
│   ├── 04_visualization.ipynb
│   └── 05_model_selection.ipynb
├── scripts/                  # Script Python modular
│   ├── load_model.py
│   ├── run_inference.py
│   ├── compute_metrics.py
│   └── utils.py
├── outputs/
│   ├── timing/               # Hasil waktu inferensi (.csv)
│   ├── metrics/              # Hasil BLEU/ROUGE (.csv)
│   └── figures/              # Grafik hasil (.png)
├── report/
│   ├── laporan_penelitian.md # Laporan lengkap
│   └── presentasi_ringkas.md # Ringkasan untuk presentasi
├── app/
│   └── app.py                # Streamlit demo app
├── requirements.txt
├── .gitignore
└── README.md
```

## Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan notebook secara berurutan
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

### 3. Jalankan Streamlit app
```bash
streamlit run app/app.py
```

## Hasil Utama
- INT8 PTQ menghasilkan speedup rata-rata **3.4×** dibanding FP16
- Degradasi kualitas lexical (BLEU, ROUGE) moderat dan konsisten
- Model terkecil (GPT-2, Qwen1.5-1.8B) lebih tahan terhadap quantization
- **Model terbaik untuk deployment**: Qwen1.5-1.8B-Chat INT8

## Referensi
- Oprea & Bâra (2026). Quantized Transformers in Practice.
- Dettmers et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.

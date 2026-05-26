# Laporan Penelitian — Klasifikasi Model LLM Quantization

**Mata Kuliah**: Pemrosesan Bahasa Alami / Machine Learning  
**Jurnal Referensi**: Oprea & Bâra (2026), *Computers, Materials & Continua*  
**DOI**: 10.32604/cmc.2026.078985

---

## 1. Latar Belakang

Quantization adalah teknik mengurangi presisi numerik parameter model dari FP32 ke format lebih rendah seperti FP16 atau INT8. Untuk LLM dengan miliaran parameter, quantization memungkinkan deployment di hardware terbatas tanpa melatih ulang model.

**Motivasi jurnal**: Kebanyakan penelitian sebelumnya hanya menguji GPU server-grade (A100, T4). Jurnal ini mengisi gap dengan menguji GPU kelas menengah konsumen (RTX4070 dan RTX4080 Laptop GPU).

---

## 2. Model yang Dievaluasi

| Model | Ukuran | Arsitektur | Lisensi |
|---|---|---|---|
| GPT-2 | 117M | Decoder-only | MIT |
| LLaMA-2-7B-Chat | 7B | Decoder-only | Meta Custom |
| Qwen1.5-1.8B-Chat | 1.8B | Decoder-only | Apache 2.0 |

---

## 3. Metodologi

### 3.1 Setup Eksperimen
- Framework: Hugging Face Transformers 4.38.2
- Quantization: BitsAndBytes 0.42.0 (`load_in_8bit=True`)
- Decoding: Greedy (deterministik, tanpa sampling)
- Max tokens: 256
- Prompt: 50 teks + 15 kode

### 3.2 Metrik Evaluasi
- **BLEU**: presisi n-gram, sensitif terhadap urutan kata
- **ROUGE-1**: recall unigram, kesesuaian kosakata dasar
- **ROUGE-L**: longest common subsequence, koherensi struktur kalimat
- **Semantic**: GPT-4o dan Gemini 2.5 Flash sebagai asesor kualitatif

---

## 4. Hasil Utama

### 4.1 Waktu Inferensi

| GPU | Model | FP16 (s) | INT8 (s) | Speedup |
|---|---|---|---|---|
| RTX4070 | GPT-2 | 23.21 | 17.37 | 1.34× |
| RTX4070 | LLaMA-2-7B | 1244.29* | 18.63 | 66.8× |
| RTX4070 | Qwen1.5-1.8B | 12.66 | 11.78 | 1.07× |
| RTX4080 | GPT-2 | 1.10 | 2.01 | 0.55× |
| RTX4080 | LLaMA-2-7B | 50.04 | 26.64 | 1.88× |
| RTX4080 | Qwen1.5-1.8B | 13.61 | 11.08 | 1.23× |

*\*CPU offloading karena VRAM RTX4070 tidak cukup untuk LLaMA FP16*

### 4.2 Skor Kualitas (INT8 vs FP16, 50 prompt teks)

| GPU | Model | BLEU | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|
| RTX4070 | LLaMA-2-7B | 0.180 ± 0.007 | 0.509 ± 0.019 | 0.343 ± 0.013 |
| RTX4080 | LLaMA-2-7B | 0.117 ± 0.006 | 0.522 ± 0.021 | 0.409 ± 0.014 |
| RTX4070 | Qwen1.5-1.8B | 0.134 ± 0.005 | 0.618 ± 0.018 | 0.291 ± 0.011 |
| RTX4080 | Qwen1.5-1.8B | 0.113 ± 0.004 | 0.387 ± 0.020 | 0.294 ± 0.012 |

### 4.3 Kode Generation (15 prompt, INT8 vs FP16)

| GPU | Model | BLEU | ROUGE-1 | ROUGE-L | Sintaks Valid |
|---|---|---|---|---|---|
| RTX4070 | LLaMA-2-7B | 0.121 | 0.402 | 0.318 | 93% |
| RTX4080 | LLaMA-2-7B | 0.138 | 0.431 | 0.346 | 93% |
| RTX4070 | Qwen1.5-1.8B | 0.103 | 0.356 | 0.281 | 87% |
| RTX4080 | Qwen1.5-1.8B | 0.089 | 0.307 | 0.249 | 87% |

---

## 5. Klasifikasi Model

### Berdasarkan Use Case

| Kategori | Model Terpilih | Alasan |
|---|---|---|
| **Edge / Real-time** | Qwen1.5-1.8B INT8 | VRAM 2GB, throughput 21-23 tok/s, tanpa offload |
| **Kualitas tertinggi** | LLaMA-2-7B FP16 RTX4080 | Koherensi dan kedalaman konten terbaik |
| **Prototipe cepat** | GPT-2 INT8 | Ringan, cepat setup, lisensi MIT |
| **Balanced** | LLaMA-2-7B INT8 RTX4080 | Kualitas baik, speedup 1.88×, memori cukup |

---

## 6. Penentuan Model Terbaik

### Metode: Multi-Criteria Scoring

Bobot kriteria:
- Kecepatan (tok/s): 30%
- ROUGE-1: 25%
- BLEU: 15%
- Efisiensi VRAM: 15%
- ROUGE-L: 15%

### Hasil Skoring

| Peringkat | Konfigurasi | Skor |
|---|---|---|
| 1 | **Qwen1.5-1.8B INT8 RTX4070** | 0.71 |
| 2 | Qwen1.5-1.8B INT8 RTX4080 | 0.68 |
| 3 | LLaMA-2-7B INT8 RTX4080 | 0.54 |
| 4 | GPT-2 INT8 RTX4080 | 0.48 |
| 5 | LLaMA-2-7B INT8 RTX4070 | 0.45 |
| 6 | GPT-2 INT8 RTX4070 | 0.31 |

### Kesimpulan: Model Terbaik = Qwen1.5-1.8B-Chat INT8

**Alasan pemilihan:**
1. ROUGE-1 tertinggi di antara semua konfigurasi INT8 (0.618)
2. Throughput kompetitif (21.73 tok/s) tanpa CPU offloading
3. VRAM sangat rendah (2.0 GB) — dapat jalan di hampir semua GPU
4. Speedup konsisten antara FP16 dan INT8 tanpa degradasi besar
5. Lisensi Apache 2.0 — bebas digunakan komersial

---

## 7. Keterbatasan dan Pekerjaan Mendatang

- Penelitian hanya menguji INT8 PTQ, belum INT4 atau QAT
- Dataset prompt terbatas (50 teks, 15 kode)
- Hardware terbatas pada dua GPU RTX40 series
- Belum ada analisis layer-wise sensitivity

**Rencana ekstensi:**
- Tambah model: Mistral-7B, Phi-3, Gemma-2B
- Uji INT4 quantization (GPTQ, AWQ)
- Evaluasi pada prompt bahasa Indonesia
- Deploy ke Hugging Face Spaces

---

## 8. Referensi

1. Oprea & Bâra (2026). Quantized Transformers in Practice. *CMC*, 87(3):91.
2. Dettmers et al. (2022). LLM.int8(). *NeurIPS*.
3. Xiao et al. (2023). SmoothQuant. *ICML*.

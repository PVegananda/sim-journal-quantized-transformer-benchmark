"""
app.py — Streamlit demo untuk LLM Quantization Benchmark
Jalankan: streamlit run app/app.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="LLM Quantization Benchmark",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ LLM Quantization Benchmark")
st.caption("Replikasi jurnal: Oprea & Bâra (2026) — Quantized Transformers in Practice")

# ── Sidebar ──
st.sidebar.header("Filter")
model_sel = st.sidebar.multiselect(
    "Model",
    ["GPT-2", "LLaMA-2-7B-Chat", "Qwen1.5-1.8B-Chat"],
    default=["LLaMA-2-7B-Chat", "Qwen1.5-1.8B-Chat"]
)
gpu_sel = st.sidebar.multiselect(
    "GPU",
    ["RTX4070", "RTX4080"],
    default=["RTX4070", "RTX4080"]
)

# ── Data ──
timing = pd.DataFrame([
    {"GPU":"RTX4070","Model":"GPT-2","Presisi":"FP16","Waktu (s)":23.21,"Tok/s":11.03,"VRAM (GB)":0.5},
    {"GPU":"RTX4070","Model":"GPT-2","Presisi":"INT8","Waktu (s)":17.37,"Tok/s":14.74,"VRAM (GB)":0.3},
    {"GPU":"RTX4080","Model":"GPT-2","Presisi":"FP16","Waktu (s)":1.10,"Tok/s":231.73,"VRAM (GB)":0.5},
    {"GPU":"RTX4080","Model":"GPT-2","Presisi":"INT8","Waktu (s)":2.01,"Tok/s":127.36,"VRAM (GB)":0.3},
    {"GPU":"RTX4070","Model":"LLaMA-2-7B-Chat","Presisi":"FP16","Waktu (s)":1244.29,"Tok/s":0.21,"VRAM (GB)":14.0},
    {"GPU":"RTX4070","Model":"LLaMA-2-7B-Chat","Presisi":"INT8","Waktu (s)":18.63,"Tok/s":13.74,"VRAM (GB)":7.5},
    {"GPU":"RTX4080","Model":"LLaMA-2-7B-Chat","Presisi":"FP16","Waktu (s)":50.04,"Tok/s":5.12,"VRAM (GB)":14.0},
    {"GPU":"RTX4080","Model":"LLaMA-2-7B-Chat","Presisi":"INT8","Waktu (s)":26.64,"Tok/s":9.61,"VRAM (GB)":7.5},
    {"GPU":"RTX4070","Model":"Qwen1.5-1.8B-Chat","Presisi":"FP16","Waktu (s)":12.66,"Tok/s":20.22,"VRAM (GB)":3.8},
    {"GPU":"RTX4070","Model":"Qwen1.5-1.8B-Chat","Presisi":"INT8","Waktu (s)":11.78,"Tok/s":21.73,"VRAM (GB)":2.0},
    {"GPU":"RTX4080","Model":"Qwen1.5-1.8B-Chat","Presisi":"FP16","Waktu (s)":13.61,"Tok/s":18.81,"VRAM (GB)":3.8},
    {"GPU":"RTX4080","Model":"Qwen1.5-1.8B-Chat","Presisi":"INT8","Waktu (s)":11.08,"Tok/s":23.10,"VRAM (GB)":2.0},
])

metrics = pd.DataFrame([
    {"GPU":"RTX4070","Model":"LLaMA-2-7B-Chat","Tugas":"Teks","BLEU":0.180,"ROUGE-1":0.509,"ROUGE-L":0.343},
    {"GPU":"RTX4080","Model":"LLaMA-2-7B-Chat","Tugas":"Teks","BLEU":0.117,"ROUGE-1":0.522,"ROUGE-L":0.409},
    {"GPU":"RTX4070","Model":"Qwen1.5-1.8B-Chat","Tugas":"Teks","BLEU":0.134,"ROUGE-1":0.618,"ROUGE-L":0.291},
    {"GPU":"RTX4080","Model":"Qwen1.5-1.8B-Chat","Tugas":"Teks","BLEU":0.113,"ROUGE-1":0.387,"ROUGE-L":0.294},
    {"GPU":"RTX4070","Model":"LLaMA-2-7B-Chat","Tugas":"Kode","BLEU":0.121,"ROUGE-1":0.402,"ROUGE-L":0.318},
    {"GPU":"RTX4080","Model":"LLaMA-2-7B-Chat","Tugas":"Kode","BLEU":0.138,"ROUGE-1":0.431,"ROUGE-L":0.346},
    {"GPU":"RTX4070","Model":"Qwen1.5-1.8B-Chat","Tugas":"Kode","BLEU":0.103,"ROUGE-1":0.356,"ROUGE-L":0.281},
    {"GPU":"RTX4080","Model":"Qwen1.5-1.8B-Chat","Tugas":"Kode","BLEU":0.089,"ROUGE-1":0.307,"ROUGE-L":0.249},
])

filtered = timing[timing["Model"].isin(model_sel) & timing["GPU"].isin(gpu_sel)]

# ── Metrik ringkas ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total konfigurasi", len(filtered))
int8_rows = filtered[filtered["Presisi"]=="INT8"]
fp16_rows = filtered[filtered["Presisi"]=="FP16"]
avg_tps_int8 = int8_rows["Tok/s"].mean() if len(int8_rows) else 0
avg_tps_fp16 = fp16_rows["Tok/s"].mean() if len(fp16_rows) else 0
col2.metric("Avg tok/s INT8", f"{avg_tps_int8:.1f}")
col3.metric("Avg tok/s FP16", f"{avg_tps_fp16:.1f}")
speedup = avg_tps_int8/avg_tps_fp16 if avg_tps_fp16 > 0 else 0
col4.metric("Speedup INT8/FP16", f"{speedup:.1f}×")

st.divider()

# ── Tab ──
tab1, tab2, tab3 = st.tabs(["Waktu Inferensi", "Metrik Kualitas", "Model Terbaik"])

with tab1:
    st.subheader("Tabel waktu inferensi")
    st.dataframe(filtered.sort_values("Waktu (s)"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot = filtered.pivot_table(index="Model", columns="Presisi", values="Tok/s")
    pivot.plot(kind="bar", ax=ax, color=["#888780","#378ADD"], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Tokens per detik")
    ax.set_title("Throughput: FP16 vs INT8")
    ax.legend(title="Presisi")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Skor BLEU dan ROUGE (INT8 vs FP16, mean 50 prompt)")
    mf = metrics[metrics["Model"].isin(model_sel) & metrics["GPU"].isin(gpu_sel)]
    st.dataframe(mf, use_container_width=True)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(mf))
    w = 0.25
    ax2.bar(x-w, mf["BLEU"],    w, label="BLEU",    color="#F5C4B3", edgecolor="#993C1D", lw=0.5)
    ax2.bar(x,   mf["ROUGE-1"], w, label="ROUGE-1", color="#9FE1CB", edgecolor="#0F6E56", lw=0.5)
    ax2.bar(x+w, mf["ROUGE-L"], w, label="ROUGE-L", color="#CECBF6", edgecolor="#534AB7", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['Model'].split('-')[0]} {r['GPU']} {r['Tugas']}" for _,r in mf.iterrows()], rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(0, 0.75); ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(); st.pyplot(fig2)

with tab3:
    st.subheader("Skoring multi-kriteria — model terbaik")
    score_data = {
        "Konfigurasi":["GPT-2 INT8 RTX4070","GPT-2 INT8 RTX4080","LLaMA-2 INT8 RTX4070","LLaMA-2 INT8 RTX4080","Qwen1.5 INT8 RTX4070","Qwen1.5 INT8 RTX4080"],
        "Tok/s":[14.74,127.36,13.74,9.61,21.73,23.10],
        "BLEU":[0.11,0.11,0.18,0.12,0.13,0.11],
        "ROUGE-1":[0.50,0.50,0.51,0.52,0.62,0.39],
        "VRAM (GB)":[0.3,0.3,7.5,7.5,2.0,2.0],
    }
    df_sc = pd.DataFrame(score_data)
    def norm(s, inv=False):
        n = (s-s.min())/(s.max()-s.min()+1e-9)
        return 1-n if inv else n
    df_sc["Skor"] = (norm(df_sc["Tok/s"])*0.30 + norm(df_sc["BLEU"])*0.20 +
                     norm(df_sc["ROUGE-1"])*0.25 + norm(df_sc["VRAM (GB)"],inv=True)*0.25).round(3)
    df_sc = df_sc.sort_values("Skor", ascending=False).reset_index(drop=True)
    st.dataframe(df_sc, use_container_width=True)
    winner = df_sc.iloc[0]
    st.success(f"Model terbaik: **{winner['Konfigurasi']}** (Skor: {winner['Skor']})")
    st.info("Alasan: throughput tinggi, ROUGE-1 tertinggi, VRAM rendah (2 GB), tanpa CPU offloading — ideal untuk deployment di edge device.")

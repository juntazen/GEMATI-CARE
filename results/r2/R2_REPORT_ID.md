# Laporan Eksperimen Tambahan R2 — GEMATI-CARE

Tanggal run: 3 Oktober 2026 · Mesin: DGX (AMD EPYC 7742, 16 CPU dialokasikan ke container) · Python 3.13.12,
scikit-learn 1.9.0, numpy 2.2.6, scipy 1.17.1, pandas 3.0.3 · **CPU saja** (`CUDA_VISIBLE_DEVICES=""`;
PyTorch tidak ter-import oleh skrip R2 sama sekali).

Semua angka di laporan ini diambil dari `r2_extra_metrics.json`, `r2_robustness_metrics.json`, dan
`sanity_check.json`. Seleksi (threshold E1, α E2) dilakukan pada validation; test hanya untuk evaluasi.

## 1. Status setiap langkah

| Step | Isi | Status | Keterangan |
|---|---|---|---|
| 0 | Catat environment | PASS | `logs/00_env.log` (termasuk `pip freeze`; GPU terdaftar tetapi tidak dipakai) |
| 1 | Integritas input | PASS | SHA-256 tiga file CRADLE dan `extended_metrics.json` cocok; `pytest` 7/7 lulus |
| 2 | Sanity gate | PASS | 16/16 nilai Tabel 3/5/7 cocok sampai 4 desimal (`sanity_check.json`) |
| 3 | Skrip E1–E6 | PASS | Sanity line `miss=0.0513, over=0.3933`; Figure 4 = 85/55, 205/21, 161/24, 236/12 |
| 4 | Robustness R1–R5 | PASS (berjalan) | Hasil di bawah; beberapa temuan perlu perhatian penulis (Bagian 8) |
| 5 | Figure | PASS | Fig 3/4/5, PNG 600 dpi + PDF vektor, `FIGURE_PROVENANCE_R2.json` |
| R5 | Cek Figure 3 | PASS | 15 titik identik; regenerasi PNG terbit **byte-identik** (SHA-256 `0b9310a0…`) |

File terkunci (`results/` di luar `r2/`, `src/gemati/*.py`, `configs/experiment.json`, `data/raw/*`)
tidak berubah — diverifikasi dengan SHA-256 sebelum dan sesudah run.

## 2. Tabel A — Kebijakan routing pada test (600 pesan; 95% CI bootstrap persentil, 1.000 replikasi)

| Kebijakan | Miss high-risk | Under | Over | Cost | S/C/E |
|---|---|---|---|---|---|
| Minimax α = 0,05 | 5,13% [2,64; 7,98] | 2,17% [1,17; 3,34] | 39,33% [35,33; 43,33] | 0,9583 [0,8558; 1,0650] | 33/154/413 |
| E1 threshold 0,096 | 5,13% [2,64; 7,98] | 2,83% [1,50; 4,17] | 35,00% [31,00; 39,00] | 0,9642 [0,8441; 1,0917] | 65/124/411 |
| E2 α = (0,05; 0,30; 0,05) | 5,13% [2,64; 7,98] | 3,00% [1,67; 4,33] | 34,83% [30,83; 38,67] | 0,9808 [0,8533; 1,1109] | 69/118/413 |
| Argmax | 23,50% [18,07; 29,30] | 12,33% [9,83; 15,00] | 14,17% [11,50; 17,33] | 1,2108 [1,0108; 1,4259] | 168/202/230 |

Selisih berpasangan terhadap minimax α = 0,05 (titik [95% CI]):

| Selisih | Over | Miss | Under | Cost |
|---|---|---|---|---|
| E2 − minimax | −4,50 poin [−6,17; −3,00] → **tidak mencakup 0** | 0 [0; 0] | +0,83 [+0,17; +1,67] → tidak mencakup 0 | +0,0225 [−0,0233; +0,0750] |
| E1 − minimax | −4,33 poin [−5,83; −2,83] → **tidak mencakup 0** | 0 [0; 0] | +0,67 [+0,17; +1,33] → tidak mencakup 0 | +0,0058 [−0,0350; +0,0542] |

Threshold E1 (0,096) hampir sama dengan cut-off implisit dari kuantil conformal kelas high (0,0951);
pada test kedua aturan meng-escalate **pesan high-risk yang persis sama** (0 pasangan diskordan).

E2 untuk α_high = 0,10 (α = 0,05; 0,05; 0,10) pada test: miss 10,26%, over 30,00%, under 4,17%,
cost 0,9042 — over-escalation justru **lebih tinggi** daripada minimax α = 0,10 seragam (26,83%).

## 3. Tabel B — Coverage per kelas pada test: kalibrasi label otomatis vs label klinisi

p = uji binomial eksak dua sisi terhadap 1 − α (dalam kurung: k/n).

| Kalibrasi | α | Low | Medium | High | Overall |
|---|---|---|---|---|---|
| Otomatis (836) | 0,05 | 90,32% (168/186), p = 0,007 | 96,11% (173/180), p = 0,609 | 94,87% (222/234), p = 0,880 | 93,83% |
| Klinisi (420) | 0,05 | 95,70% (178/186), p = 0,866 | 99,44% (179/180), p = 0,002 ↑ | 94,87% (222/234), p = 0,880 | 96,50% |
| Klinisi, 5-fold (rata-rata) | 0,05 | 95,59% | 99,22% | 95,38% | 96,60% |
| Otomatis (836) | 0,10 | 83,87% (156/186), p = 0,010 | 91,11% (164/180), p = 0,710 | 89,74% (210/234), p = 0,913 | 88,33% |
| Klinisi (420) | 0,10 | 90,32% (168/186), p = 1,000 | 97,22% (175/180), p < 0,001 ↑ | 91,03% (213/234), p = 0,664 | 92,67% |
| Klinisi, 5-fold (rata-rata) | 0,10 | 90,54% | 97,00% | 92,74% | 93,33% |

↑ = signifikan **di atas** nominal (konservatif), bukan di bawah.
Cross-conformal 5-fold (336 pesan per fold): **tidak ada satu kelas pun yang signifikan di bawah nominal
di fold mana pun** (uji satu sisi), pada α = 0,05 maupun 0,10. Low: p ≥ 0,05 di 5/5 fold pada kedua α.
Medium: signifikan di atas nominal pada 4/5 fold (α = 0,05) dan 5/5 fold (α = 0,10).
Separuh validation (210 pesan): half B pada α = 0,05 masih kurang untuk low (90,86%, p = 0,017).

Biaya rekalibrasi klinisi: over-escalation naik dari 39,33% ke 42,33% (α = 0,05; rata-rata 5-fold
43,07%) dengan miss tetap 5,13%; pada α = 0,10 miss turun 10,26% → 8,97% dan over naik 26,83% → 32,50%.

## 4. Tabel C — Stabilitas lima seed (validation, 420 pesan; rata-rata ± SD)

| α | Low | Medium | High | Miss | Over | Under |
|---|---|---|---|---|---|---|
| 0,05 | 90,60 ± 3,52% | 89,47 ± 2,40% | 93,14 ± 1,95% | 6,86 ± 1,95% | 41,81 ± 2,43% | 2,95 ± 0,96% |
| 0,10 | 79,10 ± 5,01% | 81,05 ± 2,88% | 88,26 ± 1,19% | 11,74 ± 1,19% | 33,81 ± 0,71% | 5,52 ± 0,39% |
| 0,15 | 69,55 ± 5,15% | 77,19 ± 2,97% | 83,26 ± 2,11% | 16,74 ± 2,11% | 28,38 ± 0,85% | 8,10 ± 0,84% |
| 0,20 | 62,99 ± 4,77% | 71,58 ± 3,75% | 78,84 ± 1,34% | 21,16 ± 1,34% | 24,57 ± 0,66% | 10,71 ± 0,48% |

Jumlah kasus (dari 20 kombinasi seed × α):

| Kelas | Tidak signifikan (dua sisi) | Signifikan **di bawah** nominal (satu sisi) |
|---|---|---|
| Low | 3 | 18 |
| Medium | 4 | 16 |
| High | 19 | 2 |

Kasus high yang signifikan di bawah nominal: seed 29 α = 0,05 (90,70%, p₁ = 0,013) dan seed 101
α = 0,15 (79,65%, p₁ = 0,035). Seed 42 (run naskah) termasuk seed yang paling menguntungkan untuk
kelas high (94,77/89,53/84,88/80,23%).

## 5. McNemar eksak (pesan high-risk pada test, escalate vs tidak)

| Perbandingan | Hanya A escalate | Hanya B escalate | p eksak |
|---|---|---|---|
| Minimax α = 0,05 vs argmax | 43 | 0 | 2,3 × 10⁻¹³ |
| Minimax α = 0,10 vs argmax | 31 | 0 | 9,3 × 10⁻¹⁰ |
| Minimax α = 0,05 vs threshold E1 | 0 | 0 | 1,000 |
| Minimax α = 0,10 vs expected cost | 1 | 4 | 0,375 |

## 6. Benchmark CPU pipeline TF-IDF (fresh process, 16 thread)

Cold start 1,49 s · latensi warm p50 2,35 ms · p95 5,61 ms · throughput 670,1 pesan/s (batch 64) ·
peak RSS 432 MB · ukuran model serialisasi 6,24 MB.
Catatan: yang diukur adalah `predict_proba` pipeline TF-IDF + logistic; temperature scaling, kuantil
conformal, dan routing (operasi aritmetika kecil) tidak ikut dihitung waktu.

## 7. Putusan aturan Step 6 (diterapkan mekanis)

| Eksperimen | Aturan | Putusan | Interpretasi jujur (1 baris) |
|---|---|---|---|
| E2 α per kelas | Over turun dengan CI berpasangan tidak mencakup 0 **dan** CI miss mencakup 5,13% | **LULUS** (−4,50 [−6,17; −3,00]; CI miss [2,64; 7,98]) | Penurunan nyata tetapi sama besar dengan threshold E1; jumlah ESCALATE tetap 413, under naik +0,83 poin — bukan “novelty utama”. |
| E1 matched threshold | Laporkan selisih apa pun hasilnya | Threshold **setara** (miss identik, over lebih rendah 4,33 poin, cost tidak berbeda) | Nilai conformal layer = pemilihan cut-off yang berprinsip dan tahan label-shift + tier CLARIFY, **bukan** error yang lebih rendah. |
| E3/R3 rekalibrasi | Coverage low/medium kembali nominal (p ≥ 0,05) | **Low: LULUS** (420: p = 0,87 dan 1,00; 5/5 fold). **Medium: aturan literal TIDAK terpenuhi** karena signifikan *di atas* nominal | Shortfall hilang bila label kalibrasi = label evaluasi (mendukung penjelasan label-shift); medium menjadi konservatif. |
| R1 lima seed | High tidak signifikan di bawah nominal pada ≥ 18 dari 20 kasus | **LULUS, tepat di batas** (18/20 satu sisi; 19/20 dua sisi) | Kontras high vs low/medium stabil, tetapi rata-rata high sedikit di bawah nominal (93,14% pada α = 0,05) → ubah “tetap nominal” menjadi “mendekati nominal”. |
| E4 McNemar | p < 0,05 | **LULUS** (p = 2,3 × 10⁻¹³) | Penurunan miss vs argmax signifikan pada data berpasangan. |

## 8. Hal tak terduga / perlu perhatian penulis

1. **Klaim utama R2 perlu dilunakkan.** Abstrak dan Kesimpulan R2 menulis bahwa coverage high “stayed at its
   nominal level for every tested α”. Itu benar untuk seed 42, tetapi rata-rata lima seed di bawah nominal
   (93,14% / 88,26% / 83,26% / 78,84%) dan 2 dari 20 kasus signifikan di bawah. Rumusan yang didukung:
   “close to nominal and not significantly below it in 18 of 20 seed × α cases, whereas the low and
   medium classes fell significantly short in 18 and 16 cases”.
2. **Threshold yang dituning setara/lebih baik dalam over-escalation** (E1, Figure 5: kurva threshold
   sweep berada di bawah kurva α sweep). Jangan klaim router conformal mengungguli threshold.
3. **E2 tidak mengurangi handover ke manusia** (ESCALATE 413 = 413); penghematan berasal dari CLARIFY → SUPPORT.
   Untuk α_high = 0,10, E2 malah menaikkan over-escalation (30,00% vs 26,83%).
4. **Rekalibrasi klinisi membuat medium over-coverage** dan menaikkan beban (over 42,33%). Dengan 210 pesan
   (half B), low masih kurang — ukuran kalibrasi klinisi berpengaruh.
5. Validation (420) sebelumnya juga dipakai untuk memilih representasi dan α; memakai validation untuk
   kalibrasi E3/R3 tidak menyentuh test, tetapi perlu disebut di naskah.
6. Benchmark E5 hanya mengukur classifier (lihat Bagian 6). Hasil run kedua sedikit berbeda dari run uji
   di scratchpad (p50 2,31 vs 2,35 ms) — variasi timing normal; yang dilaporkan adalah run paket ini.
7. Figure 4 dari skrip asli memakai warna di luar palet; sudah dibuat ulang dengan palet proyek
   (angka identik).
8. `results/metrics.json` (run pilot 22 Sep, MiniLM) tetap harus ditandai *superseded* di repositori.
9. Skrip `r2_extra_experiments.py` dibuat dengan bantuan AI (audit R2), dan `r2_sanity_check.py`,
   `r2_robustness.py`, `r2_figures.py` ditulis oleh agen AI (Claude Code) pada run ini. AI Use Statement dan
   Author Contributions di naskah harus menyebut ini bila hasilnya dipakai.

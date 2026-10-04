from __future__ import annotations

import json
from pathlib import Path


REFERENCES = """[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proc. ICML, vol. 70, pp. 1321–1330, 2017. https://proceedings.mlr.press/v70/guo17a.html

[2] Y. Romano, M. Sesia, and E. J. Candès, “Classification with Valid and Adaptive Coverage,” in Advances in Neural Information Processing Systems, vol. 33, pp. 3581–3591, 2020. https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html

[3] S. Bates, A. N. Angelopoulos, L. Lei, J. Malik, and M. I. Jordan, “Distribution-Free, Risk-Controlling Prediction Sets,” Journal of the ACM, vol. 68, no. 6, Art. 43, 2021. https://doi.org/10.1145/3478535

[4] I. Gibbs and E. J. Candès, “Adaptive Conformal Inference Under Distribution Shift,” in Advances in Neural Information Processing Systems, vol. 34, pp. 1660–1672, 2021.

[5] M. Minderer et al., “Revisiting the Calibration of Modern Neural Networks,” in Advances in Neural Information Processing Systems, vol. 34, pp. 15682–15694, 2021.

[6] A. N. Angelopoulos and S. Bates, “Conformal Prediction: A Gentle Introduction,” Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494–591, 2023. https://doi.org/10.1561/2200000101

[7] R. F. Barber, E. J. Candès, A. Ramdas, and R. J. Tibshirani, “Conformal Prediction Beyond Exchangeability,” Annals of Statistics, vol. 51, no. 2, pp. 816–845, 2023. https://doi.org/10.1214/23-AOS2276

[8] Z. Gu and M. Hopkins, “On the Evaluation of Neural Selective Prediction Methods for Natural Language Processing,” in Proc. ACL, pp. 7888–7899, 2023. https://doi.org/10.18653/v1/2023.acl-long.437

[9] A. Pugnana and S. Ruggieri, “A Model-Agnostic Heuristics for Selective Classification,” in Proc. AAAI, vol. 37, no. 8, pp. 9461–9469, 2023. https://doi.org/10.1609/aaai.v37i8.26133

[10] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, “Conformal Risk Control,” in Proc. ICLR, 2024. https://openreview.net/forum?id=33XGfHLtZg

[11] Y. Zhou and M. Sesia, “Conformal Classification with Equalized Coverage for Adaptively Selected Groups,” in Advances in Neural Information Processing Systems, vol. 37, 2024. https://doi.org/10.52202/079017-3454

[12] G. Byun, R. Lipschutz, S. T. Minton, A. Powers, and J. D. Choi, “CRADLE Bench: A Clinician-Annotated Benchmark for Multi-Faceted Mental Health Crisis and Safety Risk Detection,” in Proc. EACL, pp. 1572–1590, 2026. https://doi.org/10.18653/v1/2026.eacl-long.73

[13] M. F. Azmi, M. D. Al Kautsar, A. F. Wicaksono, and F. Koto, “IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages,” in Proc. EMNLP, pp. 9135–9166, 2025. https://doi.org/10.18653/v1/2025.emnlp-main.465

[14] V. Weilnhammer et al., “A Clinically Validated Framework for Auditing AI Chatbot Behavior in Mental Health Interactions,” Nature Medicine, 2026. https://doi.org/10.1038/s41591-026-04577-2

[15] A. Arnaiz-Rodriguez et al., “Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models,” JMIR Mental Health, vol. 13, Art. e88435, 2026. https://doi.org/10.2196/88435

[16] J. Qiu et al., “EmoAgent: Assessing and Safeguarding Human-AI Interaction for Mental Health Safety,” in Proc. EMNLP, pp. 11741–11756, 2025. https://doi.org/10.18653/v1/2025.emnlp-main.594

[17] C. T. Chang et al., “Red Teaming ChatGPT in Medicine to Yield Real-World Insights on Model Behavior,” npj Digital Medicine, vol. 8, Art. 149, 2025. https://doi.org/10.1038/s41746-025-01542-0

[18] W. Pichowicz, M. Kotas, and P. Piotrowski, “Performance of Mental Health Chatbot Agents in Detecting and Managing Suicidal Ideation,” Scientific Reports, vol. 15, Art. 31652, 2025. https://doi.org/10.1038/s41598-025-17242-4

[19] A. Sharma, I. W. Lin, A. S. Miner, D. C. Atkins, and T. Althoff, “Human–AI Collaboration Enables More Empathic Conversations in Text-Based Peer-to-Peer Mental Health Support,” Nature Machine Intelligence, vol. 5, pp. 46–57, 2023. https://doi.org/10.1038/s42256-022-00593-2

[20] J. W. Ayers et al., “Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum,” JAMA Internal Medicine, vol. 183, no. 6, pp. 589–596, 2023. https://doi.org/10.1001/jamainternmed.2023.1838

[21] D. Tawakalna and J. Zeniarja, “Sistem Chatbot Kesehatan Mental Berbasis LLM dengan Deteksi Emosi dan Retrieval Augmented Generation,” RABIT, vol. 11, no. 1, pp. 1398–1412, 2026. https://doi.org/10.36341/rabit.v11i1.7347

[22] N. Reimers and I. Gurevych, “Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation,” in Proc. EMNLP, pp. 4512–4525, 2020. https://doi.org/10.18653/v1/2020.emnlp-main.365

[23] F. Koto, J. H. Lau, and T. Baldwin, “IndoBERTweet: A Pretrained Language Model for Indonesian Twitter,” in Proc. EMNLP, pp. 10660–10668, 2021. https://doi.org/10.18653/v1/2021.emnlp-main.833

[24] S. Cahyawijaya et al., “NusaCrowd: Open Source Initiative for Indonesian NLP Resources,” in Findings of ACL, pp. 13745–13818, 2023. https://doi.org/10.18653/v1/2023.findings-acl.868

[25] G. I. Winata et al., “NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages,” in Proc. EACL, pp. 815–834, 2023. https://doi.org/10.18653/v1/2023.eacl-main.57

[26] B. Wilie et al., “IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding,” in Proc. AACL-IJCNLP, pp. 843–857, 2020. https://doi.org/10.18653/v1/2020.aacl-main.85

[27] B. Vasey et al., “Reporting Guideline for the Early-Stage Clinical Evaluation of Decision Support Systems Driven by Artificial Intelligence: DECIDE-AI,” Nature Medicine, vol. 28, pp. 924–933, 2022. https://doi.org/10.1038/s41591-022-01772-9

[28] G. S. Collins et al., “TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models,” BMJ, vol. 385, Art. e078378, 2024. https://doi.org/10.1136/bmj-2023-078378

[29] K. Lekadir et al., “FUTURE-AI: International Consensus Guideline for Trustworthy and Deployable Artificial Intelligence in Healthcare,” BMJ, vol. 388, Art. e081554, 2025. https://doi.org/10.1136/bmj-2024-081554
"""


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _ci(item: dict, percent: bool = False) -> str:
    if percent:
        return f"[{_pct(item['ci95_low'])}; {_pct(item['ci95_high'])}]"
    return f"[{item['ci95_low']:.4f}; {item['ci95_high']:.4f}]"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError("extended_metrics.json belum tersedia.")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "final_test_primary_alpha_0.10",
        "final_test_safety_alpha_0.05",
        "tfidf_five_seed_summary",
        "bootstrap_confidence_intervals",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"Artefak extended tidak lengkap: {sorted(missing)}")
    return data


def _common(data: dict) -> dict:
    return {
        "primary": data["final_test_primary_alpha_0.10"],
        "safety": data["final_test_safety_alpha_0.05"],
        "seeds": data["tfidf_five_seed_summary"],
        "boot": data["bootstrap_confidence_intervals"],
        "indo": data["indonesian_compatible_evaluation"],
        "temp": data["temporal_sim_vail"],
        "eff": data["efficiency_cpu"],
        "leak": data["leakage_audit"],
        "rows": data["protocol"].get("dataset_rows", {}),
        "sens": data["sensitivity"],
        "abl": data["ablation"],
    }


def _paper_id(data: dict) -> str:
    c = _common(data)
    p, s, b = c["primary"], c["safety"], c["boot"]
    r = c["rows"]
    e0, e1 = c["eff"]["fp32"], c["eff"]["dynamic_int8"]
    langs = c["indo"]["by_language"]
    return f"""# GEMATI-CARE: Routing Konformal Sensitif Biaya untuk Deteksi Risiko Krisis

[Nama Penulis 1], [Nama Penulis 2]*

[Afiliasi lengkap penulis]

*Email corresponding author: [lengkapi]

## Abstrak

Kesalahan deteksi risiko pada chatbot dukungan kesehatan mental memiliki biaya yang tidak simetris, sedangkan klasifikasi argmax memaksa satu keputusan ketika model belum pasti. Penelitian ini mengusulkan GEMATI-CARE, yaitu routing tindakan sensitif biaya di atas himpunan prediksi konformal per kelas. Protokol memisahkan data pelatihan, temperature scaling, dan conformal calibration agar tidak terjadi pemakaian ulang label kalibrasi. Model CPU dipilih pada official validation antara TF-IDF dan MiniLM, kemudian dievaluasi sekali pada official test CRADLE. TF-IDF terpilih dan memperoleh macro-F1 {p['classification']['macro_f1']:.4f}. Mode safety pada alpha 0,05 mencapai coverage {_pct(s['conformal']['coverage'])}, high-risk miss {_pct(s['routing']['high_risk_miss_rate'])}, dan under-escalation {_pct(s['routing']['under_escalation_rate'])}, dengan konsekuensi over-escalation {_pct(s['routing']['over_escalation_rate'])}. CI bootstrap 95% macro-F1 adalah {_ci(b['macro_f1'])}. Lima repeated stratified splits menghasilkan macro-F1 validation {c['seeds']['macro_f1']['mean']:.4f} ± {c['seeds']['macro_f1']['std']:.4f}. Audit leakage, variasi label mapping, SIM-VAIL, dan IndoSafety Eval2 mengungkap batas generalisasi, khususnya recall bahasa daerah yang rendah. Sistem berjalan pada CPU; kuantisasi dinamis tidak konsisten mempercepat throughput. GEMATI-CARE ditujukan sebagai prototipe routing offline, bukan diagnosis, terapi, atau triase klinis.

Kata kunci: conformal prediction; cost-sensitive routing; deteksi risiko; kesehatan mental; selective prediction

## 1. Pendahuluan

Chatbot kesehatan mental dapat menghasilkan respons yang dipersepsikan empatik, tetapi empati tidak identik dengan keselamatan [15], [17]–[20]. Dalam percakapan berisiko, kesalahan berupa kegagalan eskalasi dapat memiliki konsekuensi lebih berat daripada eskalasi yang terlalu berhati-hati. Sebaliknya, sistem yang selalu mengeskalasi menjadi tidak berguna dan dapat mengikis kepercayaan. Evaluasi berbasis accuracy atau macro-F1 saja tidak merepresentasikan kompromi tersebut.

Penelitian GEMATI terdahulu telah memadukan LLM, deteksi emosi, dan retrieval-augmented generation [21]. Mengulang arsitektur generatif tersebut bukan novelty yang memadai. Artikel ini memusatkan kontribusi pada lapisan keputusan keselamatan yang ringan, terukur, dan dapat diaudit. Conformal prediction menyediakan himpunan label yang masih masuk akal [2], [6], tetapi coverage marginal dapat menyembunyikan ketimpangan antarkelas [11]. GEMATI-CARE memakai class-conditional atau Mondrian conformal set, lalu memilih tindakan yang meminimalkan biaya terburuk pada himpunan tersebut. Istilah yang digunakan adalah cost-sensitive routing over class-conditional conformal sets, bukan conformal risk control formal [3], [10].

Kontribusi penelitian ini adalah protokol kalibrasi dua tahap yang disjoint; routing robust dengan biaya asimetris dan tie-breaking deterministik; evaluasi lima repeated stratified splits, sensitivity, ablation, bootstrap multi-metrik, serta leakage audit; dan benchmark CPU terisolasi. Pertanyaan risetnya adalah apakah routing berbasis himpunan mengurangi high-risk miss dibanding argmax, bagaimana alpha mengubah safety–utility trade-off, apakah hasil stabil terhadap split dan pemetaan label, serta apakah optimasi INT8 layak untuk deployment CPU.

## 2. Metode Penelitian

### 2.1 Batas Produk dan Dataset

GEMATI-CARE merupakan middleware penelitian untuk memilih tindakan SUPPORT, CLARIFY, atau ESCALATE. Sistem tidak menentukan diagnosis, tidak memberikan terapi, dan tidak menggantikan profesional. CRADLE Bench [12] menyediakan label krisis yang dipetakan secara transparan ke low, medium, dan high sebagai kelas rekayasa. Pemetaan bukan skor klinis. Pool publik berisi {r.get('cradle_public_train_pool', 4180)} item; seed 42 menghasilkan {r.get('model_training', 2926)} train, {r.get('probability_calibration', 418)} probability-calibration, dan {r.get('conformal_calibration', 836)} conformal-calibration. Official validation berisi {r.get('official_validation', 420)} item dan official test {r.get('official_test', 600)} item.

IndoSafety [13] digunakan hanya untuk auxiliary representation probe dengan taksonomi mental-health harm biner, bukan untuk menguji severity classifier CRADLE. Sebanyak 2.014 Eval1 yang identik dengan data latih dikeluarkan; metrik utama memakai {c['indo']['test_rows']} prompt Eval2 paralel. SIM-VAIL [14] memuat {r.get('sim_vail_vulnerable_conversations', 810)} percakapan kondisi vulnerable dan {r.get('sim_vail_control_conversations', 144)} kontrol sintetis. Karena tidak memiliki gold risk per turn, analisisnya dibatasi sebagai persistence audit.

### 2.2 Representasi dan Pemilihan Model

Kandidat pertama memakai TF-IDF kata unigram–bigram dan karakter 3–5 gram dengan regresi logistik one-vs-rest berbobot kelas. Kandidat kedua memakai frozen paraphrase-multilingual-MiniLM-L12-v2 [22] dan kepala logistik. Pemilihan dilakukan hanya pada official validation dengan macro-F1 sebagai kriteria. TF-IDF memperoleh {data['model_selection']['tfidf_macro_f1']:.4f} dan MiniLM {data['model_selection']['minilm_macro_f1']:.4f}; karena itu TF-IDF ditetapkan sebagai model utama. Keputusan ini mengutamakan bukti eksperimen dan efisiensi CPU, bukan kompleksitas model.

### 2.3 Kalibrasi dan Himpunan Konformal

Probabilitas dikalibrasi menggunakan satu temperatur yang meminimalkan negative log-likelihood [1]. Probability-calibration dan conformal-calibration tidak berbagi observasi. Untuk kelas k, nonconformity score adalah 1−p_k(x). Kuantil finite-sample dihitung terpisah pada item berlabel k. Himpunan C(x) memuat kelas yang skornya tidak melampaui kuantil kelas. Evaluasi mencakup alpha 0,05; 0,10; 0,15; dan 0,20. Alpha 0,05 dipilih sebagai safety operating point karena menjadi satu-satunya grid dengan coverage validation minimal 90%; alpha 0,10 dipertahankan sebagai utility operating point.

### 2.4 Routing Sensitif Biaya

Untuk tindakan a dan kelas k, L(a,k) adalah biaya rekayasa. Router memilih a*=argmin_a max_{{k∈C(x)}} L(a,k). Matriks utama memberi penalti terbesar pada SUPPORT ketika kelas sebenarnya high. Tie diselesaikan secara deterministik dengan prioritas konservatif. Baseline terdiri atas argmax, expected-cost dari probabilitas titik, global conformal, dan Mondrian minimax. Expected cost antarmatriks tidak dibandingkan lintas skala; analisis menggunakan distribusi tindakan serta error eskalasi.

Gambar 1 memperlihatkan jalur keputusan utama pada CPU dan menandai temporal persistence sebagai audit opsional, bukan bagian dari klaim coverage.

### 2.5 Protokol Evaluasi

Lima seed 11, 29, 42, 71, dan 101 digunakan sebagai repeated stratified development splits, bukan grouped splits. Sensitivity dan ablation seluruhnya memakai official validation; official test digunakan untuk evaluasi final. Metrik meliputi macro-F1, balanced accuracy, ECE, Brier, NLL, coverage per kelas, average set size, expected cost, under-escalation, over-escalation, dan high-risk miss. CI 95% dihitung dengan bootstrap 1.000 replikasi untuk metrik final. Audit leakage memeriksa ID, normalized exact match, dan semantic near-duplicate pada seluruh pasangan split. Efisiensi FP32 dan dynamic INT8 diukur dalam proses baru yang terpisah dengan 50 pengulangan single inference dan 12 pengulangan batch.

## 3. Hasil dan Pembahasan

### 3.1 Stabilitas dan Hasil Test

Lima repeated splits TF-IDF memberikan macro-F1 {c['seeds']['macro_f1']['mean']:.4f} ± {c['seeds']['macro_f1']['std']:.4f}, t-interval 95% untuk rerata [{c['seeds']['macro_f1']['mean_t95_low']:.4f}; {c['seeds']['macro_f1']['mean_t95_high']:.4f}]. Rata-rata high-risk miss adalah {_pct(c['seeds']['high_risk_miss']['mean'])}. Pada official test, macro-F1 mencapai {p['classification']['macro_f1']:.4f}, balanced accuracy {p['classification']['balanced_accuracy']:.4f}, ECE {p['classification']['ece']:.4f}, dan NLL {p['classification']['nll']:.4f}. Recall low, medium, dan high masing-masing {_pct(p['classification']['per_class']['low']['recall'])}, {_pct(p['classification']['per_class']['medium']['recall'])}, dan {_pct(p['classification']['per_class']['high']['recall'])}.

Tabel 1 merangkum kompromi antara coverage, kesalahan eskalasi, dan expected cost pada dua operating point serta baseline.

| Konfigurasi | Coverage | Ukuran set | High-risk miss | Under | Over | Expected cost |
|---|---:|---:|---:|---:|---:|---:|
| Utility, alpha 0,10 | {_pct(p['mondrian_metrics']['coverage'])} | {p['mondrian_metrics']['average_set_size']:.3f} | {_pct(p['routing_minimax']['high_risk_miss_rate'])} | {_pct(p['routing_minimax']['under_escalation_rate'])} | {_pct(p['routing_minimax']['over_escalation_rate'])} | {p['routing_minimax']['expected_cost']:.4f} |
| Safety, alpha 0,05 | {_pct(s['conformal']['coverage'])} | {s['conformal']['average_set_size']:.3f} | {_pct(s['routing']['high_risk_miss_rate'])} | {_pct(s['routing']['under_escalation_rate'])} | {_pct(s['routing']['over_escalation_rate'])} | {s['routing']['expected_cost']:.4f} |
| Argmax | NA | 1,000 | {_pct(p['routing_argmax']['high_risk_miss_rate'])} | {_pct(p['routing_argmax']['under_escalation_rate'])} | {_pct(p['routing_argmax']['over_escalation_rate'])} | {p['routing_argmax']['expected_cost']:.4f} |
| Expected-cost | NA | 1,000 | {_pct(p['routing_expected']['high_risk_miss_rate'])} | {_pct(p['routing_expected']['under_escalation_rate'])} | {_pct(p['routing_expected']['over_escalation_rate'])} | {p['routing_expected']['expected_cost']:.4f} |

Mode safety menurunkan high-risk miss dibanding argmax, tetapi meningkatkan over-escalation. Coverage per kelas pada mode safety adalah low {_pct(s['conformal']['per_class_coverage']['low'])}, medium {_pct(s['conformal']['per_class_coverage']['medium'])}, dan high {_pct(s['conformal']['per_class_coverage']['high'])}. Pada alpha 0,10, coverage test hanya {_pct(p['mondrian_metrics']['coverage'])}; karena itu artikel tidak mengklaim target nominal tercapai pada utility mode. Hasil ini konsisten dengan keterbatasan conformal di bawah distribution shift [4], [7].

### 3.2 Statistik, Sensitivity, dan Ablation

CI bootstrap 95% mode safety adalah macro-F1 {_ci(b['macro_f1'])}, expected cost {_ci(b['expected_cost'])}, under-escalation {_ci(b['under_escalation_rate'], True)}, dan over-escalation {_ci(b['over_escalation_rate'], True)}. CI coverage low, medium, dan high berturut-turut {_ci(b['coverage_low'], True)}, {_ci(b['coverage_medium'], True)}, dan {_ci(b['coverage_high'], True)}.

Peningkatan alpha dari 0,05 ke 0,20 pada validation menurunkan coverage dari {_pct(c['sens']['alpha']['0.05']['conformal']['coverage'])} menjadi {_pct(c['sens']['alpha']['0.2']['conformal']['coverage'])}, menurunkan over-escalation dari {_pct(c['sens']['alpha']['0.05']['routing']['over_escalation_rate'])} menjadi {_pct(c['sens']['alpha']['0.2']['routing']['over_escalation_rate'])}, tetapi menaikkan high-risk miss dari {_pct(c['sens']['alpha']['0.05']['routing']['high_risk_miss_rate'])} menjadi {_pct(c['sens']['alpha']['0.2']['routing']['high_risk_miss_rate'])}. Hasil ini memperlihatkan frontier safety–utility, bukan satu konfigurasi yang dominan.

Tanpa class weighting, macro-F1 validation turun dari {c['abl']['full']['classification']['macro_f1']:.4f} menjadi {c['abl']['without_class_weight']['classification']['macro_f1']:.4f}. Tanpa temperature scaling, label argmax dan macro-F1 tidak berubah, tetapi expected cost routing memburuk dari {c['abl']['full']['routing_minimax']['expected_cost']:.4f} menjadi {c['abl']['without_calibration']['routing_minimax']['expected_cost']:.4f}. Expected-cost routing memberi cost terendah pada validation, sedangkan minimax menghasilkan over-escalation yang lebih rendah; pemilihan kebijakan bergantung pada toleransi risiko.

### 3.3 Leakage dan Robustness Audit

ID CRADLE bersifat lokal dan berulang antar-split, tetapi normalized exact-text overlap pada seluruh pasangan split adalah nol. Dua item test memiliki kemiripan cosine minimal 0,90 terhadap train/calibration. Setelah keduanya dihapus, macro-F1 {c['leak']['sensitivity_excluding_flagged_test_rows']['classification']['macro_f1']:.4f} dan high-risk miss {_pct(c['leak']['sensitivity_excluding_flagged_test_rows']['routing']['high_risk_miss_rate'])}; kesimpulan tidak berubah secara material. Sensitivity pemetaan manual menunjukkan bahwa definisi narrow, current, dan broad mengubah macro-F1 dan miss rate, sehingga mapping wajib dipandang sebagai asumsi kebijakan.

Auxiliary probe IndoSafety Eval2 memperoleh macro-F1 {c['indo']['macro_f1']:.4f}, precision positif {c['indo']['positive_precision']:.4f}, recall positif {c['indo']['positive_recall']:.4f}, dan F1 positif {c['indo']['positive_f1']:.4f}. Recall positif per bahasa adalah formal {_pct(langs['indonesian-formal']['mental_recall'])}, kolokial {_pct(langs['colloquial']['mental_recall'])}, Jawa {_pct(langs['java']['mental_recall'])}, Minangkabau {_pct(langs['minangkabau']['mental_recall'])}, dan Sunda {_pct(langs['sunda']['mental_recall'])}. Kesenjangan ini menolak klaim robustness Indonesia dan menunjukkan kebutuhan data berlabel risiko yang kompatibel.

Pada SIM-VAIL, temporal memory menaikkan proporsi turn ESCALATE dan mengurangi de-escalation, tetapi any-escalation kontrol tetap {_pct(c['temp']['summary']['control']['temporal_any_escalation'])}. Karena kontrol sering tereskalasi dan tidak ada gold label per turn, temporal memory tidak diklaim meningkatkan deteksi atau keselamatan. Istilah temporal sengaja tidak dimasukkan dalam judul.

### 3.4 Efisiensi CPU

FP32 menghasilkan median single-inference {e0['latency']['warm_single_ms']['p50']:.2f} ms, p95 {e0['latency']['warm_single_ms']['p95']:.2f} ms, dan throughput rata-rata {e0['throughput_texts_per_second']['mean']:.2f} teks/detik. Dynamic INT8 menghasilkan median {e1['latency']['warm_single_ms']['p50']:.2f} ms, p95 {e1['latency']['warm_single_ms']['p95']:.2f} ms, dan throughput {e1['throughput_texts_per_second']['mean']:.2f} teks/detik. Ukuran state_dict turun dari {e0['serialized_encoder_state_dict_bytes']/1e6:.2f} MB menjadi {e1['serialized_encoder_state_dict_bytes']/1e6:.2f} MB, tetapi peak RSS dan throughput tidak menunjukkan keuntungan konsisten. Karena model utama TF-IDF lebih ringan, CPU direkomendasikan. Dual MIG A100 40 GB tidak diperlukan dan bukan novelty; perangkat hanya masuk akal untuk eksperimen guru–murid terpisah.

## 4. Kesimpulan

GEMATI-CARE menunjukkan bahwa novelty yang relevan bukan ukuran model, melainkan pengambilan keputusan yang memisahkan diskriminasi, ketidakpastian, dan biaya tindakan. Model TF-IDF yang dipilih pada validation mencapai macro-F1 test {p['classification']['macro_f1']:.4f}. Cost-sensitive routing pada alpha 0,05 menurunkan high-risk miss menjadi {_pct(s['routing']['high_risk_miss_rate'])}, tetapi over-escalation {_pct(s['routing']['over_escalation_rate'])} menegaskan bahwa peningkatan kehati-hatian memiliki biaya utilitas. Hasil lima seed, bootstrap, leakage exclusion, dan mapping sensitivity mendukung kestabilan kesimpulan relatif, bukan klaim keselamatan klinis. Audit Indonesia dan SIM-VAIL justru mengidentifikasi kelemahan generalisasi. Implementasi CPU memadai; GPU DGX tidak dibutuhkan. Studi lanjutan harus memperoleh label risiko Indonesia yang kompatibel, validasi profesional, dan persetujuan etik sebelum evaluasi manusia.

## Ucapan Terima Kasih

[Lengkapi sumber pendanaan, kontribusi non-penulis, konflik kepentingan, dan redaksi keputusan etik institusi. Dataset publik tidak otomatis menentukan status exempt.]

## Daftar Pustaka

{REFERENCES}
"""


def _paper_en(data: dict) -> str:
    c = _common(data)
    p, s, b = c["primary"], c["safety"], c["boot"]
    langs = c["indo"]["by_language"]
    e0, e1 = c["eff"]["fp32"], c["eff"]["dynamic_int8"]
    return f"""# GEMATI-CARE: Cost-Sensitive Conformal Routing for Crisis-Risk Detection

[Author 1], [Author 2]*

[Complete affiliations]

*Corresponding author email: [complete]

## Abstract

Errors in mental-health support chatbots have asymmetric consequences, while argmax classification forces a single decision under uncertainty. This study proposes GEMATI-CARE, a cost-sensitive action router over class-conditional conformal prediction sets. Training, temperature scaling, and conformal calibration use mutually disjoint observations. A CPU classifier was selected between TF-IDF and MiniLM using only the official CRADLE validation set and evaluated on the official test set. TF-IDF was selected and achieved test macro-F1 {p['classification']['macro_f1']:.4f}. At the safety operating point alpha 0.05, coverage was {_pct(s['conformal']['coverage'])}, high-risk miss was {_pct(s['routing']['high_risk_miss_rate'])}, and under-escalation was {_pct(s['routing']['under_escalation_rate'])}, at the cost of {_pct(s['routing']['over_escalation_rate'])} over-escalation. The 95% bootstrap interval for macro-F1 was {_ci(b['macro_f1'])}. Five repeated stratified splits yielded validation macro-F1 {c['seeds']['macro_f1']['mean']:.4f} ± {c['seeds']['macro_f1']['std']:.4f}. Leakage, mapping, SIM-VAIL, and leak-free IndoSafety Eval2 audits revealed substantial generalization limits, particularly for regional languages. The system is CPU-feasible, and dynamic INT8 did not consistently improve throughput. GEMATI-CARE is an offline routing prototype, not a diagnostic, therapeutic, or clinical-triage system.

Keywords: conformal prediction; cost-sensitive routing; crisis-risk detection; mental health; selective prediction

## 1. Introduction

Mental-health chatbots can appear empathic without being safe [15], [17]–[20]. Missing high-risk content may be more consequential than cautious escalation, but excessive escalation also undermines utility. Accuracy and macro-F1 alone cannot express this asymmetric trade-off. Prior GEMATI work addressed response generation, emotion detection, and retrieval [21]; the present work instead isolates a falsifiable safety-decision layer.

Conformal prediction represents uncertainty as a set of plausible labels [2], [6], while class-conditional construction makes class-wise coverage visible [11]. GEMATI-CARE maps a Mondrian conformal set to SUPPORT, CLARIFY, or ESCALATE by minimizing worst-case engineering cost. We deliberately call this cost-sensitive routing over class-conditional conformal sets rather than formal conformal risk control [3], [10]. Contributions comprise disjoint probability/conformal calibration; deterministic robust routing under asymmetric costs; five-split, sensitivity, ablation, bootstrap, and leakage analyses; and isolated CPU efficiency measurements.

## 2. Methods

### 2.1 Intended Use and Data

GEMATI-CARE is research middleware and is not diagnosis, therapy, or clinical triage. CRADLE [12] source labels were transparently mapped to low, medium, and high engineering classes. The public training pool was split into 70% model training, 10% probability calibration, and 20% conformal calibration. Official validation was reserved for model and operating-point selection; official test was used for final evaluation.

IndoSafety [13] was used only for an auxiliary binary taxonomy representation probe. Because 2,014 Eval1 rows exactly overlapped its training file, they were excluded and only 2,500 parallel Eval2 prompts were reported. SIM-VAIL [14] supplied synthetic multi-turn conditions for a persistence audit, not per-turn clinical ground truth.

### 2.2 Classifier and Conformal Routing

Word unigram–bigram and character 3–5-gram TF-IDF with class-weighted one-vs-rest logistic regression was compared with frozen multilingual MiniLM [22]. Validation macro-F1 selected TF-IDF ({data['model_selection']['tfidf_macro_f1']:.4f}) over MiniLM ({data['model_selection']['minilm_macro_f1']:.4f}). Temperature scaling minimized negative log-likelihood on a dedicated subset [1]. A separate subset estimated finite-sample class-conditional nonconformity quantiles using 1−p_y(x). The router selected the action minimizing maximum cost across labels in the prediction set. Alpha values 0.05, 0.10, 0.15, and 0.20, three cost matrices, calibration-set sizes, thresholds, and tie policies were evaluated on validation.

Figure 1 depicts the main CPU decision path and marks temporal persistence as an optional audit rather than part of the coverage claim.

### 2.3 Evaluation

Five repeated stratified splits used seeds 11, 29, 42, 71, and 101. Metrics included macro-F1, balanced accuracy, ECE, Brier score, NLL, class-wise coverage, set size, expected cost, under-/over-escalation, and high-risk miss. A 1,000-repetition bootstrap estimated final uncertainty. Leakage audits covered IDs, normalized exact matches, and MiniLM cosine near-duplicates. FP32 and dynamic INT8 were benchmarked in separate fresh processes using repeated single and batch inference.

## 3. Results and Discussion

### 3.1 Final Performance and Safety–Utility Trade-off

Across five validation splits, TF-IDF macro-F1 was {c['seeds']['macro_f1']['mean']:.4f} ± {c['seeds']['macro_f1']['std']:.4f}. On official test, macro-F1 was {p['classification']['macro_f1']:.4f}, balanced accuracy {p['classification']['balanced_accuracy']:.4f}, ECE {p['classification']['ece']:.4f}, and NLL {p['classification']['nll']:.4f}.

Table 1 summarizes coverage, escalation errors, and expected cost for the two operating points and the argmax baseline.

| Operating point | Coverage | Set size | High-risk miss | Under | Over | Expected cost |
|---|---:|---:|---:|---:|---:|---:|
| Utility, alpha 0.10 | {_pct(p['mondrian_metrics']['coverage'])} | {p['mondrian_metrics']['average_set_size']:.3f} | {_pct(p['routing_minimax']['high_risk_miss_rate'])} | {_pct(p['routing_minimax']['under_escalation_rate'])} | {_pct(p['routing_minimax']['over_escalation_rate'])} | {p['routing_minimax']['expected_cost']:.4f} |
| Safety, alpha 0.05 | {_pct(s['conformal']['coverage'])} | {s['conformal']['average_set_size']:.3f} | {_pct(s['routing']['high_risk_miss_rate'])} | {_pct(s['routing']['under_escalation_rate'])} | {_pct(s['routing']['over_escalation_rate'])} | {s['routing']['expected_cost']:.4f} |
| Argmax | NA | 1.000 | {_pct(p['routing_argmax']['high_risk_miss_rate'])} | {_pct(p['routing_argmax']['under_escalation_rate'])} | {_pct(p['routing_argmax']['over_escalation_rate'])} | {p['routing_argmax']['expected_cost']:.4f} |

Alpha 0.10 under-covered empirically and must not be presented as reaching the 90% nominal target. Alpha 0.05 achieved class coverage of {_pct(s['conformal']['per_class_coverage']['low'])}, {_pct(s['conformal']['per_class_coverage']['medium'])}, and {_pct(s['conformal']['per_class_coverage']['high'])} for low, medium, and high, respectively, but produced substantial over-escalation. This is a safety–utility frontier, not an unqualified improvement.

### 3.2 Statistical and Robustness Analyses

Safety-mode bootstrap intervals were macro-F1 {_ci(b['macro_f1'])}, expected cost {_ci(b['expected_cost'])}, under-escalation {_ci(b['under_escalation_rate'], True)}, and over-escalation {_ci(b['over_escalation_rate'], True)}. Removing two test near-duplicates changed macro-F1 to {c['leak']['sensitivity_excluding_flagged_test_rows']['classification']['macro_f1']:.4f} without changing the high-risk-miss conclusion. Label-mapping variants changed both class balance and outcomes, demonstrating policy sensitivity.

The leak-free IndoSafety Eval2 probe achieved macro-F1 {c['indo']['macro_f1']:.4f} and positive recall {_pct(c['indo']['positive_recall'])}. Positive recall was {_pct(langs['indonesian-formal']['mental_recall'])} for formal Indonesian, {_pct(langs['colloquial']['mental_recall'])} for colloquial Indonesian, {_pct(langs['java']['mental_recall'])} for Javanese, {_pct(langs['minangkabau']['mental_recall'])} for Minangkabau, and {_pct(langs['sunda']['mental_recall'])} for Sundanese. These gaps contradict any claim of Indonesian robustness. SIM-VAIL controls also showed {_pct(c['temp']['summary']['control']['temporal_any_escalation'])} temporal any-escalation; temporal memory is therefore reported only as persistence behavior and omitted from the title.

### 3.3 CPU Efficiency

FP32 median and p95 single latency were {e0['latency']['warm_single_ms']['p50']:.2f} and {e0['latency']['warm_single_ms']['p95']:.2f} ms, with {e0['throughput_texts_per_second']['mean']:.2f} texts/s. INT8 values were {e1['latency']['warm_single_ms']['p50']:.2f} ms, {e1['latency']['warm_single_ms']['p95']:.2f} ms, and {e1['throughput_texts_per_second']['mean']:.2f} texts/s. Serialized encoder size decreased from {e0['serialized_encoder_state_dict_bytes']/1e6:.2f} to {e1['serialized_encoder_state_dict_bytes']/1e6:.2f} MB, but memory and throughput gains were not consistent. CPU is recommended; dual A100 MIG is unnecessary for the reported contribution.

## 4. Conclusion

GEMATI-CARE separates risk discrimination, uncertainty representation, and response action. The validation-selected TF-IDF classifier achieved test macro-F1 {p['classification']['macro_f1']:.4f}. Safety-mode conformal routing reduced high-risk miss to {_pct(s['routing']['high_risk_miss_rate'])} while increasing over-escalation to {_pct(s['routing']['over_escalation_rate'])}. Repeated splits, bootstrap intervals, leakage exclusion, and mapping sensitivity support the relative conclusion, but not clinical safety. Indonesian and temporal audits reveal major generalization limitations. The reported method is CPU-feasible and does not require DGX hardware. Future work requires compatible Indonesian crisis labels, professional validation, and ethics approval before any human-facing study.

## Acknowledgements

[Complete funding, conflicts of interest, non-author contributions, and the institutional ethics determination. Public data alone do not establish exemption.]

## References

{REFERENCES}
"""


def generate_papers(metrics_path: Path, output_dir: Path) -> list[Path]:
    data = _load(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "DRAFT_PAPER_RESTI_ID.md", output_dir / "DRAFT_PAPER_RESTI_EN.md"]
    paths[0].write_text(_paper_id(data), encoding="utf-8")
    paths[1].write_text(_paper_en(data), encoding="utf-8")
    return paths

# GEMATI-CARE

Uncertainty-aware conformal routing for mental-health crisis escalation.

This repository accompanies the manuscript **"GEMATI-CARE: Uncertainty-Aware Conformal Routing Reduces Missed Mental-Health Crisis Escalations"**, submitted to *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)*. It contains the source code, configuration, trained model bundle, machine-readable results, and figure scripts behind every number in the paper.

Repository: https://github.com/juntazen/GEMATI-CARE

> **Scope.** GEMATI-CARE is an offline research prototype. It does not diagnose, treat, or triage people, and it must not be deployed in a human-facing service without clinical governance and ethics approval.

## What the method does

1. A word + character TF-IDF logistic-regression classifier estimates low / medium / high routing risk.
2. Temperature scaling calibrates the probabilities on a dedicated probability-calibration subset (n = 418).
3. Class-conditional (Mondrian) conformal prediction builds a prediction set C(x) on a separate conformal-calibration subset (n = 836).
4. A minimax router chooses SUPPORT, CLARIFY, or ESCALATE by minimizing the worst-case engineering cost over C(x) (`configs/experiment.json`, `cost_matrix`).

## Main result (official CRADLE Bench test set, n = 600)

| Policy | High-risk miss | Under-escalation | Over-escalation |
|---|---|---|---|
| Argmax | 23.50% | 12.33% | 14.17% |
| GEMATI-CARE, alpha = 0.05 | 5.13% | 2.17% | 39.33% |

Test macro-F1 0.7314 and ECE 0.0323. Empirical coverage at alpha = 0.05 was 93.83%.

## Repository layout

```
src/gemati/            pipeline (data mapping, model, conformal sets, routing, metrics, extended audits, CPU benchmark)
configs/experiment.json  seeds, features, alpha, cost matrix, persistence parameters
models/gemati_care.joblib  serialized model bundle
results/               machine-readable results and the paper's reliability / trade-off figures
  extended_metrics_public.json  all paper metrics (message-text previews redacted, see below)
  metrics.json          SUPERSEDED pilot run (22 Sep, MiniLM, 80/20 split); not used in the paper
  dataset_summary.json, progress files
  r2/                   additional analyses added during revision (see below)
scripts/               figure regeneration, compare_with_paper.py, and the r2_*.py revision analyses
figures/               Figure 1 (final architecture schematic, 600 dpi) and Figure 4 of the initial submission
                       (miss vs over-escalation; its counts are reported in Section 3.2 of the revised paper)
data/manifests/        file sizes and SHA-256 hashes of the downloaded public datasets
tests/                 unit tests (label mapping, conformal sets, router, temperature scaling)
```

Note on Figure 1: `figures/fig1_architecture.png` is the final schematic used in the revised manuscript. Its layout was
redrawn from the authors' design (see the AI Use Statement of the article); `scripts/make_paper_figures_1_4.py`
reproduces the earlier layout of the initial submission.

## Reproducing the experiments

Requirements: Python 3.11 or newer, CPU only, about 3 GB free disk, and an internet connection for the first run.

```bash
git clone https://github.com/juntazen/GEMATI-CARE.git
cd GEMATI-CARE
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,encoder]"
python -m pytest -q                        # 7 unit tests should pass
```

Before running the extended audits, download the SIM-VAIL repository as a ZIP (https://github.com/veithweilnhammer/sim-vail, "Code" -> "Download ZIP") and extract it so that the folder `data/external/sim-vail-main/data/` exists.

```bash
python -m gemati experiment   # core experiment; downloads CRADLE Bench, IndoSafety, and MiniLM
python -m gemati extended     # all analyses reported in the paper
python -m gemati status       # progress and ETA (use in a second terminal)
python scripts/compare_with_paper.py       # PASS/FAIL against the values in the paper
```

The run overwrites `models/gemati_care.joblib` and writes `results/extended_metrics.json`. The reference values used in the paper are kept in `results/extended_metrics_public.json`. Seeds and every analysis setting are fixed in `configs/experiment.json`. Dataset sources and checksums are listed in `data/SOURCE_DATASETS.json` and `data/manifests/data_manifest.csv`.

## Additional analyses added during revision (`results/r2/`)

These analyses evaluate the frozen seed-42 system. Thresholds and significance levels are selected on the
validation set; the test set is used only for evaluation. They run on a CPU in about two minutes and need the
three CRADLE parquet files in `data/raw/` (produced by `python -m gemati experiment`).

```bash
export CUDA_VISIBLE_DEVICES=""                      # CPU only, as in the paper
PYTHONPATH=src python scripts/r2_sanity_check.py      # must print SANITY GATE: PASS (16 locked values)
PYTHONPATH=src python scripts/r2_extra_experiments.py # E1-E6
PYTHONPATH=src python scripts/r2_robustness.py        # R1-R5 (needs the E1-E6 output)
PYTHONPATH=src python scripts/r2_figures.py           # Figures 3-5, 600 dpi PNG + PDF
```

| Analysis | Main result (official test, n = 600) |
|---|---|
| Matched threshold (E1) | t = 0.096 escalates exactly the same high-risk messages as the router at alpha = 0.05 (miss 5.13%); over-escalation 35.00% vs 39.33% |
| Class-specific alpha (E2) | over-escalation 34.83% (paired difference -4.50 points, 95% CI -6.17 to -3.00) at the same miss; escalations unchanged (413) |
| Clinician-label recalibration (E3, R3) | low-risk coverage 90.32% -> 95.70% (alpha = 0.05); no class significantly below nominal in any of five folds |
| McNemar (E4) | router alpha = 0.05 vs argmax: 43 vs 0 discordant, p = 2.3e-13 |
| TF-IDF CPU benchmark (E5) | median 2.35 ms, p95 5.61 ms, 670 msg/s, peak RSS 432 MB |
| Five-seed coverage (R1) | high-risk coverage not significantly below nominal in 18 of 20 seed x alpha cases |

`results/r2/R2_REPORT_ID.md` (Indonesian) gives every table with confidence intervals and p-values,
`results/r2/sanity_check.json` the reproduction gate, and `results/r2/SHA256SUMS.txt` the checksums.
Local paths, the host name, and GPU identifiers were redacted from `results/r2/logs/`. The revision scripts
were drafted with AI assistance (Claude) and reviewed and run by the first author; they call the pipeline
functions in `src/gemati/` without modifying them.

## Data

The datasets are **not redistributed** here. Obtain them from their original providers and follow their licenses:

- CRADLE Bench (Byun et al., EACL 2026), doi:10.18653/v1/2026.eacl-long.73
- IndoSafety (Azmi et al., EMNLP 2025), doi:10.18653/v1/2025.emnlp-main.465
- SIM-VAIL (Weilnhammer et al., Nature Medicine 2026), doi:10.1038/s41591-026-04577-2

Because these corpora contain sensitive mental-health disclosures, the text previews stored by the leakage audit were replaced with a redaction marker in `results/extended_metrics_public.json`. All numeric values are unchanged. The full, unredacted file is regenerated locally when the pipeline is run.

## Citation

If you use this code, please cite the article. Bibliographic details will be updated after publication. See also `CITATION.cff`.

## License

Code: MIT License (see `LICENSE`). Results and figures: CC BY 4.0.

## Authors and contributors

Software: **Junta Zeniarja** (Universitas Dian Nuswantoro), who also maintains this repository.
Contributors to the study: Erwin Yudi Hidayat (supervision), Egia Rosi Subhiyakto (data curation), and
Yani Parti Astuti (investigation and visualization).

The article that this repository accompanies is authored by Junta Zeniarja, Erwin Yudi Hidayat, Egia Rosi
Subhiyakto, and Yani Parti Astuti; please cite it when you use this code (see `CITATION.cff`).

## Contact

Junta Zeniarja, Faculty of Computer Science, Universitas Dian Nuswantoro, Semarang, Indonesia. junta@dsn.dinus.ac.id

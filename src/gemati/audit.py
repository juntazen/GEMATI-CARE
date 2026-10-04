from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit(root: Path) -> tuple[dict, str]:
    metrics_path = root / "results" / "extended_metrics.json"
    progress_path = root / "results" / "extended_progress.json"
    if not metrics_path.exists() or not progress_path.exists():
        raise FileNotFoundError("Artefak extended belum lengkap.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    primary = metrics["final_test_primary_alpha_0.10"]
    safety = metrics["final_test_safety_alpha_0.05"]
    bootstrap = metrics["bootstrap_confidence_intervals"]
    indonesia = metrics["indonesian_compatible_evaluation"]
    temporal = metrics["temporal_sim_vail"]["summary"]
    leakage = metrics["leakage_audit"]
    efficiency = metrics["efficiency_cpu"]

    checks = {
        "extended_complete": progress.get("status") == "complete",
        "all_numeric_values_finite": _finite(metrics),
        "five_tfidf_repeated_splits": len(metrics.get("tfidf_five_seed", {})) == 5,
        "required_alpha_grid": set(metrics["sensitivity"]["alpha"])
        == {"0.05", "0.1", "0.15", "0.2"},
        "disjoint_probability_and_conformal_calibration": bool(
            metrics["protocol"]["probability_and_conformal_calibration_disjoint"]
        ),
        "development_uses_validation": metrics["sensitivity"]["evaluation_split"]
        == "official_validation",
        "safety_coverage_at_least_90pct": safety["conformal"]["coverage"] >= 0.90,
        "bootstrap_has_all_required_metrics": {
            "macro_f1",
            "expected_cost",
            "under_escalation_rate",
            "over_escalation_rate",
            "coverage_low",
            "coverage_medium",
            "coverage_high",
        }.issubset(bootstrap),
        "all_split_exact_text_overlap_zero": all(
            item["unique_normalized_texts"] == 0
            for item in leakage["exact_overlap_all_pairs"].values()
        ),
        "near_duplicate_sensitivity_present": "sensitivity_excluding_flagged_test_rows"
        in leakage,
        "indosafety_eval1_excluded": indonesia.get(
            "excluded_eval1_rows_due_to_exact_train_overlap"
        )
        == 2014,
        "fresh_process_efficiency": efficiency["fp32"]["fresh_process_pid"]
        != efficiency["dynamic_int8"]["fresh_process_pid"],
        "benchmark_repetitions_adequate": min(
            efficiency["fp32"]["latency"]["warm_single_ms"]["repetitions"],
            efficiency["dynamic_int8"]["latency"]["warm_single_ms"]["repetitions"],
        )
        >= 50,
    }

    warnings = []
    if primary["mondrian_metrics"]["coverage"] < 0.90:
        warnings.append(
            "Alpha 0,10 mengalami under-coverage empiris; jangan klaim target 90% tercapai."
        )
    if safety["routing"]["over_escalation_rate"] > 0.35:
        warnings.append(
            "Mode safety alpha 0,05 menurunkan miss risiko tinggi tetapi over-escalation tinggi."
        )
    if temporal["control"]["temporal_any_escalation"] > 0.50:
        warnings.append(
            "Audit SIM-VAIL mengeskalasi mayoritas kontrol; temporal hanya boleh diklaim sebagai audit persistence."
        )
    worst_recall = min(
        value["mental_recall"]
        for value in indonesia["by_language"].values()
        if value["mental_recall"] is not None
    )
    if worst_recall < 0.50:
        warnings.append(
            "Representation probe IndoSafety lemah pada setidaknya satu bahasa daerah; jangan klaim robustness Indonesia."
        )
    int8 = efficiency["dynamic_int8"]
    fp32 = efficiency["fp32"]
    if int8["throughput_texts_per_second"]["mean"] <= fp32[
        "throughput_texts_per_second"
    ]["mean"]:
        warnings.append(
            "INT8 tidak meningkatkan throughput pada benchmark ini; jangan klaim percepatan umum."
        )
    warnings.extend(
        [
            "CRADLE memakai ID lokal yang berulang antar-split; audit leakage harus memakai teks, bukan ID saja.",
            "Dataset publik tidak otomatis membebaskan penetapan etik institusi.",
            "Nama penulis, afiliasi, corresponding author, pendanaan, konflik kepentingan, dan redaksi etik belum tersedia.",
            "Review bahasa dan inspeksi visual DOCX/PDF tetap memerlukan manusia sebelum unggah.",
        ]
    )

    experiments_pass = all(checks.values())
    submission_blockers = [
        "identitas penulis dan afiliasi belum diberikan",
        "nomor/redaksi keputusan etik institusi belum diberikan",
        "inspeksi visual akhir template dan proofread manusia belum selesai",
    ]
    verdict = {
        "experiments": "PASS_WITH_LIMITATIONS" if experiments_pass else "FAIL",
        "scientific_manuscript": "READY_FOR_AUTHOR_COMPLETION_AND_INTERNAL_REVIEW"
        if experiments_pass
        else "NOT_READY",
        "submission": "NOT_FINAL_READY",
        "acceptance_probability_claim": (
            "99% is not defensible; editorial fit and peer review cannot be guaranteed or "
            "estimated reliably from these artifacts alone."
        ),
        "checks": checks,
        "warnings": warnings,
        "submission_blockers": submission_blockers,
    }

    tfidf_seeds = metrics["tfidf_five_seed_summary"]
    leak_exclusion = leakage["sensitivity_excluding_flagged_test_rows"]
    md = f"""# Audit Akhir Eksperimen dan Kelayakan Submission GEMATI-CARE

## Putusan

Eksperimen: **{verdict['experiments']}**. Naskah ilmiah: **{verdict['scientific_manuscript']}**. Submission final: **{verdict['submission']}**. Klaim peluang diterima 99% tidak dapat dipertanggungjawabkan secara ilmiah karena keputusan editor, kecocokan scope, kualitas penulisan, dan penilaian reviewer berada di luar hasil eksperimen.

## Hasil Utama yang Lolos Verifikasi

Model dipilih hanya dari official validation: TF-IDF word+character memperoleh macro-F1 {metrics['model_selection']['tfidf_macro_f1']:.4f}, sedangkan MiniLM {metrics['model_selection']['minilm_macro_f1']:.4f}. Pada official test, TF-IDF memperoleh macro-F1 {primary['classification']['macro_f1']:.4f}, ECE {primary['classification']['ece']:.4f}, dan balanced accuracy {primary['classification']['balanced_accuracy']:.4f}. Lima repeated stratified splits pada validation menghasilkan macro-F1 rata-rata {tfidf_seeds['macro_f1']['mean']:.4f} ± {tfidf_seeds['macro_f1']['std']:.4f}, dengan t-interval 95% untuk rerata [{tfidf_seeds['macro_f1']['mean_t95_low']:.4f}, {tfidf_seeds['macro_f1']['mean_t95_high']:.4f}].

Pada alpha 0,10, coverage test adalah {_pct(primary['mondrian_metrics']['coverage'])}, di bawah nominal 90%, sehingga konfigurasi ini hanya boleh disebut utility operating point. Mode safety alpha 0,05 yang dipilih berdasarkan validation mencapai coverage {_pct(safety['conformal']['coverage'])}, high-risk miss {_pct(safety['routing']['high_risk_miss_rate'])}, under-escalation {_pct(safety['routing']['under_escalation_rate'])}, dan over-escalation {_pct(safety['routing']['over_escalation_rate'])}. Trade-off over-escalation harus ditulis eksplisit.

Bootstrap 1.000 replikasi pada mode safety memberikan CI 95% macro-F1 [{bootstrap['macro_f1']['ci95_low']:.4f}, {bootstrap['macro_f1']['ci95_high']:.4f}], expected cost [{bootstrap['expected_cost']['ci95_low']:.4f}, {bootstrap['expected_cost']['ci95_high']:.4f}], under-escalation [{_pct(bootstrap['under_escalation_rate']['ci95_low'])}, {_pct(bootstrap['under_escalation_rate']['ci95_high'])}], dan over-escalation [{_pct(bootstrap['over_escalation_rate']['ci95_low'])}, {_pct(bootstrap['over_escalation_rate']['ci95_high'])}].

Audit leakage CRADLE menemukan {len(leakage['test_ids_flagged_at_0.90_against_train_or_calibration'])} test item dengan cosine ≥0,90. Setelah keduanya dikeluarkan, macro-F1 menjadi {leak_exclusion['classification']['macro_f1']:.4f} dan high-risk miss tetap {_pct(leak_exclusion['routing']['high_risk_miss_rate'])}; kesimpulan utama tidak berubah. ID CRADLE bersifat lokal dan berulang antar-split, tetapi normalized exact-text overlap pada semua pasangan split bernilai nol.

Audit IndoSafety mengecualikan 2.014 Eval1 yang identik dengan train, lalu menilai 2.500 Eval2. Macro-F1 auxiliary taxonomy probe adalah {indonesia['macro_f1']:.4f} dan positive-class recall {_pct(indonesia['positive_recall'])}. Hasil ini bukan validasi eksternal classifier risiko. Audit SIM-VAIL juga bukan evaluasi klinis; kontrol memiliki temporal any-escalation {_pct(temporal['control']['temporal_any_escalation'])}.

## Gate Otomatis

"""
    md += "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in checks.items()
    )
    md += "\n\n## Peringatan Wajib dalam Naskah\n\n"
    md += "\n".join(f"- {item}" for item in warnings)
    md += "\n\n## Blocker Submission Final\n\n"
    md += "\n".join(f"- {item}" for item in submission_blockers)
    md += "\n"
    return verdict, md


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final GEMATI-CARE")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    verdict, markdown = audit(args.root.resolve())
    results = args.root.resolve() / "results"
    (results / "FINAL_AUDIT.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (results / "FINAL_AUDIT.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_cradle, load_indosafety, save_dataset_summary, write_manifest
from .encoder import MiniLMRiskModel
from .metrics import classification_metrics, conformal_metrics, grouped_bootstrap_difference, routing_metrics
from .model import GlobalConformal, MondrianConformal, TemperatureScaler, build_tfidf_classifier
from .progress import ProgressTracker
from .routing import argmax_actions, route_sets


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_splits(cradle: dict[str, pd.DataFrame], seed: int) -> tuple[pd.DataFrame, ...]:
    if "train" not in cradle:
        raise ValueError(f"CRADLE tidak menyediakan split train. Split: {list(cradle)}")
    full_train = cradle["train"]
    train, calibration = train_test_split(
        full_train,
        test_size=0.20,
        random_state=seed,
        stratify=full_train["risk"],
    )
    dev_key = next((key for key in ("dev", "validation", "valid") if key in cradle), None)
    test_key = "test" if "test" in cradle else None
    if dev_key is None or test_key is None:
        remaining_key = next((key for key in cradle if key != "train"), None)
        if remaining_key is None:
            validation, test = train_test_split(
                calibration,
                test_size=0.5,
                random_state=seed,
                stratify=calibration["risk"],
            )
            calibration, _ = train_test_split(
                train,
                test_size=0.20,
                random_state=seed + 1,
                stratify=train["risk"],
            )
        else:
            validation, test = train_test_split(
                cradle[remaining_key],
                test_size=0.5,
                random_state=seed,
                stratify=cradle[remaining_key]["risk"],
            )
    else:
        validation, test = cradle[dev_key], cradle[test_key]
    return tuple(frame.reset_index(drop=True) for frame in (train, calibration, validation, test))


def _safe_performance_metric(actions: np.ndarray, labels: np.ndarray) -> float:
    high = labels == "high"
    if not high.any():
        return 0.0
    return float(np.mean(actions[high] != "ESCALATE"))


def _predict_in_class_order(model, texts, classes: list[str]) -> np.ndarray:
    """Return probabilities in the explicit policy order, never estimator order."""
    raw = model.predict_proba(texts)
    estimator_classes = list(model.classes_)
    missing = set(classes) - set(estimator_classes)
    if missing:
        raise ValueError(f"Model tidak memiliki kelas kebijakan: {sorted(missing)}")
    return raw[:, [estimator_classes.index(label) for label in classes]]


def _reliability_plot(probabilities: np.ndarray, labels: list[str], classes: list[str], path: Path) -> None:
    y_index = np.asarray([classes.index(label) for label in labels])
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_index
    edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
            ns.append(int(mask.sum()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(xs, ys, marker="o", label="GEMATI-CARE")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), fontsize=7)
    ax.set(xlabel="Confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _risk_coverage_plot(
    probabilities: np.ndarray,
    labels: list[str],
    classes: list[str],
    cost_matrix: dict,
    path: Path,
) -> None:
    rows = []
    for alpha in np.linspace(0.02, 0.30, 15):
        # Descriptive test-time curve. The final alpha remains fixed on calibration.
        confidence = probabilities.max(axis=1)
        covered = confidence >= (1 - alpha)
        if covered.any():
            predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
            risk = float(np.mean(predicted[covered] != np.asarray(labels)[covered]))
        else:
            risk = np.nan
        rows.append((float(covered.mean()), risk))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([row[0] for row in rows], [row[1] for row in rows], marker="o")
    ax.set(xlabel="Automatic coverage", ylabel="Selective classification risk")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _external_indosafety_audit(model, scaler, frame: pd.DataFrame, classes: list[str]) -> dict:
    raw = _predict_in_class_order(model, frame["text"].tolist(), classes)
    probabilities = scaler.transform(raw) if scaler is not None else raw
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    output = frame.copy()
    output["predicted_risk"] = predicted
    category_text = output["category"].str.lower()
    mental_mask = category_text.str.contains("mental|self|suic|overreliance", regex=True)
    by_language = (
        output.groupby("language")["predicted_risk"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(output)),
        "mental_health_related_rows_by_keyword": int(mental_mask.sum()),
        "prediction_distribution": output["predicted_risk"].value_counts().to_dict(),
        "mental_subset_prediction_distribution": output.loc[mental_mask, "predicted_risk"].value_counts().to_dict(),
        "by_language": by_language,
        "note": "Distributional audit only; IndoSafety categories are not remapped as CRADLE gold labels.",
    }


def run_experiment(root: Path, config_path: Path) -> dict:
    config = load_config(config_path)
    results_dir = root / "results"
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    model_dir = root / "models"
    for directory in (results_dir, raw_dir, processed_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(results_dir / "progress.json", total_steps=13)
    started = time.perf_counter()
    try:
        progress.update("data_cradle", "Mengunduh/memuat CRADLE Bench tanpa token")
        cradle = load_cradle(raw_dir)
        progress.update("data_cradle", "Snapshot CRADLE tersimpan", increment=True)

        progress.update("data_indosafety", "Mengunduh IndoSafety Eval-1 dan Eval-2")
        indosafety = load_indosafety(raw_dir)
        progress.update("data_indosafety", f"IndoSafety siap: {len(indosafety)} baris", increment=True)

        summary_frames = {f"cradle_{name}": frame for name, frame in cradle.items()}
        summary_frames["indosafety"] = indosafety
        save_dataset_summary(summary_frames, results_dir / "dataset_summary.json")
        write_manifest(raw_dir, root / "data" / "manifests" / "data_manifest.csv")
        progress.update("manifest", "Manifest SHA-256 dan ringkasan dataset dibuat", increment=True)

        train, calibration, validation, test = _select_splits(cradle, config["seed"])
        split_frames = {"train": train, "calibration": calibration, "validation": validation, "test": test}
        for name, frame in split_frames.items():
            frame.to_parquet(processed_dir / f"{name}.parquet", index=False)
        progress.update("split", "Split train/calibration/validation/test dibekukan", increment=True)

        tfidf_model = build_tfidf_classifier(config)
        progress.update("baseline_training", f"Melatih baseline TF-IDF word+char pada {len(train)} sampel")
        tfidf_model.fit(train["text"], train["risk"])
        progress.update("baseline_training", "Baseline TF-IDF selesai", increment=True)

        progress.update("embedding", "Memuat frozen multilingual MiniLM pada CPU")
        model = MiniLMRiskModel(
            cache_dir=root / ".cache" / "embeddings",
            model_name=config["encoder_model"],
            batch_size=config["encoder_batch_size"],
            max_length=config["encoder_max_length"],
            c=config["logistic_c"],
            max_iter=config["max_iter"],
            seed=config["seed"],
            progress_callback=lambda message: progress.update("embedding", message),
        )
        progress.update("training", f"Melatih kepala risiko MiniLM pada {len(train)} sampel")
        model.fit(train["text"].tolist(), train["risk"].tolist())
        classes = list(config["risk_labels"])
        progress.update("training", "Model utama MiniLM + risk head selesai", increment=True)

        raw_calibration = _predict_in_class_order(model, calibration["text"], classes)
        cal_index = np.asarray([classes.index(label) for label in calibration["risk"]])
        scaler = TemperatureScaler().fit(raw_calibration, cal_index)
        probabilities_cal = scaler.transform(raw_calibration)
        mondrian = MondrianConformal(config["alpha"], classes).fit(probabilities_cal, calibration["risk"])
        global_cp = GlobalConformal(config["alpha"], classes).fit(probabilities_cal, calibration["risk"])
        progress.update(
            "calibration",
            f"Temperature={scaler.temperature:.4f}; kuantil conformal dibekukan",
            increment=True,
        )

        validation_probabilities = scaler.transform(
            _predict_in_class_order(model, validation["text"], classes)
        )
        validation_sets = mondrian.predict_sets(validation_probabilities)
        validation_actions = route_sets(
            validation_sets, classes, config["actions"], config["cost_matrix"]
        )
        validation_metrics = {
            "classification": classification_metrics(validation["risk"].tolist(), validation_probabilities, classes),
            "conformal": conformal_metrics(validation_sets, validation["risk"].tolist(), classes),
            "routing": routing_metrics(
                validation_actions,
                validation["risk"].tolist(),
                classes,
                config["cost_matrix"],
            ),
        }
        progress.update("validation", "Validasi dan pemeriksaan konfigurasi selesai", increment=True)

        test_probabilities_raw = _predict_in_class_order(model, test["text"], classes)
        test_probabilities = scaler.transform(test_probabilities_raw)
        test_sets = mondrian.predict_sets(test_probabilities)
        global_sets = global_cp.predict_sets(test_probabilities)
        cstcr_actions = route_sets(test_sets, classes, config["actions"], config["cost_matrix"])
        global_actions = route_sets(global_sets, classes, config["actions"], config["cost_matrix"])
        baseline_actions = argmax_actions(test_probabilities, classes)
        raw_actions = argmax_actions(test_probabilities_raw, classes)

        test_metrics = {
            "uncalibrated_classification": classification_metrics(
                test["risk"].tolist(), test_probabilities_raw, classes
            ),
            "classification": classification_metrics(test["risk"].tolist(), test_probabilities, classes),
            "mondrian_conformal": conformal_metrics(test_sets, test["risk"].tolist(), classes),
            "global_conformal": conformal_metrics(global_sets, test["risk"].tolist(), classes),
            "routing_cstcr": routing_metrics(
                cstcr_actions, test["risk"].tolist(), classes, config["cost_matrix"]
            ),
            "routing_global_conformal": routing_metrics(
                global_actions, test["risk"].tolist(), classes, config["cost_matrix"]
            ),
            "routing_argmax_calibrated": routing_metrics(
                baseline_actions, test["risk"].tolist(), classes, config["cost_matrix"]
            ),
            "routing_argmax_uncalibrated": routing_metrics(
                raw_actions, test["risk"].tolist(), classes, config["cost_matrix"]
            ),
        }
        progress.update("test", f"Test final selesai pada {len(test)} contoh", increment=True)

        tfidf_test_probabilities = _predict_in_class_order(tfidf_model, test["text"], classes)
        tfidf_test_actions = argmax_actions(tfidf_test_probabilities, classes)
        tfidf_test_metrics = {
            "classification": classification_metrics(
                test["risk"].tolist(), tfidf_test_probabilities, classes
            ),
            "routing_argmax": routing_metrics(
                tfidf_test_actions, test["risk"].tolist(), classes, config["cost_matrix"]
            ),
        }

        bootstrap = grouped_bootstrap_difference(
            test["risk"].tolist(),
            cstcr_actions,
            baseline_actions,
            test["id"].tolist(),
            _safe_performance_metric,
            config["bootstrap_repetitions"],
            config["seed"],
        )
        progress.update("statistics", "Grouped bootstrap selesai", increment=True)

        external = _external_indosafety_audit(model, scaler, indosafety, classes)
        tfidf_external = _external_indosafety_audit(tfidf_model, None, indosafety, classes)
        progress.update("external_audit", "Audit distribusi IndoSafety selesai", increment=True)

        _reliability_plot(
            test_probabilities, test["risk"].tolist(), classes, results_dir / "reliability_diagram.png"
        )
        _risk_coverage_plot(
            test_probabilities,
            test["risk"].tolist(),
            classes,
            config["cost_matrix"],
            results_dir / "risk_coverage_curve.png",
        )
        progress.update("figures", "Gambar hasil eksperimen dibuat", increment=True)

        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "conformal": mondrian,
                "config": config,
                "probability_order": classes,
                "estimator_order": list(model.classes_),
            },
            model_dir / "gemati_care.joblib",
            compress=3,
        )
        progress.update("artifacts", "Model dan kebijakan routing disimpan", increment=True)

        results = {
            "run": {
                "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "seed": config["seed"],
                "alpha": config["alpha"],
                "classes": classes,
                "temperature": scaler.temperature,
                "conformal_quantiles": mondrian.quantiles,
            },
            "data": {
                name: {"rows": int(len(frame)), "risk_distribution": frame["risk"].value_counts().to_dict()}
                for name, frame in split_frames.items()
            },
            "validation": validation_metrics,
            "test": test_metrics,
            "tfidf_baseline": {
                "test": tfidf_test_metrics,
                "indosafety_external": tfidf_external,
            },
            "bootstrap_high_risk_miss_difference_cstcr_minus_argmax": bootstrap,
            "indosafety_external": external,
            "limitations": [
                "Risk labels are engineering routing categories derived from CRADLE labels, not clinical diagnoses.",
                "IndoSafety is used as a distributional external audit because its gold taxonomy differs.",
                "No patient-level or real-world safety claim is supported by this offline experiment.",
            ],
        }
        (results_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        progress.complete("Eksperimen CPU selesai; metrics.json siap untuk pembuatan paper")
        return results
    except Exception as exc:
        progress.fail(f"{type(exc).__name__}: {exc}")
        raise

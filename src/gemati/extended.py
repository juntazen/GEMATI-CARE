from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import t as student_t

from .data import download_file, map_cradle_risk
from .encoder import MiniLMRiskModel
from .metrics import classification_metrics, conformal_metrics, routing_metrics
from .model import GlobalConformal, MondrianConformal, TemperatureScaler, build_tfidf_classifier
from .progress import ProgressTracker
from .routing import (
    TemporalRiskState,
    argmax_actions,
    expected_cost_actions,
    route_sets,
    threshold_actions,
)


SEEDS = [11, 29, 42, 71, 101]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
CALIBRATION_FRACTIONS = [0.10, 0.20, 0.30]
HIGH_THRESHOLDS = [0.30, 0.50, 0.70]
INDOSAFETY_TRAIN_URL = (
    "https://raw.githubusercontent.com/falensiazmi/IndoSafety/"
    "main/dataset/IndoSafety-Train.xlsx"
)

COST_MATRICES = {
    "balanced": {
        "SUPPORT": [0.0, 2.0, 8.0],
        "CLARIFY": [1.0, 0.5, 4.0],
        "ESCALATE": [3.0, 1.0, 0.0],
    },
    "safety_first": {
        "SUPPORT": [0.0, 4.0, 16.0],
        "CLARIFY": [1.0, 0.5, 7.0],
        "ESCALATE": [2.5, 1.0, 0.0],
    },
    "utility_first": {
        "SUPPORT": [0.0, 2.0, 8.0],
        "CLARIFY": [0.75, 0.5, 4.0],
        "ESCALATE": [6.0, 3.0, 0.0],
    },
}

TIE_COST_MATRIX = {
    "SUPPORT": [0.0, 3.0, 12.0],
    "CLARIFY": [1.0, 0.5, 3.0],
    "ESCALATE": [3.0, 1.0, 0.0],
}

TIE_POLICIES = {
    "conservative": ["ESCALATE", "CLARIFY", "SUPPORT"],
    "neutral": ["CLARIFY", "ESCALATE", "SUPPORT"],
    "lenient": ["SUPPORT", "CLARIFY", "ESCALATE"],
}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _head(seed: int, weighted: bool = True, c: float = 2.0, max_iter: int = 1500):
    return OneVsRestClassifier(
        LogisticRegression(
            C=c,
            class_weight="balanced" if weighted else None,
            max_iter=max_iter,
            solver="liblinear",
            random_state=seed,
        )
    )


def _reorder(probabilities: np.ndarray, source_classes: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    return probabilities[:, [list(source_classes).index(label) for label in classes]]


def _fit_evaluate(
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    calibration_embeddings: np.ndarray,
    calibration_labels: Sequence[str],
    test_embeddings: np.ndarray,
    test_labels: Sequence[str],
    classes: list[str],
    actions: list[str],
    costs: dict,
    seed: int,
    alpha: float = 0.10,
    weighted: bool = True,
    calibrate: bool = True,
) -> dict:
    """Fit with disjoint probability- and conformal-calibration subsets."""
    pool_labels = np.asarray(list(calibration_labels))
    temp_idx, conformal_idx = train_test_split(
        np.arange(len(pool_labels)),
        train_size=1.0 / 3.0,
        random_state=seed,
        stratify=pool_labels,
    )
    classifier = _head(seed, weighted=weighted)
    classifier.fit(train_embeddings, train_labels)
    raw_temp = _reorder(
        classifier.predict_proba(calibration_embeddings[temp_idx]), classifier.classes_, classes
    )
    raw_conformal = _reorder(
        classifier.predict_proba(calibration_embeddings[conformal_idx]), classifier.classes_, classes
    )
    raw_test = _reorder(classifier.predict_proba(test_embeddings), classifier.classes_, classes)
    scaler = TemperatureScaler()
    if calibrate:
        y_temp = np.asarray([classes.index(label) for label in pool_labels[temp_idx]])
        scaler.fit(raw_temp, y_temp)
        cal_probs = scaler.transform(raw_conformal)
        test_probs = scaler.transform(raw_test)
    else:
        cal_probs, test_probs = raw_conformal, raw_test
    conformal_labels = pool_labels[conformal_idx].tolist()
    mondrian = MondrianConformal(alpha, classes).fit(cal_probs, conformal_labels)
    global_cp = GlobalConformal(alpha, classes).fit(cal_probs, conformal_labels)
    mondrian_sets = mondrian.predict_sets(test_probs)
    global_sets = global_cp.predict_sets(test_probs)
    minimax = route_sets(mondrian_sets, classes, actions, costs)
    expected = expected_cost_actions(test_probs, classes, actions, costs)
    argmax = argmax_actions(test_probs, classes)
    return {
        "classifier": classifier,
        "scaler": scaler,
        "mondrian": mondrian,
        "test_probabilities": test_probs,
        "calibration_probabilities": cal_probs,
        "conformal_labels": conformal_labels,
        "classification": classification_metrics(test_labels, test_probs, classes),
        "mondrian_metrics": conformal_metrics(mondrian_sets, test_labels, classes),
        "global_metrics": conformal_metrics(global_sets, test_labels, classes),
        "routing_minimax": routing_metrics(minimax, test_labels, classes, costs),
        "routing_global_minimax": routing_metrics(
            route_sets(global_sets, classes, actions, costs), test_labels, classes, costs
        ),
        "routing_expected": routing_metrics(expected, test_labels, classes, costs),
        "routing_argmax": routing_metrics(argmax, test_labels, classes, costs),
        "actions_minimax": minimax,
        "actions_expected": expected,
        "actions_argmax": argmax,
        "sets": mondrian_sets,
        "global_sets": global_sets,
        "temperature": scaler.temperature if calibrate else None,
        "calibration_protocol": {
            "probability_calibration_n": int(len(temp_idx)),
            "conformal_calibration_n": int(len(conformal_idx)),
            "disjoint": True,
        },
    }


def _fit_evaluate_tfidf(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    calibration_texts: Sequence[str],
    calibration_labels: Sequence[str],
    evaluation_texts: Sequence[str],
    evaluation_labels: Sequence[str],
    classes: list[str],
    actions: list[str],
    costs: dict,
    config: dict,
    seed: int,
    alpha: float = 0.10,
    calibrate: bool = True,
    weighted: bool = True,
) -> dict:
    pool_labels = np.asarray(list(calibration_labels))
    temp_idx, conformal_idx = train_test_split(
        np.arange(len(pool_labels)),
        train_size=1.0 / 3.0,
        random_state=seed,
        stratify=pool_labels,
    )
    local_config = dict(config)
    local_config["seed"] = seed
    classifier = build_tfidf_classifier(local_config)
    if not weighted:
        classifier.named_steps["classifier"].estimator.class_weight = None
    classifier.fit(list(train_texts), list(train_labels))
    calibration_texts = np.asarray(list(calibration_texts), dtype=object)
    raw_temp = _reorder(
        classifier.predict_proba(calibration_texts[temp_idx]), classifier.classes_, classes
    )
    raw_conformal = _reorder(
        classifier.predict_proba(calibration_texts[conformal_idx]), classifier.classes_, classes
    )
    raw_evaluation = _reorder(
        classifier.predict_proba(list(evaluation_texts)), classifier.classes_, classes
    )
    y_temp = np.asarray([classes.index(label) for label in pool_labels[temp_idx]])
    scaler = TemperatureScaler()
    if calibrate:
        scaler.fit(raw_temp, y_temp)
        conformal_probabilities = scaler.transform(raw_conformal)
        evaluation_probabilities = scaler.transform(raw_evaluation)
    else:
        conformal_probabilities = raw_conformal
        evaluation_probabilities = raw_evaluation
    conformal_labels = pool_labels[conformal_idx].tolist()
    mondrian = MondrianConformal(alpha, classes).fit(
        conformal_probabilities, conformal_labels
    )
    global_cp = GlobalConformal(alpha, classes).fit(
        conformal_probabilities, conformal_labels
    )
    sets = mondrian.predict_sets(evaluation_probabilities)
    global_sets = global_cp.predict_sets(evaluation_probabilities)
    minimax = route_sets(sets, classes, actions, costs)
    expected = expected_cost_actions(evaluation_probabilities, classes, actions, costs)
    argmax = argmax_actions(evaluation_probabilities, classes)
    return {
        "classifier": classifier,
        "scaler": scaler,
        "mondrian": mondrian,
        "test_probabilities": evaluation_probabilities,
        "calibration_probabilities": conformal_probabilities,
        "conformal_labels": conformal_labels,
        "classification": classification_metrics(
            evaluation_labels, evaluation_probabilities, classes
        ),
        "mondrian_metrics": conformal_metrics(sets, evaluation_labels, classes),
        "global_metrics": conformal_metrics(global_sets, evaluation_labels, classes),
        "routing_minimax": routing_metrics(minimax, evaluation_labels, classes, costs),
        "routing_global_minimax": routing_metrics(
            route_sets(global_sets, classes, actions, costs),
            evaluation_labels,
            classes,
            costs,
        ),
        "routing_expected": routing_metrics(expected, evaluation_labels, classes, costs),
        "routing_argmax": routing_metrics(argmax, evaluation_labels, classes, costs),
        "actions_minimax": minimax,
        "actions_expected": expected,
        "actions_argmax": argmax,
        "sets": sets,
        "global_sets": global_sets,
        "temperature": scaler.temperature if calibrate else None,
        "calibration_protocol": {
            "probability_calibration_n": int(len(temp_idx)),
            "conformal_calibration_n": int(len(conformal_idx)),
            "disjoint": True,
        },
    }


def _strip_model_objects(run: dict) -> dict:
    excluded = {
        "classifier",
        "scaler",
        "mondrian",
        "test_probabilities",
        "calibration_probabilities",
        "conformal_labels",
        "actions_minimax",
        "actions_expected",
        "actions_argmax",
        "sets",
        "global_sets",
    }
    return {key: _json_ready(value) for key, value in run.items() if key not in excluded}


def _bootstrap_interval(values: Sequence[float]) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _seed_summary(values: Sequence[float]) -> dict:
    """Descriptive repeated-split summary with a t interval for the mean."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = (
        float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_t95_low": mean - half,
        "mean_t95_high": mean + half,
        "n_repeated_splits": int(len(values)),
    }


def _bootstrap_all(
    labels: Sequence[str],
    probabilities: np.ndarray,
    actions: Sequence[str],
    sets: Sequence[set[str]],
    classes: list[str],
    costs: dict,
    repetitions: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    actions_arr = np.asarray(actions)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    covered = {label: np.asarray([label in item for item in sets]) for label in classes}
    values = defaultdict(list)
    n = len(labels_arr)
    for _ in range(repetitions):
        idx = rng.integers(0, n, size=n)
        y = labels_arr[idx]
        pred = predicted[idx]
        act = actions_arr[idx]
        values["macro_f1"].append(f1_score(y, pred, labels=classes, average="macro", zero_division=0))
        route = routing_metrics(act, y, classes, costs)
        for metric in ("expected_cost", "under_escalation_rate", "over_escalation_rate"):
            values[metric].append(route[metric])
        for label in classes:
            mask = y == label
            if mask.any():
                values[f"coverage_{label}"].append(float(covered[label][idx][mask].mean()))
    return {metric: _bootstrap_interval(items) for metric, items in values.items()}


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _leakage_audit(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_embeddings: np.ndarray,
    calibration_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
) -> dict:
    frames = {
        "train": train.reset_index(drop=True),
        "calibration_pool": calibration.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    embeddings = {
        "train": train_embeddings,
        "calibration_pool": calibration_embeddings,
        "validation": validation_embeddings,
        "test": test_embeddings,
    }
    thresholds = [0.90, 0.95, 0.98, 0.99]
    exact, id_overlap, semantic = {}, {}, {}
    names = list(frames)
    for left_i, left_name in enumerate(names):
        for right_name in names[left_i + 1 :]:
            key = f"{left_name}__{right_name}"
            left, right = frames[left_name], frames[right_name]
            left_norm = left["text"].map(_normalized_text)
            right_norm = right["text"].map(_normalized_text)
            shared = set(left_norm) & set(right_norm)
            exact[key] = {
                "unique_normalized_texts": int(len(shared)),
                "left_rows_in_overlap": int(left_norm.isin(shared).sum()),
                "right_rows_in_overlap": int(right_norm.isin(shared).sum()),
            }
            id_overlap[key] = int(
                len(set(left["id"].astype(str)) & set(right["id"].astype(str)))
            )
            similarities = embeddings[right_name] @ embeddings[left_name].T
            maxima = similarities.max(axis=1)
            best = similarities.argmax(axis=1)
            top_idx = np.argsort(maxima)[-10:][::-1]
            semantic[key] = {
                "counts": {str(t): int((maxima >= t).sum()) for t in thresholds},
                "max_similarity": float(maxima.max()),
                "median_max_similarity": float(np.median(maxima)),
                "top_pairs": [
                    {
                        "left_id": str(left.iloc[int(best[i])]["id"]),
                        "right_id": str(right.iloc[i]["id"]),
                        "left_label": str(left.iloc[int(best[i])].get("risk", "")),
                        "right_label": str(right.iloc[i].get("risk", "")),
                        "label_agreement": bool(
                            left.iloc[int(best[i])].get("risk", None)
                            == right.iloc[i].get("risk", None)
                        ),
                        "cosine_similarity": float(maxima[i]),
                        "left_text_preview": str(left.iloc[int(best[i])]["text"])[:160],
                        "right_text_preview": str(right.iloc[i]["text"])[:160],
                    }
                    for i in top_idx
                ],
            }
    reference = np.vstack([train_embeddings, calibration_embeddings])
    test_maxima = (test_embeddings @ reference.T).max(axis=1)
    return {
        "method": "normalized exact match and MiniLM cosine nearest-neighbour audit",
        "semantic_thresholds": thresholds,
        "exact_overlap_all_pairs": exact,
        "id_overlap_all_pairs": id_overlap,
        "semantic_overlap_all_pairs": semantic,
        "test_ids_flagged_at_0.90_against_train_or_calibration": test.loc[
            test_maxima >= 0.90, "id"
        ].astype(str).tolist(),
    }


def _error_taxonomy(test: pd.DataFrame, probabilities: np.ndarray, classes: list[str]) -> dict:
    patterns = {
        "negation": r"\b(no|not|never|don't|doesn't|didn't|isn't|wasn't|without)\b",
        "reported_speech_or_quote": r"[\"“”]|\b(said|told|asked|according to|my friend|my partner)\b",
        "past_or_historical": r"\b(ago|used to|previously|last year|past|when i was|history of)\b",
        "slang": r"\b(idk|lol|lmao|gonna|wanna|tbh|imo|kinda|sorta|bc|rn)\b",
    }
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    output = {}
    for name, pattern in patterns.items():
        mask = test["text"].str.lower().str.contains(pattern, regex=True, na=False).to_numpy()
        if not mask.any():
            output[name] = {"n": 0}
            continue
        y = test["risk"].to_numpy()[mask]
        p = predicted[mask]
        high_mask = y == "high"
        output[name] = {
            "n": int(mask.sum()),
            "macro_f1": float(f1_score(y, p, labels=classes, average="macro", zero_division=0)),
            "high_recall": float((p[high_mask] == "high").mean()) if high_mask.any() else None,
            "error_rate": float((p != y).mean()),
        }
    return output


def _mapping(label: str, variant: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "_", str(label).lower())
    if "no_crisis" in compact:
        return "low"
    if variant == "broad":
        return "high" if "ongoing" in compact else "medium"
    if variant == "narrow":
        severe = (
            "suicideideation_active_ongoing",
            "selfharm_ongoing",
            "self_harm_ongoing",
            "domesticviolence_ongoing",
            "rape_ongoing",
            "childabuse_ongoing",
        )
        return "high" if any(item in compact for item in severe) else "medium"
    return map_cradle_risk(label)


def _load_indosafety_train(root: Path) -> pd.DataFrame:
    path = download_file(INDOSAFETY_TRAIN_URL, root / "data" / "raw" / "indosafety_train.xlsx")
    frame = pd.read_excel(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    text_col = next(c for c in frame.columns if "prompt" in c or "instruction" in c)
    category_cols = [
        c for c in frame.columns if c in ("risk_area", "types_of_harm", "specific_harms", "category")
    ]
    category = frame[category_cols].fillna("").astype(str).agg(" | ".join, axis=1)
    return pd.DataFrame(
        {
            "text": frame[text_col].fillna("").astype(str),
            "category": category,
            "is_mental_health_harm": category.str.lower().str.contains(
                "mental|self.harm|suic|over.reliance", regex=True
            ),
        }
    )


def _indonesian_compatible_evaluation(
    root: Path,
    model: MiniLMRiskModel,
    external: pd.DataFrame,
    seed: int,
) -> dict:
    train = _load_indosafety_train(root)
    train_x = model.encode(train["text"].tolist())
    test_x = model.encode(external["text"].tolist())
    train_y = train["is_mental_health_harm"].astype(int).to_numpy()
    test_y = (
        external["category"]
        .str.lower()
        .str.contains("mental|self.harm|suic|over.reliance", regex=True)
        .astype(int)
        .to_numpy()
    )
    train_norm = set(train["text"].map(_normalized_text))
    external_norm = external["text"].map(_normalized_text)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="liblinear", random_state=seed
    )
    classifier.fit(train_x, train_y)
    pred_all = classifier.predict(test_x)
    eval2_mask = (~external_norm.isin(train_norm)).to_numpy()
    eval_frame = external.loc[eval2_mask].reset_index(drop=True)
    pred = pred_all[eval2_mask]
    test_y_eval = test_y[eval2_mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y_eval, pred, labels=[0, 1], zero_division=0
    )
    by_language = {}
    for language, indices in eval_frame.groupby("language").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        by_language[str(language)] = {
            "n": int(len(idx)),
            "accuracy": float(accuracy_score(test_y_eval[idx], pred[idx])),
            "mental_recall": float(
                (pred[idx][test_y_eval[idx] == 1] == 1).mean()
            )
            if (test_y_eval[idx] == 1).any()
            else None,
            "macro_f1": float(
                f1_score(test_y_eval[idx], pred[idx], average="macro", zero_division=0)
            ),
        }
    eval2 = eval_frame.copy()
    trailing_id = eval2["id"].astype(str).str.rsplit(":", n=1).str[-1].astype(int)
    base_prompt_id = trailing_id.to_numpy(copy=True)
    formal_mask = eval2["language"].eq("indonesian-formal").to_numpy()
    base_prompt_id[formal_mask] = np.arange(int(formal_mask.sum()))
    eval2["base_prompt_id"] = base_prompt_id.astype(str)
    eval2["prediction"] = pred
    eval2["target"] = test_y_eval
    consistency = eval2.groupby("base_prompt_id")["prediction"].nunique()
    groups = eval2["base_prompt_id"].unique()
    group_confusions = []
    for group in groups:
        part = eval2[eval2["base_prompt_id"] == group]
        target = part["target"].to_numpy()
        prediction = part["prediction"].to_numpy()
        group_confusions.append(
            [
                int(((target == 0) & (prediction == 0)).sum()),
                int(((target == 0) & (prediction == 1)).sum()),
                int(((target == 1) & (prediction == 0)).sum()),
                int(((target == 1) & (prediction == 1)).sum()),
            ]
        )
    group_confusions = np.asarray(group_confusions, dtype=int)
    rng = np.random.default_rng(42)
    clustered_macro, clustered_recall = [], []
    for _ in range(1000):
        confusion = group_confusions[
            rng.integers(0, len(groups), size=len(groups))
        ].sum(axis=0)
        tn, fp, fn, tp = confusion
        f1_negative = 2 * tn / max(1, 2 * tn + fp + fn)
        f1_positive = 2 * tp / max(1, 2 * tp + fp + fn)
        clustered_macro.append(float((f1_negative + f1_positive) / 2))
        clustered_recall.append(float(tp / max(1, tp + fn)))
    test_norm = external_norm
    similarities = test_x @ train_x.T
    maxima = similarities.max(axis=1)
    maxima_eval2 = maxima[eval2_mask]
    return {
        "task": "auxiliary mental-health-harm taxonomy representation probe, not crisis severity",
        "train_rows": int(len(train)),
        "test_rows": int(len(eval_frame)),
        "excluded_eval1_rows_due_to_exact_train_overlap": int((~eval2_mask).sum()),
        "positive_test_rows": int(test_y_eval.sum()),
        "accuracy": float(accuracy_score(test_y_eval, pred)),
        "macro_f1": float(
            f1_score(test_y_eval, pred, average="macro", zero_division=0)
        ),
        "positive_precision": float(precision[1]),
        "positive_recall": float(recall[1]),
        "positive_f1": float(f1[1]),
        "by_language": by_language,
        "worst_language_macro_f1": min(item["macro_f1"] for item in by_language.values()),
        "parallel_translation_audit": {
            "eval2_base_prompt_groups": int(len(groups)),
            "expected_variants_per_group": 5,
            "unanimous_prediction_rate": float((consistency == 1).mean()),
            "macro_f1_cluster_bootstrap_ci95": [
                float(np.nanquantile(clustered_macro, 0.025)),
                float(np.nanquantile(clustered_macro, 0.975)),
            ],
            "positive_recall_cluster_bootstrap_ci95": [
                float(np.nanquantile(clustered_recall, 0.025)),
                float(np.nanquantile(clustered_recall, 0.975)),
            ],
        },
        "train_test_leakage_audit": {
            "scope_all_eval1_plus_eval2": int(len(external)),
            "exact_normalized_test_rows": int(test_norm.isin(train_norm).sum()),
            "semantic_test_rows_ge_0.90": int((maxima >= 0.90).sum()),
            "semantic_test_rows_ge_0.95": int((maxima >= 0.95).sum()),
            "max_cosine_similarity": float(maxima.max()),
            "eval2_semantic_rows_ge_0.90": int((maxima_eval2 >= 0.90).sum()),
            "eval2_semantic_rows_ge_0.95": int((maxima_eval2 >= 0.95).sum()),
            "eval2_max_cosine_similarity": float(maxima_eval2.max()),
        },
    }


def _sim_vail_label_map(root: Path, control: bool = False) -> list[dict]:
    suffix = "_control" if control else ""
    path = (
        root
        / "data"
        / "external"
        / "sim-vail-main"
        / "repro"
        / "petri_overrides"
        / "src"
        / "petri"
        / "tasks"
        / "v1"
        / f"auditor_instruction_labels_v1{suffix}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _match_sim_condition(description: str, labels: list[dict]) -> dict:
    for item in labels:
        instruction = str(item.get("instruction_text", ""))
        if instruction and instruction in description:
            return item
    return {"phenotype": "unknown", "intention_label": "unknown", "id": "unknown"}


def _load_sim_vail(root: Path) -> tuple[list[dict], list[dict]]:
    base = root / "data" / "external" / "sim-vail-main" / "data"
    primary_labels = _sim_vail_label_map(root, False)
    control_labels = _sim_vail_label_map(root, True)
    outputs = []
    for condition, pattern, labels in (
        ("vulnerable", str(base / "v1" / "*" / "transcript_*.json"), primary_labels),
        ("control", str(base / "v1_control" / "*" / "transcript_*.json"), control_labels),
    ):
        conversations = []
        for path in sorted(glob.glob(pattern)):
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            user_turns = [
                str(message.get("content", ""))
                for message in data.get("target_messages", [])
                if message.get("role") == "user" and str(message.get("content", "")).strip()
            ]
            if not user_turns:
                continue
            matched = _match_sim_condition(str(metadata.get("description", "")), labels)
            conversations.append(
                {
                    "id": metadata.get("transcript_id", Path(path).stem),
                    "target_model": metadata.get("target_model", Path(path).parent.name),
                    "phenotype": matched.get("phenotype", "unknown"),
                    "intention": matched.get("intention_label", matched.get("intention", "unknown")),
                    "turns": user_turns,
                }
            )
        outputs.append(conversations)
    return outputs[0], outputs[1]


def _temporal_audit(
    model: MiniLMRiskModel,
    scaler: TemperatureScaler,
    conformal: MondrianConformal,
    vulnerable: list[dict],
    controls: list[dict],
    classes: list[str],
    actions: list[str],
    costs: dict,
    rho: float,
    high_threshold: float,
    low_threshold: float,
) -> dict:
    all_conversations = vulnerable + controls
    flat_texts = [turn for conversation in all_conversations for turn in conversation["turns"]]
    raw = _reorder(model.predict_proba(flat_texts), model.classes_, classes)
    probabilities = scaler.transform(raw)
    cursor = 0
    records = []
    for conversation in all_conversations:
        length = len(conversation["turns"])
        probs = probabilities[cursor : cursor + length]
        cursor += length
        static_sets = conformal.predict_sets(probs)
        static_actions = route_sets(static_sets, classes, actions, costs)
        state = TemporalRiskState(classes, rho=rho, high_threshold=high_threshold, low_threshold=low_threshold)
        temporal_actions = []
        for row in probs:
            accumulated = state.update(row)
            normalized = accumulated / accumulated.sum()
            pred_set = conformal.predict_sets(normalized.reshape(1, -1))[0]
            action = route_sets([pred_set], classes, actions, costs)[0]
            temporal_actions.append(state.apply_hysteresis(action))
        def first(items):
            return next((i + 1 for i, value in enumerate(items) if value == "ESCALATE"), None)
        records.append(
            {
                **{k: conversation[k] for k in ("id", "target_model", "phenotype", "intention")},
                "condition": "vulnerable" if conversation in vulnerable else "control",
                "turns": length,
                "static_any_escalate": "ESCALATE" in static_actions,
                "temporal_any_escalate": "ESCALATE" in temporal_actions,
                "static_escalate_rate": static_actions.count("ESCALATE") / length,
                "temporal_escalate_rate": temporal_actions.count("ESCALATE") / length,
                "static_deescalations": sum(
                    a == "ESCALATE" and b != "ESCALATE"
                    for a, b in zip(static_actions[:-1], static_actions[1:])
                ),
                "temporal_deescalations": sum(
                    a == "ESCALATE" and b != "ESCALATE"
                    for a, b in zip(temporal_actions[:-1], temporal_actions[1:])
                ),
                "static_first_escalation": first(static_actions),
                "temporal_first_escalation": first(temporal_actions),
            }
        )
    frame = pd.DataFrame(records)
    summary = {}
    for condition, group in frame.groupby("condition"):
        cluster_key = (
            group["target_model"].astype(str)
            + "|"
            + group["phenotype"].astype(str)
            + "|"
            + group["intention"].astype(str)
        )
        clusters = list(cluster_key.unique())
        cluster_aggregates = np.asarray(
            [
                [
                    int((cluster_key == key).sum()),
                    float(group.loc[cluster_key == key, "temporal_any_escalate"].sum()),
                    float(group.loc[cluster_key == key, "temporal_escalate_rate"].sum()),
                ]
                for key in clusters
            ],
            dtype=float,
        )
        rng = np.random.default_rng(42)
        boot_any, boot_rate = [], []
        for _ in range(1000):
            sampled = cluster_aggregates[
                rng.integers(0, len(clusters), size=len(clusters))
            ].sum(axis=0)
            boot_any.append(float(sampled[1] / sampled[0]))
            boot_rate.append(float(sampled[2] / sampled[0]))
        summary[condition] = {
            "conversations": int(len(group)),
            "static_any_escalation": float(group["static_any_escalate"].mean()),
            "temporal_any_escalation": float(group["temporal_any_escalate"].mean()),
            "static_mean_escalate_rate": float(group["static_escalate_rate"].mean()),
            "temporal_mean_escalate_rate": float(group["temporal_escalate_rate"].mean()),
            "static_total_deescalations": int(group["static_deescalations"].sum()),
            "temporal_total_deescalations": int(group["temporal_deescalations"].sum()),
            "cluster_definition": "target_model × phenotype × intention",
            "cluster_count": int(len(clusters)),
            "temporal_any_escalation_cluster_ci95": [
                float(np.quantile(boot_any, 0.025)),
                float(np.quantile(boot_any, 0.975)),
            ],
            "temporal_mean_escalate_rate_cluster_ci95": [
                float(np.quantile(boot_rate, 0.025)),
                float(np.quantile(boot_rate, 0.975)),
            ],
        }
    phenotype = {}
    vulnerable_frame = frame[frame["condition"] == "vulnerable"]
    for name, group in vulnerable_frame.groupby("phenotype"):
        phenotype[str(name)] = {
            "n": int(len(group)),
            "temporal_any_escalation": float(group["temporal_any_escalate"].mean()),
            "temporal_mean_escalate_rate": float(group["temporal_escalate_rate"].mean()),
        }
    return {
        "scope_note": (
            "SIM-VAIL conditions are synthetic conversation-level vulnerability conditions, "
            "not per-turn clinical gold labels. Temporal scores are renormalized before "
            "set construction, but conformal coverage is not claimed under temporal dependence."
        ),
        "summary": summary,
        "by_phenotype": phenotype,
        "worst_phenotype_temporal_any_escalation": min(
            (item["temporal_any_escalation"] for item in phenotype.values()), default=None
        ),
        "deescalation_reduction": int(
            frame["static_deescalations"].sum() - frame["temporal_deescalations"].sum()
        ),
    }


def _model_cache_size(model_name: str) -> int:
    slug = "models--" + model_name.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / slug
    if not root.exists():
        return 0
    unique = {}
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                unique[str(path.resolve())] = path.stat().st_size
            except OSError:
                pass
    return int(sum(unique.values()))


def _efficiency_benchmark(
    root: Path,
    model: MiniLMRiskModel,
    test_texts: Sequence[str],
    optimized: bool,
) -> dict:
    del test_texts
    variant = "int8" if optimized else "fp32"
    descriptor, name = tempfile.mkstemp(prefix=f"gemati_{variant}_", suffix=".json")
    os.close(descriptor)
    output_path = Path(name)
    env = os.environ.copy()
    source_path = str(root / "src")
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "gemati.benchmark_worker",
        "--root",
        str(root),
        "--variant",
        variant,
        "--model-name",
        model.model_name,
        "--max-length",
        str(model.max_length),
        "--encoder-batch-size",
        str(model.batch_size),
        "--output",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["worker_stderr"] = completed.stderr.strip()
        return result
    finally:
        output_path.unlink(missing_ok=True)


def run_extended(root: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    results_dir = root / "results"
    tracker = ProgressTracker(results_dir / "extended_progress.json", total_steps=12)
    classes = list(config["risk_labels"])
    actions = list(config["actions"])
    costs = config["cost_matrix"]
    train_full = pd.read_parquet(root / "data" / "raw" / "cradle_train.parquet")
    validation = pd.read_parquet(root / "data" / "raw" / "cradle_validation.parquet")
    test = pd.read_parquet(root / "data" / "raw" / "cradle_test.parquet")
    external = pd.read_parquet(root / "data" / "raw" / "indosafety_external.parquet").reset_index(drop=True)
    bundle = joblib.load(root / "models" / "gemati_care.joblib")
    model: MiniLMRiskModel = bundle["model"]
    model.progress_callback = lambda message: tracker.update("embedding", message)

    tracker.update("embedding", "Membentuk/memuat embedding penuh CRADLE")
    full_x = model.encode(train_full["text"].tolist())
    validation_x = model.encode(validation["text"].tolist())
    test_x = model.encode(test["text"].tolist())
    tracker.update("embedding", "Embedding penuh siap", increment=True)

    five_seed = {}
    tfidf_five_seed = {}
    seed_internal = {}
    tfidf_seed_internal = {}
    for seed in SEEDS:
        train_idx, cal_idx = train_test_split(
            np.arange(len(train_full)),
            test_size=0.30,
            random_state=seed,
            stratify=train_full["risk"],
        )
        run = _fit_evaluate(
            full_x[train_idx],
            train_full.iloc[train_idx]["risk"].tolist(),
            full_x[cal_idx],
            train_full.iloc[cal_idx]["risk"].tolist(),
            validation_x,
            validation["risk"].tolist(),
            classes,
            actions,
            costs,
            seed,
        )
        five_seed[str(seed)] = _strip_model_objects(run)
        tfidf_run = _fit_evaluate_tfidf(
            train_full.iloc[train_idx]["text"].tolist(),
            train_full.iloc[train_idx]["risk"].tolist(),
            train_full.iloc[cal_idx]["text"].tolist(),
            train_full.iloc[cal_idx]["risk"].tolist(),
            validation["text"].tolist(),
            validation["risk"].tolist(),
            classes,
            actions,
            costs,
            config,
            seed,
        )
        tfidf_five_seed[str(seed)] = _strip_model_objects(tfidf_run)
        tfidf_seed_internal[seed] = tfidf_run
        seed_internal[seed] = (run, train_idx, cal_idx)
        tracker.update("five_seed", f"Seed {seed} selesai")
    five_summary = {
        "macro_f1": _seed_summary([five_seed[str(s)]["classification"]["macro_f1"] for s in SEEDS]),
        "expected_cost": _seed_summary(
            [five_seed[str(s)]["routing_minimax"]["expected_cost"] for s in SEEDS]
        ),
        "under_escalation": _seed_summary(
            [five_seed[str(s)]["routing_minimax"]["under_escalation_rate"] for s in SEEDS]
        ),
        "over_escalation": _seed_summary(
            [five_seed[str(s)]["routing_minimax"]["over_escalation_rate"] for s in SEEDS]
        ),
        "high_risk_miss": _seed_summary(
            [five_seed[str(s)]["routing_minimax"]["high_risk_miss_rate"] for s in SEEDS]
        ),
    }
    tfidf_five_summary = {
        "macro_f1": _seed_summary(
            [tfidf_five_seed[str(s)]["classification"]["macro_f1"] for s in SEEDS]
        ),
        "expected_cost": _seed_summary(
            [tfidf_five_seed[str(s)]["routing_minimax"]["expected_cost"] for s in SEEDS]
        ),
        "under_escalation": _seed_summary(
            [
                tfidf_five_seed[str(s)]["routing_minimax"]["under_escalation_rate"]
                for s in SEEDS
            ]
        ),
        "over_escalation": _seed_summary(
            [
                tfidf_five_seed[str(s)]["routing_minimax"]["over_escalation_rate"]
                for s in SEEDS
            ]
        ),
        "high_risk_miss": _seed_summary(
            [
                tfidf_five_seed[str(s)]["routing_minimax"]["high_risk_miss_rate"]
                for s in SEEDS
            ]
        ),
    }
    tracker.update("five_seed", "Lima repeated stratified splits selesai", increment=True)

    fixed_dev, fixed_train_idx, fixed_cal_idx = seed_internal[42]
    fixed_tfidf_dev = tfidf_seed_internal[42]
    final_test = _fit_evaluate(
        full_x[fixed_train_idx],
        train_full.iloc[fixed_train_idx]["risk"].tolist(),
        full_x[fixed_cal_idx],
        train_full.iloc[fixed_cal_idx]["risk"].tolist(),
        test_x,
        test["risk"].tolist(),
        classes,
        actions,
        costs,
        42,
    )
    final_tfidf = _fit_evaluate_tfidf(
        train_full.iloc[fixed_train_idx]["text"].tolist(),
        train_full.iloc[fixed_train_idx]["risk"].tolist(),
        train_full.iloc[fixed_cal_idx]["text"].tolist(),
        train_full.iloc[fixed_cal_idx]["risk"].tolist(),
        test["text"].tolist(),
        test["risk"].tolist(),
        classes,
        actions,
        costs,
        config,
        42,
    )
    safety_cp = MondrianConformal(0.05, classes).fit(
        final_tfidf["calibration_probabilities"], final_tfidf["conformal_labels"]
    )
    safety_sets = safety_cp.predict_sets(final_tfidf["test_probabilities"])
    safety_actions = route_sets(safety_sets, classes, actions, costs)
    final_tfidf_safety = {
        "alpha": 0.05,
        "selection_rule": "lowest tested alpha achieving validation coverage >= 0.90",
        "classification": final_tfidf["classification"],
        "conformal": conformal_metrics(safety_sets, test["risk"], classes),
        "routing": routing_metrics(safety_actions, test["risk"], classes, costs),
    }
    alpha_sensitivity = {}
    for alpha in ALPHAS:
        cp = MondrianConformal(alpha, classes).fit(
            fixed_tfidf_dev["calibration_probabilities"],
            fixed_tfidf_dev["conformal_labels"],
        )
        sets = cp.predict_sets(fixed_tfidf_dev["test_probabilities"])
        routed = route_sets(sets, classes, actions, costs)
        alpha_sensitivity[str(alpha)] = {
            "conformal": conformal_metrics(sets, validation["risk"].tolist(), classes),
            "routing": routing_metrics(routed, validation["risk"].tolist(), classes, costs),
        }

    cost_sensitivity = {}
    for name, matrix in COST_MATRICES.items():
        routed = route_sets(fixed_tfidf_dev["sets"], classes, actions, matrix)
        cost_sensitivity[name] = routing_metrics(
            routed, validation["risk"].tolist(), classes, matrix
        )

    tie_sensitivity = {}
    for name, priority in TIE_POLICIES.items():
        routed = route_sets(
            fixed_tfidf_dev["sets"], classes, actions, TIE_COST_MATRIX, priority
        )
        tie_sensitivity[name] = routing_metrics(
            routed, validation["risk"].tolist(), classes, TIE_COST_MATRIX
        )

    threshold_sensitivity = {}
    for threshold in HIGH_THRESHOLDS:
        routed = threshold_actions(
            fixed_tfidf_dev["test_probabilities"], classes, threshold
        )
        threshold_sensitivity[str(threshold)] = routing_metrics(
            routed, validation["risk"].tolist(), classes, costs
        )

    pool_train_idx, calibration_pool_idx = train_test_split(
        np.arange(len(train_full)),
        test_size=0.40,
        random_state=42,
        stratify=train_full["risk"],
    )
    temp_idx, conformal_pool_idx = train_test_split(
        calibration_pool_idx,
        train_size=0.25,
        random_state=42,
        stratify=train_full.iloc[calibration_pool_idx]["risk"],
    )
    pool_head = build_tfidf_classifier(config)
    pool_head.fit(
        train_full.iloc[pool_train_idx]["text"],
        train_full.iloc[pool_train_idx]["risk"],
    )
    raw_temp = _reorder(
        pool_head.predict_proba(train_full.iloc[temp_idx]["text"]),
        pool_head.classes_,
        classes,
    )
    temp_y = np.asarray([classes.index(label) for label in train_full.iloc[temp_idx]["risk"]])
    size_scaler = TemperatureScaler().fit(raw_temp, temp_y)
    validation_p = size_scaler.transform(
        _reorder(pool_head.predict_proba(validation["text"]), pool_head.classes_, classes)
    )
    calibration_size = {}
    for fraction in CALIBRATION_FRACTIONS:
        desired = int(round(len(train_full) * fraction))
        if desired >= len(conformal_pool_idx):
            selected = conformal_pool_idx
        else:
            selected, _ = train_test_split(
                conformal_pool_idx,
                train_size=desired,
                random_state=42,
                stratify=train_full.iloc[conformal_pool_idx]["risk"],
            )
        raw_cal = _reorder(
            pool_head.predict_proba(train_full.iloc[selected]["text"]),
            pool_head.classes_,
            classes,
        )
        cal_p = size_scaler.transform(raw_cal)
        cp = MondrianConformal(0.10, classes).fit(cal_p, train_full.iloc[selected]["risk"])
        sets = cp.predict_sets(validation_p)
        routed = route_sets(sets, classes, actions, costs)
        calibration_size[str(fraction)] = {
            "probability_calibration_n": int(len(temp_idx)),
            "conformal_calibration_n": int(len(selected)),
            "conformal": conformal_metrics(sets, validation["risk"], classes),
            "routing": routing_metrics(routed, validation["risk"], classes, costs),
        }
    tracker.update("sensitivity", "Alpha, biaya, calibration size, threshold, dan tie policy selesai", increment=True)

    no_calibration = _fit_evaluate_tfidf(
        train_full.iloc[fixed_train_idx]["text"],
        train_full.iloc[fixed_train_idx]["risk"],
        train_full.iloc[fixed_cal_idx]["text"],
        train_full.iloc[fixed_cal_idx]["risk"],
        validation["text"],
        validation["risk"],
        classes,
        actions,
        costs,
        config,
        42,
        calibrate=False,
    )
    no_weight = _fit_evaluate_tfidf(
        train_full.iloc[fixed_train_idx]["text"],
        train_full.iloc[fixed_train_idx]["risk"],
        train_full.iloc[fixed_cal_idx]["text"],
        train_full.iloc[fixed_cal_idx]["risk"],
        validation["text"],
        validation["risk"],
        classes,
        actions,
        costs,
        config,
        42,
        weighted=False,
    )
    ablation = {
        "evaluation_split": "official_validation",
        "full": _strip_model_objects(fixed_tfidf_dev),
        "without_calibration": _strip_model_objects(no_calibration),
        "without_class_weight": _strip_model_objects(no_weight),
        "global_vs_mondrian": {
            "global": fixed_tfidf_dev["global_metrics"],
            "mondrian": fixed_tfidf_dev["mondrian_metrics"],
        },
        "routing_policy": {
            "argmax": fixed_tfidf_dev["routing_argmax"],
            "expected_cost": fixed_tfidf_dev["routing_expected"],
            "minimax": fixed_tfidf_dev["routing_minimax"],
        },
        "tfidf_vs_minilm": {
            "tfidf": fixed_tfidf_dev["classification"],
            "minilm": fixed_dev["classification"],
        },
    }
    tracker.update("ablation", "Ablation kalibrasi, conformal, routing, bobot, dan encoder selesai", increment=True)

    optimized_model = MiniLMRiskModel(
        cache_dir=root / ".cache" / "embeddings",
        model_name=config["encoder_model"],
        batch_size=config["encoder_batch_size"],
        max_length=config["encoder_max_length"],
        c=config["logistic_c"],
        max_iter=config["max_iter"],
        seed=42,
        optimized=True,
        progress_callback=lambda message: tracker.update("int8_embedding", message),
    )
    optimized_x = optimized_model.encode(train_full["text"].tolist())
    optimized_run = _fit_evaluate(
        optimized_x[fixed_train_idx],
        train_full.iloc[fixed_train_idx]["risk"],
        optimized_x[fixed_cal_idx],
        train_full.iloc[fixed_cal_idx]["risk"],
        optimized_model.encode(validation["text"].tolist()),
        validation["risk"],
        classes,
        actions,
        costs,
        42,
    )
    ablation["fp32_vs_int8"] = {
        "fp32": _strip_model_objects(fixed_dev),
        "dynamic_int8": _strip_model_objects(optimized_run),
    }
    tracker.update("int8", "Dynamic INT8 ablation selesai", increment=True)

    vulnerable, controls = _load_sim_vail(root)
    model.classifier = final_test["classifier"]
    temporal = _temporal_audit(
        model,
        final_test["scaler"],
        final_test["mondrian"],
        vulnerable,
        controls,
        classes,
        actions,
        costs,
        config["temporal_rho"],
        config["temporal_high"],
        config["temporal_low"],
    )
    temporal_sensitivity = {}
    for rho, high, low in (
        (0.00, 0.65, 0.35),
        (0.70, 0.65, 0.35),
        (0.85, 0.55, 0.25),
        (0.85, 0.75, 0.45),
        (0.95, 0.65, 0.35),
    ):
        key = f"rho={rho:.2f};high={high:.2f};low={low:.2f}"
        temporal_sensitivity[key] = _temporal_audit(
            model,
            final_test["scaler"],
            final_test["mondrian"],
            vulnerable,
            controls,
            classes,
            actions,
            costs,
            rho,
            high,
            low,
        )["summary"]
    ablation["without_vs_with_temporal_memory"] = temporal["summary"]
    tracker.update("temporal", "Audit temporal SIM-VAIL selesai", increment=True)

    efficiency = {
        "fp32": _efficiency_benchmark(root, model, test["text"].tolist(), False),
        "dynamic_int8": _efficiency_benchmark(
            root, optimized_model, test["text"].tolist(), True
        ),
    }
    tracker.update("efficiency", "Benchmark latency, throughput, RAM, size, dan cold start selesai", increment=True)

    bootstrap = _bootstrap_all(
        test["risk"].tolist(),
        final_tfidf["test_probabilities"],
        safety_actions,
        safety_sets,
        classes,
        costs,
        config["bootstrap_repetitions"],
        42,
    )
    tracker.update("statistics", "Bootstrap multi-metrik selesai", increment=True)

    fixed_train = train_full.iloc[fixed_train_idx].reset_index(drop=True)
    fixed_cal = train_full.iloc[fixed_cal_idx].reset_index(drop=True)
    leakage = _leakage_audit(
        fixed_train,
        fixed_cal,
        validation,
        test,
        full_x[fixed_train_idx],
        full_x[fixed_cal_idx],
        validation_x,
        test_x,
    )
    flagged_ids = set(leakage["test_ids_flagged_at_0.90_against_train_or_calibration"])
    retained = ~test["id"].astype(str).isin(flagged_ids).to_numpy()
    retained_sets = [item for item, keep in zip(safety_sets, retained) if keep]
    retained_actions = [
        item for item, keep in zip(safety_actions, retained) if keep
    ]
    retained_labels = test.loc[retained, "risk"].tolist()
    leakage["sensitivity_excluding_flagged_test_rows"] = {
        "excluded_n": int((~retained).sum()),
        "retained_n": int(retained.sum()),
        "classification": classification_metrics(
            retained_labels, final_tfidf["test_probabilities"][retained], classes
        ),
        "conformal": conformal_metrics(retained_sets, retained_labels, classes),
        "routing": routing_metrics(retained_actions, retained_labels, classes, costs),
    }
    tracker.update("leakage", "Audit exact dan semantic near-duplicate selesai", increment=True)

    indonesia = _indonesian_compatible_evaluation(root, model, external, 42)
    error_analysis = {
        "cradle_linguistic_phenomena": _error_taxonomy(
            test, final_tfidf["test_probabilities"], classes
        ),
        "indosafety_by_language": indonesia["by_language"],
        "sim_vail_by_phenotype": temporal["by_phenotype"],
        "worst_groups": {
            "risk_class_recall": {
                label: final_tfidf["classification"]["per_class"][label]["recall"]
                for label in classes
            },
            "indosafety_worst_language_macro_f1": indonesia["worst_language_macro_f1"],
            "sim_vail_worst_phenotype_escalation": temporal[
                "worst_phenotype_temporal_any_escalation"
            ],
        },
    }
    tracker.update("indonesia_error", "Evaluasi Indonesia dan error/worst-group analysis selesai", increment=True)

    mapping_sensitivity = {}
    for variant in ("narrow", "current", "broad"):
        train_labels = train_full["source_label"].map(lambda item: _mapping(item, variant))
        validation_labels = validation["source_label"].map(lambda item: _mapping(item, variant))
        tr_idx, ca_idx = train_test_split(
            np.arange(len(train_full)),
            test_size=0.30,
            random_state=42,
            stratify=train_labels,
        )
        run = _fit_evaluate(
            full_x[tr_idx],
            train_labels.iloc[tr_idx],
            full_x[ca_idx],
            train_labels.iloc[ca_idx],
            validation_x,
            validation_labels,
            classes,
            actions,
            costs,
            42,
        )
        mapping_sensitivity[variant] = {
            "train_distribution": train_labels.value_counts().to_dict(),
            "validation_distribution": validation_labels.value_counts().to_dict(),
            "results": _strip_model_objects(run),
        }
    tracker.update("mapping", "Sensitivity pemetaan label manual selesai", increment=True)

    results = {
        "protocol": {
            "seeds": SEEDS,
            "alphas": ALPHAS,
            "calibration_fractions": CALIBRATION_FRACTIONS,
            "high_thresholds": HIGH_THRESHOLDS,
            "bootstrap_repetitions": config["bootstrap_repetitions"],
            "cpu_only": True,
            "development_evaluation_split": "official_validation",
            "final_evaluation_split": "official_test evaluated once after fixing seed 42 protocol",
            "split_protocol": "70% train; 10% probability calibration; 20% conformal calibration",
            "probability_and_conformal_calibration_disjoint": True,
            "dataset_rows": {
                "cradle_public_train_pool": int(len(train_full)),
                "model_training": int(len(fixed_train_idx)),
                "probability_calibration": int(
                    final_tfidf["calibration_protocol"]["probability_calibration_n"]
                ),
                "conformal_calibration": int(
                    final_tfidf["calibration_protocol"]["conformal_calibration_n"]
                ),
                "official_validation": int(len(validation)),
                "official_test": int(len(test)),
                "indosafety_eval2": int(indonesia["test_rows"]),
                "sim_vail_vulnerable_conversations": int(len(vulnerable)),
                "sim_vail_control_conversations": int(len(controls)),
            },
        },
        "five_seed": five_seed,
        "five_seed_summary": five_summary,
        "tfidf_five_seed": tfidf_five_seed,
        "tfidf_five_seed_summary": tfidf_five_summary,
        "model_selection": {
            "split": "official_validation",
            "criterion": "macro_f1",
            "selected": "TF-IDF word+character n-grams",
            "tfidf_macro_f1": fixed_tfidf_dev["classification"]["macro_f1"],
            "minilm_macro_f1": fixed_dev["classification"]["macro_f1"],
        },
        "final_test_primary_alpha_0.10": _strip_model_objects(final_tfidf),
        "final_test_safety_alpha_0.05": _json_ready(final_tfidf_safety),
        "final_test_minilm_secondary": _strip_model_objects(final_test),
        "sensitivity": {
            "evaluation_split": "official_validation",
            "alpha": alpha_sensitivity,
            "cost_matrix": cost_sensitivity,
            "calibration_size": calibration_size,
            "threshold": threshold_sensitivity,
            "tie_policy": tie_sensitivity,
        },
        "ablation": _json_ready(ablation),
        "temporal_sim_vail": _json_ready(temporal),
        "temporal_sensitivity": _json_ready(temporal_sensitivity),
        "efficiency_cpu": _json_ready(efficiency),
        "bootstrap_operating_point": "TF-IDF safety mode, alpha=0.05",
        "bootstrap_confidence_intervals": _json_ready(bootstrap),
        "leakage_audit": _json_ready(leakage),
        "indonesian_compatible_evaluation": _json_ready(indonesia),
        "error_analysis": _json_ready(error_analysis),
        "label_mapping_sensitivity": _json_ready(mapping_sensitivity),
        "strict_limitations": [
            "No experiment establishes clinical safety or therapeutic effectiveness.",
            "SIM-VAIL provides synthetic condition-level audits, not per-turn clinical labels.",
            "SIM-VAIL temporal analysis measures persistence only; it does not establish improved detection or safety.",
            "IndoSafety evaluation is an auxiliary representation probe for taxonomy membership, not external validation of crisis severity routing.",
            "CRADLE release lacks source/subreddit fields, so source-wise analysis is unavailable.",
            "Five repeated stratified development splits use official validation; they are not grouped splits.",
            "Public data use does not itself determine ethical exemption; institutional determination may still be required.",
        ],
    }
    output = results_dir / "extended_metrics.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tracker.complete("Seluruh suite eksperimen tambahan selesai")
    return results

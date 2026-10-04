from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from .routing import expected_cost


def expected_calibration_error(probabilities: np.ndarray, y_index: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == y_index
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def multiclass_brier(probabilities: np.ndarray, y_index: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[y_index]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(
    labels_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]
) -> dict:
    y_index = np.asarray([list(classes).index(label) for label in labels_true])
    predicted_index = probabilities.argmax(axis=1)
    predicted_labels = np.asarray(classes)[predicted_index]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_true, predicted_labels, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels_true, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_true, predicted_labels)),
        "macro_f1": float(f1_score(labels_true, predicted_labels, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probabilities, y_index),
        "brier": multiclass_brier(probabilities, y_index),
        "nll": float(log_loss(y_index, probabilities, labels=np.arange(len(classes)))),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(labels_true, predicted_labels, labels=classes).tolist(),
    }


def conformal_metrics(prediction_sets: Sequence[set[str]], labels_true: Sequence[str], classes: Sequence[str]) -> dict:
    covered = np.asarray([label in pred_set for label, pred_set in zip(labels_true, prediction_sets)])
    sizes = np.asarray([len(pred_set) for pred_set in prediction_sets])
    result = {
        "coverage": float(covered.mean()),
        "average_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "empty_rate": float((sizes == 0).mean()),
        "per_class_coverage": {},
    }
    labels_array = np.asarray(labels_true)
    for label in classes:
        mask = labels_array == label
        result["per_class_coverage"][label] = float(covered[mask].mean()) if mask.any() else None
    return result


def routing_metrics(
    actions_pred: Sequence[str],
    labels_true: Sequence[str],
    labels: Sequence[str],
    cost_matrix: dict[str, Sequence[float]],
) -> dict:
    true_action = {"low": "SUPPORT", "medium": "CLARIFY", "high": "ESCALATE"}
    severity = {"SUPPORT": 0, "CLARIFY": 1, "ESCALATE": 2}
    expected = [true_action[label] for label in labels_true]
    under = [severity[pred] < severity[truth] for pred, truth in zip(actions_pred, expected)]
    over = [severity[pred] > severity[truth] for pred, truth in zip(actions_pred, expected)]
    high_mask = np.asarray(labels_true) == "high"
    high_missed = np.asarray([action != "ESCALATE" for action in actions_pred])[high_mask]
    return {
        "action_accuracy": float(np.mean(np.asarray(actions_pred) == np.asarray(expected))),
        "under_escalation_rate": float(np.mean(under)),
        "over_escalation_rate": float(np.mean(over)),
        "high_risk_miss_rate": float(high_missed.mean()) if len(high_missed) else None,
        "expected_cost": expected_cost(actions_pred, labels_true, labels, cost_matrix),
        "action_distribution": {
            action: int(sum(item == action for item in actions_pred))
            for action in ("SUPPORT", "CLARIFY", "ESCALATE")
        },
    }


def grouped_bootstrap_difference(
    labels_true: Sequence[str],
    action_a: Sequence[str],
    action_b: Sequence[str],
    groups: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    repetitions: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels_true)
    a_array = np.asarray(action_a)
    b_array = np.asarray(action_b)
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    differences = []
    for _ in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(group_array == group) for group in sampled])
        differences.append(metric(a_array[indices], labels_array[indices]) - metric(b_array[indices], labels_array[indices]))
    values = np.asarray(differences)
    return {
        "mean_difference": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "repetitions": repetitions,
    }


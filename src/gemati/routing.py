from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def robust_cost_action(
    prediction_set: set[str],
    labels: Sequence[str],
    actions: Sequence[str],
    cost_matrix: dict[str, Sequence[float]],
    priority: Sequence[str] = ("ESCALATE", "CLARIFY", "SUPPORT"),
) -> str:
    if not prediction_set:
        return "CLARIFY"
    index = {label: i for i, label in enumerate(labels)}
    worst_costs = {
        action: max(float(cost_matrix[action][index[label]]) for label in prediction_set)
        for action in actions
    }
    minimum = min(worst_costs.values())
    candidates = [action for action in actions if worst_costs[action] == minimum]
    # Conservative deterministic tie-breaking.
    return next(action for action in priority if action in candidates)


def route_sets(
    prediction_sets: Sequence[set[str]],
    labels: Sequence[str],
    actions: Sequence[str],
    cost_matrix: dict[str, Sequence[float]],
    priority: Sequence[str] = ("ESCALATE", "CLARIFY", "SUPPORT"),
) -> list[str]:
    return [robust_cost_action(item, labels, actions, cost_matrix, priority) for item in prediction_sets]


def expected_cost_actions(
    probabilities: np.ndarray,
    labels: Sequence[str],
    actions: Sequence[str],
    cost_matrix: dict[str, Sequence[float]],
    priority: Sequence[str] = ("ESCALATE", "CLARIFY", "SUPPORT"),
) -> list[str]:
    output = []
    for row in probabilities:
        costs = {
            action: float(np.dot(row, np.asarray(cost_matrix[action], dtype=float)))
            for action in actions
        }
        minimum = min(costs.values())
        candidates = [action for action in actions if np.isclose(costs[action], minimum)]
        output.append(next(action for action in priority if action in candidates))
    return output


def threshold_actions(
    probabilities: np.ndarray,
    classes: Sequence[str],
    high_threshold: float,
) -> list[str]:
    base = argmax_actions(probabilities, classes)
    high_index = list(classes).index("high")
    return [
        "ESCALATE" if row[high_index] >= high_threshold else action
        for row, action in zip(probabilities, base)
    ]


def argmax_actions(probabilities: np.ndarray, classes: Sequence[str]) -> list[str]:
    mapping = {"low": "SUPPORT", "medium": "CLARIFY", "high": "ESCALATE"}
    return [mapping[classes[index]] for index in probabilities.argmax(axis=1)]


@dataclass
class TemporalRiskState:
    labels: Sequence[str]
    rho: float = 0.85
    high_threshold: float = 0.65
    low_threshold: float = 0.35
    risk: np.ndarray = field(init=False)
    escalated: bool = False

    def __post_init__(self) -> None:
        self.risk = np.zeros(len(self.labels), dtype=float)

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        self.risk = np.maximum(probabilities, self.rho * self.risk)
        high_index = list(self.labels).index("high")
        if self.risk[high_index] >= self.high_threshold:
            self.escalated = True
        elif self.risk[high_index] < self.low_threshold:
            self.escalated = False
        return self.risk.copy()

    def apply_hysteresis(self, action: str) -> str:
        return "ESCALATE" if self.escalated else action


def expected_cost(
    actions_pred: Sequence[str],
    true_labels: Sequence[str],
    labels: Sequence[str],
    cost_matrix: dict[str, Sequence[float]],
) -> float:
    index = {label: i for i, label in enumerate(labels)}
    costs = [float(cost_matrix[action][index[label]]) for action, label in zip(actions_pred, true_labels)]
    return float(np.mean(costs))

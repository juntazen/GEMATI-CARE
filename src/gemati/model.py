from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline


EPS = 1e-12


def build_tfidf_classifier(config: dict) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=config["max_features_word"],
                    min_df=config["min_df"],
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=config["max_features_char"],
                    min_df=config["min_df"],
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=config["logistic_c"],
            class_weight="balanced",
            max_iter=config["max_iter"],
            solver="liblinear",
            random_state=config["seed"],
        )
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPS, 1 - EPS)
    logits = np.log(clipped)
    return logits - logits.mean(axis=1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, probabilities: np.ndarray, y_index: np.ndarray) -> "TemperatureScaler":
        logits = _probabilities_to_logits(probabilities)

        def objective(log_temperature: float) -> float:
            temperature = math.exp(log_temperature)
            calibrated = _softmax(logits / temperature)
            chosen = calibrated[np.arange(len(y_index)), y_index]
            return float(-np.mean(np.log(np.clip(chosen, EPS, 1.0))))

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
        self.temperature = float(math.exp(result.x))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return _softmax(_probabilities_to_logits(probabilities) / self.temperature)


@dataclass
class MondrianConformal:
    alpha: float
    classes: Sequence[str]
    quantiles: dict[str, float] | None = None

    @staticmethod
    def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
        if len(scores) == 0:
            return 1.0
        rank = math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)
        level = min(1.0, rank)
        return float(np.quantile(scores, level, method="higher"))

    def fit(self, probabilities: np.ndarray, labels: Sequence[str]) -> "MondrianConformal":
        class_to_index = {label: i for i, label in enumerate(self.classes)}
        labels_array = np.asarray(labels)
        self.quantiles = {}
        for label in self.classes:
            mask = labels_array == label
            scores = 1.0 - probabilities[mask, class_to_index[label]]
            self.quantiles[label] = self._finite_sample_quantile(scores, self.alpha)
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[set[str]]:
        if self.quantiles is None:
            raise RuntimeError("MondrianConformal belum di-fit.")
        prediction_sets: list[set[str]] = []
        for row in probabilities:
            selected = {
                label
                for index, label in enumerate(self.classes)
                if 1.0 - row[index] <= self.quantiles[label]
            }
            prediction_sets.append(selected)
        return prediction_sets


@dataclass
class GlobalConformal:
    alpha: float
    classes: Sequence[str]
    quantile: float | None = None

    def fit(self, probabilities: np.ndarray, labels: Sequence[str]) -> "GlobalConformal":
        class_to_index = {label: i for i, label in enumerate(self.classes)}
        indices = np.asarray([class_to_index[label] for label in labels])
        scores = 1.0 - probabilities[np.arange(len(indices)), indices]
        self.quantile = MondrianConformal._finite_sample_quantile(scores, self.alpha)
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[set[str]]:
        if self.quantile is None:
            raise RuntimeError("GlobalConformal belum di-fit.")
        return [
            {label for index, label in enumerate(self.classes) if 1.0 - row[index] <= self.quantile}
            for row in probabilities
        ]

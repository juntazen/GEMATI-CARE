import numpy as np

from gemati.data import map_cradle_risk
from gemati.experiment import _predict_in_class_order
from gemati.model import MondrianConformal, TemperatureScaler, build_tfidf_classifier
from gemati.routing import TemporalRiskState, robust_cost_action


def test_label_mapping_is_conservative():
    assert map_cradle_risk("no_crisis") == "low"
    assert map_cradle_risk("suicideideation_active_ongoing") == "high"
    assert map_cradle_risk("suicideideation_passive_past") == "medium"


def test_robust_router_escalates_when_low_and_high_are_possible():
    costs = {
        "SUPPORT": [0.0, 3.0, 12.0],
        "CLARIFY": [1.0, 0.5, 5.0],
        "ESCALATE": [3.0, 1.0, 0.0],
    }
    action = robust_cost_action(
        {"low", "high"}, ["low", "medium", "high"], ["SUPPORT", "CLARIFY", "ESCALATE"], costs
    )
    assert action == "ESCALATE"


def test_empty_conformal_set_abstains():
    costs = {"SUPPORT": [0, 3, 12], "CLARIFY": [1, 0.5, 5], "ESCALATE": [3, 1, 0]}
    assert (
        robust_cost_action(set(), ["low", "medium", "high"], list(costs), costs) == "CLARIFY"
    )


def test_temperature_scaler_preserves_probability_simplex():
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]])
    scaler = TemperatureScaler().fit(probabilities, np.asarray([0, 1, 2]))
    transformed = scaler.transform(probabilities)
    assert np.allclose(transformed.sum(axis=1), 1.0)
    assert scaler.temperature > 0


def test_mondrian_sets_include_calibration_truth_at_reasonable_rate():
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
    )
    labels = ["low", "low", "medium", "high"]
    cp = MondrianConformal(0.1, ["low", "medium", "high"]).fit(probabilities, labels)
    sets = cp.predict_sets(probabilities)
    assert all(label in prediction for label, prediction in zip(labels, sets))


def test_temporal_hysteresis_prevents_immediate_deescalation():
    state = TemporalRiskState(["low", "medium", "high"], rho=0.85, high_threshold=0.65, low_threshold=0.35)
    state.update(np.asarray([0.05, 0.10, 0.85]))
    assert state.apply_hysteresis("SUPPORT") == "ESCALATE"
    state.update(np.asarray([0.90, 0.08, 0.02]))
    assert state.apply_hysteresis("SUPPORT") == "ESCALATE"


def test_classifier_supports_three_classes():
    config = {
        "max_features_word": 100,
        "max_features_char": 100,
        "min_df": 1,
        "logistic_c": 1.0,
        "max_iter": 100,
        "seed": 42,
    }
    model = build_tfidf_classifier(config)
    model.fit(
        ["semua baik", "tidak yakin dan takut", "bahaya sekarang"] * 2,
        ["low", "medium", "high"] * 2,
    )
    assert set(model.classes_) == {"low", "medium", "high"}
    canonical = ["low", "medium", "high"]
    reordered = _predict_in_class_order(model, ["bahaya sekarang"], canonical)
    raw = model.predict_proba(["bahaya sekarang"])
    for i, label in enumerate(canonical):
        assert np.isclose(reordered[0, i], raw[0, list(model.classes_).index(label)])

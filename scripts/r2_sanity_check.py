#!/usr/bin/env python3
"""GEMATI-CARE R2 — Step 2 sanity gate.

Reproduces the locked manuscript numbers (Tables 3 and 5, Table 7 validation alpha = 0.05)
with the package functions and compares them to 4 decimals. Writes
results/r2/sanity_check.json, and results/r2/SANITY_FAIL.md on any mismatch (exit code 1).

    PYTHONPATH=src python scripts/r2_sanity_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemati.extended import _fit_evaluate_tfidf  # noqa: E402
from gemati.metrics import classification_metrics, conformal_metrics, routing_metrics  # noqa: E402
from gemati.model import MondrianConformal  # noqa: E402
from gemati.routing import argmax_actions, route_sets  # noqa: E402

OUT = ROOT / "results" / "r2"

EXPECTED = {
    "confusion_matrix": [[123, 34, 29], [19, 139, 22], [26, 29, 179]],
    "macro_f1": 0.7314,
    "ece": 0.0323,
    "temperature": 0.5344,
    "minimax_0.05_miss": 0.0513,
    "minimax_0.05_under": 0.0217,
    "minimax_0.05_over": 0.3933,
    "minimax_0.05_cost": 0.9583,
    "minimax_0.05_SCE": [33, 154, 413],
    "minimax_0.10_coverage": 0.8833,
    "minimax_0.10_miss": 0.1026,
    "minimax_0.10_over": 0.2683,
    "argmax_miss": 0.2350,
    "argmax_over": 0.1417,
    "argmax_cost": 1.2108,
    "validation_0.05_high_coverage": 0.9477,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/experiment.json").read_text())
    classes, actions, costs = config["risk_labels"], config["actions"], config["cost_matrix"]
    train = pd.read_parquet(ROOT / "data/raw/cradle_train.parquet")
    val = pd.read_parquet(ROOT / "data/raw/cradle_validation.parquet")
    test = pd.read_parquet(ROOT / "data/raw/cradle_test.parquet")
    train_idx, cal_idx = train_test_split(
        np.arange(len(train)), test_size=0.30, random_state=42, stratify=train["risk"])

    def fit(eval_df):
        return _fit_evaluate_tfidf(
            train.iloc[train_idx]["text"].tolist(), train.iloc[train_idx]["risk"].tolist(),
            train.iloc[cal_idx]["text"].tolist(), train.iloc[cal_idx]["risk"].tolist(),
            eval_df["text"].tolist(), eval_df["risk"].tolist(), classes, actions, costs, config, 42)

    run_t = fit(test)
    run_v = fit(val)
    yt, yv = test["risk"].tolist(), val["risk"].tolist()
    pt = run_t["test_probabilities"]
    cls = classification_metrics(yt, pt, classes)

    def minimax(alpha, probs):
        cp = MondrianConformal(alpha, classes).fit(run_t["calibration_probabilities"], run_t["conformal_labels"])
        sets = cp.predict_sets(probs)
        return sets, route_sets(sets, classes, actions, costs)

    s05, a05 = minimax(0.05, pt)
    s10, a10 = minimax(0.10, pt)
    m05 = routing_metrics(a05, yt, classes, costs)
    m10 = routing_metrics(a10, yt, classes, costs)
    am = routing_metrics(argmax_actions(pt, classes), yt, classes, costs)
    cpv = MondrianConformal(0.05, classes).fit(run_v["calibration_probabilities"], run_v["conformal_labels"])
    cov_v = conformal_metrics(cpv.predict_sets(run_v["test_probabilities"]), yv, classes)

    observed = {
        "confusion_matrix": cls["confusion_matrix"],
        "macro_f1": cls["macro_f1"],
        "ece": cls["ece"],
        "temperature": run_t["temperature"],
        "minimax_0.05_miss": m05["high_risk_miss_rate"],
        "minimax_0.05_under": m05["under_escalation_rate"],
        "minimax_0.05_over": m05["over_escalation_rate"],
        "minimax_0.05_cost": m05["expected_cost"],
        "minimax_0.05_SCE": [m05["action_distribution"][k] for k in ("SUPPORT", "CLARIFY", "ESCALATE")],
        "minimax_0.10_coverage": conformal_metrics(s10, yt, classes)["coverage"],
        "minimax_0.10_miss": m10["high_risk_miss_rate"],
        "minimax_0.10_over": m10["over_escalation_rate"],
        "argmax_miss": am["high_risk_miss_rate"],
        "argmax_over": am["over_escalation_rate"],
        "argmax_cost": am["expected_cost"],
        "validation_0.05_high_coverage": cov_v["per_class_coverage"]["high"],
    }

    rows, failures = [], []
    for key, exp in EXPECTED.items():
        obs = observed[key]
        ok = obs == exp if isinstance(exp, list) else round(float(obs), 4) == exp
        rows.append({"quantity": key, "expected": exp, "observed": obs, "match": ok})
        print(f"{'PASS' if ok else 'FAIL'}  {key}: expected {exp}, observed {obs}")
        if not ok:
            failures.append(rows[-1])

    (OUT / "sanity_check.json").write_text(json.dumps(
        {"all_match": not failures, "checks": rows}, indent=1, default=float))
    if failures:
        lines = ["# SANITY FAIL", "", "| Quantity | Expected | Observed |", "|---|---|---|"]
        lines += [f"| {r['quantity']} | {r['expected']} | {r['observed']} |" for r in failures]
        (OUT / "SANITY_FAIL.md").write_text("\n".join(lines) + "\n")
        print(f"SANITY GATE: FAIL ({len(failures)} mismatches)")
        return 1
    print("SANITY GATE: PASS (all values match to 4 decimals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

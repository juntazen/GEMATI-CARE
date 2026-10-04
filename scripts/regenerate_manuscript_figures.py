#!/usr/bin/env python3
"""Regenerate manuscript plots from the final TF-IDF protocol."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemati.extended import _fit_evaluate_tfidf
from gemati.metrics import conformal_metrics, routing_metrics
from gemati.model import MondrianConformal
from gemati.routing import route_sets
NAVY, CYAN, TEAL, AMBER, GRID = "#123B66", "#19A7CE", "#13A89E", "#F2A900", "#DDE8F2"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def style(ax):
    ax.set_facecolor("white")
    ax.grid(color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#9BB3C9")
    ax.tick_params(colors=NAVY, labelsize=8)

def main():
    config = json.loads((ROOT / "configs/experiment.json").read_text())
    train = pd.read_parquet(ROOT / "data/raw/cradle_train.parquet")
    test = pd.read_parquet(ROOT / "data/raw/cradle_test.parquet")
    classes, actions, costs = config["risk_labels"], config["actions"], config["cost_matrix"]
    train_idx, cal_idx = train_test_split(
        np.arange(len(train)), test_size=0.30, random_state=42, stratify=train["risk"]
    )
    run = _fit_evaluate_tfidf(
        train.iloc[train_idx]["text"].tolist(), train.iloc[train_idx]["risk"].tolist(),
        train.iloc[cal_idx]["text"].tolist(), train.iloc[cal_idx]["risk"].tolist(),
        test["text"].tolist(), test["risk"].tolist(), classes, actions, costs, config, 42,
    )
    probabilities, labels = run["test_probabilities"], test["risk"].tolist()
    y_index = np.asarray([classes.index(label) for label in labels])
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_index
    output = ROOT / "results/manuscript_figures"
    output.mkdir(parents=True, exist_ok=True)
    xs, ys, ns = [], [], []
    edges = np.linspace(0, 1, 11)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
            ns.append(int(mask.sum()))
    reliability = output / "reliability_tfidf_final.png"
    fig, ax = plt.subplots(figsize=(4.8, 3.6), facecolor="white")
    style(ax)
    ax.plot([0, 1], [0, 1], "--", color="#8CA3B8", label="Ideal")
    ax.plot(xs, ys, marker="o", linewidth=2.2, color=CYAN, label="TF-IDF final")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.5)
    ax.set(xlabel="Confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(reliability, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rows = []
    for alpha in np.linspace(0.02, 0.30, 15):
        cp = MondrianConformal(float(alpha), classes).fit(
            run["calibration_probabilities"], run["conformal_labels"]
        )
        sets = cp.predict_sets(probabilities)
        routed = route_sets(sets, classes, actions, costs)
        conformal = conformal_metrics(sets, labels, classes)
        routing = routing_metrics(routed, labels, classes, costs)
        rows.append((conformal["coverage"], routing["high_risk_miss_rate"], routing["over_escalation_rate"]))
    tradeoff = output / "conformal_tradeoff_tfidf_final.png"
    fig, ax = plt.subplots(figsize=(4.8, 3.6), facecolor="white")
    style(ax)
    coverage = np.asarray([row[0] for row in rows]) * 100
    ax.plot(coverage, np.asarray([row[1] for row in rows]) * 100, marker="o", linewidth=2.2, color=AMBER, label="High-risk miss")
    ax.plot(coverage, np.asarray([row[2] for row in rows]) * 100, marker="s", linewidth=2.2, color=TEAL, label="Over-escalation")
    ax.set(xlabel="Empirical set coverage (%)", ylabel="Routing rate (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(tradeoff, dpi=300, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "protocol": "final TF-IDF; seed 42; disjoint probability/conformal calibration",
        "metrics_check": run["classification"],
        "figures": {str(path.relative_to(ROOT)): sha256(path) for path in (reliability, tradeoff)},
    }
    (output / "FIGURE_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()

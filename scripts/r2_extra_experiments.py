#!/usr/bin/env python3
"""GEMATI-CARE — additional experiments requested by the R2 audit (3 Oct 2026).

Run from the root of the GEMATI_RESTI package (the folder that contains src/, configs/
and data/raw/cradle_*.parquet):

    PYTHONPATH=src python3 scripts/r2_extra_experiments.py

Outputs (all new files, nothing existing is overwritten):
    results/r2/r2_extra_metrics.json        machine-readable results of E1-E6
    results/r2/R2_SUMMARY.md                 human-readable summary
    results/r2/fig3_tradeoff_with_markers.png  Figure 3 with alpha=0.05 and 0.10 marked
    results/r2/fig5_pareto_policies.png      miss vs over-escalation frontier of all policies
    results/r2/fig4_counts.png               Figure 4 regenerated from the same run

Experiments
    E1  matched-threshold baseline (threshold tuned on validation to the minimax miss level)
    E2  class-specific significance levels (alpha_high = crisis-miss budget, alpha_low/medium free)
    E3  conformal recalibration on clinician-annotated validation labels (split and 2-fold cross)
    E4  paired exact McNemar tests for high-risk misses
    E5  fresh-process CPU benchmark of the primary TF-IDF pipeline
    E6  policy frontier (threshold sweep, alpha sweep, cost-scaled expected-cost routing)

The protocol (seed 42, 70/10/20 split of the training pool, TF-IDF head, temperature scaling,
Mondrian sets, primary cost matrix) is identical to the locked manuscript run, so E1-E6 reuse
exactly the probabilities that produced Tables 3-5.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemati.extended import _fit_evaluate_tfidf  # noqa: E402
from gemati.metrics import conformal_metrics, routing_metrics  # noqa: E402
from gemati.model import MondrianConformal  # noqa: E402
from gemati.routing import expected_cost_actions, route_sets, threshold_actions, argmax_actions  # noqa: E402

OUT = ROOT / "results" / "r2"


# ----------------------------------------------------------------------------- helpers
def class_specific_sets(cal_p, cal_y, eval_p, classes, alphas):
    """Mondrian sets with one significance level per class (same quantile rule as the package)."""
    quantiles = {}
    cal_y = np.asarray(cal_y)
    for k, label in enumerate(classes):
        scores = 1.0 - cal_p[cal_y == label, k]
        quantiles[label] = MondrianConformal._finite_sample_quantile(scores, alphas[label])
    return [
        {label for k, label in enumerate(classes) if 1.0 - row[k] <= quantiles[label]}
        for row in eval_p
    ], quantiles


def mondrian_sets(cal_p, cal_y, eval_p, classes, alpha):
    cp = MondrianConformal(alpha, classes).fit(cal_p, list(cal_y))
    return cp.predict_sets(eval_p), cp.quantiles


def cov_tests(sets, labels, classes, alpha):
    out = {}
    labels = np.asarray(labels)
    for label in classes:
        mask = labels == label
        covered = int(sum(label in s for s, m in zip(sets, mask) if m))
        n = int(mask.sum())
        out[label] = {
            "covered": covered, "n": n, "coverage": covered / n,
            "p_two_sided_vs_nominal": float(binomtest(covered, n, 1 - alpha).pvalue),
        }
    covered = int(sum(l in s for s, l in zip(sets, labels)))
    out["overall"] = {"covered": covered, "n": len(labels), "coverage": covered / len(labels),
                      "p_two_sided_vs_nominal": float(binomtest(covered, len(labels), 1 - alpha).pvalue)}
    return out


def mcnemar(actions_a, actions_b, labels):
    """Exact McNemar on high-risk messages: escalated (1) vs missed (0)."""
    labels = np.asarray(labels)
    a = np.asarray([x == "ESCALATE" for x in actions_a])[labels == "high"]
    b = np.asarray([x == "ESCALATE" for x in actions_b])[labels == "high"]
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0
    return {"escalated_by_first_only": only_a, "escalated_by_second_only": only_b,
            "discordant": n, "p_exact": p}


def scaled_costs(costs, factor):
    """Scale the cost of under-serving a high-risk message (SUPPORT/CLARIFY on high)."""
    out = {a: list(v) for a, v in costs.items()}
    out["SUPPORT"][2] *= factor
    out["CLARIFY"][2] *= factor
    return out


def summarize(actions, labels, classes, costs):
    m = routing_metrics(actions, labels, classes, costs)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}


# ----------------------------------------------------------------------------- E5 benchmark
BENCH_CODE = r"""
import json, sys, time, resource, joblib
t0 = time.perf_counter()
model = joblib.load(sys.argv[1])
texts = json.load(open(sys.argv[2]))
model.predict_proba(texts[:1])
cold = time.perf_counter() - t0
single = []
for i in range(50):
    t = time.perf_counter(); model.predict_proba([texts[i % len(texts)]]); single.append((time.perf_counter() - t) * 1000)
batch = []
for r in range(12):
    chunk = [texts[(r * 64 + j) % len(texts)] for j in range(64)]
    t = time.perf_counter(); model.predict_proba(chunk); batch.append(time.perf_counter() - t)
import numpy as np
print(json.dumps({
    "cold_start_s": cold,
    "warm_single_ms_p50": float(np.percentile(single, 50)),
    "warm_single_ms_p95": float(np.percentile(single, 95)),
    "throughput_msgs_per_s_mean": float(np.mean([64 / b for b in batch])),
    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
}))
"""


def benchmark_tfidf(classifier, texts):
    import joblib
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "tfidf.joblib"
        text_path = Path(tmp) / "texts.json"
        joblib.dump(classifier, model_path)
        json.dump(list(texts)[:256], open(text_path, "w"))
        size_mb = model_path.stat().st_size / 1e6
        env = dict(os.environ, OMP_NUM_THREADS="16", OPENBLAS_NUM_THREADS="16", MKL_NUM_THREADS="16")
        res = subprocess.run([sys.executable, "-c", BENCH_CODE, str(model_path), str(text_path)],
                             capture_output=True, text=True, env=env, check=True)
        out = json.loads(res.stdout.strip().splitlines()[-1])
        out["serialized_model_mb"] = size_mb
        return out


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/experiment.json").read_text())
    classes, actions, costs = config["risk_labels"], config["actions"], config["cost_matrix"]
    train = pd.read_parquet(ROOT / "data/raw/cradle_train.parquet")
    val = pd.read_parquet(ROOT / "data/raw/cradle_validation.parquet")
    test = pd.read_parquet(ROOT / "data/raw/cradle_test.parquet")

    train_idx, cal_idx = train_test_split(
        np.arange(len(train)), test_size=0.30, random_state=42, stratify=train["risk"])
    eval_texts = val["text"].tolist() + test["text"].tolist()
    eval_labels = val["risk"].tolist() + test["risk"].tolist()
    run = _fit_evaluate_tfidf(
        train.iloc[train_idx]["text"].tolist(), train.iloc[train_idx]["risk"].tolist(),
        train.iloc[cal_idx]["text"].tolist(), train.iloc[cal_idx]["risk"].tolist(),
        eval_texts, eval_labels, classes, actions, costs, config, 42)
    nv = len(val)
    P = run["test_probabilities"]
    pv, pt = P[:nv], P[nv:]
    yv, yt = val["risk"].tolist(), test["risk"].tolist()
    cal_p, cal_y = run["calibration_probabilities"], run["conformal_labels"]
    results = {"protocol": "identical to the locked manuscript run (seed 42, 70/10/20, TF-IDF, T-scaling, Mondrian)"}

    # sanity check against the manuscript (Table 5, alpha = 0.05)
    sets_t05, q05 = mondrian_sets(cal_p, cal_y, pt, classes, 0.05)
    mm05_t = route_sets(sets_t05, classes, actions, costs)
    sets_v05, _ = mondrian_sets(cal_p, cal_y, pv, classes, 0.05)
    mm05_v = route_sets(sets_v05, classes, actions, costs)
    results["sanity_minimax_alpha_0.05_test"] = summarize(mm05_t, yt, classes, costs)
    results["quantiles_alpha_0.05"] = q05

    # ---------------- E1 matched threshold
    target = routing_metrics(mm05_v, yv, classes, costs)["high_risk_miss_rate"]
    grid = np.round(np.arange(0.01, 0.991, 0.001), 3)
    best_t = None
    for t in grid:  # largest threshold whose validation miss does not exceed the minimax miss
        miss = routing_metrics(threshold_actions(pv, classes, t), yv, classes, costs)["high_risk_miss_rate"]
        if miss <= target + 1e-12:
            best_t = float(t)
    thr_t = threshold_actions(pt, classes, best_t)
    results["E1_matched_threshold"] = {
        "validation_target_miss": target, "selected_threshold": best_t,
        "test_matched_threshold": summarize(thr_t, yt, classes, costs),
        "test_minimax_alpha_0.05": results["sanity_minimax_alpha_0.05_test"],
        "implied_threshold_from_conformal_high_quantile": 1.0 - q05["high"],
    }

    # ---------------- E2 class-specific alphas (selected on validation)
    e2 = []
    for a_high in (0.05, 0.10):
        for a_low in (0.05, 0.10, 0.15, 0.20, 0.30):
            for a_med in (0.05, 0.10, 0.15, 0.20, 0.30):
                alphas = {"low": a_low, "medium": a_med, "high": a_high}
                sv, _ = class_specific_sets(cal_p, cal_y, pv, classes, alphas)
                st, _ = class_specific_sets(cal_p, cal_y, pt, classes, alphas)
                e2.append({"alphas": alphas,
                           "validation": summarize(route_sets(sv, classes, actions, costs), yv, classes, costs),
                           "test": summarize(route_sets(st, classes, actions, costs), yt, classes, costs)})
    results["E2_class_specific_alpha_grid"] = e2
    chosen = {}
    for a_high in (0.05, 0.10):
        cands = [r for r in e2 if r["alphas"]["high"] == a_high]
        cands.sort(key=lambda r: (r["validation"]["under_escalation_rate"] > 0.05,
                                  r["validation"]["over_escalation_rate"]))
        chosen[str(a_high)] = cands[0]
    results["E2_selected_on_validation"] = {
        "rule": "for each alpha_high, minimize validation over-escalation subject to validation "
                "under-escalation <= 5%", "selected": chosen}

    # ---------------- E3 recalibration on clinician labels (validation -> test)
    e3 = {}
    for alpha in (0.05, 0.10):
        full_sets, _ = mondrian_sets(pv, yv, pt, classes, alpha)
        a_idx, b_idx = train_test_split(np.arange(nv), test_size=0.5, random_state=42, stratify=yv)
        folds = []
        for idx in (a_idx, b_idx):
            s, _ = mondrian_sets(pv[idx], [yv[i] for i in idx], pt, classes, alpha)
            folds.append({"coverage": cov_tests(s, yt, classes, alpha),
                          "routing": summarize(route_sets(s, classes, actions, costs), yt, classes, costs)})
        auto_sets, _ = mondrian_sets(cal_p, cal_y, pt, classes, alpha)
        e3[str(alpha)] = {
            "calibrated_on_automatic_labels_836": {
                "coverage": cov_tests(auto_sets, yt, classes, alpha),
                "routing": summarize(route_sets(auto_sets, classes, actions, costs), yt, classes, costs)},
            "calibrated_on_clinician_validation_420": {
                "coverage": cov_tests(full_sets, yt, classes, alpha),
                "routing": summarize(route_sets(full_sets, classes, actions, costs), yt, classes, costs)},
            "calibrated_on_clinician_validation_half_A_B": folds,
        }
    results["E3_clinician_recalibration"] = e3

    # ---------------- E4 McNemar
    argmax_t = argmax_actions(pt, classes)
    expected_t = expected_cost_actions(pt, classes, actions, costs)
    sets_t10, _ = mondrian_sets(cal_p, cal_y, pt, classes, 0.10)
    mm10_t = route_sets(sets_t10, classes, actions, costs)
    results["E4_mcnemar_high_risk"] = {
        "minimax_0.05_vs_argmax": mcnemar(mm05_t, argmax_t, yt),
        "minimax_0.10_vs_argmax": mcnemar(mm10_t, argmax_t, yt),
        "minimax_0.05_vs_matched_threshold": mcnemar(mm05_t, thr_t, yt),
        "minimax_0.10_vs_expected_cost": mcnemar(mm10_t, expected_t, yt),
    }

    # ---------------- E5 CPU benchmark of the primary pipeline
    try:
        results["E5_tfidf_cpu_benchmark"] = benchmark_tfidf(run["classifier"], test["text"].tolist())
    except Exception as exc:  # benchmark must not block the other results
        results["E5_tfidf_cpu_benchmark"] = {"error": repr(exc)}

    # ---------------- E6 frontier + figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frontier = {"threshold": [], "minimax_alpha": [], "expected_cost_scaled": []}
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        m = routing_metrics(threshold_actions(pt, classes, t), yt, classes, costs)
        frontier["threshold"].append({"t": float(t), "miss": m["high_risk_miss_rate"], "over": m["over_escalation_rate"]})
    alphas_grid = sorted(set(np.round(np.linspace(0.02, 0.30, 15), 3).tolist() + [0.05]))
    for a in alphas_grid:
        s, _ = mondrian_sets(cal_p, cal_y, pt, classes, a)
        c = conformal_metrics(s, yt, classes)
        m = routing_metrics(route_sets(s, classes, actions, costs), yt, classes, costs)
        frontier["minimax_alpha"].append({"alpha": a, "coverage": c["coverage"],
                                          "miss": m["high_risk_miss_rate"], "over": m["over_escalation_rate"]})
    for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        sc = scaled_costs(costs, f)
        m = routing_metrics(expected_cost_actions(pt, classes, actions, sc), yt, classes, costs)
        frontier["expected_cost_scaled"].append({"factor": f, "miss": m["high_risk_miss_rate"], "over": m["over_escalation_rate"]})
    results["E6_frontier_test"] = frontier

    # Figure 3 with markers
    rows = frontier["minimax_alpha"]
    cov = np.array([r["coverage"] for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(cov, [r["miss"] * 100 for r in rows], "o-", color="#F2A900", label="High-risk miss")
    ax.plot(cov, [r["over"] * 100 for r in rows], "s-", color="#13A89E", label="Over-escalation")
    for r in rows:
        if r["alpha"] in (0.05, 0.10):
            x = r["coverage"] * 100
            ax.axvline(x, color="#123B66", lw=0.8, ls=":")
            ax.annotate(f"α = {r['alpha']:.2f}", (x, 44), ha="center", fontsize=7, color="#123B66")
    ax.set(xlabel="Empirical set coverage (%)", ylabel="Routing rate (%)")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig3_tradeoff_with_markers.png", dpi=600); plt.close(fig)

    # Figure 5 frontier
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for key, mk, col, lab in (("threshold", "^", "#8C6BB1", "Threshold sweep"),
                              ("expected_cost_scaled", "D", "#E07B39", "Expected cost (scaled)"),
                              ("minimax_alpha", "o", "#19A7CE", "Minimax over Mondrian sets")):
        pts = sorted(frontier[key], key=lambda r: r["over"])
        ax.plot([r["over"] * 100 for r in pts], [r["miss"] * 100 for r in pts], mk + "-", ms=3.5, lw=1.2, color=col, label=lab)
    am = routing_metrics(argmax_t, yt, classes, costs)
    ax.plot(am["over_escalation_rate"] * 100, am["high_risk_miss_rate"] * 100, "kx", ms=7, label="Argmax")
    ax.set(xlabel="Over-escalation (%)", ylabel="High-risk miss (%)")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig5_pareto_policies.png", dpi=600); plt.close(fig)

    # Figure 4 regenerated
    pol = {"Argmax": argmax_t, "Expected cost": expected_t, "Minimax α=0.10": mm10_t, "Minimax α=0.05": mm05_t}
    names = list(pol)[::-1]
    yt_arr = np.asarray(yt)
    missed = [int(np.sum((yt_arr == "high") & (np.asarray(pol[n]) != "ESCALATE"))) for n in names]
    sev = {"SUPPORT": 0, "CLARIFY": 1, "ESCALATE": 2}
    ref = {"low": 0, "medium": 1, "high": 2}
    over = [int(sum(sev[a] > ref[l] for a, l in zip(pol[n], yt))) for n in names]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    y = np.arange(len(names))
    ax.barh(y + 0.2, over, 0.38, color="#2AA0C4", label="Over-escalated messages (of 600)")
    ax.barh(y - 0.2, missed, 0.38, color="#D04A7A", label="Missed high-risk messages (of 234)")
    for i in range(len(names)):
        ax.text(over[i] + 3, y[i] + 0.2, str(over[i]), va="center", fontsize=7)
        ax.text(missed[i] + 3, y[i] - 0.2, str(missed[i]), va="center", fontsize=7)
    ax.set_yticks(y, names); ax.set_xlabel("Number of test messages")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig4_counts.png", dpi=600); plt.close(fig)
    results["figure4_counts"] = {n: {"missed": m, "over": o} for n, m, o in zip(names, missed, over)}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, set):
            return sorted(o)
        return o
    (OUT / "r2_extra_metrics.json").write_text(json.dumps(clean(results), indent=1))

    # summary
    s = results
    lines = ["# R2 extra experiments — summary", "",
             f"Sanity (must equal Table 5, alpha=0.05): miss={s['sanity_minimax_alpha_0.05_test']['high_risk_miss_rate']:.4f}, "
             f"over={s['sanity_minimax_alpha_0.05_test']['over_escalation_rate']:.4f}", "",
             "## E1 matched threshold",
             f"threshold={s['E1_matched_threshold']['selected_threshold']}  test: {s['E1_matched_threshold']['test_matched_threshold']}", "",
             "## E2 class-specific alpha (selected on validation)"]
    for k, v in s["E2_selected_on_validation"]["selected"].items():
        lines.append(f"alpha_high={k}: alphas={v['alphas']} test={v['test']}")
    lines += ["", "## E3 recalibration on clinician labels"]
    for a, v in s["E3_clinician_recalibration"].items():
        lines.append(f"alpha={a} auto-labels: {json.dumps({c: round(v['calibrated_on_automatic_labels_836']['coverage'][c]['coverage'], 4) for c in ['low','medium','high','overall']})}")
        lines.append(f"alpha={a} clinician 420: {json.dumps({c: round(v['calibrated_on_clinician_validation_420']['coverage'][c]['coverage'], 4) for c in ['low','medium','high','overall']})}")
    lines += ["", "## E4 McNemar", json.dumps(s["E4_mcnemar_high_risk"], indent=1), "",
              "## E5 TF-IDF CPU benchmark", json.dumps(s["E5_tfidf_cpu_benchmark"], indent=1)]
    (OUT / "R2_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

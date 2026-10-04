#!/usr/bin/env python3
"""GEMATI-CARE R2 — robustness extension (Step 4 of the DGX agent prompt).

    PYTHONPATH=src python scripts/r2_robustness.py

R1  five-seed stability of class-wise coverage on validation (seeds 11, 29, 42, 71, 101)
R2  paired bootstrap on test for minimax alpha=0.05, the E1 threshold, the E2 configuration, argmax
R3  five-fold cross-conformal calibration on the 420 clinician-annotated validation messages
R4  exact binomial tests (per class, two-sided) for E1, E2, E3 and R3 on test
R5  check that the alpha grid of fig3_tradeoff_with_markers reproduces the published Figure 3

Selection (E1 threshold, E2 alphas) uses validation only; test is evaluation only.
Output: results/r2/r2_robustness_metrics.json (nothing outside results/r2 is written).
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gemati.extended import SEEDS, _fit_evaluate_tfidf  # noqa: E402
from gemati.metrics import conformal_metrics, routing_metrics  # noqa: E402
from gemati.model import MondrianConformal  # noqa: E402
from gemati.routing import argmax_actions, route_sets, threshold_actions  # noqa: E402
from r2_extra_experiments import class_specific_sets  # noqa: E402

OUT = ROOT / "results" / "r2"
ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_BOOT = 1000
SEVERITY = {"SUPPORT": 0, "CLARIFY": 1, "ESCALATE": 2}
REFERENCE = {"low": 0, "medium": 1, "high": 2}


# ----------------------------------------------------------------------------- helpers
def class_tests(sets, labels, classes, nominal):
    """Per-class coverage with exact binomial tests; nominal is a float or a dict per class."""
    labels = np.asarray(labels)
    out = {}
    for label in classes:
        target = nominal[label] if isinstance(nominal, dict) else nominal
        mask = labels == label
        covered = int(sum(label in s for s, m in zip(sets, mask) if m))
        n = int(mask.sum())
        out[label] = {
            "covered": covered, "n": n, "coverage": covered / n, "nominal": target,
            "p_two_sided": float(binomtest(covered, n, target).pvalue),
            "p_one_sided_below": float(binomtest(covered, n, target, alternative="less").pvalue),
        }
    return out


def tidy(metrics):
    return {k: v for k, v in metrics.items()}


def per_message(actions, labels, classes, costs):
    labels = np.asarray(labels)
    idx = {c: i for i, c in enumerate(classes)}
    sev = np.asarray([SEVERITY[a] for a in actions])
    ref = np.asarray([REFERENCE[l] for l in labels])
    return {
        "over": (sev > ref).astype(float),
        "under": (sev < ref).astype(float),
        "cost": np.asarray([costs[a][idx[l]] for a, l in zip(actions, labels)], dtype=float),
        "high": labels == "high",
        "missed": (labels == "high") & (sev < 2),
    }


def boot_metrics(pm, ix):
    high = pm["high"][ix]
    return {
        "miss": float(pm["missed"][ix].sum() / high.sum()),
        "over": float(pm["over"][ix].mean()),
        "under": float(pm["under"][ix].mean()),
        "cost": float(pm["cost"][ix].mean()),
    }


def ci(values):
    v = np.asarray(values)
    return {"ci95_low": float(np.quantile(v, 0.025)), "ci95_high": float(np.quantile(v, 0.975))}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return o


# ----------------------------------------------------------------------------- main
def main():
    config = json.loads((ROOT / "configs/experiment.json").read_text())
    classes, actions, costs = config["risk_labels"], config["actions"], config["cost_matrix"]
    train = pd.read_parquet(ROOT / "data/raw/cradle_train.parquet")
    val = pd.read_parquet(ROOT / "data/raw/cradle_validation.parquet")
    test = pd.read_parquet(ROOT / "data/raw/cradle_test.parquet")
    yv, yt = val["risk"].tolist(), test["risk"].tolist()
    extra = json.loads((OUT / "r2_extra_metrics.json").read_text())
    results = {"protocol": "locked seed-42 TF-IDF protocol; selection on validation only; "
                           f"{N_BOOT} bootstrap replications with numpy default_rng(42)"}

    # ---------------- R1 five-seed stability (validation only)
    r1 = {}
    seed42 = None
    for seed in SEEDS:
        train_idx, cal_idx = train_test_split(
            np.arange(len(train)), test_size=0.30, random_state=seed, stratify=train["risk"])
        eval_texts = val["text"].tolist() + test["text"].tolist()
        run = _fit_evaluate_tfidf(
            train.iloc[train_idx]["text"].tolist(), train.iloc[train_idx]["risk"].tolist(),
            train.iloc[cal_idx]["text"].tolist(), train.iloc[cal_idx]["risk"].tolist(),
            eval_texts, yv + yt, classes, actions, costs, config, seed)
        pv = run["test_probabilities"][:len(val)]
        if seed == 42:
            seed42 = run
        r1[str(seed)] = {}
        for alpha in ALPHAS:
            cp = MondrianConformal(alpha, classes).fit(run["calibration_probabilities"], run["conformal_labels"])
            sets = cp.predict_sets(pv)
            r1[str(seed)][str(alpha)] = {
                "class_coverage": class_tests(sets, yv, classes, 1 - alpha),
                "routing": tidy(routing_metrics(route_sets(sets, classes, actions, costs), yv, classes, costs)),
            }
        print(f"R1 seed {seed} done", flush=True)
    summary = {}
    for alpha in ALPHAS:
        a = str(alpha)
        row = {}
        for label in classes:
            covs = [r1[str(s)][a]["class_coverage"][label]["coverage"] for s in SEEDS]
            row[f"{label}_coverage_mean"] = float(np.mean(covs))
            row[f"{label}_coverage_sd"] = float(np.std(covs, ddof=1))
        for key, name in (("high_risk_miss_rate", "miss"), ("over_escalation_rate", "over"),
                          ("under_escalation_rate", "under")):
            vals = [r1[str(s)][a]["routing"][key] for s in SEEDS]
            row[f"{name}_mean"] = float(np.mean(vals))
            row[f"{name}_sd"] = float(np.std(vals, ddof=1))
        summary[a] = row
    counts = {}
    for label in classes:
        cases = [r1[str(s)][str(a)]["class_coverage"][label] for s in SEEDS for a in ALPHAS]
        counts[label] = {
            "cases": len(cases),
            "not_significant_two_sided": int(sum(c["p_two_sided"] >= 0.05 for c in cases)),
            "significantly_below_nominal_one_sided": int(sum(c["p_one_sided_below"] < 0.05 for c in cases)),
        }
    results["R1_five_seed_validation"] = {"per_seed": r1, "summary_mean_sd": summary,
                                          "non_significant_case_counts": counts}

    # ---------------- seed-42 objects reused below (identical to the locked run)
    nv = len(val)
    pv, pt = seed42["test_probabilities"][:nv], seed42["test_probabilities"][nv:]
    cal_p, cal_y = seed42["calibration_probabilities"], seed42["conformal_labels"]

    def mondrian(alpha, cp_probs, cp_labels, probs):
        return MondrianConformal(alpha, classes).fit(cp_probs, list(cp_labels)).predict_sets(probs)

    sets_v05 = mondrian(0.05, cal_p, cal_y, pv)
    sets_t05 = mondrian(0.05, cal_p, cal_y, pt)
    mm05_v = route_sets(sets_v05, classes, actions, costs)
    mm05_t = route_sets(sets_t05, classes, actions, costs)

    # E1 threshold re-selected on validation with the delivered rule
    target = routing_metrics(mm05_v, yv, classes, costs)["high_risk_miss_rate"]
    best_t = None
    for t in np.round(np.arange(0.01, 0.991, 0.001), 3):
        if routing_metrics(threshold_actions(pv, classes, t), yv, classes, costs)["high_risk_miss_rate"] <= target + 1e-12:
            best_t = float(t)
    assert best_t == extra["E1_matched_threshold"]["selected_threshold"], "E1 threshold differs from E1 run"
    e1_t = threshold_actions(pt, classes, best_t)

    # E2 configuration selected on validation for alpha_high = 0.05 (delivered rule)
    e2_alphas = extra["E2_selected_on_validation"]["selected"]["0.05"]["alphas"]
    e2_sets_t, _ = class_specific_sets(cal_p, cal_y, pt, classes, e2_alphas)
    e2_t = route_sets(e2_sets_t, classes, actions, costs)
    argmax_t = argmax_actions(pt, classes)

    # ---------------- R2 paired bootstrap on test
    policies = {"minimax_alpha_0.05": mm05_t, "E1_matched_threshold": e1_t,
                "E2_class_specific_alpha": e2_t, "argmax": argmax_t}
    pms = {k: per_message(v, yt, classes, costs) for k, v in policies.items()}
    rng = np.random.default_rng(42)
    draws = {k: {m: [] for m in ("miss", "over", "under", "cost")} for k in policies}
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(yt), len(yt))
        for k, pm in pms.items():
            for m, v in boot_metrics(pm, ix).items():
                draws[k][m].append(v)
    full = np.arange(len(yt))
    r2 = {"policies": {}, "paired_differences": {}}
    for k, pm in pms.items():
        point = boot_metrics(pm, full)
        r2["policies"][k] = {
            m: {"point": point[m], **ci(draws[k][m])} for m in point}
        r2["policies"][k]["actions_S_C_E"] = [sum(a == x for a in policies[k]) for x in ("SUPPORT", "CLARIFY", "ESCALATE")]
    for other in ("E2_class_specific_alpha", "E1_matched_threshold"):
        name = f"{other} - minimax_alpha_0.05"
        r2["paired_differences"][name] = {}
        for m in ("over", "miss", "under", "cost"):
            diff = np.asarray(draws[other][m]) - np.asarray(draws["minimax_alpha_0.05"][m])
            point = boot_metrics(pms[other], full)[m] - boot_metrics(pms["minimax_alpha_0.05"], full)[m]
            c = ci(diff)
            r2["paired_differences"][name][m] = {
                "point": point, **c, "excludes_zero": bool(c["ci95_low"] > 0 or c["ci95_high"] < 0)}
    r2["E1_threshold"] = best_t
    r2["E2_alphas"] = e2_alphas
    results["R2_paired_bootstrap_test"] = r2

    # ---------------- R3 five-fold cross-conformal on clinician labels
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    yv_arr = np.asarray(yv)
    r3 = {}
    for alpha in (0.05, 0.10):
        folds = []
        for k, (fit_idx, _) in enumerate(skf.split(pv, yv_arr)):
            sets = mondrian(alpha, pv[fit_idx], yv_arr[fit_idx], pt)
            folds.append({
                "fold": k, "n_calibration": int(len(fit_idx)),
                "class_coverage": class_tests(sets, yt, classes, 1 - alpha),
                "overall_coverage": conformal_metrics(sets, yt, classes)["coverage"],
                "average_set_size": conformal_metrics(sets, yt, classes)["average_set_size"],
                "routing": tidy(routing_metrics(route_sets(sets, classes, actions, costs), yt, classes, costs)),
            })
        mean = {f"{label}_coverage": float(np.mean([f["class_coverage"][label]["coverage"] for f in folds]))
                for label in classes}
        mean["overall_coverage"] = float(np.mean([f["overall_coverage"] for f in folds]))
        for key in ("high_risk_miss_rate", "over_escalation_rate", "under_escalation_rate", "expected_cost"):
            mean[key] = float(np.mean([f["routing"][key] for f in folds]))
        fold_counts = {label: {
            "folds_not_significant_two_sided": int(sum(f["class_coverage"][label]["p_two_sided"] >= 0.05 for f in folds)),
            "folds_significantly_below_one_sided": int(sum(f["class_coverage"][label]["p_one_sided_below"] < 0.05 for f in folds)),
        } for label in classes}
        auto_sets = mondrian(alpha, cal_p, cal_y, pt)
        r3[str(alpha)] = {
            "folds": folds, "mean_over_folds": mean, "fold_counts": fold_counts,
            "automatic_label_calibration_836": {
                "class_coverage": class_tests(auto_sets, yt, classes, 1 - alpha),
                "overall_coverage": conformal_metrics(auto_sets, yt, classes)["coverage"],
                "routing": tidy(routing_metrics(route_sets(auto_sets, classes, actions, costs), yt, classes, costs)),
            },
        }
    results["R3_cross_conformal_clinician"] = r3

    # ---------------- R4 exact binomial tests for every reported configuration (test)
    r4 = {}
    escalated_high = [a == "ESCALATE" for a, l in zip(e1_t, yt) if l == "high"]
    k = int(sum(escalated_high))
    r4["E1_matched_threshold"] = {
        "note": "a threshold rule has no prediction sets; the high-class quantity is the escalation rate "
                "of high-risk messages, tested against the 0.95 budget it was matched to",
        "high": {"escalated": k, "n": len(escalated_high), "rate": k / len(escalated_high),
                 "p_two_sided": float(binomtest(k, len(escalated_high), 0.95).pvalue)},
    }
    r4["E2_class_specific_alpha"] = {
        "alphas": e2_alphas,
        "class_coverage": class_tests(e2_sets_t, yt, classes, {c: 1 - e2_alphas[c] for c in classes}),
    }
    e3 = {}
    a_idx, b_idx = train_test_split(np.arange(nv), test_size=0.5, random_state=42, stratify=yv)
    for alpha in (0.05, 0.10):
        e3[str(alpha)] = {
            "clinician_420": class_tests(mondrian(alpha, pv, yv, pt), yt, classes, 1 - alpha),
            "clinician_half_A": class_tests(mondrian(alpha, pv[a_idx], yv_arr[a_idx], pt), yt, classes, 1 - alpha),
            "clinician_half_B": class_tests(mondrian(alpha, pv[b_idx], yv_arr[b_idx], pt), yt, classes, 1 - alpha),
            "automatic_836": class_tests(mondrian(alpha, cal_p, cal_y, pt), yt, classes, 1 - alpha),
        }
    r4["E3_recalibration"] = e3
    r4["R3_cross_conformal"] = {a: [f["class_coverage"] for f in v["folds"]] for a, v in r3.items()}
    results["R4_binomial_tests_test"] = r4

    # ---------------- R5 Figure 3 check
    published_alphas = np.linspace(0.02, 0.30, 15)
    published = []
    for alpha in published_alphas:
        s = mondrian(float(alpha), cal_p, cal_y, pt)
        published.append(conformal_metrics(s, yt, classes)["coverage"])
    delivered = {round(r["alpha"], 3): r["coverage"] for r in extra["E6_frontier_test"]["minimax_alpha"]}
    comparisons = []
    for alpha, cov in zip(published_alphas, published):
        d = delivered.get(round(float(alpha), 3))
        comparisons.append({"alpha": round(float(alpha), 3), "published_protocol_coverage": cov,
                            "fig3_r2_coverage": d, "match": d is not None and abs(d - cov) < 1e-12})
    # regenerate the published figure byte-for-byte in a temporary folder
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from regenerate_manuscript_figures import AMBER, TEAL, style
    rows = []
    for alpha in published_alphas:
        s = mondrian(float(alpha), cal_p, cal_y, pt)
        m = routing_metrics(route_sets(s, classes, actions, costs), yt, classes, costs)
        rows.append((conformal_metrics(s, yt, classes)["coverage"], m["high_risk_miss_rate"], m["over_escalation_rate"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conformal_tradeoff_tfidf_final.png"
        fig, ax = plt.subplots(figsize=(4.8, 3.6), facecolor="white")
        style(ax)
        coverage = np.asarray([r[0] for r in rows]) * 100
        ax.plot(coverage, np.asarray([r[1] for r in rows]) * 100, marker="o", linewidth=2.2, color=AMBER, label="High-risk miss")
        ax.plot(coverage, np.asarray([r[2] for r in rows]) * 100, marker="s", linewidth=2.2, color=TEAL, label="Over-escalation")
        ax.set(xlabel="Empirical set coverage (%)", ylabel="Routing rate (%)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        regenerated_hash = sha256(path)
    published_png = ROOT / "results/manuscript_figures/conformal_tradeoff_tfidf_final.png"
    if not published_png.exists():  # layout of the public GitHub/Zenodo package
        published_png = ROOT / "results/figures/conformal_tradeoff_tfidf_final.png"
    published_hash = sha256(published_png)
    results["R5_figure3_check"] = {
        "points": comparisons,
        "all_15_points_match": all(c["match"] for c in comparisons),
        "added_point_alpha_0.05_coverage": delivered.get(0.05),
        "published_png_sha256": published_hash,
        "regenerated_png_sha256": regenerated_hash,
        "published_png_byte_identical_to_regeneration": published_hash == regenerated_hash,
    }

    (OUT / "r2_robustness_metrics.json").write_text(json.dumps(clean(results), indent=1))

    # console summary
    print("\n== R1 counts (20 seed x alpha cases per class)")
    print(json.dumps(counts, indent=1))
    print("== R1 mean +- SD")
    for a, row in summary.items():
        print(a, {k: round(v, 4) for k, v in row.items()})
    print("== R2 policies (point [CI])")
    for k, v in r2["policies"].items():
        print(k, {m: f"{v[m]['point']:.4f} [{v[m]['ci95_low']:.4f}, {v[m]['ci95_high']:.4f}]" for m in ("miss", "over", "under", "cost")}, v["actions_S_C_E"])
    print("== R2 paired differences")
    for k, v in r2["paired_differences"].items():
        print(k, {m: f"{d['point']:+.4f} [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] excl0={d['excludes_zero']}" for m, d in v.items()})
    print("== R3 cross-conformal (mean over folds)")
    for a, v in r3.items():
        print(a, {k: round(x, 4) for k, x in v["mean_over_folds"].items()}, v["fold_counts"])
    print("== R5", {k: v for k, v in results["R5_figure3_check"].items() if k != "points"})


if __name__ == "__main__":
    main()

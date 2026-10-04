#!/usr/bin/env python3
"""GEMATI-CARE R2 — publication figures (Step 5 of the DGX agent prompt).

Reads results/r2/r2_extra_metrics.json and results/r2/r2_robustness_metrics.json and writes
600-dpi PNG plus vector PDF versions of Figure 3 (revised), Figure 4 and Figure 5, and
results/r2/FIGURE_PROVENANCE_R2.json.

    PYTHONPATH=src python scripts/r2_figures.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r2"
NAVY, CYAN, TEAL, AMBER, GRID = "#123B66", "#19A7CE", "#13A89E", "#F2A900", "#DDE8F2"
SIZE = (4.8, 3.6)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
                     "svg.hashsalt": "gemati", "path.simplify": False})
METADATA_PNG = {"Software": None}
METADATA_PDF = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}


def style(ax):
    ax.set_facecolor("white")
    ax.grid(color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#9BB3C9")
    ax.tick_params(colors=NAVY)


def save(fig, stem):
    fig.tight_layout()
    paths = []
    for ext, meta in (("png", METADATA_PNG), ("pdf", METADATA_PDF)):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, metadata=meta)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    extra = json.loads((OUT / "r2_extra_metrics.json").read_text())
    robust = json.loads((OUT / "r2_robustness_metrics.json").read_text())
    made = []

    # Figure 3 (revised): coverage trade-off with alpha = 0.05 and 0.10 marked
    rows = sorted(extra["E6_frontier_test"]["minimax_alpha"], key=lambda r: r["alpha"])
    cov = [r["coverage"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=SIZE, facecolor="white")
    style(ax)
    ax.plot(cov, [r["miss"] * 100 for r in rows], marker="o", ms=4, lw=1.8, color=AMBER, label="High-risk miss")
    ax.plot(cov, [r["over"] * 100 for r in rows], marker="s", ms=4, lw=1.8, color=TEAL, label="Over-escalation")
    for r in rows:
        if round(r["alpha"], 3) in (0.05, 0.10):
            x = r["coverage"] * 100
            ax.axvline(x, color=NAVY, lw=0.8, ls=":")
            ax.annotate(f"α = {r['alpha']:.2f}\n({x:.2f}%)", (x, 47), ha="center", va="bottom",
                        fontsize=7, color=NAVY)
    ax.set(xlabel="Empirical set coverage (%)", ylabel="Routing rate (%)", ylim=(0, 55))
    ax.legend(frameon=False, loc="upper left")
    made += save(fig, "fig3_tradeoff_with_markers")

    # Figure 4: missed high-risk and over-escalated test messages per policy
    counts = extra["figure4_counts"]
    names = ["Minimax α=0.05", "Minimax α=0.10", "Expected cost", "Argmax"]
    labels = ["Minimax α = 0.05", "Minimax α = 0.10", "Expected cost", "Argmax"]
    fig, ax = plt.subplots(figsize=SIZE, facecolor="white")
    style(ax)
    ys = range(len(names))
    over = [counts[n]["over"] for n in names]
    missed = [counts[n]["missed"] for n in names]
    ax.barh([y + 0.2 for y in ys], over, 0.38, color=CYAN, label="Over-escalated (of 600)")
    ax.barh([y - 0.2 for y in ys], missed, 0.38, color=AMBER, label="Missed high-risk (of 234)")
    for y, o, m in zip(ys, over, missed):
        ax.text(o + 3, y + 0.2, str(o), va="center", fontsize=7, color=NAVY)
        ax.text(m + 3, y - 0.2, str(m), va="center", fontsize=7, color=NAVY)
    ax.set_yticks(list(ys), labels)
    ax.set_xlabel("Number of test messages")
    ax.set_xlim(0, max(over) * 1.15)
    ax.legend(frameon=False, loc="lower right")
    made += save(fig, "fig4_counts")

    # Figure 5: miss vs over-escalation frontier with E1 and E2 highlighted
    frontier = extra["E6_frontier_test"]
    pol = robust["R2_paired_bootstrap_test"]["policies"]
    fig, ax = plt.subplots(figsize=SIZE, facecolor="white")
    style(ax)
    for key, mk, col, lab in (("threshold", "^", TEAL, "Threshold sweep"),
                              ("expected_cost_scaled", "D", AMBER, "Expected cost, scaled under-service cost"),
                              ("minimax_alpha", "o", CYAN, "Minimax over Mondrian sets (α sweep)")):
        pts = sorted(frontier[key], key=lambda r: r["over"])
        ax.plot([r["over"] * 100 for r in pts], [r["miss"] * 100 for r in pts], mk + "-", ms=3, lw=1.1,
                color=col, label=lab)
    for key, mk, lab in (("minimax_alpha_0.05", "o", "Minimax α = 0.05"),
                         ("E1_matched_threshold", "^", "E1 matched threshold"),
                         ("E2_class_specific_alpha", "*", "E2 class-specific α"),
                         ("argmax", "X", "Argmax")):
        ax.plot(pol[key]["over"]["point"] * 100, pol[key]["miss"]["point"] * 100, mk, ms=8 if mk == "*" else 6.5,
                mfc="white" if key != "argmax" else NAVY, mec=NAVY, mew=1.3, ls="none", label=lab)
    ax.set(xlabel="Over-escalation (%)", ylabel="High-risk miss (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    made += save(fig, "fig5_pareto_policies")

    provenance = {
        "script": "scripts/r2_figures.py",
        "inputs": {p: hashlib.sha256((OUT / p).read_bytes()).hexdigest()
                   for p in ("r2_extra_metrics.json", "r2_robustness_metrics.json")},
        "settings": {"dpi": 600, "size_in": list(SIZE), "palette": [NAVY, CYAN, TEAL, AMBER]},
        "figures": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in made},
    }
    (OUT / "FIGURE_PROVENANCE_R2.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

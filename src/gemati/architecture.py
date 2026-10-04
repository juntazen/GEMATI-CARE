from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BG_TOP = "#F4F9FF"
BG_BOTTOM = "#FFFFFF"
NAVY = "#123B66"
INK = "#20354A"
MUTED = "#60788F"
CYAN = "#13A8D3"
TEAL = "#16A394"
BLUE = "#4E7DE9"
AMBER = "#F3A83B"
ROSE = "#E8708A"
BORDER = "#D8E5F1"
AUDIT_BG = "#EDF6FF"


def _rounded(ax, x, y, w, h, *, face="white", edge=BORDER, radius=0.16, lw=1.15, z=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    patch.set_path_effects([
        pe.SimplePatchShadow(offset=(1.4, -1.4), shadow_rgbFace="#B8CADB", alpha=0.18),
        pe.Normal(),
    ])
    ax.add_patch(patch)
    return patch


def _chip(ax, x, y, text, color, width=0.45):
    pill = FancyBboxPatch(
        (x, y), width, 0.27,
        boxstyle="round,pad=0.01,rounding_size=0.12",
        facecolor=color, edgecolor="none", zorder=6,
    )
    ax.add_patch(pill)
    ax.text(x + width / 2, y + 0.135, text, ha="center", va="center",
            color="white", fontsize=6.9, weight="bold", zorder=7)


def _stage_card(ax, x, y, w, h, number, title, lines, color):
    _rounded(ax, x, y, w, h)
    ax.add_patch(FancyBboxPatch(
        (x, y + h - 0.10), w, 0.10,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        facecolor=color, edgecolor="none", zorder=5,
    ))
    _chip(ax, x + 0.13, y + h - 0.42, number, color)
    ax.text(x + 0.13, y + h - 0.62, title, ha="left", va="top",
            fontsize=9.1, weight="bold", color=NAVY, zorder=6)
    ax.text(x + 0.13, y + h - 0.94, lines, ha="left", va="top",
            fontsize=7.35, color=MUTED, linespacing=1.28, zorder=6)


def _arrow(ax, start, end, *, color=CYAN, rad=0.0, lw=1.9, z=2):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0, zorder=z,
    )
    ax.add_patch(arrow)
    return arrow


def _output_card(ax, x, y, w, h, title, subtitle, color):
    _rounded(ax, x, y, w, h, face="white", edge=BORDER, radius=0.12, lw=1.0, z=4)
    ax.add_patch(FancyBboxPatch(
        (x, y), 0.12, h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=color, edgecolor="none", zorder=5,
    ))
    ax.text(x + 0.25, y + h * 0.61, title, ha="left", va="center",
            fontsize=8.6, weight="bold", color=NAVY, zorder=6)
    ax.text(x + 0.25, y + h * 0.28, subtitle, ha="left", va="center",
            fontsize=6.7, color=MUTED, zorder=6)


def _audit_card(ax, x, y, w, title, subtitle, color):
    _rounded(ax, x, y, w, 0.78, face="white", edge="#D6E8F7", radius=0.12, lw=0.9, z=4)
    ax.add_patch(FancyBboxPatch(
        (x + 0.12, y + 0.48), 0.30, 0.18,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        facecolor=color, edgecolor="none", zorder=5,
    ))
    ax.text(x + 0.52, y + 0.55, title, ha="left", va="center",
            fontsize=7.6, weight="bold", color=NAVY, zorder=6)
    ax.text(x + 0.14, y + 0.24, subtitle, ha="left", va="center",
            fontsize=6.25, color=MUTED, zorder=6)


def generate_architecture_figure(output_dir: Path) -> list[Path]:
    """Generate the publication Figure 1 from the implemented GEMATI-CARE protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.0, 7.8), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    # Quiet light gradient keeps the figure modern without reducing print legibility.
    top = np.array([244, 249, 255]) / 255
    bottom = np.array([255, 255, 255]) / 255
    gradient = np.linspace(top, bottom, 320).reshape(320, 1, 3)
    ax.imshow(gradient, extent=(0, 16, 0, 7.8), aspect="auto", zorder=0)

    ax.text(0.55, 7.30, "GEMATI–CARE", fontsize=18.5, weight="bold", color=NAVY, va="center")
    ax.text(3.06, 7.30, "Cost-sensitive conformal routing for crisis-risk research",
            fontsize=11.0, color=MUTED, va="center")
    ax.add_patch(FancyBboxPatch(
        (13.55, 7.02), 1.85, 0.50,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor="#E3F7F4", edgecolor="#B8E5DE", linewidth=0.9, zorder=4,
    ))
    ax.text(14.475, 7.27, "CPU-FIRST  •  AUDITABLE", ha="center", va="center",
            fontsize=7.6, weight="bold", color="#13796F", zorder=5)

    ax.text(0.62, 6.63, "PRIMARY ROUTING PATH", fontsize=8.0, weight="bold", color=CYAN)
    ax.text(2.55, 6.63, "evaluated on public CRADLE data with disjoint calibration",
            fontsize=7.6, color=MUTED)

    y, h = 4.18, 1.78
    cards = [
        (0.55, 1.35, "01", "PUBLIC TEXT", "CRADLE message\nlow • medium • high", CYAN),
        (2.20, 1.55, "02", "TF–IDF", "word 1–2 grams\ncharacter 3–5 grams", BLUE),
        (4.05, 1.70, "03", "RISK HEAD", "class-weighted OvR\nlogistic regression", TEAL),
        (6.05, 1.62, "04", "CALIBRATION", "temperature scaling\nprobability set  n=418", AMBER),
        (7.97, 1.68, "05", "MONDRIAN SET", "class-conditional CP\nconformal set  n=836", CYAN),
        (9.95, 1.78, "06", "ROBUST ROUTER", "minimax over C(x)\nengineering cost L(a,k)", ROSE),
    ]

    # Connectors are drawn first and terminate exactly on card boundaries.
    for left, right in zip(cards[:-1], cards[1:]):
        x1, w1 = left[0], left[1]
        x2 = right[0]
        _arrow(ax, (x1 + w1, y + h / 2), (x2, y + h / 2), color="#42A9C9", lw=1.8)

    for x, w, num, title, lines, color in cards:
        _stage_card(ax, x, y, w, h, num, title, lines, color)

    ax.text(12.16, 6.63, "ROUTING ACTION", fontsize=8.0, weight="bold", color=ROSE)
    outputs = [
        (5.34, "SUPPORT", "routine supportive path", TEAL),
        (4.58, "CLARIFY", "selective abstention", AMBER),
        (3.82, "ESCALATE", "human review pathway", ROSE),
    ]
    out_x, out_w, out_h = 12.23, 3.10, 0.61
    router_right = cards[-1][0] + cards[-1][1]
    router_y = y + h / 2
    for out_y, title, subtitle, color in outputs:
        target = (out_x, out_y + out_h / 2)
        rad = -0.12 if out_y > y + 0.4 else (0.12 if out_y < y else 0.0)
        _arrow(ax, (router_right, router_y), target, color=color, rad=rad, lw=1.7)
        _output_card(ax, out_x, out_y, out_w, out_h, title, subtitle, color)

    # Evaluation layer: solid connectors, clearly separated from the deployed decision path.
    band = FancyBboxPatch(
        (0.55, 0.42), 14.78, 2.42,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor=AUDIT_BG, edgecolor="#CFE3F4", linewidth=1.0, zorder=1,
    )
    ax.add_patch(band)
    ax.text(0.83, 2.48, "EVALUATION & AUDIT LAYER", fontsize=8.0, weight="bold", color=NAVY)
    ax.text(3.17, 2.48, "diagnostic evidence only • excluded from the conformal coverage claim",
            fontsize=7.25, color=MUTED)

    # One continuous audit bus connected to the measured primary path.
    bus_y = 2.13
    source_x = 8.81
    ax.plot([source_x, source_x], [y, bus_y], color="#8BBBD4", linewidth=1.35, zorder=2)
    ax.plot([1.13, 14.76], [bus_y, bus_y], color="#8BBBD4", linewidth=1.35, zorder=2)
    ax.scatter([source_x], [bus_y], s=28, color=CYAN, edgecolor="white", linewidth=0.9, zorder=5)

    audits = [
        (0.82, 2.12, "STABILITY", "5 repeated\nstratified splits", BLUE),
        (3.20, 2.12, "UNCERTAINTY", "1,000 bootstrap\nreplications", CYAN),
        (5.58, 2.12, "LEAKAGE", "exact + semantic\nnear-duplicates", TEAL),
        (7.96, 2.12, "TRANSFER", "IndoSafety\nregional languages", AMBER),
        (10.34, 2.12, "PERSISTENCE", "SIM–VAIL\nconversation audit", ROSE),
        (12.72, 2.12, "EFFICIENCY", "FP32 / INT8\nCPU benchmark", NAVY),
    ]
    audit_y = 0.82
    for x, w, title, subtitle, color in audits:
        center = x + w / 2
        _arrow(ax, (center, bus_y), (center, audit_y + 0.78), color="#8BBBD4", lw=1.2, z=2)
        _audit_card(ax, x, audit_y, w, title, subtitle, color)

    ax.text(15.31, 0.18, "Offline research prototype  •  no diagnosis  •  no autonomous clinical triage",
            ha="right", va="center", fontsize=6.9, color="#8A3C4A")

    fig.tight_layout(pad=0.25)
    wide_paths = [output_dir / "arsitektur_gemati_care_wide.png", output_dir / "arsitektur_gemati_care_wide.pdf"]
    fig.savefig(wide_paths[0], dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(wide_paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # The manuscript uses a compact, column-readable composition of the same pipeline.
    main_paths = _generate_column_figure(output_dir)
    return main_paths + wide_paths


def _generate_column_figure(output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.6, 4.0), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10.4)
    ax.axis("off")
    top = np.array([244, 249, 255]) / 255
    bottom = np.array([255, 255, 255]) / 255
    gradient = np.linspace(top, bottom, 260).reshape(260, 1, 3)
    ax.imshow(gradient, extent=(0, 12, 0, 10.4), aspect="auto", zorder=0)

    ax.text(0.45, 9.93, "GEMATI–CARE", fontsize=11.8, weight="bold", color=NAVY, va="center")
    ax.text(4.72, 9.93, "CPU-FIRST CONFORMAL ROUTING", fontsize=6.6, weight="bold", color=TEAL, va="center")
    ax.text(0.47, 9.43, "PRIMARY PATH  •  disjoint calibration  •  solid, traceable decisions",
            fontsize=6.6, color=MUTED, va="center")

    card_w, card_h = 3.22, 1.55
    top_y, lower_y = 7.48, 5.28
    stages = [
        (0.45, top_y, "01", "PUBLIC TEXT", "CRADLE messages\nlow • medium • high", CYAN),
        (4.39, top_y, "02", "TF–IDF", "word 1–2 grams\ncharacter 3–5 grams", BLUE),
        (8.33, top_y, "03", "RISK HEAD", "class-weighted OvR\nlogistic regression", TEAL),
        (8.33, lower_y, "04", "CALIBRATION", "temperature scaling\nprobability n=418", AMBER),
        (4.39, lower_y, "05", "MONDRIAN SET", "class-conditional CP\nconformal n=836", CYAN),
        (0.45, lower_y, "06", "ROBUST ROUTER", "minimax over C(x)\nengineering cost L(a,k)", ROSE),
    ]

    def compact_card(x, y, number, title, lines, color):
        _rounded(ax, x, y, card_w, card_h, radius=0.18, lw=1.0)
        ax.add_patch(FancyBboxPatch(
            (x, y + card_h - 0.10), card_w, 0.10,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=color, edgecolor="none", zorder=5,
        ))
        _chip(ax, x + 0.16, y + card_h - 0.42, number, color, width=0.68)
        ax.text(x + 1.02, y + card_h - 0.29, title, fontsize=7.7, weight="bold",
                color=NAVY, va="center", zorder=6)
        ax.text(x + 0.18, y + 0.53, lines, fontsize=6.35, color=MUTED,
                va="center", linespacing=1.25, zorder=6)

    # Continuous snake connector with exact edge-to-edge termination.
    _arrow(ax, (0.45 + card_w, top_y + card_h / 2), (4.39, top_y + card_h / 2), color="#42A9C9", lw=1.55)
    _arrow(ax, (4.39 + card_w, top_y + card_h / 2), (8.33, top_y + card_h / 2), color="#42A9C9", lw=1.55)
    _arrow(ax, (8.33 + card_w / 2, top_y), (8.33 + card_w / 2, lower_y + card_h), color="#42A9C9", lw=1.55)
    _arrow(ax, (8.33, lower_y + card_h / 2), (4.39 + card_w, lower_y + card_h / 2), color="#42A9C9", lw=1.55)
    _arrow(ax, (4.39, lower_y + card_h / 2), (0.45 + card_w, lower_y + card_h / 2), color="#42A9C9", lw=1.55)
    for stage in stages:
        compact_card(*stage)

    # Router branches into three explicitly governed actions.
    bus_y = 4.65
    router_center = 0.45 + card_w / 2
    ax.plot([router_center, router_center], [lower_y, bus_y], color=ROSE, linewidth=1.5, zorder=2)
    ax.plot([1.93, 10.07], [bus_y, bus_y], color=ROSE, linewidth=1.5, zorder=2)
    action_specs = [
        (0.45, "SUPPORT", "routine path", TEAL),
        (4.39, "CLARIFY", "selective abstention", AMBER),
        (8.33, "ESCALATE", "human review", ROSE),
    ]
    for x, title, subtitle, color in action_specs:
        center = x + card_w / 2
        _arrow(ax, (center, bus_y), (center, 4.07), color=color, lw=1.35)
        _output_card(ax, x, 3.30, card_w, 0.77, title, subtitle, color)

    band = FancyBboxPatch(
        (0.45, 0.48), 11.10, 2.24,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor=AUDIT_BG, edgecolor="#CFE3F4", linewidth=1.0, zorder=1,
    )
    ax.add_patch(band)
    ax.text(0.76, 2.43, "EVALUATION-ONLY AUDITS", fontsize=7.1, weight="bold", color=NAVY)
    ax.text(0.76, 2.12, "Diagnostic evidence only; excluded from the conformal coverage claim", fontsize=5.65, color=MUTED)
    audit_items = [
        (0.75, 1.43, "5 seeds", BLUE), (4.26, 1.43, "1,000 bootstrap", CYAN), (7.77, 1.43, "leakage audit", TEAL),
        (0.75, 0.72, "IndoSafety", AMBER), (4.26, 0.72, "SIM–VAIL", ROSE), (7.77, 0.72, "CPU FP32 / INT8", NAVY),
    ]
    for x, y, label, color in audit_items:
        pill = FancyBboxPatch(
            (x, y), 3.05, 0.47,
            boxstyle="round,pad=0.015,rounding_size=0.15",
            facecolor="white", edgecolor="#D6E8F7", linewidth=0.85, zorder=4,
        )
        ax.add_patch(pill)
        ax.add_patch(FancyBboxPatch(
            (x + 0.13, y + 0.12), 0.23, 0.23,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=color, edgecolor="none", zorder=5,
        ))
        ax.text(x + 0.50, y + 0.235, label, fontsize=6.15, weight="bold",
                color=INK, va="center", zorder=6)

    ax.text(11.47, 0.18, "offline prototype • no diagnosis • no autonomous triage",
            ha="right", fontsize=5.6, color="#8A3C4A")
    fig.tight_layout(pad=0.12)
    paths = [output_dir / "arsitektur_gemati_care.png", output_dir / "arsitektur_gemati_care.pdf"]
    fig.savefig(paths[0], dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths

"""Compare a fresh run (results/extended_metrics.json) with the values reported in the paper.

Usage:  python scripts/compare_with_paper.py
Prints PASS/FAIL for each headline metric. Tolerance 0.001 (0.1 percentage point).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "results" / "extended_metrics.json"
REF = ROOT / "results" / "extended_metrics_public.json"

CHECKS = [
    ("Validation macro-F1, TF-IDF", ["model_selection", "tfidf_macro_f1"]),
    ("Test macro-F1", ["final_test_primary_alpha_0.10", "classification", "macro_f1"]),
    ("Test ECE", ["final_test_primary_alpha_0.10", "classification", "ece"]),
    ("Argmax high-risk miss", ["final_test_primary_alpha_0.10", "routing_argmax", "high_risk_miss_rate"]),
    ("Router a=0.05 coverage", ["final_test_safety_alpha_0.05", "conformal", "coverage"]),
    ("Router a=0.05 high-risk miss", ["final_test_safety_alpha_0.05", "routing", "high_risk_miss_rate"]),
    ("Router a=0.05 under-escalation", ["final_test_safety_alpha_0.05", "routing", "under_escalation_rate"]),
    ("Router a=0.05 over-escalation", ["final_test_safety_alpha_0.05", "routing", "over_escalation_rate"]),
]


def get(d, path):
    for key in path:
        d = d[key]
    return float(d)


def main() -> int:
    if not NEW.exists():
        print(f"Not found: {NEW}\nRun 'python -m gemati experiment' and 'python -m gemati extended' first.")
        return 2
    new, ref = json.loads(NEW.read_text()), json.loads(REF.read_text())
    failed = 0
    for name, path in CHECKS:
        a, b = get(new, path), get(ref, path)
        ok = abs(a - b) <= 1e-3
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} paper={b:.4f}  your run={a:.4f}")
    print("\nAll headline metrics reproduced." if not failed else f"\n{failed} metric(s) differ. See the guide, section Troubleshooting.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

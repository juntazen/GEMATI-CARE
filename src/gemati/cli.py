from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .architecture import generate_architecture_figure
from .audit import audit
from .docx_report import generate_bilingual_docx
from .experiment import run_experiment
from .extended import run_extended
from .progress import read_progress
from .report import generate_papers


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemati", description="GEMATI-CARE research automation")
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("experiment", help="download data and run the CPU experiment")
    sub.add_parser("paper", help="generate Indonesian and English RESTI drafts")
    sub.add_parser("architecture", help="generate publication-ready architecture figures")
    sub.add_parser("extended", help="run full submission-grade experiment suite")
    sub.add_parser("audit", help="audit final artifacts and submission readiness")
    docx = sub.add_parser("docx", help="fill TemplateRESTI2026.docx with both paper languages")
    docx.add_argument("--template", type=Path, default=None)
    sub.add_parser("all", help="run experiment, figures, and bilingual papers")
    sub.add_parser("status", help="print current progress and ETA")
    monitor = sub.add_parser("monitor", help="continuously print progress until completion/failure")
    monitor.add_argument("--interval", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    config = args.config or root / "configs" / "experiment.json"
    progress_path = root / "results" / "progress.json"

    if args.command in ("experiment", "all"):
        run_experiment(root, config)
    if args.command in ("extended", "all"):
        run_extended(root, config)
    if args.command in ("architecture", "all"):
        for path in generate_architecture_figure(root / "docs" / "figures"):
            print(path)
    if args.command in ("paper", "all"):
        for path in generate_papers(root / "results" / "extended_metrics.json", root / "paper"):
            print(path)
    if args.command == "docx":
        template = args.template or root.parent / "resti" / "TemplateRESTI2026.docx"
        for path in generate_bilingual_docx(root, template):
            print(path)
    if args.command == "all":
        template = root.parent / "resti" / "TemplateRESTI2026.docx"
        if template.exists():
            for path in generate_bilingual_docx(root, template):
                print(path)
        else:
            print(f"DOCX dilewati: template belum tersedia di {template}")
    if args.command == "status":
        extended_path = root / "results" / "extended_progress.json"
        state = {
            "core": read_progress(progress_path),
            "extended": read_progress(extended_path),
        }
        print(json.dumps(state, indent=2, ensure_ascii=False))
    if args.command == "monitor":
        previous = None
        while True:
            state = {
                "core": read_progress(progress_path),
                "extended": read_progress(root / "results" / "extended_progress.json"),
            }
            serialized = json.dumps(state, sort_keys=True, ensure_ascii=False)
            if serialized != previous:
                print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)
                previous = serialized
            statuses = [item.get("status") for item in state.values()]
            if all(status in ("complete", "failed") for status in statuses):
                break
            time.sleep(max(1.0, args.interval))
    if args.command == "audit":
        verdict, markdown = audit(root)
        results = root / "results"
        (results / "FINAL_AUDIT.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (results / "FINAL_AUDIT.md").write_text(markdown, encoding="utf-8")
        print(markdown)


if __name__ == "__main__":
    main()

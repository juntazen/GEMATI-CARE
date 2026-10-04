from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd


CRADLE_DATASET = "SungJoo/Cradle-Bench"
INDOSAFETY_FILES = {
    "eval1": "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/dataset/IndoSafety-Eval-1.xlsx",
    "eval2": "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/dataset/IndoSafety-Eval-2.xlsx",
}
EMOTION_URL = (
    "https://raw.githubusercontent.com/meisaputri21/"
    "Indonesian-Twitter-Emotion-Dataset/master/Twitter_Emotion_Dataset.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    request = Request(url, headers={"User-Agent": "gemati-care-research/0.1"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def _flatten_label(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value).lower()
    return str(value).lower()


def map_cradle_risk(value: object) -> str:
    """Map CRADLE clinical categories into non-clinical routing severity.

    This is a transparent engineering mapping, not a clinical risk score.
    """
    label = _flatten_label(value)
    compact = re.sub(r"[^a-z0-9]+", "_", label)
    if "no_crisis" in compact and not any(
        token in compact
        for token in ("suicide", "selfharm", "self_harm", "violence", "rape", "abuse", "harassment")
    ):
        return "low"
    high_tokens = (
        "active_ongoing",
        "selfharm_ongoing",
        "self_harm_ongoing",
        "domesticviolence_ongoing",
        "domestic_violence_ongoing",
        "rape_ongoing",
        "childabuse_ongoing",
        "child_abuse_ongoing",
        "endangerment_ongoing",
    )
    if any(token in compact for token in high_tokens):
        return "high"
    if "ongoing" in compact and any(
        token in compact for token in ("suicide", "selfharm", "violence", "rape", "abuse")
    ):
        return "high"
    return "medium"


def load_cradle(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load a public, ungated Hugging Face dataset and persist immutable parquet snapshots."""
    from datasets import load_dataset

    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(CRADLE_DATASET)
    output: dict[str, pd.DataFrame] = {}
    for split_name, split in dataset.items():
        frame = split.to_pandas()
        text_col = next((c for c in ("question_text", "text", "prompt") if c in frame.columns), None)
        label_col = next((c for c in ("final_labels", "labels", "label") if c in frame.columns), None)
        if text_col is None or label_col is None:
            raise ValueError(f"Kolom CRADLE tidak dikenali: {list(frame.columns)}")
        clean = pd.DataFrame(
            {
                "id": frame.get("question_id", frame.index.astype(str)).astype(str),
                "text": frame[text_col].fillna("").astype(str),
                "source_label": frame[label_col].map(_flatten_label),
                "risk": frame[label_col].map(map_cradle_risk),
                "source": "cradle",
                "official_split": split_name,
            }
        )
        clean = clean[clean["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)
        path = raw_dir / f"cradle_{split_name}.parquet"
        clean.to_parquet(path, index=False)
        output[split_name] = clean
    return output


def load_indosafety(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    language_columns = ("indonesian-formal", "colloquial", "minangkabau", "java", "sunda")
    for name, url in INDOSAFETY_FILES.items():
        path = download_file(url, raw_dir / f"indosafety_{name}.xlsx")
        workbook = pd.ExcelFile(path)
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet)
            frame.columns = [str(c).strip().lower() for c in frame.columns]
            text_col = next(
                (c for c in frame.columns if any(k in c for k in ("prompt", "instruction", "question"))),
                None,
            )
            category_cols = [
                c
                for c in frame.columns
                if c in ("risk_area", "types_of_harm", "specific_harms", "category")
            ]
            if category_cols:
                category = frame[category_cols].fillna("").astype(str).agg(" | ".join, axis=1)
            else:
                category = pd.Series(["unknown"] * len(frame), index=frame.index)
            lang_col = next((c for c in frame.columns if "lang" in c or "bahasa" in c), None)
            if text_col is not None:
                out = pd.DataFrame(
                    {
                        "id": [f"{name}:{sheet}:{i}" for i in range(len(frame))],
                        "text": frame[text_col].fillna("").astype(str),
                        "category": category,
                        "language": frame[lang_col].fillna("indonesian-formal").astype(str)
                        if lang_col
                        else "indonesian-formal",
                        "source": "indosafety",
                    }
                )
                frames.append(out[out["text"].str.len() > 0])
            else:
                for language in language_columns:
                    if language not in frame.columns:
                        continue
                    out = pd.DataFrame(
                        {
                            "id": [f"{name}:{sheet}:{language}:{i}" for i in range(len(frame))],
                            "text": frame[language].fillna("").astype(str),
                            "category": category,
                            "language": language,
                            "source": "indosafety",
                        }
                    )
                    frames.append(out[out["text"].str.len() > 0])
    if not frames:
        raise ValueError("Tidak menemukan kolom prompt dalam berkas IndoSafety.")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    result.to_parquet(raw_dir / "indosafety_external.parquet", index=False)
    return result


def load_emotion(raw_dir: Path) -> pd.DataFrame:
    path = download_file(EMOTION_URL, raw_dir / "indonesian_emotion.csv")
    frame = pd.read_csv(path)
    frame.columns = [str(c).lower().strip() for c in frame.columns]
    text_col = next(c for c in frame.columns if c in ("tweet", "text"))
    label_col = next(c for c in frame.columns if c in ("label", "emotion"))
    result = frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "emotion"})
    result["source"] = "indonesian_twitter_emotion"
    result.to_parquet(raw_dir / "emotion.parquet", index=False)
    return result


def write_manifest(raw_dir: Path, output: Path) -> pd.DataFrame:
    records = []
    for path in sorted(raw_dir.glob("*")):
        if path.is_file():
            records.append(
                {
                    "file": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def save_dataset_summary(datasets: dict[str, pd.DataFrame], path: Path) -> None:
    summary = {
        name: {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "risk_distribution": frame["risk"].value_counts().to_dict() if "risk" in frame else None,
        }
        for name, frame in datasets.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_texts(frames: Iterable[pd.DataFrame]) -> Iterable[str]:
    for frame in frames:
        yield from frame["text"].astype(str)

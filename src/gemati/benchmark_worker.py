"""Isolated CPU benchmark worker for the GEMATI encoder.

Run exactly one encoder variant per process.  Keeping the worker single-variant
is important: ``resource.getrusage(...).ru_maxrss`` is cumulative for a process,
so benchmarking FP32 and INT8 in the same interpreter produces invalid memory
comparisons.

Example
-------
PYTHONPATH=src python3 -m gemati.benchmark_worker --variant fp32
PYTHONPATH=src python3 -m gemati.benchmark_worker --variant int8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .encoder import MiniLMRiskModel


def _rss_mb() -> float:
    """Return current RSS when psutil is available, otherwise current peak RSS."""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except (ImportError, OSError):
        return _peak_rss_mb()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    if platform.system() == "Darwin":
        return value / (1024**2)
    return value / 1024.0


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "repetitions": int(len(array)),
    }


def _stable_sample(texts: Sequence[str], size: int, seed: int) -> list[str]:
    """Choose a deterministic sample independent of input row ordering."""
    unique = list(dict.fromkeys(str(text) for text in texts if str(text).strip()))
    ranked = sorted(
        unique,
        key=lambda text: hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest(),
    )
    return ranked[: min(size, len(ranked))]


def _cyclic_batch(texts: Sequence[str], start: int, size: int) -> list[str]:
    return [texts[(start + offset) % len(texts)] for offset in range(size)]


def _serialized_state_dict_size(encoder) -> int:
    """Serialize the in-memory variant and remove the temporary file."""
    import torch

    descriptor, name = tempfile.mkstemp(prefix="gemati_encoder_", suffix=".pt")
    os.close(descriptor)
    path = Path(name)
    try:
        torch.save(encoder.state_dict(), path)
        return int(path.stat().st_size)
    finally:
        path.unlink(missing_ok=True)


def benchmark(
    root: Path,
    variant: str,
    model_name: str,
    max_length: int,
    encoder_batch_size: int,
    sample_size: int,
    seed: int,
    warmup_items: int,
    single_repetitions: int,
    batch_repetitions: int,
    benchmark_batch_size: int,
) -> dict:
    import torch
    import transformers

    test_path = root / "data" / "raw" / "cradle_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test snapshot tidak ditemukan: {test_path}")
    frame = pd.read_parquet(test_path, columns=["text"])
    sample = _stable_sample(frame["text"].tolist(), sample_size, seed)
    minimum = max(2, warmup_items + 1, benchmark_batch_size)
    if len(sample) < minimum:
        raise ValueError(f"Sampel valid hanya {len(sample)}; minimal {minimum} diperlukan")

    baseline_rss = _rss_mb()
    baseline_peak = _peak_rss_mb()
    model = MiniLMRiskModel(
        cache_dir=root / ".cache" / "benchmark_worker_unused",
        model_name=model_name,
        batch_size=encoder_batch_size,
        max_length=max_length,
        optimized=variant == "int8",
    )

    # This includes tokenizer/model loading from the local HF cache and the
    # first single-text forward pass.  ``encode_uncached`` deliberately avoids
    # the project's persistent .npy embedding cache.
    started = time.perf_counter()
    first_vector = model.encode_uncached([sample[0]])
    cold_load_first_inference_seconds = time.perf_counter() - started
    if first_vector.shape[0] != 1:
        raise RuntimeError("Encoder tidak menghasilkan satu vektor untuk satu input")

    warmup = sample[1 : 1 + warmup_items]
    if warmup:
        model.encode_uncached(warmup)

    single_ms: list[float] = []
    for repetition in range(single_repetitions):
        text = sample[(1 + warmup_items + repetition) % len(sample)]
        started = time.perf_counter()
        model.encode_uncached([text])
        single_ms.append((time.perf_counter() - started) * 1000.0)

    batch_ms: list[float] = []
    batch_throughput: list[float] = []
    offset = 1 + warmup_items + single_repetitions
    for repetition in range(batch_repetitions):
        batch = _cyclic_batch(sample, offset + repetition * benchmark_batch_size, benchmark_batch_size)
        started = time.perf_counter()
        model.encode_uncached(batch)
        elapsed = time.perf_counter() - started
        batch_ms.append(elapsed * 1000.0)
        batch_throughput.append(benchmark_batch_size / elapsed)

    runtime_current_rss = _rss_mb()
    runtime_peak_rss = _peak_rss_mb()
    state_dict_bytes = _serialized_state_dict_size(model._encoder)
    post_serialization_peak_rss = _peak_rss_mb()

    return {
        "variant": variant,
        "optimized_dynamic_int8": variant == "int8",
        "fresh_process_pid": os.getpid(),
        "cold_scope": (
            "fresh Python process; tokenizer/model load from Hugging Face cache plus first inference; "
            "persistent embedding cache bypassed"
        ),
        "sample": {
            "source": str(test_path),
            "selection": "SHA-256 deterministic ranking of unique non-empty texts",
            "seed": seed,
            "available_rows": int(len(frame)),
            "selected_unique_texts": int(len(sample)),
            "sha256": hashlib.sha256("\0".join(sample).encode("utf-8")).hexdigest(),
        },
        "settings": {
            "model_name": model_name,
            "max_length": max_length,
            "encoder_batch_size": encoder_batch_size,
            "warmup_items": warmup_items,
            "single_repetitions": single_repetitions,
            "batch_repetitions": batch_repetitions,
            "benchmark_batch_size": benchmark_batch_size,
            "torch_threads": int(torch.get_num_threads()),
        },
        "latency": {
            "cold_load_first_inference_seconds": cold_load_first_inference_seconds,
            "warm_single_ms": _percentiles(single_ms),
            "warm_batch_ms": _percentiles(batch_ms),
        },
        "throughput_texts_per_second": _percentiles(batch_throughput),
        "memory_mb": {
            "baseline_current_rss": baseline_rss,
            "baseline_peak_rss": baseline_peak,
            "runtime_current_rss": runtime_current_rss,
            "runtime_peak_rss": runtime_peak_rss,
            "runtime_peak_delta_from_baseline_current": max(0.0, runtime_peak_rss - baseline_rss),
            "post_serialization_peak_rss": post_serialization_peak_rss,
        },
        "serialized_encoder_state_dict_bytes": state_dict_bytes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "psutil_available": _psutil_available(),
        },
    }


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh-process GEMATI FP32/INT8 CPU benchmark")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--variant", choices=("fp32", "int8"), required=True)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-items", type=int, default=8)
    parser.add_argument("--single-repetitions", type=int, default=50)
    parser.add_argument("--batch-repetitions", type=int, default=12)
    parser.add_argument("--benchmark-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = benchmark(
        root=args.root.resolve(),
        variant=args.variant,
        model_name=args.model_name,
        max_length=args.max_length,
        encoder_batch_size=args.encoder_batch_size,
        sample_size=args.sample_size,
        seed=args.seed,
        warmup_items=args.warmup_items,
        single_repetitions=args.single_repetitions,
        batch_repetitions=args.batch_repetitions,
        benchmark_batch_size=args.benchmark_batch_size,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

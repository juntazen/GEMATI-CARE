from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class MiniLMRiskModel:
    """Frozen multilingual encoder plus a lightweight CPU risk head.

    Transformer weights are intentionally omitted from joblib serialization and
    lazily loaded from the public model snapshot/cache during inference.
    """

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
        max_length: int = 128,
        c: float = 2.0,
        max_iter: int = 1500,
        seed: int = 42,
        progress_callback: Callable[[str], None] | None = None,
        optimized: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.c = c
        self.max_iter = max_iter
        self.seed = seed
        self.progress_callback = progress_callback
        self.optimized = optimized
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=max_iter,
                solver="liblinear",
                random_state=seed,
            )
        )
        self._tokenizer = None
        self._encoder = None

    @property
    def classes_(self):
        return self.classifier.classes_

    def _load_encoder(self):
        if self._encoder is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._encoder = AutoModel.from_pretrained(self.model_name)
            self._encoder.to("cpu")
            self._encoder.eval()
            if getattr(self, "optimized", False):
                self._encoder = torch.ao.quantization.quantize_dynamic(
                    self._encoder, {torch.nn.Linear}, dtype=torch.qint8
                )

    def _fingerprint(self, texts: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode())
        digest.update(str(self.max_length).encode())
        digest.update(str(getattr(self, "optimized", False)).encode())
        for text in texts:
            digest.update(str(text).encode("utf-8", errors="replace"))
            digest.update(b"\0")
        return digest.hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(item) for item in texts]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self._fingerprint(texts)
        cache_path = self.cache_dir / f"{fingerprint}.npy"
        if cache_path.exists():
            if self.progress_callback:
                self.progress_callback(f"Memuat cache embedding {cache_path.name[:12]}")
            return np.load(cache_path)

        matrix = self.encode_uncached(texts)
        np.save(cache_path, matrix)
        return matrix

    def encode_uncached(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(item) for item in texts]
        self._load_encoder()
        import torch

        vectors = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
                batch = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                output = self._encoder(**encoded)
                token_embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype(np.float32))
                if self.progress_callback and (
                    batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0
                ):
                    self.progress_callback(
                        f"Embedding batch {batch_index}/{total_batches} ({min(start + len(batch), len(texts))}/{len(texts)})"
                    )
        return np.concatenate(vectors, axis=0)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "MiniLMRiskModel":
        self.classifier.fit(self.encode(texts), labels)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.encode(texts))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_encoder"] = None
        state["progress_callback"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "optimized" not in self.__dict__:
            self.optimized = False

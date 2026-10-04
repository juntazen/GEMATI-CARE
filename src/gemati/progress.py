from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ProgressState:
    status: str
    stage: str
    completed_steps: int
    total_steps: int
    percent: float
    started_at: str
    updated_at: str
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    message: str


class ProgressTracker:
    def __init__(self, path: Path, total_steps: int) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.started = time.monotonic()
        self.started_iso = self._now()
        self.completed = 0
        self.update("initializing", "Pipeline dimulai", status="running")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def update(self, stage: str, message: str, *, status: str = "running", increment: bool = False) -> None:
        if increment:
            self.completed = min(self.total_steps, self.completed + 1)
        elapsed = time.monotonic() - self.started
        rate = elapsed / self.completed if self.completed else None
        remaining = rate * (self.total_steps - self.completed) if rate is not None else None
        state = ProgressState(
            status=status,
            stage=stage,
            completed_steps=self.completed,
            total_steps=self.total_steps,
            percent=round(100 * self.completed / self.total_steps, 2),
            started_at=self.started_iso,
            updated_at=self._now(),
            elapsed_seconds=round(elapsed, 2),
            estimated_remaining_seconds=round(remaining, 2) if remaining is not None else None,
            message=message,
        )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        print(f"[{state.percent:6.2f}%] {stage}: {message}", flush=True)

    def complete(self, message: str) -> None:
        self.completed = self.total_steps
        self.update("complete", message, status="complete")

    def fail(self, message: str) -> None:
        self.update("failed", message, status="failed")


def read_progress(path: Path) -> dict:
    if not path.exists():
        return {"status": "not_started", "message": "Belum ada eksperimen."}
    return json.loads(path.read_text(encoding="utf-8"))


"""Persistence helpers for trace replay."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile


class TraceRepository:
    """Stores trace payloads locally for replay and auditing."""

    def __init__(self, traces_dir: Path) -> None:
        self.traces_dir = traces_dir
        self.fallback_traces_dir = Path(tempfile.gettempdir()) / "topic10_entity_linking_agent" / "traces"
        self._cache: dict[str, dict] = {}
        try:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def write_trace(self, payload: dict) -> bool:
        trace_id = payload["trace_id"]
        self._cache[trace_id] = payload
        for traces_dir in (self.traces_dir, self.fallback_traces_dir):
            target = self._build_path(trace_id, traces_dir)
            try:
                traces_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True
            except OSError:
                continue
        return False

    def read_trace(self, trace_id: str) -> dict:
        for traces_dir in (self.traces_dir, self.fallback_traces_dir):
            target = self._build_path(trace_id, traces_dir)
            if target.exists():
                return json.loads(target.read_text(encoding="utf-8"))

        cached = self._cache.get(trace_id)
        if cached is None:
            raise FileNotFoundError(trace_id)
        return cached

    @staticmethod
    def _build_path(trace_id: str, traces_dir: Path) -> Path:
        return traces_dir / f"{trace_id}.json"

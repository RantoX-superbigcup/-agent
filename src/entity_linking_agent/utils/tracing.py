"""Trace helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def build_trace_id(prefix: str = "t10") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"

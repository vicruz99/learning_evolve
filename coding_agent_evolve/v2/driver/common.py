"""Shared bits for the driver: cell loading, event log, fault signatures, turn record."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# Text that means the model was never reached. Any of these in a turn's stderr/error makes
# the turn "infrastructure", never a model stop (campaign lesson: weka-enospc, dead relay).
FAULT_SIGNATURES = (
    "PgClient", "Failed to connect", "Connection refused", "Connection reset", "ECONNREFUSED",
    "ECONNRESET", "unreachable", "Bad Gateway", "Service Unavailable", "No space left",
    "Invalid HTTP request", "APIConnectionError", "no live vLLM upstream", "ETIMEDOUT",
    "socket hang up", "fetch failed", "Timed out waiting for the shared backend",
)
INFRA_FAST_S = 60.0          # a turn shorter than this that errored never reached the model


def harness_fault(text: str) -> str | None:
    for sig in FAULT_SIGNATURES:
        if sig.lower() in (text or "").lower():
            return sig
    return None


def load_cell(cell_dir: Path) -> dict:
    return json.loads((Path(cell_dir) / "cell.json").read_text())


class EventLog:
    """Append-only JSONL of everything the harness reports; one file per cell."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "a", buffering=1)

    def write(self, kind: str, **fields) -> None:
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        self._fh.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


@dataclass
class TurnEnd:
    stop_reason: str                 # end_turn | max_tokens | cancelled | error | killed ...
    ran_s: float
    tools: int = 0                   # tool calls observed during the turn
    text_chars: int = 0
    error: str | None = None
    rc: int | None = None
    stderr_tail: str = ""
    extra: dict = field(default_factory=dict)

    def is_error(self) -> bool:
        return self.stop_reason in ("error", "acp_error") or self.error is not None or (self.rc not in (None, 0))


def tail(path: Path, n: int = 4000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log(cell_dir: Path, msg: str) -> None:
    line = f"[driver {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(Path(cell_dir) / "driver.log", "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass

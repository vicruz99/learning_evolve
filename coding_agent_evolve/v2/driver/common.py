"""Shared bits for the driver: cell loading, event log, fault signatures, turn record."""
from __future__ import annotations

import json
import sys
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
        self._fh = None
        try:
            self._fh = open(self.path, "a", buffering=1)
        except OSError as exc:         # ENOSPC at start-up: open lazily on the first write instead
            print(f"[events] open failed ({exc}); will retry on the first write", file=sys.stderr, flush=True)

    def write(self, kind: str, **fields) -> None:
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        line = json.dumps(rec, default=str) + "\n"
        # 2026-09-15: a transient ENOSPC burst on the scratch filesystem (weka, 445 TB free) raised here
        # and killed all six running Claude Code cells at 16:17:26 -- the events stream is diagnostics,
        # never worth the run. Drop the record, remember how many, and keep going.
        try:
            if self._fh is None or self._fh.closed:
                # 2026-09-15 (2): after a failed reopen the handle stayed CLOSED and the next write raised
                # ValueError, which the except below did not catch -- every Claude Code cell died of it.
                self._fh = open(self.path, "a", buffering=1)
            self._fh.write(line)
        except (OSError, ValueError) as exc:
            self.dropped = getattr(self, "dropped", 0) + 1
            if self.dropped in (1, 10, 100, 1000) or self.dropped % 10000 == 0:
                print(f"[events] write failed ({exc}); {self.dropped} record(s) dropped so far", file=sys.stderr, flush=True)
            try:                      # the handle may be poisoned after ENOSPC; reopen lazily
                if self._fh is not None:
                    self._fh.close()
            except (OSError, ValueError):
                pass
            self._fh = None

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except (OSError, ValueError):
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
    try:
        print(line, flush=True)      # stdout is LSF's lsf.out on weka: ENOSPC bursts raise here too (2026-09-15)
    except OSError:
        pass
    try:
        with open(Path(cell_dir) / "driver.log", "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass

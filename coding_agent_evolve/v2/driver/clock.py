"""Persistent budget clock: a cell's budget is ACTIVE time, accumulated across relaunches.

`clock.json` in the cell dir survives a killed job, a node reboot and a relaunch on another host.
Without it every relaunch restarted the 18 h budget from zero (2026-09-08 campaign), so cells
that crashed got more wall time than cells that did not. A heartbeat writes the file every
HEARTBEAT_S seconds; a hard kill loses at most that much accounted time.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

HEARTBEAT_S = 30
# 2026-09-21: paused time (LLM server outage, an operator PAUSE file) is NOT active time. The q38ac_r3
# cells lost 7.7 h of their 18 h budget to a server outage the clock kept charging.


class Clock:
    def __init__(self, cell_dir: Path, budget_s: float):
        self.path = Path(cell_dir) / "clock.json"
        self.budget_s = float(budget_s)
        prev = {}
        if self.path.is_file():
            try:
                prev = json.loads(self.path.read_text())
            except ValueError:
                prev = {}
        self.started = time.time()
        self.first_started = float(prev.get("first_started", self.started))
        self.active_before = float(prev.get("active_s", 0.0))
        self.launches = int(prev.get("launches", 0)) + 1
        self.paused_before = float(prev.get("paused_s", 0.0))   # total paused seconds of earlier launches
        self.paused_this = 0.0                                   # closed pauses of this launch
        self.pause_started: float | None = None                  # open pause, if any
        self.pauses: list[dict] = list(prev.get("pauses", []))
        self.remaining_at_start = max(0.0, self.budget_s - self.active_before)
        self.save()

    @property
    def relaunch(self) -> bool:
        return self.launches > 1

    @property
    def paused(self) -> bool:
        return self.pause_started is not None

    def _paused_now(self) -> float:
        return self.paused_this + ((time.time() - self.pause_started) if self.pause_started else 0.0)

    def elapsed(self) -> float:
        """Active seconds so far, across launches, minus every pause."""
        return self.active_before + (time.time() - self.started) - self._paused_now()

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    @property
    def deadline(self) -> float:
        """Wall time at which the budget runs out if the clock keeps running from now."""
        return time.time() + self.remaining()

    def pause(self, why: str) -> None:
        if self.pause_started is None:
            self.pause_started = time.time()
            self.pauses.append({"start": self.pause_started, "why": why})
            self.save()

    def resume(self) -> float:
        """Close the open pause; returns its length in seconds."""
        if self.pause_started is None:
            return 0.0
        dur = time.time() - self.pause_started
        self.paused_this += dur
        self.pauses[-1]["end"] = time.time(); self.pauses[-1]["s"] = round(dur, 1)
        self.pause_started = None
        self.save()
        return dur

    def save(self) -> None:
        rec = {"first_started": self.first_started, "last_started": self.started, "launches": self.launches,
               "active_s": round(self.elapsed(), 1), "budget_s": self.budget_s, "saved": time.time(),
               "paused_s": round(self.paused_before + self._paused_now(), 1), "paused_now": self.paused,
               "pauses": self.pauses[-200:]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2) + "\n")
        tmp.replace(self.path)

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            try:
                self.save()
            except OSError:
                pass   # a quota outage must not kill the driver; the next beat retries

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
        self.remaining_at_start = max(0.0, self.budget_s - self.active_before)
        self.deadline = self.started + self.remaining_at_start
        self.save()

    @property
    def relaunch(self) -> bool:
        return self.launches > 1

    def elapsed(self) -> float:
        """Active seconds so far, across launches."""
        return self.active_before + (time.time() - self.started)

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    def save(self) -> None:
        rec = {"first_started": self.first_started, "last_started": self.started, "launches": self.launches,
               "active_s": round(self.elapsed(), 1), "budget_s": self.budget_s, "saved": time.time()}
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

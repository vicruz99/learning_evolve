"""Continue / stop policy for one cell, ported from adrs_acp/run.py::run_session.

The agent's turn ending is the stop signal. The host then decides:

  * under the hard budget and no STOP file, otherwise the run is over;
  * the first stop of a run is always continued;
  * while the minima are unmet (official evals < min_evals, or elapsed < min_hours, or nothing
    better than the start scored yet) continue immediately;
  * once they are met: keep_going -> continue, throttled to one nudge per nudge_min_gap_s;
    not keep_going -> let the run end.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class State:
    elapsed_s: float
    hard_budget_s: float
    evals: int
    banked: bool
    continues: int
    since_last_nudge_s: float
    stop_reason: str | None


@dataclass
class Rules:
    min_evals: int
    min_hours: float
    keep_going: bool
    nudge_min_gap_s: float


def decide(s: State, r: Rules) -> tuple[bool, str]:
    if s.stop_reason:
        return False, f"stop file: {s.stop_reason}"
    if s.elapsed_s >= s.hard_budget_s:
        return False, "hard budget reached"
    if s.continues == 0:
        return True, "first stop"
    if s.evals < r.min_evals:
        return True, f"evals {s.evals} < minimum {r.min_evals}"
    if s.elapsed_s < r.min_hours * 3600:
        return True, f"elapsed {s.elapsed_s / 3600:.2f}h < minimum {r.min_hours}h"
    if not s.banked:
        return True, "nothing better than the start scored yet"
    if not r.keep_going:
        return False, "minima met and keep_going is off"
    if s.since_last_nudge_s >= r.nudge_min_gap_s:
        return True, f"{int(s.since_last_nudge_s)}s since last nudge"
    # keep_going: the gap is a throttle, not a stop -- the caller waits it out, then nudges
    return True, f"WAIT {int(r.nudge_min_gap_s - s.since_last_nudge_s)}s (minima met; throttled to one nudge per {int(r.nudge_min_gap_s)}s)"


def host_line(remaining_s: float, hours: float, evals: int, min_evals: int, max_evals: int,
              best: float | None, metric_word: str) -> str:
    rem = max(0, int(remaining_s)) // 60
    cap = f", cap {max_evals}" if max_evals else ""
    best_txt = f"best official {metric_word} so far {best!r}" if best is not None else "no valid official score yet"
    return (f"[host] {rem} min remaining of the {hours:g}h budget; {evals} official evaluations so far "
            f"(minimum {min_evals}{cap}); {best_txt}.")

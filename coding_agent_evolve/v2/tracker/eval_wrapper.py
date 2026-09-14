"""Official scoring wrapper. Do not edit. The real grader is _official_evaluator.py.

Installed as eval.py in every agent workspace (see tracker.py). Two roles:

  * `from eval import evaluate_sequence` (or verify_c5_solution) -- re-exported from the
    official grader, unrecorded. Inner-loop calls cost nothing and consume no budget.
  * `python eval.py candidate.npy` -- the OFFICIAL score. The candidate is snapshotted to
    <cell>/submissions/NNN.npy, one row is appended to <cell>/iterations.jsonl, best.npy /
    best.json are replaced when the score improves, and STOP is written at the eval cap.
    Output and exit code are exactly those of the official grader.

Adapted from arxiv_graph/baselines/adrs_acp/tracker.py (Tim's ADRS wrapper): our graders take
a .npy and print a float, and reject an invalid construction with a non-zero exit -- such a
call is logged with score null and still counts as an attempt.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _official_evaluator import *  # noqa: F401,F403  -- re-export the grading function(s)

_HERE = Path(__file__).resolve().parent
_META = _HERE / "_adrs_track.json"
_OFFICIAL = _HERE / "_official_evaluator.py"


def _lock(path: Path, timeout: float = 60.0):
    """mkdir-based lock: atomic on every filesystem, no flock semantics to trust."""
    deadline = time.time() + timeout
    while True:
        try:
            os.mkdir(path)
            return
        except FileExistsError:
            if time.time() > deadline:
                # a crashed holder must not wedge every later evaluation
                try:
                    if time.time() - path.stat().st_mtime > timeout:
                        os.rmdir(path)
                        continue
                except OSError:
                    pass
                raise TimeoutError(f"tracker lock held too long: {path}")
            time.sleep(0.05)


def _unlock(path: Path) -> None:
    try:
        os.rmdir(path)
    except OSError:
        pass


def _better(score, best, maximize: bool) -> bool:
    if score is None:
        return False
    if best is None:
        return True
    return score > best if maximize else score < best


def _grader_timeout() -> float:
    """The candidate PROGRAM has always had a time budget (budget_s / kill_s in the prompt); the
    grader itself had none, and its cost grows with the sequence length. In the 2026-09 campaign 40
    of 13490 official calls graded for more than 1000 s (up to 6.5 h, sequences of 10^7-10^8
    entries) for no gain in score. Cap grading at kill_s: such a call scores nothing, and the message
    below is what the agent sees."""
    try:
        return float(json.loads(_META.read_text()).get("grader_timeout_s", 1100))
    except Exception:
        return 1100.0
def _run_official(argv: list[str]) -> tuple[int, str, str]:
    t = _grader_timeout()
    try:
        proc = subprocess.run([sys.executable, str(_OFFICIAL), *argv], cwd=str(_HERE),
                              capture_output=True, text=True, timeout=t)
    except subprocess.TimeoutExpired:
        return 124, "", (f"official grader killed after {t:.0f} s: this construction is too long to grade "
                         f"within the candidate time budget and scores nothing. Keep sequences short enough "
                         f"that `eval.py` finishes well inside {t:.0f} s.\n")
    return proc.returncode, proc.stdout, proc.stderr


def _parse_score(stdout: str, pattern: str):
    m = re.search(pattern, stdout, flags=re.MULTILINE)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def main(argv: list[str]) -> int:
    meta = json.loads(_META.read_text())
    if os.environ.get("TRACKER_HOST") or len(argv) != 1:
        rc, out, err = _run_official(argv)
        sys.stdout.write(out); sys.stderr.write(err)
        return rc

    cell = Path(meta["cell_dir"])
    log = cell / "iterations.jsonl"
    snaps = cell / "submissions"
    lock = cell / ".tracker.lock"
    src = Path(argv[0])
    if not src.is_absolute():
        src = Path.cwd() / src
    t_call = time.time()

    # 1. reserve an index and snapshot BEFORE scoring, so a candidate that crashes the grader
    #    still leaves a record.
    _lock(lock)
    try:
        counter = cell / ".tracker.counter"
        n = int(counter.read_text()) + 1 if counter.is_file() else 1
        counter.write_text(str(n))
        snaps.mkdir(exist_ok=True)
        snap = snaps / f"{n:03d}{src.suffix or '.npy'}"
        if src.is_file():
            shutil.copy2(src, snap)
    finally:
        _unlock(lock)

    # 2. score with the untouched official grader, mirroring its output exactly.
    rc, out, err = _run_official([str(src)])
    sys.stdout.write(out); sys.stderr.write(err)
    score = _parse_score(out, meta["score_regex"]) if rc == 0 else None

    # 3. record, and promote the best. One append under the lock; O_APPEND keeps lines intact.
    rec = {"i": n, "t": round(t_call, 3), "elapsed_s": round(t_call - meta["t0"], 1),
           "eval_s": round(time.time() - t_call, 1), "score": score, "snap": snap.name,
           "src": str(src)}
    if score is None:
        rec["error"] = (err.strip().splitlines() or ["non-zero exit"])[-1][:300]
    _lock(lock)
    try:
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        best_json = cell / "best.json"
        best_score = None
        if best_json.is_file():
            try:
                best_score = json.loads(best_json.read_text()).get("score")
            except ValueError:
                best_score = None
        if _better(score, best_score, meta["maximize"]) and snap.is_file():
            shutil.copy2(snap, cell / "best.npy")
            best_json.write_text(json.dumps({**rec, "best": True}, indent=2) + "\n")
        cap = int(meta.get("max_evals") or 0)
        if cap and n >= cap:
            (cell / "STOP").write_text("iteration_limit\n")
    finally:
        _unlock(lock)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

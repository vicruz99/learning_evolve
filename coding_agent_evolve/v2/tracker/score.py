#!/usr/bin/env python3
"""Host-side authoritative score of a finished cell: re-grade the tracker's best snapshot.

    score.py <cell_dir> [--python /path/to/python]

Never scores anything the agent copied by hand -- best.json -> submissions/NNN.npy only
(adrs_acp/run.py::write_score records why). Writes <cell>/score.json.
"""
from __future__ import annotations

import argparse
import json
import time
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker import best_record, read_iterations, stop_reason_file  # noqa: E402



def _write_retry(path, text, tries=60, delay=30.0):
    """2026-09-15: weka ENOSPC bursts; score.json is the run's result, wait up to 30 min for it."""
    for i in range(tries):
        try:
            path.write_text(text)
            return
        except OSError as exc:
            if i == tries - 1:
                raise
            print(f"score.py: writing {path.name} failed ({exc}); retrying in {delay:.0f}s", file=sys.stderr, flush=True)
            time.sleep(delay)

def score_cell(cell_dir: Path, python: str, timeout: int = int(os.environ.get("SCORE_TIMEOUT_S", "10800"))) -> dict:
    cell_dir = cell_dir.resolve()
    ws = cell_dir / "workspace"
    meta = json.loads((ws / "_adrs_track.json").read_text())
    rec = best_record(cell_dir)
    out = {"cell": str(cell_dir), "metric": meta["metric"], "maximize": meta["maximize"],
           "n_evals": len(read_iterations(cell_dir)), "cutoff": stop_reason_file(cell_dir),
           "score": None, "ok": False}
    if rec is None:
        out["error"] = "no official evaluation ever improved on nothing -- best.json missing"
        _write_retry(cell_dir / "score.json", json.dumps(out, indent=2) + "\n")
        return out
    snap = cell_dir / "submissions" / rec["snap"]
    out["snapshot"] = str(snap.relative_to(cell_dir))
    out["agent_reported"] = rec.get("score")
    # Pin the grader's threads: an unpinned numpy on a 256-core node spawned ~46 threads and took
    # minutes for a construction the agent's 2-thread call graded in 9 s (thread thrash).
    threads = str(int(os.environ.get("SCORE_THREADS", "2")))
    env = dict(os.environ, TRACKER_HOST="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads)
    try:
        proc = subprocess.run([python, str(ws / "_official_evaluator.py"), str(snap)], cwd=str(ws),
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        # The tracker already graded this snapshot with the same grader when the agent submitted it
        # (host and tracker scores agreed to 1e-6 in 1114 of 1165 checked cases). Keep that number,
        # but say so: ok stays False and the flag is explicit.
        out["error"] = f"grader timed out after {timeout}s; agent_reported score kept"
        out["score"] = rec.get("score"); out["agent_reported_used"] = True
        _write_retry(cell_dir / "score.json", json.dumps(out, indent=2) + "\n")
        return out
    out["returncode"] = proc.returncode
    out["stdout"] = proc.stdout[-2000:]
    out["stderr"] = proc.stderr[-2000:]
    m = re.search(meta["score_regex"], proc.stdout, flags=re.MULTILINE)
    if proc.returncode == 0 and m:
        try:
            v = float(m.group(1))
            if math.isfinite(v):
                out["score"] = v
                out["ok"] = True
        except ValueError:
            pass
    _write_retry(cell_dir / "score.json", json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cell", type=Path)
    ap.add_argument("--python", default=os.environ.get("AGENT_PYTHON", "/home/crv1pi/venvs/agent-eval/bin/python"))
    a = ap.parse_args()
    r = score_cell(a.cell, a.python)
    print(json.dumps({k: r.get(k) for k in ("score", "ok", "agent_reported", "n_evals", "cutoff", "snapshot", "error")}))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

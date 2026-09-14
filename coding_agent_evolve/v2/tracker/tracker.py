"""Install and read the official-scoring tracker for one cell.

Cell layout (host files one level above the agent's cwd, as in adrs_acp):

    <cell>/                 iterations.jsonl  submissions/NNN.npy  best.npy  best.json  STOP
    <cell>/workspace/       eval.py (wrapper)  _official_evaluator.py  _adrs_track.json  <npy>

Ported from arxiv_graph/baselines/adrs_acp/tracker.py. Differences: the wrapper is a file
(eval_wrapper.py) rather than an inline string; state that changes during the run lives in the
host dir so the three workspace files can be read-only; host-mode scoring is the TRACKER_HOST
env var instead of a flag written into the meta file.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPER_SRC = HERE / "eval_wrapper.py"
PROTECTED = ("eval.py", "_official_evaluator.py", "_adrs_track.json")


def load_meta(path: Path) -> dict:
    """problems/<P>/meta.yaml is flat scalars; parse it without requiring PyYAML."""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(Path(path).read_text())
    except ImportError:
        pass
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        line = line.split(" #", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        elif v.lower() in ("true", "false"):
            v = v.lower() == "true"
        elif v.lstrip("-").isdigit():
            v = int(v)
        out[k.strip()] = v
    return out


def _chmod(path: Path, writable: bool) -> None:
    mode = path.stat().st_mode
    ro = mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    path.chmod(ro | stat.S_IWUSR if writable else ro)


def install_tracker(cell_dir: Path, official_eval: Path, meta: dict, *, max_evals: int = 0,
                    t0: float | None = None) -> Path:
    """Copy the real grader as _official_evaluator.py, install the wrapper as eval.py, write meta."""
    cell_dir = cell_dir.resolve()
    ws = cell_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    for name in PROTECTED:
        p = ws / name
        if p.exists():
            _chmod(p, True)
    shutil.copy2(official_eval, ws / "_official_evaluator.py")
    shutil.copy2(WRAPPER_SRC, ws / "eval.py")
    track = {
        "cell_dir": str(cell_dir),
        "metric": meta["metric"],
        "maximize": bool(meta["maximize"]),
        "score_regex": meta["score_regex"],
        "max_evals": int(max_evals or 0),
        "grader_timeout_s": float(meta.get("kill_s", 1100)),   # eval_wrapper caps the OFFICIAL grade at this
        "t0": t0 if t0 is not None else time.time(),
    }
    (ws / "_adrs_track.json").write_text(json.dumps(track, indent=2) + "\n")
    (cell_dir / "submissions").mkdir(exist_ok=True)
    for name in PROTECTED:
        _chmod(ws / name, False)
    return ws / "_official_evaluator.py"


def set_t0(cell_dir: Path, t0: float) -> None:
    """The driver calls this when the agent session actually starts."""
    p = Path(cell_dir) / "workspace" / "_adrs_track.json"
    meta = json.loads(p.read_text())
    meta["t0"] = t0
    _chmod(p, True)
    p.write_text(json.dumps(meta, indent=2) + "\n")
    _chmod(p, False)


def tracker_installed(cell_dir: Path) -> bool:
    ws = Path(cell_dir) / "workspace"
    return all((ws / n).is_file() for n in PROTECTED)


def read_iterations(cell_dir: Path) -> list[dict]:
    log = Path(cell_dir) / "iterations.jsonl"
    if not log.is_file():
        return []
    rows = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a torn line from a crashed writer is not worth failing the host over
    return rows


def stop_reason_file(cell_dir: Path) -> str | None:
    stop = Path(cell_dir) / "STOP"
    if not stop.is_file():
        return None
    return stop.read_text().strip() or "stopped"


def better(score, best, maximize: bool) -> bool:
    if score is None:
        return False
    if best is None:
        return True
    return score > best if maximize else score < best


def best_record(cell_dir: Path) -> dict | None:
    p = Path(cell_dir) / "best.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def improvement_banked(cell_dir: Path, start_score: float, maximize: bool) -> bool:
    """True once an official evaluation beat the starting construction."""
    for row in read_iterations(cell_dir):
        if better(row.get("score"), start_score, maximize):
            return True
    return False

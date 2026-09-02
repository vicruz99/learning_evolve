"""ShinkaEvolve evaluator for the GPU-mode TriMul kernel task.

Shinka runs this as   python evaluate.py --program_path <candidate.py> --results_dir <dir>
and reads back  <dir>/metrics.json  (combined_score, public, private)  and
                <dir>/correct.json  ({"correct": bool, "error": str|None}).

Unlike the math ports this does NOT use shinka.core.run_shinka_eval: the candidate cannot be
imported into the Shinka venv (no torch there, and a kernel must never be graded by anything but
the pinned torch 2.7.1 / triton 3.3.1 interpreter). Instead the candidate file is handed, as-is,
to the frozen GPU-mode harness -- the same `coding_agent_evolve/gpumode/evaluate.py` the ICL arm
(`src/envs/kernel_trimul.py`) and the coding-agent arm shell out to -- and its JSON is turned into
Shinka's two files. Reward shaping follows TTT-Discover: combined_score = 1500 / runtime_us, 0 on
any failure. The raw geomean runtime is kept in public.score_us.

The two upstream gates from kernel_trimul.py are applied before the GPU is touched: the code must
contain `@triton.jit` and must not contain `identity` (banned anywhere, comments included).

Environment (set by src/jobs/marvin_shinka_run.bsub; defaults work on Marvin):
  KPY                  grading interpreter        ~/venvs/kernel-eval/bin/python
  TRIMUL_EVALUATE_PY   the frozen harness         <repo>/coding_agent_evolve/gpumode/evaluate.py
  TRIMUL_EVAL_MODE     test|benchmark|leaderboard leaderboard (the ranked path)
  TRIMUL_EVAL_TIMEOUT  seconds                    1500 (== ICL eval_timeout)
The grader inherits CUDA_VISIBLE_DEVICES: inside an LSF job that is the one card the job holds.
Never pass --gpu.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# .../learning_evolve/src/shinka_baseline/ttt_discover_kernel/trimul -> repo root is 4 up.
# realpath() first: Shinka runs us through the ShinkaEvolve/examples/... symlink.
REPO = HERE.parents[3]

KPY = os.path.expanduser(os.environ.get("KPY", "~/venvs/kernel-eval/bin/python"))
GRADER = Path(os.path.expanduser(os.environ.get(
    "TRIMUL_EVALUATE_PY", str(REPO / "coding_agent_evolve" / "gpumode" / "evaluate.py"))))
MODE = os.environ.get("TRIMUL_EVAL_MODE", "leaderboard")
TIMEOUT = int(os.environ.get("TRIMUL_EVAL_TIMEOUT", "1500"))
REWARD_SCALE = 1500.0            # TTT-Discover: reward = 1500 / runtime_us

_REQUIRED_TOKEN = "@triton.jit"
_BANNED_TOKEN = "identity"


def _write(results_dir: Path, metrics: dict, correct: bool, error: str | None) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (results_dir / "correct.json").write_text(json.dumps({"correct": correct, "error": error}))


def _fail(results_dir: Path, reason: str, failure_type: str, extra: dict | None = None) -> None:
    metrics = {
        "combined_score": 0.0,
        "public": {"score_us": None, "failure_type": failure_type, **(extra or {})},
        "private": {"reason": reason[:4000]},
    }
    _write(results_dir, metrics, False, reason[:2000])
    print(f"FAILED ({failure_type}): {reason[:500]}")


def _reap(proc: subprocess.Popen) -> None:
    """Kill the grader's whole process group (evaluate.py, eval.py, the CUDA worker)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


def main(program_path: str, results_dir: str) -> None:
    results = Path(results_dir)
    code = Path(program_path).read_text()

    if _REQUIRED_TOKEN not in code:
        return _fail(results, "Code must contain @triton.jit.", "invalid_result")
    if _BANNED_TOKEN in code:
        return _fail(results, "Identity kernel is not allowed.", "invalid_result")
    if not GRADER.exists():
        return _fail(results, f"grader missing: {GRADER}", "harness_error")
    if not Path(KPY).exists():
        return _fail(results, f"grading interpreter missing: {KPY}", "harness_error")

    with tempfile.TemporaryDirectory(prefix="shinka-trimul-") as tmp:
        cand = Path(tmp) / "candidate.py"
        cand.write_text(code)
        out_json = Path(tmp) / "result.json"
        cmd = [KPY, str(GRADER), str(cand), "--task", "trimul", "--mode", MODE, "--json", str(out_json)]
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            _reap(proc)
            return _fail(results, f"grader timed out after {TIMEOUT}s (process group killed)",
                         "eval_timeout", {"eval_seconds": round(time.perf_counter() - t0, 1)})
        eval_seconds = round(time.perf_counter() - t0, 1)

        if not out_json.exists():
            tail = (stderr or "")[-3000:] or (stdout or "")[-3000:]
            return _fail(results, f"grader produced no JSON (rc={proc.returncode}):\n{tail}",
                         "process_crash", {"eval_seconds": eval_seconds})
        runs = json.loads(out_json.read_text())

    run = runs[0]
    failed_phase = run.get("failed_phase")
    score_us = run.get("score_us")
    benchmarks = (run.get("phases", {}).get(MODE, {}) or {}).get("benchmarks", [])
    if failed_phase is not None or score_us is None:
        tail = (stdout or "")[-3000:]
        return _fail(results, f"FAILED in the '{failed_phase}' phase -- no score.\n{tail}",
                     "process_crash", {"eval_seconds": eval_seconds, "failed_phase": failed_phase})

    metrics = {
        "combined_score": REWARD_SCALE / float(score_us),
        "public": {
            "score_us": float(score_us),
            "eval_seconds": eval_seconds,
            "mode": MODE,
            "benchmarks": benchmarks,
        },
        "private": {"stdout_tail": (stdout or "")[-2000:]},
    }
    _write(results, metrics, True, None)
    print(f"SCORE (geom of {len(benchmarks)} benchmarks): {score_us:.1f} us  "
          f"-> combined_score {metrics['combined_score']:.4f}  ({eval_seconds}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TriMul evaluator for ShinkaEvolve (external grader)")
    ap.add_argument("--program_path", type=str, default="initial.py")
    ap.add_argument("--results_dir", type=str, default="results")
    a = ap.parse_args()
    main(a.program_path, a.results_dir)

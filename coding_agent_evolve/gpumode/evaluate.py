#!/usr/bin/env python3
"""Evaluate a GPU-mode kernel submission on a local GPU and report its score.

Self-contained: depends only on this folder plus torch/triton/pyyaml. It does
not import the TTT-Discover repo. It reimplements the parts of
`libkernelbot.run_eval` that actually matter for a local run, so the numbers it
prints match the official harness:

  * the task's `task.yml` supplies the file set, the test/benchmark shapes and
    the timeouts;
  * each shape is serialised to the exact `k: v; k: v` line format `eval.py`
    parses (`build_test_string`);
  * every run gets its own temp dir, since `eval.py` and friends are written
    under fixed names;
  * `eval.py` reports through the `POPCORN_FD` pipe, not stdout;
  * the score is the geometric mean of `benchmark.<i>.mean` over the leaderboard
    run, matching `libkernelbot.submission.compute_score`.

Usage
-----
    python evaluate.py candidate.py                     # score it (leaderboard)
    python evaluate.py candidate.py --mode test         # correctness only
    python evaluate.py candidate.py --gpu 1 --repeats 3 # pin GPU, repeat
    python evaluate.py candidate.py --json out.json

Modes
-----
    test        correctness over every shape in `tests:`. No timings.
    benchmark   time the `benchmarks:` shapes, correctness checked once up front.
    leaderboard the official path: run `tests:` first, and only if they all pass,
                time the `benchmarks:` shapes re-checking correctness every rep.
                This is what produces the ranked score.

Notes
-----
Timing on a shared GPU is meaningless -- give this an idle one. Run-to-run
spread is ~1% on an idle card but the floor for a *believable* difference is
around 2%, so use --repeats before trusting a small win.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("pyyaml is required: pip install pyyaml")

HERE = Path(__file__).resolve().parent

# eval.py's own exit codes; VALIDATE_FAIL means "ran fine, kernel was wrong"
EXIT_SUCCESS = 0
EXIT_VALIDATE_FAIL = 112

COMPILE_TIMEOUT = 120


# ----------------------------------------------------------------------
# task definition
# ----------------------------------------------------------------------
@dataclasses.dataclass
class Task:
    name: str
    root: Path
    files: dict[str, str]  # filename -> contents ("@SUBMISSION@" is the hole)
    main: str
    tests: list[dict]
    benchmarks: list[dict]
    ranking_by: str
    test_timeout: int
    benchmark_timeout: int
    ranked_timeout: int
    seed: int | None


def find_task_dir(task: str) -> Path:
    """Locate a task folder (one holding task.yml) inside this directory."""
    if Path(task).is_dir() and (Path(task) / "task.yml").exists():
        return Path(task).resolve()
    for cand in (HERE / task, HERE / "tasks" / task, HERE / "test" / task):
        if (cand / "task.yml").exists():
            return cand.resolve()
    searched = ", ".join(str(p) for p in (HERE / task, HERE / "tasks" / task, HERE / "test" / task))
    sys.exit(f"no task.yml for task '{task}'; looked in: {searched}")


def load_task(task: str) -> Task:
    root = find_task_dir(task)
    raw = yaml.safe_load((root / "task.yml").read_text())

    files: dict[str, str] = {}
    for spec in raw["files"]:
        src = spec["source"]
        files[spec["name"]] = "@SUBMISSION@" if src == "@SUBMISSION@" else (root / src).read_text()

    if raw.get("lang", "py") != "py":
        sys.exit(f"only python tasks are supported here, got lang={raw.get('lang')}")

    return Task(
        name=root.name,
        root=root,
        files=files,
        main=raw["config"]["main"],
        tests=raw.get("tests", []),
        benchmarks=raw.get("benchmarks", []),
        ranking_by=raw.get("ranking_by", "last"),
        test_timeout=int(raw.get("test_timeout", 180)),
        benchmark_timeout=int(raw.get("benchmark_timeout", 180)),
        ranked_timeout=int(raw.get("ranked_timeout", 180)),
        seed=raw.get("seed"),
    )


def build_case_string(cases: list[dict]) -> str:
    """Serialise shapes the way eval.py's parser expects: 'k: v; k: v' per line.

    Mirrors libkernelbot.run_eval.build_test_string exactly, including how
    non-string values are rendered.
    """
    out = []
    for case in cases:
        out.append("; ".join(f"{k}: {v}" for k, v in case.items()))
    return "\n".join(out) + ("\n" if out else "")


def filter_cases(cases: list[dict], max_bytes: float | None) -> list[dict]:
    """Drop shapes whose input tensor alone exceeds max_bytes (shared-GPU escape)."""
    if max_bytes is None:
        return cases
    return [c for c in cases if c["bs"] * c["seqlen"] ** 2 * c["dim"] * 4 <= max_bytes]


# ----------------------------------------------------------------------
# running
# ----------------------------------------------------------------------
@dataclasses.dataclass
class RunResult:
    success: bool
    passed: bool
    exit_code: int
    duration: float
    result: dict[str, str]
    stdout: str
    stderr: str


def run_program(args: list[str], cwd: Path, timeout: int, seed: int | None) -> RunResult:
    """Run a harness command, reading its verdict off the POPCORN_FD pipe."""
    env = os.environ.copy()
    pipe_read, pipe_write = os.pipe()
    env["POPCORN_FD"] = str(pipe_write)
    if seed is not None:
        env["POPCORN_SEED"] = str(seed)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            args, cwd=cwd, env=env, capture_output=True, text=True,
            check=False, pass_fds=[pipe_write], timeout=timeout,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        os.close(pipe_write)
        os.close(pipe_read)
        return RunResult(False, False, -1, timeout, {},
                         e.stdout or "", f"TIMEOUT after {timeout}s")
    duration = time.perf_counter() - start

    os.close(pipe_write)
    with os.fdopen(pipe_read, "r") as f:
        raw = f.read()

    result: dict[str, str] = {}
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        if key or value:
            result[key.strip()] = value.strip()

    return RunResult(
        success=exit_code in (EXIT_SUCCESS, EXIT_VALIDATE_FAIL),
        passed=result.get("check") == "pass",
        exit_code=exit_code,
        duration=duration,
        result=result,
        stdout=stdout[-4000:],
        stderr=stderr[-4000:],
    )


def run_phase(task: Task, submission: str, mode: str, cases: list[dict],
              python: str, workdir: Path) -> RunResult:
    """Materialise the task into workdir and run one eval.py phase."""
    for name, content in task.files.items():
        (workdir / name).write_text(submission if content == "@SUBMISSION@" else content)

    # The harness executes the submission once first so any load_inline/JIT
    # compilation is cached before timing starts. Failures here are not fatal.
    run_program([python, "submission.py"], workdir, COMPILE_TIMEOUT, seed=1)

    timeout = {
        "test": task.test_timeout,
        "benchmark": task.benchmark_timeout,
        "leaderboard": task.ranked_timeout,
    }[mode]

    case_file = workdir / f"{mode}_cases.txt"
    case_file.write_text(build_case_string(cases))

    return run_program([python, task.main, mode, str(case_file)], workdir, timeout, task.seed)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def collect_benchmarks(res: dict[str, str]) -> list[dict]:
    out = []
    for i in range(int(res.get("benchmark-count", 0))):
        if f"benchmark.{i}.mean" in res:
            out.append({
                "spec": res.get(f"benchmark.{i}.spec", ""),
                "mean_us": float(res[f"benchmark.{i}.mean"]) / 1e3,
                "best_us": float(res.get(f"benchmark.{i}.best", "nan")) / 1e3,
                "err_pct": float(res[f"benchmark.{i}.err"]) / float(res[f"benchmark.{i}.mean"]) * 100
                if res.get(f"benchmark.{i}.mean") not in (None, "0") else float("nan"),
            })
        else:
            out.append({
                "spec": res.get(f"benchmark.{i}.spec", ""),
                "error": res.get(f"benchmark.{i}.error", "unknown failure"),
            })
    return out


def score_of(benchmarks: list[dict], ranking_by: str) -> float | None:
    """Geometric mean of per-shape means, in us. Matches compute_score."""
    means = [b["mean_us"] for b in benchmarks if "mean_us" in b]
    if not means or len(means) != len(benchmarks):
        return None
    if ranking_by == "last":
        return means[-1]
    if ranking_by == "mean":
        return statistics.fmean(means)
    return math.exp(statistics.fmean(math.log(m) for m in means))


def collect_tests(res: dict[str, str]) -> dict:
    """Split tests into passed / failed / never-reported.

    A shape only counts as passed if eval.py explicitly said so. If the harness
    dies partway (a kernel that raises, an OOM, a timeout) it has already logged
    `test-count` and the specs but no status, and treating those as passes would
    turn a crash into a clean 18/18.
    """
    n = int(res.get("test-count", 0))
    failed, incomplete = [], []
    passed = 0
    for i in range(n):
        status = res.get(f"test.{i}.status")
        spec = res.get(f"test.{i}.spec", "")
        if status == "pass":
            passed += 1
        elif status == "fail":
            failed.append({"idx": i, "spec": spec, "error": res.get(f"test.{i}.error", "")})
        else:
            incomplete.append({"idx": i, "spec": spec})
    return {"count": n, "passed": passed, "failed": failed, "incomplete": incomplete}


def report(run: dict, verbose: bool) -> None:
    if "tests" in run:
        t = run["tests"]
        mark = "PASS" if t["count"] and t["passed"] == t["count"] else "FAIL"
        print(f"  correctness: {mark}  {t['passed']}/{t['count']} shapes passed")
        for f in t["failed"][:5]:
            print(f"    wrong  [{f['idx']}] {f['spec']}")
            print(f"           {f['error'][:200]}")
        if t["incomplete"]:
            first = t["incomplete"][0]
            print(f"    never reported: {len(t['incomplete'])} shapes "
                  f"(harness died at [{first['idx']}] {first['spec'][:60]})")

    if run.get("benchmarks"):
        print(f"  {'shape':<62}{'mean':>11}{'best':>11}{'err':>8}")
        for b in run["benchmarks"]:
            if "error" in b:
                print(f"  {b['spec'][:60]:<62}{'FAILED':>11}   {b['error'][:40]}")
            else:
                print(f"  {b['spec'][:60]:<62}{b['mean_us']:10.1f}us{b['best_us']:10.1f}us"
                      f"{b['err_pct']:7.2f}%")

    # on failure the stderr tail is usually the only thing that explains why,
    # so show it without needing -v
    if run.get("stderr") and (verbose or not run.get("passed", True)):
        tail = run["stderr"] if verbose else run["stderr"][-1200:]
        print("  stderr tail:")
        for line in tail.strip().splitlines():
            print(f"    {line}")


# ----------------------------------------------------------------------
def evaluate(task: Task, submission: str, mode: str, python: str,
             max_bytes: float | None) -> dict:
    """Run one full evaluation. Returns a JSON-able summary including 'score_us'."""
    tests = filter_cases(task.tests, max_bytes)
    benchmarks = filter_cases(task.benchmarks, max_bytes)

    out: dict = {"task": task.name, "mode": mode, "phases": {}, "score_us": None}

    with tempfile.TemporaryDirectory(prefix=f"gpumode-{task.name}-") as tmp:
        workdir = Path(tmp)

        # leaderboard is gated on the correctness tier, same as the real harness
        phases = [("test", tests)] if mode == "leaderboard" else []
        phases.append((mode, tests if mode == "test" else benchmarks))

        for phase, cases in phases:
            if not cases:
                sys.exit(f"no cases left for phase '{phase}' after filtering")
            r = run_phase(task, submission, phase, cases, python, workdir)
            entry = {
                "success": r.success,
                "passed": r.passed,
                "exit_code": r.exit_code,
                "duration_s": round(r.duration, 1),
                "stderr": r.stderr,
            }
            if phase == "test":
                entry["tests"] = collect_tests(r.result)
            else:
                entry["benchmarks"] = collect_benchmarks(r.result)
            out["phases"][phase] = entry

            if not r.passed:
                out["failed_phase"] = phase
                return out

        scoring = out["phases"][mode].get("benchmarks")
        if scoring:
            out["score_us"] = score_of(scoring, task.ranking_by)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("submission", type=Path, help="path to the candidate .py")
    ap.add_argument("--task", default="trimul", help="task name or path to a task dir")
    ap.add_argument("--mode", default="leaderboard", choices=["test", "benchmark", "leaderboard"])
    ap.add_argument("--gpu", default=None,
                    help="CUDA_VISIBLE_DEVICES value. OMIT under a scheduler (LSF/Slurm) that has "
                         "already assigned a card via CUDA_VISIBLE_DEVICES: overriding it here can "
                         "re-point the eval at a GPU that belongs to someone else's job.")
    ap.add_argument("--repeats", type=int, default=1, help="repeat the whole eval N times")
    ap.add_argument("--max-input-gib", type=float, default=None,
                    help="skip shapes whose input tensor exceeds this")
    ap.add_argument("--json", type=Path, default=None, help="write full results here")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the harness")
    ap.add_argument("-v", "--verbose", action="store_true", help="show stderr tails")
    args = ap.parse_args()

    if not args.submission.exists():
        sys.exit(f"no such submission: {args.submission}")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Triton's JIT cache must live on local disk -- this folder is often on NFS,
    # where cache writes cost more than the compiles they save. Key it on this
    # folder so parallel runs out of different directories never share a cache.
    cache_key = hashlib.sha1(str(HERE).encode()).hexdigest()[:8]
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / f"gpumode-triton-{os.getuid()}-{cache_key}"),
    )

    task = load_task(args.task)
    submission = args.submission.read_text()
    max_bytes = args.max_input_gib * 1024**3 if args.max_input_gib else None

    print(f"task       : {task.name}  ({task.root})")
    print(f"submission : {args.submission}")
    gpu_shown = args.gpu if args.gpu is not None else (
        f"inherited (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset: all cards')})")
    print(f"mode       : {args.mode}   ranking_by: {task.ranking_by}   GPU: {gpu_shown}")

    runs, scores = [], []
    for i in range(args.repeats):
        if args.repeats > 1:
            print(f"\n--- run {i + 1}/{args.repeats} ---")
        out = evaluate(task, submission, args.mode, args.python, max_bytes)
        runs.append(out)
        for name, phase in out["phases"].items():
            print(f"\n[{name}]  {phase['duration_s']}s")
            report(phase, args.verbose)
        if out["score_us"] is not None:
            scores.append(out["score_us"])

    ok = all(r.get("failed_phase") is None for r in runs)
    print()
    if not ok:
        bad = next(r for r in runs if r.get("failed_phase"))
        print(f"RESULT: FAILED in the '{bad['failed_phase']}' phase -- no score.")
    elif scores:
        print(f"SCORE ({task.ranking_by} of {len(runs[0]['phases'][args.mode]['benchmarks'])} "
              f"benchmarks): {statistics.fmean(scores):.1f} us")
        if len(scores) > 1:
            spread = (max(scores) - min(scores)) / statistics.fmean(scores) * 100
            print(f"  across {len(scores)} runs: "
                  f"min={min(scores):.1f} max={max(scores):.1f} spread={spread:.1f}%")
    else:
        print("RESULT: passed, but this mode produces no score (use --mode leaderboard).")

    if args.json:
        args.json.write_text(json.dumps(runs, indent=2))
        print(f"wrote {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

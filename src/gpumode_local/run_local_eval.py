"""Run a GPU-mode (trimul / mla_decode) submission on a local GPU.

TTT-Discover only ships a ModalLauncher, but the measurement core
(`libkernelbot.run_eval.run_config`) is pure-local: it writes the task sources
into the CWD and shells out to `python3 eval.py <mode> <casefile>`. Modal is
just an RPC wrapper around it. This script calls that core directly.

Nothing in discover/ is modified; we only put its `lib` dir on sys.path.

Isolation notes:
  * `_create_files` writes fixed filenames (submission.py, eval.py, ...) into
    os.getcwd(), so every eval gets its own temp dir and we chdir into it.
  * the eval subprocess inherits our env, so CUDA_VISIBLE_DEVICES pins the GPU
    and PATH is prepended with our venv bin dir so `python3` is the right one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_MODE = REPO_ROOT / "discover" / "examples" / "gpu_mode"
GPU_MODE_LIB = GPU_MODE / "lib"

# libkernelbot's internal imports are absolute (`from libkernelbot.consts ...`),
# so its parent dir has to be on sys.path, not the repo root.
sys.path.insert(0, str(GPU_MODE_LIB))

from libkernelbot.consts import SubmissionMode  # noqa: E402
from libkernelbot.run_eval import run_config  # noqa: E402
from libkernelbot.task import build_task_config, make_task_definition  # noqa: E402

TASK_YAML = {
    "trimul": GPU_MODE / "lib" / "bioml" / "trimul" / "task.yml",
    "mla_decode_nvidia": GPU_MODE / "lib" / "mla-decode" / "task.yml",
}


@contextlib.contextmanager
def working_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def filter_cases(task, max_bytes: float | None):
    """Drop cases whose input tensor alone exceeds max_bytes.

    The largest trimul configs (seqlen=1024, dim=768) need ~3 GiB for the input
    and well over 20 GiB once the reference and allclose temporaries are live.
    On a shared GPU that OOMs, so allow trimming to the cases that fit.
    """
    if max_bytes is None:
        return task.tests, task.benchmarks

    def fits(case):
        nbytes = case["bs"] * case["seqlen"] ** 2 * case["dim"] * 4
        return nbytes <= max_bytes

    return [t for t in task.tests if fits(t)], [b for b in task.benchmarks if fits(b)]


def run_once(
    submission: str,
    task_name: str,
    mode: str,
    max_bytes: float | None,
    backend: str = "local",
    modal_app: str = "discord-bot-runner",
    modal_gpu: str = "a100",
) -> dict:
    definition = make_task_definition(TASK_YAML[task_name])
    task = definition.task
    task.tests, task.benchmarks = filter_cases(task, max_bytes)

    config = build_task_config(
        task=task,
        submission_content=submission,
        arch=None,
        mode=SubmissionMode(mode),
    )

    t0 = time.perf_counter()
    if backend == "local":
        # run_config writes fixed filenames into the CWD, so isolate each eval
        with tempfile.TemporaryDirectory(prefix="gpumode-eval-") as tmp:
            with working_dir(Path(tmp)):
                result = run_config(config)
    else:
        import modal

        fn = modal.Function.from_name(modal_app, f"run_pytorch_script_{modal_gpu}")
        result = fn.remote(config=config)
    wall = time.perf_counter() - t0

    return {"result": result, "wall_seconds": wall, "task": task}


def summarize(result, run_key: str) -> dict:
    """Pull per-benchmark means (ns) and the geometric mean out of a run."""
    if run_key not in result.runs:
        return {}
    res = result.runs[run_key].run.result
    count = int(res.get("benchmark-count", 0))
    means_ns, specs = [], []
    for i in range(count):
        key = f"benchmark.{i}.mean"
        if key in res:
            means_ns.append(float(res[key]))
            specs.append(res.get(f"benchmark.{i}.spec", ""))
    out = {
        "benchmarks": [
            {"spec": s, "mean_us": m / 1e3} for s, m in zip(specs, means_ns, strict=False)
        ]
    }
    if means_ns:
        # matches compute_score's `geom` criterion, reported in microseconds
        out["geomean_us"] = math.pow(math.prod(means_ns), 1.0 / len(means_ns)) / 1e3
    return out


def describe(result, wall: float) -> dict:
    info = {"success": result.success, "wall_seconds": round(wall, 1), "phases": {}}
    if not result.success:
        info["error"] = (result.error or "")[:2000]
    for key, run in result.runs.items():
        r = run.run
        phase = {
            "success": r.success,
            "passed": r.passed,
            "exit_code": int(r.exit_code),
            "duration_s": round(r.duration, 1),
        }
        if key == "test":
            n = int(r.result.get("test-count", 0))
            phase["tests"] = {
                "count": n,
                "failed": [
                    {
                        "idx": i,
                        "spec": r.result.get(f"test.{i}.spec", ""),
                        "error": (r.result.get(f"test.{i}.error", "") or "")[:300],
                    }
                    for i in range(n)
                    if r.result.get(f"test.{i}.status") == "fail"
                ],
            }
        else:
            phase.update(summarize(result, key))
        if not r.passed and r.stderr:
            phase["stderr_tail"] = r.stderr[-1500:]
        info["phases"][key] = phase
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", type=Path, help="path to a submission .py")
    ap.add_argument("--task", default="trimul", choices=sorted(TASK_YAML))
    ap.add_argument(
        "--mode",
        default="benchmark",
        choices=["test", "benchmark", "leaderboard"],
        help="test=correctness only; benchmark=time without per-rep recheck; "
        "leaderboard=official (recheck every rep, ~100x slower)",
    )
    ap.add_argument(
        "--backend",
        default="local",
        choices=["local", "modal"],
        help="local GPU, or the deployed Modal runner (identical config either way)",
    )
    ap.add_argument("--modal-app", default="discord-bot-runner")
    ap.add_argument("--modal-gpu", default="a100", help="suffix of the deployed function")
    ap.add_argument("--gpu", default="0", help="value for CUDA_VISIBLE_DEVICES (local only)")
    ap.add_argument("--repeats", type=int, default=1, help="repeat the whole eval N times")
    ap.add_argument(
        "--max-input-gib",
        type=float,
        default=None,
        help="skip cases whose input tensor exceeds this (use on a shared GPU)",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # keep triton's JIT cache off NFS
    os.environ.setdefault("TRITON_CACHE_DIR", "/scratch/vicstorage/.triton-cache")
    # run_pytorch_script shells out to `python3`; make sure it's this interpreter's
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"

    submission = args.submission.read_text()
    max_bytes = args.max_input_gib * 1024**3 if args.max_input_gib else None

    runs = []
    for i in range(args.repeats):
        out = run_once(
            submission,
            args.task,
            args.mode,
            max_bytes,
            backend=args.backend,
            modal_app=args.modal_app,
            modal_gpu=args.modal_gpu,
        )
        info = describe(out["result"], out["wall_seconds"])
        runs.append(info)
        print(f"\n=== run {i + 1}/{args.repeats} ===")
        print(json.dumps(info, indent=2))

    if args.repeats > 1:
        geos = [
            r["phases"][k]["geomean_us"]
            for r in runs
            for k in ("leaderboard", "benchmark")
            if k in r["phases"] and "geomean_us" in r["phases"][k]
        ]
        if len(geos) > 1:
            spread = (max(geos) - min(geos)) / statistics.fmean(geos) * 100
            print(
                f"\ngeomean across {len(geos)} runs: "
                f"mean={statistics.fmean(geos):.1f}us "
                f"stdev={statistics.stdev(geos):.1f}us spread={spread:.1f}%"
            )

    if args.json_out:
        args.json_out.write_text(json.dumps(runs, indent=2))
        print(f"\nwrote {args.json_out}")

    return 0 if all(r["success"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

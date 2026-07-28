"""Re-grade a finished run's candidates through the real evaluator, on THIS machine.

Why: `eval_seconds` is only comparable across machines as a PAIRED measurement — the same saved
program, the same evaluator, the same cap. Per-core benchmarks (ray_doctor section 6) are too noisy
to explain a slow box, and aggregate percentiles from two campaigns are confounded by which
candidates each model happened to write.

Run the same command on both boxes and diff the output:

    python -m sandbox.regrade runs/bon_qwen/cp26_s1 --n 10
    python -m sandbox.regrade runs/bon_qwen/cp26_s1 --n 10 --out inesc.json

Also answers "is this program reproducible?": --repeat N grades each candidate N times. A spread of
exactly 0 means the program is deterministic on this machine, so a DIFFERENT result on another
machine is a library/platform difference, not randomness.

Reading the ratio column:
  * a tight ratio across all candidates      -> per-core speed difference (hardware)
  * a wide spread (e.g. 0.03x to 1.5x)       -> program-specific, i.e. the software path. Compare
    scipy versions first: a solver that runs to maxiter instead of hitting its convergence test
    costs 10-40x at the SAME final answer, and these candidates are almost all scipy.optimize.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time


class _Dummy:  # Environment needs a sampler; grading never uses it
    def update_states(self, *a, **k): pass
    def record_failed_rollout(self, *a, **k): pass


def _sample(run_dir: str, n: int, kind: str) -> list[dict]:
    """Stratified across the recorded eval_seconds range, so the pairing spans cheap and expensive."""
    events = [json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl"))]
    pool = [e for e in events if e.get("eval_seconds") and e.get("completion_file")]
    if kind == "success":
        pool = [e for e in pool if (e.get("failure_type") or "") == ""]
    elif kind == "timeout":
        pool = [e for e in pool if e.get("failure_type") == "eval_timeout"]
    pool.sort(key=lambda e: e["eval_seconds"])
    if not pool or n >= len(pool):
        return pool
    step = (len(pool) - 1) / (n - 1) if n > 1 else 0
    return [pool[round(i * step)] for i in range(n)]


async def _grade(sem, spec, initial, run_dir, code, eval_timeout):
    async with sem:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            from envs import EnvConfig
            cfg = EnvConfig(problem_type=spec.problem_type, log_path=td,
                            num_cpus_per_task=spec.num_cpus_per_task,
                            eval_timeout=eval_timeout, timeout=eval_timeout + 120)
            env = spec.env_type(initial_state=initial, sampler=_Dummy(), config=cfg)
            out = await env.check_answer(code, step=0)
        m = out.metrics or {}
        return {"eval": m.get("eval_seconds"), "wall": round(time.time() - t0, 2),
                "score": out.raw_score if out.correctness == 1.0 else None,
                "outcome": out.failure_type or "SUCCESS"}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a finished run directory (containing events.jsonl)")
    ap.add_argument("--n", type=int, default=10, help="candidates to re-grade (default 10)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="grade each candidate this many times to test determinism (default 1)")
    ap.add_argument("--kind", choices=["all", "success", "timeout"], default="success",
                    help="which candidates to sample (default: success — they have a score to compare)")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="parallel grades; keep low so this measures the program, not contention")
    ap.add_argument("--eval-timeout", type=int, default=None,
                    help="override the cap; default is the problem's registry value")
    ap.add_argument("--out", default=None, help="also write the rows as JSON here")
    args = ap.parse_args()

    from envs import get_problem
    from sandbox import init_ray

    cfg_path = os.path.join(args.run_dir, "config.json")
    problem = json.load(open(cfg_path))["problem"]
    spec = get_problem(problem)
    cap = args.eval_timeout or spec.eval_timeout
    init_ray(spec.num_cpus_per_task)
    initial = spec.env_type.create_initial_state(spec.problem_type)

    sample = _sample(args.run_dir, args.n, args.kind)
    if not sample:
        print(f"no candidates matching --kind {args.kind} in {args.run_dir}")
        return 1

    print(f"host {os.uname().nodename}   problem {problem}   cap {cap}s   "
          f"{len(sample)} candidates x {args.repeat}")
    try:
        print(f"load average {os.getloadavg()[0]:.1f} on "
              f"{len(os.sched_getaffinity(0))} cores")
    except (OSError, AttributeError):
        pass
    print(f"\n{'recorded':>9} {'now':>9} {'ratio':>7} {'spread':>8}  outcome        file")

    sem = asyncio.Semaphore(args.concurrency)
    rows, ratios = [], []
    for e in sample:
        code = open(os.path.join(args.run_dir, e["completion_file"])).read()
        reps = await asyncio.gather(*[_grade(sem, spec, initial, args.run_dir, code, cap)
                                     for _ in range(args.repeat)])
        evals = [r["eval"] for r in reps if r["eval"] is not None]
        scores = [r["score"] for r in reps if r["score"] is not None]
        now = statistics.median(evals) if evals else None
        ratio = (now / e["eval_seconds"]) if now and e["eval_seconds"] else None
        spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
        if ratio:
            ratios.append(ratio)
        row = {"file": e["completion_file"], "recorded_eval": e["eval_seconds"],
               "recorded_score": e.get("raw_score"), "now_eval": now,
               "now_scores": scores, "outcomes": [r["outcome"] for r in reps]}
        rows.append(row)
        print(f"{e['eval_seconds']:>8.1f}s {(now if now else -1):>8.1f}s "
              f"{(f'{ratio:.3f}' if ratio else '-'):>7} "
              f"{(f'{spread:.1e}' if args.repeat > 1 else '-'):>8}  "
              f"{reps[0]['outcome']:<14} {e['completion_file']}")
        sys.stdout.flush()

    if ratios:
        ratios.sort()
        lo, hi = ratios[0], ratios[-1]
        print(f"\nratio now/recorded: p50 {statistics.median(ratios):.3f}  min {lo:.3f}  max {hi:.3f}")
        if hi / max(lo, 1e-9) > 5:
            print(f"  spread is {hi / lo:.0f}x across candidates -> program-specific, so NOT a single "
                  "hardware factor. Beware: two outliers in a small sample can manufacture this. "
                  "Re-run with a larger --n before believing it.")
        else:
            print("  ratio is roughly uniform -> ONE systematic cause, but this does not say which.")
        # Uniformity alone does NOT mean hardware. Cross-check against ray_doctor section 6: if this
        # box benchmarks FASTER per core yet grades candidates slower, the factor is in the software
        # path, and the first thing to compare is scipy -- an SLSQP that runs to maxiter instead of
        # meeting its convergence test costs 10-40x at the same final answer.
        print("  Cross-check `python -m sandbox.ray_doctor` section 6 before concluding hardware:")
        print("      a box that is FASTER per core but SLOWER per candidate is a software difference.")
        print("      python -c 'import scipy, numpy; print(scipy.__version__, numpy.__version__)'")
    changed = [r for r in rows if r["recorded_score"] is not None and r["now_scores"]
               and abs(r["now_scores"][0] - r["recorded_score"]) > 1e-9]
    if changed:
        print(f"\n{len(changed)}/{len(rows)} candidates produced a DIFFERENT score here than recorded "
              "— the programs are not portable across these two environments.")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

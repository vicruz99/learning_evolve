"""Retroactively reclassify — and optionally re-run — the FAILED candidates of a past ICL run,
to tell whether each failure was a **genuine** bad/slow solution or an **infrastructure** artefact
(CPU starvation / contention-induced timeout), WITHOUT repeating the whole experiment.

Two passes:

1. **Static reclassification** (no execution, works on old runs even before the ``failure_type``
   instrumentation existed): read ``events.jsonl``, and for every failed candidate derive a
   ``failure_type`` — from the stored field if present, else from the failure ``msg`` /traceback via
   :func:`sandbox.classify_failure`. Prints a count table.

2. **One-by-one re-run** (needs Ray + the model completions on disk): for the selected failure types
   (default ``eval_timeout,cpu_starvation``), re-extract each candidate's code from its saved
   completion and grade it **sequentially** with a generous timeout. Running one at a time means
   there is no CPU contention, so:

     - originally ``cpu_starvation``/``eval_timeout`` but now **valid & fast**  -> it was INFRA/contention
     - still ``eval_timeout`` at the generous limit                            -> GENUINELY slow
     - ``process_crash`` / ``invalid_result``                                  -> GENUINE bad solution

   Writes ``<run_dir>/recheck_report.json`` and prints a per-candidate table.

Usage (from ``src/``):

    python -m results.recheck_failures runs/<run> \
        [--rerun-types eval_timeout,cpu_starvation] [--rerun-timeout 300] \
        [--max-rerun 50] [--no-rerun] [--problem circle_packing_26]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter

from envs import EnvConfig, get_problem
from envs.codeblock import last_codeblock_postprocess
from sandbox.sandbox_reward_evaluator import classify_failure


class _DummySampler:
    """``Environment.__init__`` requires a non-None sampler; grading never touches it."""
    def update_states(self, *a, **k):
        pass

    def record_failed_rollout(self, *a, **k):
        pass


def static_classify(rec: dict) -> str:
    """Best-effort failure_type for one events.jsonl record (failed candidate)."""
    ft = rec.get("failure_type")
    if ft:
        return ft
    msg = rec.get("msg", "") or ""
    if "Cannot extract python code" in msg or "Invalid code" in msg:
        return "no_code"
    if "Packing is not valid" in msg or "Invalid solution" in msg:
        return "invalid_result"
    if "Timeout grading" in msg:
        return "grade_timeout"
    return classify_failure(msg)


def load_failed(run_dir: str) -> list[dict]:
    """Read events.jsonl and return the failed (correctness==0) candidate records, annotated with
    a ``_type`` (static classification) and ``_loc`` (gen/parent/child)."""
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No events.jsonl in {run_dir}")
    failed = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if float(rec.get("correctness", 0.0)) > 0:
                continue
            rec["_type"] = static_classify(rec)
            rec["_loc"] = f"gen{rec.get('generation')}/p{rec.get('parent_slot')}/c{rec.get('child')}"
            failed.append(rec)
    return failed


def _verdict(original_type: str, outcome: str) -> str:
    if outcome == "valid":
        if original_type in ("cpu_starvation", "eval_timeout"):
            return "INFRA / contention (ran fine & fast when uncontended)"
        return f"now valid uncontended (was {original_type})"
    if outcome == "eval_timeout":
        return "GENUINE: still times out at the generous limit -> truly slow"
    if outcome == "process_crash":
        return "GENUINE: code crashes"
    if outcome == "invalid_result":
        return "GENUINE: runs but produces an invalid result"
    if outcome == "cpu_starvation":
        return "still cpu_starvation (unexpected when running sequentially)"
    if outcome == "no_code":
        return "no extractable code in completion"
    return outcome


async def rerun_one(env_factory, run_dir: str, rec: dict) -> dict:
    """Re-extract and re-grade a single failed candidate; return an outcome dict."""
    loc = rec["_loc"]
    comp_rel = rec.get("completion_file")
    if not comp_rel:
        return {"loc": loc, "original_type": rec["_type"], "outcome": "no_completion",
                "wall_s": None, "score": None, "verdict": "no completion saved -> cannot re-run"}
    comp_path = os.path.join(run_dir, comp_rel)
    if not os.path.exists(comp_path):
        return {"loc": loc, "original_type": rec["_type"], "outcome": "no_completion",
                "wall_s": None, "score": None, "verdict": f"missing {comp_rel}"}

    with open(comp_path) as f:
        completion = f.read()
    # Reproduce the harness's runtime extraction (envs/base.rollout_step order).
    parsed = last_codeblock_postprocess(completion, codeblock_seps=["python"], keep_separators=True)
    if not parsed:
        return {"loc": loc, "original_type": rec["_type"], "outcome": "no_code",
                "wall_s": None, "score": None, "verdict": _verdict(rec["_type"], "no_code")}

    env = env_factory()
    t0 = time.perf_counter()
    outs = await env.check_answer(parsed, step=0)
    wall = time.perf_counter() - t0
    if outs.correctness > 0:
        outcome, score = "valid", outs.raw_score
    else:
        outcome, score = (outs.failure_type or "unknown"), None
    return {"loc": loc, "original_type": rec["_type"], "outcome": outcome,
            "wall_s": round(wall, 2), "score": score, "verdict": _verdict(rec["_type"], outcome)}


async def _run(args) -> None:
    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config.json")
    problem = args.problem
    if problem is None:
        if not os.path.exists(cfg_path):
            raise SystemExit(f"--problem not given and no config.json in {run_dir}")
        problem = json.load(open(cfg_path)).get("problem")
    spec = get_problem(problem)

    failed = load_failed(run_dir)
    counts = Counter(r["_type"] for r in failed)
    total_events = sum(1 for _ in open(os.path.join(run_dir, "events.jsonl")))

    print(f"\nrun            : {run_dir}")
    print(f"problem        : {problem}")
    print(f"events (total) : {total_events}   failed: {len(failed)}")
    print("\n=== static failure classification (no re-run) ===")
    infra = {"cpu_starvation", "results_missing"}
    for t, n in counts.most_common():
        tag = "  <-- INFRA" if t in infra else ("  <-- infra?/slow" if t == "eval_timeout" else "")
        print(f"  {n:5d}  {t}{tag}")

    report = {
        "run_dir": run_dir, "problem": problem,
        "total_events": total_events, "total_failed": len(failed),
        "static_counts": dict(counts), "rerun": [],
    }

    rerun_types = set() if args.no_rerun else {t.strip() for t in args.rerun_types.split(",") if t.strip()}
    to_rerun = [r for r in failed if r["_type"] in rerun_types]
    if args.max_rerun and len(to_rerun) > args.max_rerun:
        print(f"\n(capping re-run at --max-rerun={args.max_rerun} of {len(to_rerun)} eligible)")
        to_rerun = to_rerun[:args.max_rerun]

    if to_rerun:
        from sandbox import init_ray
        init_ray(spec.num_cpus_per_task)

        def env_factory():
            initial = spec.env_type.create_initial_state(spec.problem_type)
            cfg = EnvConfig(
                problem_type=spec.problem_type, log_path=run_dir,
                num_cpus_per_task=spec.num_cpus_per_task,
                eval_timeout=args.rerun_timeout, timeout=args.rerun_timeout + 60,
            )
            return spec.env_type(initial_state=initial, sampler=_DummySampler(), config=cfg)

        print(f"\n=== re-running {len(to_rerun)} candidate(s) sequentially "
              f"(eval_timeout={args.rerun_timeout}s, contention-free) ===")
        print(f"  {'loc':22s} {'was':16s} {'now':16s} {'wall':>8s}  verdict")
        for rec in to_rerun:
            out = await rerun_one(env_factory, run_dir, rec)
            report["rerun"].append(out)
            wall = "-" if out["wall_s"] is None else f"{out['wall_s']:.1f}s"
            print(f"  {out['loc']:22s} {out['original_type']:16s} {out['outcome']:16s} "
                  f"{wall:>8s}  {out['verdict']}")

        vc = Counter(o["verdict"] for o in report["rerun"])
        print("\n=== re-run verdict summary ===")
        for v, n in vc.most_common():
            print(f"  {n:5d}  {v}")
    elif not args.no_rerun:
        print(f"\n(no failed candidates matched --rerun-types={sorted(rerun_types)}; nothing to re-run)")

    out_path = os.path.join(run_dir, "recheck_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="Path to a run directory (contains events.jsonl + config.json).")
    p.add_argument("--problem", default=None, help="Registry key; default: read from config.json.")
    p.add_argument("--rerun-types", default="eval_timeout,cpu_starvation",
                   help="Comma-separated failure_types to re-run (default: eval_timeout,cpu_starvation). "
                        "Add process_crash/invalid_result to reconfirm genuine failures.")
    p.add_argument("--rerun-timeout", type=int, default=300,
                   help="Generous per-candidate eval timeout for the re-run (seconds).")
    p.add_argument("--max-rerun", type=int, default=0, help="Cap number of re-runs (0 = no cap).")
    p.add_argument("--no-rerun", action="store_true", help="Static reclassification only; no execution.")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

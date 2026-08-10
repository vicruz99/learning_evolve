#!/usr/bin/env python
"""CLI entrypoint for an ICL discovery run.

Example:
    python run_icl.py --problem ac2 --n-context 32 --num-generations 20 \
        --vllm-base-url http://localhost:8000/v1 --model openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime

from envs.registry import REGISTRY
from context import STRATEGIES
from icl.config import ICLConfig
from icl.loop import run


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface of a single ICL run.

    Exposed separately from ``parse_args`` so ``run_sweep.py`` can validate a sweep file's keys
    against the real flags (and their bool/negative-flag pairings) instead of duplicating the list.
    """
    p = argparse.ArgumentParser(description="ICL discovery run (PUCT buffer + in-context past solutions).")
    p.add_argument("--problem", required=True, choices=sorted(REGISTRY), help="Problem to run.")
    p.add_argument("--log-path", default=None,
                   help="Output dir (default: ./runs/<problem>_<strategy>_n<ctx>_g<gs>x<gpb>_<timestamp>).")

    p.add_argument("--model", dest="model_name", default="openai/gpt-oss-120b")
    p.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    p.add_argument("--reasoning-effort", default="medium", help="gpt-oss; 'none' to disable (e.g. non-gpt-oss models).")
    p.add_argument("--thinking-token-budget", type=int, default=None,
                   help="Qwen3: cap reasoning tokens; vLLM forces </think> once hit. "
                        "Needs the server launched with --reasoning-parser qwen3.")
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false", default=None,
                   help="Qwen3: disable thinking entirely (chat_template_kwargs enable_thinking=false).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=26000)
    p.add_argument("--max-gen-concurrency", type=int, default=8)
    p.add_argument("--grade-chunk-size", type=int, default=None,
                   help="Completions per vLLM request within a parent's group; each chunk is graded "
                        "as soon as it returns (overlaps grading with generation). Default: whole "
                        "group in one request (grade only after all children arrive). Raise "
                        "--max-gen-concurrency to groups_per_batch*ceil(group_size/chunk) when set.")

    p.add_argument("--llm-max-wait", type=float, default=3600.0, metavar="SECONDS",
                   help="How long to keep retrying an unreachable model server before stopping the "
                        "run at its last COMPLETE generation (0 = wait indefinitely). The run never "
                        "walks past a generation whose groups did not reach the model: those parents "
                        "produce no children, so every later prompt is conditioned on a different "
                        "buffer than the configuration being compared.")

    p.add_argument("--max-empty-generations", type=int, default=3, metavar="N",
                   help="Stop the run after N consecutive generations in which NO candidate graded "
                        "valid (0 disables). Such generations are structurally perfect and entirely "
                        "worthless, so a run full of them verifies as complete and --resume skips "
                        "it; the cause is almost always the evaluator (a starved cpu_scheduler, a "
                        "full disk, a Ray head that lost its workers), not the problem.")

    p.add_argument("--groups-per-batch", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--num-generations", type=int, default=50)

    p.add_argument("--puct-c", type=float, default=1.0)
    p.add_argument("--max-buffer-size", type=int, default=1000)
    p.add_argument("--topk-children", type=int, default=2)
    p.add_argument("--parent-source", default="puct", choices=["puct", "initial"],
                   help="Where each generation's parents come from. 'puct': select from the buffer "
                        "(TTT-Discover's search). 'initial': always the problem's seed solution => "
                        "Best-of-N; with --n-context 0 that is the no-past-experience baseline.")
    p.add_argument("--seed", type=int, default=None,
                   help="Replicate seed. Seeds the sampler's random initial construction (ac1/ac2) and "
                        "the 'random' context strategy unless --context-seed is given. Recorded in "
                        "config.json. Does NOT make a run bit-reproducible — vLLM sampling is unseeded.")

    p.add_argument("--context-strategy", default="best", choices=sorted(STRATEGIES),
                   help="Which past-solution selector to inject into the prompt (see docs/strategies/).")
    p.add_argument("--n-context", type=int, default=32,
                   help="Number of past solutions to include in context (the main hyperparameter).")
    p.add_argument("--max-context-tokens", type=int, default=None)
    # strategy knobs (used only by the strategies that read them)
    p.add_argument("--mix-fraction", type=float, default=0.5,
                   help="x: fraction of n-context from the 'best' pool (best_worst/best_jump/per_lineage/contrastive).")
    p.add_argument("--mmr-lambda", type=float, default=0.7,
                   help="MMR quality<->diversity (best_diverse/informative/contrastive); 1=quality only, 0=spread only.")
    p.add_argument("--jump-alpha", type=float, default=0.5,
                   help="informative: value(alpha) vs improvement-over-parent(1-alpha) blend.")
    p.add_argument("--context-seed", type=int, default=None, help="Seed for the 'random' strategy.")
    p.add_argument("--exclude-parent", dest="exclude_parent_from_context",
                   action="store_true", default=True,
                   help="Drop each parent from its own context block (default on): it is already "
                        "shown as the current solution in the prompt tail.")
    p.add_argument("--no-exclude-parent", dest="exclude_parent_from_context", action="store_false",
                   help="Let a parent also appear among its own past solutions. Combined with "
                        "--context-seed this makes the context block IDENTICAL for every parent in a "
                        "generation, so vLLM prefills it once for the whole generation instead of once "
                        "per parent — at the cost of prompt diversity between parents.")
    # rendering (orthogonal to selection)
    p.add_argument("--include-code", dest="include_code", action="store_true", default=True,
                   help="Show each context solution's code (default on).")
    p.add_argument("--no-include-code", dest="include_code", action="store_false",
                   help="Hide code (use with --include-strategy for a strategy-only context).")
    p.add_argument("--include-strategy", dest="include_strategy", action="store_true", default=False,
                   help="Show each context solution's <strategy> reasoning block (default off).")
    p.add_argument("--save-completions", dest="save_completions", action="store_true", default=True,
                   help="Save full raw completions per candidate (default on).")
    p.add_argument("--no-save-completions", dest="save_completions", action="store_false",
                   help="Do not save raw completions (smaller runs).")
    p.add_argument("--save-reasoning", dest="save_reasoning", action="store_true", default=True,
                   help="Save each candidate's reasoning_content to child_NN.reasoning.txt (default on).")
    p.add_argument("--no-save-reasoning", dest="save_reasoning", action="store_false",
                   help="Do not save reasoning traces (smaller runs; token counts are still recorded).")

    p.add_argument("--eval-timeout", type=int, default=None)
    p.add_argument("--num-cpus-per-task", type=int, default=None)
    p.add_argument("--ray-num-cpus", type=int, default=None, metavar="N",
                   help="Cores for the Ray cluster THIS run starts when no head is up (default: the "
                        "whole box). Ignored when attaching to an existing head, which cannot be "
                        "resized — under run_sweep.py use its --ray-num-cpus instead.")
    p.add_argument("--grade-timeout", type=float, default=8000.0)

    # --- trimul_* only: the machine-specific half of kernel grading -------------------------------
    # These are properties of the BOX, not of the experiment, but they belong in the sweep file all
    # the same: they are validated there like every other key, they show up in --print-cmds, and
    # to_dict() records them in the run's config.json -- so a finished run says which interpreter and
    # which card produced its timings. The TRIMUL_* environment variables still work and are the
    # fallback when a flag is not given; a flag always wins.
    p.add_argument("--trimul-eval-python", default=None, metavar="PATH",
                   help="Interpreter that runs the kernel harness (needs torch 2.7.1 / triton 3.3.1, "
                        "which the ICL venv does NOT have). Keep it a separate venv: the pin is a "
                        "measurement dependency. Env fallback: TRIMUL_EVAL_PYTHON.")
    p.add_argument("--trimul-eval-gpu", default=None, metavar="IDX",
                   help="CUDA_VISIBLE_DEVICES value for the eval. Must be a card nothing else is "
                        "using -- a contended GPU makes every timing meaningless. Env fallback: "
                        "TRIMUL_EVAL_GPU.")
    p.add_argument("--trimul-evaluate-py", default=None, metavar="PATH",
                   help="Path to coding_agent_evolve/gpumode/evaluate.py. Env: TRIMUL_EVALUATE_PY.")
    p.add_argument("--trimul-eval-mode", default=None, choices=["test", "benchmark", "leaderboard"],
                   help="Grading mode. `benchmark` (default) scores; `test` is correctness-only so "
                        "reward becomes binary; `leaderboard` is ~100x slower. Env: TRIMUL_EVAL_MODE.")
    p.add_argument("--memory-stop-fraction", type=float, default=0.85, metavar="F",
                   help="Stop at the next generation boundary once the job's process-tree RSS "
                        "reaches this fraction of its detected memory ceiling (LSF MEMLIMIT or "
                        "cgroup). Landing the stop on a boundary keeps the run resumable, which a "
                        "TERM_MEMLIMIT kill mid-generation does not. 0 disables the stop; the "
                        "per-generation memory line is logged either way.")

    p.add_argument("--resume-step", default=None, metavar="N|auto",
                   help="Continue an interrupted run in --log-path: N restarts from generation N, "
                        "'auto' uses the last generation that run's own files can back. Anything "
                        "after the resume point is moved to <log-path>/stale_<timestamp>/ so the "
                        "relaunch does not write on top of the attempt it replaces.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                   help="Console log level; icl.log always captures DEBUG.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build & print one assembled prompt (base + context block), then exit. No server/ray.")

    return p


def _resolve_resume_step(raw: str | None, log_path: str) -> int | None:
    """Turn ``--resume-step`` into a generation index the run dir can actually back.

    ``auto`` asks ``results.resume`` where the run's artifacts stop being trustworthy. An explicit N is
    honoured, but checked and reported first: the ways it used to go wrong were silent or cryptic — a
    missing buffer snapshot surfaced as a FileNotFoundError from inside the sampler, and a context pool
    that did not reach N left every later prompt with an empty context block. Either way the tail after
    the resume point is set aside, because a resumed run appends to events.jsonl / progress.csv /
    solutions/ and would otherwise interleave with the attempt it replaces.
    """
    if raw is None:
        return None
    from results.resume import inspect_run, rewind, tail_exists

    prog = inspect_run(log_path)
    for line in prog.damage:
        print(f"[resume] {line}")
    if str(raw).strip().lower() in ("auto", "last"):
        step = prog.resume_step
        print(f"[resume] {log_path}: {prog.describe()}")
    else:
        try:
            step = int(raw)
        except ValueError:
            raise SystemExit(f"--resume-step: expected a generation number or 'auto', got {raw!r}")
        if step < 0:
            raise SystemExit("--resume-step: must be >= 0")
        if step and step not in prog.snapshots:
            have = ", ".join(str(s) for s in prog.snapshots) or "none"
            raise SystemExit(
                f"--resume-step {step}: {log_path} has no loadable PUCT snapshot for that step "
                f"(have: {have}). Use --resume-step auto to continue from {prog.resume_step}.")
        if step > prog.good_generations:
            print(f"[resume] WARNING: only {prog.good_generations} generation(s) of {log_path} are "
                  f"verifiable, but you asked to continue from {step} — generations "
                  f"{prog.good_generations}..{step - 1} are damaged or missing.")
    if tail_exists(log_path, step):
        for line in rewind(log_path, step):
            print(f"[resume] moved {line}")
    return step or None


def parse_args() -> ICLConfig:
    a = build_parser().parse_args()
    if a.log_path:
        log_path = a.log_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # A Best-of-N run has no context strategy in play; label it as such rather than inheriting the
        # (unused) --context-strategy default, which would read as an ICL run in the runs/ listing.
        tag = "bon" if a.parent_source == "initial" else a.context_strategy
        seed = f"_s{a.seed}" if a.seed is not None else ""
        auto = (f"{a.problem}_{tag}_n{a.n_context}_g{a.group_size}x{a.groups_per_batch}{seed}_{ts}")
        log_path = os.path.join("runs", auto)
    reasoning_effort = None if a.reasoning_effort.lower() == "none" else a.reasoning_effort

    return ICLConfig(
        problem=a.problem,
        log_path=log_path,
        model_name=a.model_name,
        vllm_base_url=a.vllm_base_url,
        reasoning_effort=reasoning_effort,
        thinking_token_budget=a.thinking_token_budget,
        enable_thinking=a.enable_thinking,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        max_gen_concurrency=a.max_gen_concurrency,
        grade_chunk_size=a.grade_chunk_size,
        llm_max_wait=a.llm_max_wait,
        max_empty_generations=a.max_empty_generations,
        groups_per_batch=a.groups_per_batch,
        group_size=a.group_size,
        num_generations=a.num_generations,
        puct_c=a.puct_c,
        max_buffer_size=a.max_buffer_size,
        topk_children=a.topk_children,
        parent_source=a.parent_source,
        seed=a.seed,
        context_strategy=a.context_strategy,
        n_context=a.n_context,
        max_context_tokens=a.max_context_tokens,
        mix_fraction=a.mix_fraction,
        mmr_lambda=a.mmr_lambda,
        jump_alpha=a.jump_alpha,
        context_seed=a.context_seed,
        exclude_parent_from_context=a.exclude_parent_from_context,
        include_code=a.include_code,
        include_strategy=a.include_strategy,
        save_completions=a.save_completions,
        save_reasoning=a.save_reasoning,
        log_level=a.log_level,
        eval_timeout=a.eval_timeout,
        num_cpus_per_task=a.num_cpus_per_task,
        ray_num_cpus=a.ray_num_cpus,
        grade_timeout=a.grade_timeout,
        memory_stop_fraction=a.memory_stop_fraction,
        # Only the flags actually given are forwarded, so an unset flag leaves the evaluator's own
        # env-var fallback in charge rather than overriding it with None.
        evaluator_options={
            k: v for k, v in (
                ("eval_python", a.trimul_eval_python),
                ("eval_gpu", a.trimul_eval_gpu),
                ("evaluate_py", a.trimul_evaluate_py),
                ("eval_mode", a.trimul_eval_mode),
            ) if v is not None
        },
        # --dry-run only prints a prompt: never let it rewind a run dir.
        resume_step=None if a.dry_run else _resolve_resume_step(a.resume_step, log_path),
        dry_run=a.dry_run,
    )


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.dry_run:
        from icl.loop import dry_run
        dry_run(cfg)
    else:
        asyncio.run(run(cfg))

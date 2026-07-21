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


def parse_args() -> ICLConfig:
    p = argparse.ArgumentParser(description="ICL discovery run (PUCT buffer + in-context past solutions).")
    p.add_argument("--problem", required=True, choices=sorted(REGISTRY), help="Problem to run.")
    p.add_argument("--log-path", default=None,
                   help="Output dir (default: ./runs/<problem>_<strategy>_n<ctx>_g<gs>x<gpb>_<timestamp>).")

    p.add_argument("--model", dest="model_name", default="openai/gpt-oss-120b")
    p.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    p.add_argument("--reasoning-effort", default="high", help="'none' to disable (e.g. non-gpt-oss models).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=26000)
    p.add_argument("--max-gen-concurrency", type=int, default=8)

    p.add_argument("--groups-per-batch", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--num-generations", type=int, default=50)

    p.add_argument("--puct-c", type=float, default=1.0)
    p.add_argument("--max-buffer-size", type=int, default=1000)
    p.add_argument("--topk-children", type=int, default=2)

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

    p.add_argument("--eval-timeout", type=int, default=None)
    p.add_argument("--num-cpus-per-task", type=int, default=None)
    p.add_argument("--grade-timeout", type=float, default=8000.0)

    p.add_argument("--resume-step", type=int, default=None)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                   help="Console log level; icl.log always captures DEBUG.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build & print one assembled prompt (base + context block), then exit. No server/ray.")

    a = p.parse_args()
    if a.log_path:
        log_path = a.log_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto = f"{a.problem}_{a.context_strategy}_n{a.n_context}_g{a.group_size}x{a.groups_per_batch}_{ts}"
        log_path = os.path.join("runs", auto)
    reasoning_effort = None if a.reasoning_effort.lower() == "none" else a.reasoning_effort

    return ICLConfig(
        problem=a.problem,
        log_path=log_path,
        model_name=a.model_name,
        vllm_base_url=a.vllm_base_url,
        reasoning_effort=reasoning_effort,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        max_gen_concurrency=a.max_gen_concurrency,
        groups_per_batch=a.groups_per_batch,
        group_size=a.group_size,
        num_generations=a.num_generations,
        puct_c=a.puct_c,
        max_buffer_size=a.max_buffer_size,
        topk_children=a.topk_children,
        context_strategy=a.context_strategy,
        n_context=a.n_context,
        max_context_tokens=a.max_context_tokens,
        mix_fraction=a.mix_fraction,
        mmr_lambda=a.mmr_lambda,
        jump_alpha=a.jump_alpha,
        context_seed=a.context_seed,
        include_code=a.include_code,
        include_strategy=a.include_strategy,
        save_completions=a.save_completions,
        log_level=a.log_level,
        eval_timeout=a.eval_timeout,
        num_cpus_per_task=a.num_cpus_per_task,
        grade_timeout=a.grade_timeout,
        resume_step=a.resume_step,
        dry_run=a.dry_run,
    )


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.dry_run:
        from icl.loop import dry_run
        dry_run(cfg)
    else:
        asyncio.run(run(cfg))

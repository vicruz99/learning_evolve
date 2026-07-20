"""Configuration for an ICL discovery run.

Mirrors the relevant knobs of TTT-Discover's ``DiscoverConfig`` (search shape, sampling) and drops
the RL-only ones (lora_rank, learning_rate, kl_penalty_coef, ...). Adds the ICL-specific
``n_context`` / ``max_context_tokens`` and the vLLM generation settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ICLConfig:
    # --- problem / logging ---
    problem: str                                    # registry key: erdos | circle_packing_26|32 | ac1 | ac2
    log_path: str

    # --- generation (local vLLM OpenAI-compatible server) ---
    model_name: str = "openai/gpt-oss-120b"
    vllm_base_url: str = "http://localhost:8000/v1"
    reasoning_effort: str | None = "high"           # gpt-oss; set None for models without it
    temperature: float = 1.0
    max_tokens: int = 26000                          # matches upstream phase1_max_tokens
    max_gen_concurrency: int = 8                      # in-flight requests to the vLLM server

    # --- search shape (matches TTT-Discover: 8 parents x 64 children = 512/generation) ---
    groups_per_batch: int = 8
    group_size: int = 64
    num_generations: int = 50

    # --- PUCT buffer ---
    puct_c: float = 1.0
    max_buffer_size: int = 1000
    topk_children: int = 2

    # --- ICL context ---
    context_strategy: str = "best"                   # selector in context.STRATEGIES: best | recent
    n_context: int = 32                              # number of past solutions injected into the prompt
    max_context_tokens: int | None = None            # None = no trimming (rely on n_context)

    # --- results storage ---
    save_completions: bool = True                    # write full raw completions per candidate

    # --- logging ---
    log_level: str = "INFO"                          # console level; icl.log always captures DEBUG

    # --- evaluation (override registry defaults if set) ---
    eval_timeout: int | None = None                  # sandbox per-candidate timeout, seconds
    num_cpus_per_task: int | None = None
    grade_timeout: float = 8000.0                    # async grading wall-clock timeout

    # --- resume ---
    resume_step: int | None = None

    # --- debug ---
    dry_run: bool = False   # build & print one assembled prompt, then exit (no ray, no server)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

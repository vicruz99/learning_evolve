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
    thinking_token_budget: int | None = None        # Qwen3: cap reasoning tokens (needs server --reasoning-parser qwen3)
    enable_thinking: bool | None = None              # Qwen3: False disables thinking entirely; None = model default
    temperature: float = 1.0
    max_tokens: int = 26000                          # matches upstream phase1_max_tokens
    max_gen_concurrency: int = 8                      # in-flight requests to the vLLM server
    # How many completions of a parent's group to request per vLLM call. Each chunk is graded as soon
    # as it returns, so smaller chunks let early-finishing completions grade while slower ones are
    # still being generated (a vLLM request only returns once its slowest sequence finishes, so a big
    # chunk withholds every completion until the slowest sibling is done). None/0 = whole group in one
    # request = grade only after ALL group_size children arrive (original behavior). NOTE: with
    # chunking there are groups_per_batch * ceil(group_size/chunk) concurrent requests, so
    # max_gen_concurrency must be raised to that many or vLLM can't co-batch them (a warning fires).
    grade_chunk_size: int | None = None
    # How long to keep waiting for an unreachable model server before giving up on the RUN.
    #
    # A generation whose groups did not reach the model is not a slow generation, it is a different
    # experiment: those parents produced no children, so the buffer, the context pool and every later
    # prompt differ from the configuration being compared. The old behaviour — swallow the error,
    # record an empty group, carry on — spent the rest of the run's budget producing exactly that, and
    # results.resume then refused to trust anything after it anyway. So: wait the server out (a vLLM
    # job being requeued is the common case and costs only wall clock), and if it does not come back,
    # stop at the last COMPLETE generation instead of walking on. 0 = wait indefinitely.
    llm_max_wait: float = 3600.0
    # Consecutive generations that may yield NO valid candidate before the run stops itself.
    #
    # A generation where all `groups_per_batch * group_size` candidates fail to grade is not a hard
    # problem, it is a broken evaluator — a starved cpu_scheduler, a full disk, a Ray head that lost
    # its workers. The run used to carry on to the end and record every one of those generations as
    # ordinary, which is worse than a crash: the generations are structurally perfect (full groups,
    # full complement of children, all invalid), so `results.resume` verifies the run as COMPLETE and
    # `--resume` skips it. A sweep can hand back twelve green runs holding nothing.
    #
    # 0 disables the stop. Keep it above 1: an early generation can legitimately come back empty on a
    # hard problem before the buffer has anything good in it.
    max_empty_generations: int = 3


    # --- search shape (matches TTT-Discover: 8 parents x 64 children = 512/generation) ---
    groups_per_batch: int = 8
    group_size: int = 64
    num_generations: int = 50

    # --- PUCT buffer ---
    puct_c: float = 1.0
    max_buffer_size: int = 1000
    topk_children: int = 2
    # Where each generation's parents come from, i.e. WHICH solution the prompt hands the model as
    # "the current solution to improve upon":
    #   "puct"    — PUCT-select from the buffer (TTT-Discover's search; the default)
    #   "initial" — always the problem's seed solution => Best-of-N. Combined with n_context=0 this is
    #               the no-past-experience baseline: no history via parent selection, none via prompt.
    #   "best"    — always the buffer's best-so-far solution => greedy hill-climbing. Same prompt
    #               shape as "puct", so the difference between the two is exactly what PUCT's
    #               exploration term (under-visited states, lineage spreading) is worth.
    #   "none"    — NO current solution in the prompt at all: the "improve upon this" framing is
    #               removed and the tail states only the objective and the target
    #               (envs.base.objective_only_prompt). Past experience then reaches the model through
    #               the ICL context block and nothing else, so this is the parent source that
    #               measures a context strategy on its own. With n_context=0 it is a from-scratch
    #               zero-shot arm.
    #
    # "none" is still a search-neutral choice, not a search variant: children are attributed to the
    # seed (as in "initial"), and the evaluator still receives that state — so the constructions the
    # sandbox pre-imports (ac1/ac2's height_sequence_1, erdos' initial_h_values) are unchanged.
    parent_source: str = "puct"

    # --- memory guard ---
    # Fraction of the job's detected memory ceiling at which the run stops itself at the next
    # generation boundary rather than being killed mid-generation. LSF enforces its ceiling by
    # polling the process tree's total RSS and then sending SIGINT/SIGTERM/SIGKILL 10 s apart, which
    # lands wherever it lands; stopping ourselves lands on a boundary where the sampler and tracker
    # have just been flushed, so --resume has a clean generation to continue from. 0 disables the
    # stop (the per-generation rss line is always logged either way).
    memory_stop_fraction: float = 0.85

    # --- reproducibility ---
    # Seeds the sampler's only stochastic surface (the AC problems' random initial construction) and,
    # unless --context-seed is given explicitly, the `random` context strategy. It does NOT make a run
    # bit-reproducible: vLLM sampling is temperature>0 and unseeded, which is where replicate-to-
    # replicate variation actually comes from. Its job is provenance + distinct replicate identities.
    seed: int | None = None

    # --- ICL context ---
    context_strategy: str = "best"                   # selector in context.STRATEGIES (see docs/strategies/)
    n_context: int = 32                              # number of past solutions injected into the prompt
    max_context_tokens: int | None = None            # None = no trimming (rely on n_context)
    # strategy knobs (each strategy reads only the ones it needs):
    mix_fraction: float = 0.5                        # x: share of n_context from the primary/"best" pool
    mmr_lambda: float = 0.7                          # MMR quality<->diversity (1=quality only, 0=spread only)
    jump_alpha: float = 0.5                          # informative: value(alpha) vs jump(1-alpha) blend
    context_seed: int | None = None                  # seed for the `random` strategy
    # Whether a parent is dropped from its own context block. True (default) = a parent never sees
    # itself listed as a "past solution" it should improve on -- it is already rendered, once, as the
    # current solution in the prompt tail.
    #
    # Setting this False is also the only way to make the context block IDENTICAL across a
    # generation's parents (together with a fixed `context_seed`, which pins tie-breaking): the block
    # then becomes a genuine shared prefix that vLLM prefills once for the whole generation instead of
    # once per parent. The cost is less prompt diversity between a generation's parents.
    exclude_parent_from_context: bool = True
    # rendering (orthogonal to selection; apply to every strategy):
    include_code: bool = True                        # show each solution's code
    include_strategy: bool = False                   # show each solution's <strategy> reasoning block

    # --- results storage ---
    save_completions: bool = True                    # write full raw completions per candidate
    # Write each candidate's reasoning_content to child_NN.reasoning.txt. Reasoning is the bulk of the
    # decode cost and is invisible in the completion text, so keeping it is what makes the token
    # accounting interpretable; SFT/RL variants need it too. Off = smaller runs.
    save_reasoning: bool = True

    # --- logging ---
    log_level: str = "INFO"                          # console level; icl.log always captures DEBUG

    # --- evaluation (override registry defaults if set) ---
    eval_timeout: int | None = None                  # sandbox per-candidate timeout, seconds
    num_cpus_per_task: int | None = None
    # Cores the Ray cluster may use, when THIS run has to start one (no head up). Ignored when a head
    # is already running -- its size was fixed at `ray start` and a client cannot change it. Under
    # run_sweep.py the launcher owns the head, so use its --ray-num-cpus instead.
    ray_num_cpus: int | None = None
    grade_timeout: float = 8000.0                    # async grading wall-clock timeout

    # Extra kwargs for the problem's reward evaluator (see envs.base.EnvConfig.evaluator_options).
    # Only trimul uses it today, for the machine-specific half of grading: which GPU, and which
    # interpreter runs the kernel harness. It goes through to_dict() into the run's config.json, so a
    # finished run records the interpreter that produced its timings -- which an environment variable
    # would not, and kernel timings are only meaningful against a known stack.
    evaluator_options: dict[str, Any] = field(default_factory=dict)

    # --- resume ---
    resume_step: int | None = None

    # --- debug ---
    dry_run: bool = False   # build & print one assembled prompt, then exit (no ray, no server)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

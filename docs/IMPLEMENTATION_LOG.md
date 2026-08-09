# Implementation Log

> A dated diary of what was built, the design decisions behind it, and the problems faced / how they
> were solved. Kept intentionally terse — scan it to recall *what* we did, *why* we did it that way,
> and *what bit us*. For the research framing see `PROJECT_CONTEXT.md`; for the roadmap see
> `EXPERIMENT_PLAN.md`.
>
> Started 2026-07-24 (retroactively reconstructed from git history + working notes for the earlier
> days — better late than never).

---

## 2026-07-20 — Project scaffolding + the tinker-free ICL harness

**Built**
- Context docs for coding agents: `PROJECT_CONTEXT.md`, `EXPERIMENT_PLAN.md`, `CLAUDE.md`.
- The whole ICL harness under `src/`: `puct/` (State + PUCTSampler), `sandbox/` (Ray evaluators +
  cpu_scheduler), `envs/` (grading base + erdos / circle_packing 26·32 / ac1·ac2 + registry),
  `generation/` (vLLM client), `context/` (best/recent selectors), `icl/` (config + search loop),
  `results/` (per-run tracker + cross-run index + analysis), `run_icl.py` CLI.

**Design decisions & why**
- **Vendor TTT-Discover's search machinery into `src/`, decoupled from tinker/torch/transformers.**
  Reason: `ttt_discover`'s package root eagerly imports tinker, but PUCT/buffer/State/evaluators don't
  need it. Vendoring drops the dependency footprint to ray + numpy + scipy + cvxpy + openai.
- **Replace the RL head (tinker sampling + GRPO/LoRA) with a frozen-model generate-and-score loop.**
  No weights change — improvement comes purely from search + in-context conditioning. The loop mirrors
  `do_sync_training` minus the gradient step.
- **Same 8×64 shape as the baseline:** PUCT selects `groups_per_batch` parents, model proposes
  `group_size` children each generation.
- **Vendor the problem definitions verbatim** so prompts and seed solutions match TTT-Discover exactly
  (fair-comparison requirement from the plan).
- **Self-descriptive metric names** in our outputs (`puct_expansions`, `valid_candidates`,
  `best_so_far_score`) instead of raw PUCT symbols; per-parent phase logging with timing; Ray/vendored
  console noise suppressed behind `--log-level`.

**End-of-day summary**
- ✅ Repo scaffolding + full ICL harness stood up and verified (unit tests + stubbed and real
  end-to-end runs against a live gpt-oss-120b server).
- 🔎 Key call: vendor-and-decouple rather than depend on `ttt_discover` — keeps us tinker-free.

---

## 2026-07-21 — Pluggable context-selection strategies

**Built**
- Expanded the context layer from 2 ad-hoc selectors (best/recent) to **10 presets over 4 reusable
  engines**: `topk` (random, best, recent, biggest_jump), `mix` (best_worst, best_jump),
  `per_lineage`, `mmr` (best_diverse, informative, contrastive). Shared helpers `context/ranking.py`
  and `context/lineage.py`. One explanation doc per strategy under `src/docs/strategies/`.

**Design decisions & why**
- **Selection and Rendering are orthogonal.** Selection = *which* past solutions enter the prompt
  (→ `SelectionResult` = positive block + optional secondary block). Rendering = *what* of each is
  shown (`include_code` / `include_strategy`; "strategy-only" = strategy on, code off, falling back to
  code so a solution never renders empty).
- **Diversity comes from the search tree already in `State.parents`** — no embeddings. `per_lineage` =
  hard skip of ancestors/descendants (the path relationship the user specified; siblings NOT
  penalized). MMR = soft tree-distance similarity `1/(1+dist)`, which penalizes both direct lineage and
  siblings.
- **Capture the model's `<strategy>…</strategy>` reasoning at rollout** — new `State.strategy` field
  (backward-compatible `from_dict`) + `parse_strategy_block`, so context can show reasoning instead of
  / alongside code.
- `informative` blends value with improvement-over-parent (`jump`) in the MMR quality term;
  `contrastive` pairs diverse bests with low-scoring negatives.

**Problems faced**
- **Negatives can only be low-scoring *valid* solutions** — the PUCT buffer stores no failed rollouts.
  Genuine-failure injection (broken code + error messages) deferred.

**End-of-day summary**
- ✅ 10 strategies / 4 engines landed with per-strategy docs and tests.
- 🔎 Kept selection vs rendering orthogonal so any strategy composes with any render mode.
- ⏳ Deferred: genuine-failure negatives; GRPO-style group strategy.

---

## 2026-07-22 — Context pool fix, tie-breaking, seed handling, toy env, thinking config

**Built / changed**
- **Context now draws from a harness-side pool of *every valid graded solution*** (mirrored to
  `buffer/context_pool.jsonl` for `--resume-step`), separate from the PUCT search buffer. PUCT search
  (parent sampling, buffer) is untouched.
- Toy in-process (no-Ray) smoke-test env `toy_ee` + `Environment.uses_sandbox` flag; the loop skips
  `init_ray` for sandbox-free problems.
- Qwen3.6 thinking config: enable/disable thinking, configurable thinking-token budget.
- First real trial experiments run.

**Problems faced & how solved**
- **`best_worst` / `contrastive` had no genuine low-scoring negatives.** Cause: the PUCT buffer is
  pruned by `topk_children` / `max_buffer_size` down to high scorers. Fix: build context from the
  separate all-valid-solutions pool (above), so negatives actually exist.
- **Tied equal-score solutions were always taken in buffer-insertion order.** Fix: random tie-breaking
  in `_candidates` (entropy-seeded, or reproducible via `context_seed`).
- **Seed states were duplicated in context.** The sampler seeds the buffer with `groups_per_batch`
  identical copies of the initial state (distinct uuids). Rule adopted: the seed *may* appear as
  context, but **never duplicated**, never the copy that is the current PUCT parent, and **not at all
  when we're starting *from* the seed** (all of gen 0 → blank context). Implemented as
  `dedupe_seeds(states, initial_ids, drop_initial=…)`, applied in the loop before selection.
- **Trailing ``` fence leaked into stored solutions / `State.code` / context.** Fix: strip a same-line
  trailing fence in `last_codeblock_postprocess`.
- Rewrote `contrastive` negatives to reuse the MMR-lineage engine worst-first (dropped the earlier
  stratified low→mid ladder and its now-dead helpers).

**End-of-day summary**
- ✅ Context pool decoupled from the PUCT buffer → negatives work; seeds de-duplicated; tie-breaking
  randomized; a fast no-Ray toy env for smoke tests; Qwen thinking configurable.
- 🔎 Key call: **selection reads an all-valid-solutions pool, not the pruned search buffer** — the
  buffer is for search, the pool is for context.

---

## 2026-07-24 — PUCT verification, extraction/failure semantics, docs reorg

**Done**
- **Verified the PUCT sampler** against the TTT-Discover paper (§A.2) and the upstream repo. It is
  byte-faithful to TTT-Discover's own code. One accepted discrepancy vs. the *paper's* wording: `T` and
  `n(s)` increment **per rollout (per child)** rather than once per parent-expansion, because
  `update_states` / `record_failed_rollout` are called once per completion. `m(s)` is unaffected (max
  is idempotent). Left as-is — faithful to the reference implementation. Decision: **do nothing.**
- **Confirmed code / strategy extraction + failure semantics** (no code change, just nailed down the
  behaviour):
  - Code = last fenced ```python block via `last_codeblock_postprocess`; **no/empty block → `''`**.
  - Strategy = last `<strategy>…</strategy>` via `parse_strategy_block`; absent → `""`.
  - **Unextractable/empty code = a failed rollout**: `check_format` fails, the sandbox never runs,
    `record_failed_rollout` advances PUCT counters, and **no child is buffered**. (Same failure bucket
    as code that extracts but crashes/times out/scores 0 — the difference is only whether the sandbox
    ran.)
  - **Missing `<strategy>` is harmless** — pure metadata; the solution still counts, and rendering
    falls back to code.
- **Docs reorg:** created `docs/`, moved `PROJECT_CONTEXT.md` + `EXPERIMENT_PLAN.md` into it, kept
  `CLAUDE.md` at the repo root (Claude Code auto-loads it from root) and updated its paths. Started this
  log.

**End-of-day summary**
- ✅ PUCT confirmed faithful (one documented, accepted per-rollout counting quirk).
- ✅ Extraction/failure rules pinned down and documented.
- ✅ Docs consolidated under `docs/`.

---

## 2026-07-24 — Pipeline bottleneck analysis + failure-type instrumentation

**Analysis (measured, from `runs/cp_26_best_n_30_g5x12_gen30` + live `localhost:8001` gpt-oss-120b on 1×A100)**
- **Grading dominates, not generation.** gen 1: 1319s total = 317s generate + ~1000s grade. Grading
  is gated by the **slowest candidate** because of hard barriers (per-group `asyncio.gather`, then a
  per-generation barrier for PUCT). Eval-time distribution of *valid* circle-packing solutions:
  p50=0.8s, p90=41s, p95=72s, max>120s — median is trivial, the tail is everything.
- **`eval_timeout=530` (inherited from TTT-Discover) is ~7× beyond p95 of valid solves** → pure dead
  weight on the tail. Recommended ~90s (keeps valid-but-slow, chops runaways). Not yet applied.
- **LLM throughput scales with batch:** 127 tok/s (1 seq) → 519 (8) → 1023 (32), still climbing.
  Biggest generation lever is the reasoning-token budget (`max_tokens`/`reasoning_effort`), and
  `--max-model-len` on the server (131k starves KV cache / batch size).
- **Ray is not the compute culprit.** It's a distributed process pool + CPU-affinity scheduler; the
  real costs are the long-tail timeouts + the GPU/CPU serialization (GPU idle during grading, CPUs
  idle during generation). The one Ray-design cost that *does* bite is **CPU starvation** (below).

**Built — failure taxonomy so "failed" is no longer one opaque bucket**
- `sandbox.classify_failure(text)` → `{no_code, invalid_result, process_crash, eval_timeout,
  cpu_starvation, results_missing, grade_timeout, grade_error, unknown}`. `cpu_starvation` /
  `results_missing` = **infra**; the rest = **genuine** (though `eval_timeout` can be contention-induced).
- Threaded a `failure_type` field through `execute_code` (now returns `(result, msg, failure_type)`),
  `_get_failure_entry`, the 3 env `get_reward`s, `VerifyResult`/`RolloutResult`, and the tracker.
- **Stopped truncating the failure `msg`** (was `[:200]`/`[:500]` in `tracker.py`, which clipped the
  *terminal* exception of the Ray traceback — the exact bytes that distinguish infra from genuine).
  New cap `MAX_MSG_CHARS=4000`. `summary.json` now carries `totals.failure_types` + per-generation
  `failure_types`.

**Built — `src/results/recheck_failures.py`** (retroactive, no full rerun)
- Static pass reclassifies a past run's failures from `events.jsonl` (works on old runs via the msg
  traceback frame). Re-run pass re-extracts each candidate's code from its saved completion and grades
  it **one-at-a-time** (contention-free) with a generous timeout → verdict: infra/contention vs.
  genuinely-slow vs. genuine-crash.
- Validated on `cp_26_best_n_30_g5x12_gen30`: static = 23 process_crash, 22 invalid_result,
  3 cpu_starvation, 1 no_code, 7 unknown(=truncated-msg crashes), **0 genuine "Process timed out"**
  (so the "timeouts" the user recalled were mostly crashes here). Re-running the 3 cpu_starvations:
  **1 was a genuine infra loss** (graded valid in 20s uncontended — a real solution thrown away),
  **2 were genuinely slow** (still time out at 120s). Confirms both failure modes are real and now
  separable.

**ShinkaEvolve eval-architecture comparison** (source: `shinka/core/async_runner.py`,
`shinka/launch/{local.py,scheduler.py}`, `shinka/core/wrap_eval.py`, `examples/ttt_discover_math/circle_packing_26/`)

| Aspect | Our harness | ShinkaEvolve |
|---|---|---|
| Process layers / candidate | **2** (Ray task → inner `subprocess.Popen`) | **1** (`subprocess.Popen python evaluate.py`; candidate imported in-process) |
| Scheduler | Ray + `cpu_scheduler` **fixed CPU groups**, acquire-or-fail within `eval_timeout+10s` | asyncio + threadpool; **no CPU groups, no acquire step** |
| CPU allocation | hard affinity pin to a reserved 1-core group | **soft**: BLAS/OMP thread-cap env (`cpu_count // max_evaluation_jobs`); OS spreads |
| Can a candidate fail for lack of CPU? | **Yes** → `cpu_starvation` | **No** — it just waits its turn |
| Per-eval timeout (circle packing) | `eval_timeout=530s` | wall-clock poll + `process.kill()`, `job_time` 5min (dev) / 15min (full) |
| Eval body cost | validate + sum radii (all cost is in the evolved code) | same |
| Local LLM | OpenAI-compatible `--vllm-base-url` | `local/<model>@<url>` grammar; also swap `meta/novelty/embedding` models + `SHINKA_PRICING_MODE=offline` |

Takeaway: the **infra-timeout failure mode is specific to our fixed-group acquire-timeout design**;
ShinkaEvolve's soft-thread-cap + no-reservation model cannot produce it. Motivates a deferred
root-cause fix (decouple the CPU-group queue-wait from `eval_timeout` so a merely-queued candidate
never *fails*; and/or drop the double-process nesting). Head-to-head timing run left for later.

**Deferred (explicitly out of scope this pass)**
- Cutting `eval_timeout` 530→~90, pipelining generation↔grading (overlap GPU/CPU), and the
  CPU-request root-cause fix. All measured/justified above; not yet applied.

**End-of-day summary**
- ✅ Failures now self-classify (infra vs genuine) with untruncated messages; `summary.json` shows the split.
- ✅ `recheck_failures.py` reclassifies + re-runs past failures without repeating an experiment.
- ✅ ShinkaEvolve eval model documented; confirms our CPU-starvation mode is design-specific.
- ⏳ Deferred: eval_timeout cut, gen↔grade pipelining, CPU-request root-cause fix.

---

## 2026-07-24 — Grade-as-ready pipelining (`grade_chunk_size`) + evaluation design notes

**Built — configurable grade-as-ready chunking**
- New `ICLConfig.grade_chunk_size` (CLI `--grade-chunk-size`, default `None` = whole group = original
  behavior). It sets how many of a parent's `group_size` completions are requested per vLLM call;
  each chunk is graded the moment it returns. `icl/loop.py`: `_chunk_sizes()` + `_run_group()` now
  runs the per-chunk gen→grade coroutines concurrently, so one chunk's grading (CPU/sandbox) overlaps
  another chunk's in-flight generation (GPU). A startup **warning** fires when `max_gen_concurrency`
  is below `groups_per_batch * ceil(group_size/chunk)` (else vLLM can't co-batch the chunk requests).

**Correction — when grading actually starts (this drove the design)**
- Earlier I claimed per-parent grading overlaps other parents' generation. **Wrong.** The logs show
  all parent requests are co-batched by vLLM and each returns only when *its slowest sequence*
  finishes; the global-slowest sequences are spread across parents, so **all parent requests return
  near-simultaneously** and grading is a hard phase *after* all generation. The real waste: a
  completion ready at 300 s waits for its slowest sibling (~960 s) before it can be graded.
- Chunking fixes it by **decoupling each completion from the slowest-in-request**: smaller `n` per
  request → early finishers return and grade while slow ones still generate. Measured on a
  `5×12` gen: generate ~960–1150 s, grade ~530 s, total ~1500–1680 s → overlapping should pull total
  toward ~1000–1150 s.

**vLLM / prefix-caching notes (for running with chunking)**
- Automatic Prefix Caching reuses a shared prompt prefix **across requests**, so a parent's whole
  prompt (incl. its big context block) is prefilled once and reused across *that parent's* chunks;
  cross-parent, only the shared base-question head (~800 tok) is reused (context blocks diverge).
- **Caveat at large context:** firing many chunk requests concurrently can *race* the prefill before
  APC caches it → redundant prefill of the (67k–103k-tok) prompt. So on big-context gens use a
  **moderate** chunk (~8), not `n=1`; optional "prime the prefix" (send 1 chunk, await, then fan out)
  would remove the race — not yet built.
- Suggested server flags: `--enable-prefix-caching` (V1 default on), `--kv-cache-dtype fp8`
  (~2× KV capacity → more concurrent long-context seqs + better cache retention), size
  `--max-model-len` to actual use (131k only justified because context reaches ~103k), keep
  `--async-scheduling` + chunked prefill. Client `--max-gen-concurrency` should match `--max-num-seqs`.
- Observed: prompt grows to **~103k tokens by gen 4** with `n_context=30` — prefill dominates
  generation time. Independent lever: `max_context_tokens` / `n_context`.

**Considerations & deferred decisions (evaluation), captured for later**
- **Parallel experiments were the real starvation cause.** Each `run_icl.py` starts its *own* Ray +
  its *own* `cpu_scheduler`, each assuming it owns all 96 cores; three runs launched together (03:00)
  triple-booked the cores → contention + starvation. A single run is fine (96 groups ≥ 60 candidates).
  Re-check proved it: `cp_26_random` lost **17/22** starved candidates that were actually valid
  solutions; the contemporaneous `v2` runs (machine freer) had **0** starvation.
  - **Fix (deferred, agreed):** run concurrent experiments against **one shared Ray cluster**
    (`ray start --head` + `RAY_ADDRESS=auto`; `init_ray` already honors it) so they share one 96-slot
    scheduler → no oversubscription. Add a `--ray-address` flag for convenience. Same-problem runs
    only until the detached scheduler is keyed by `num_cpus_per_task`.
- **Keep the CPU-acquire timeout, don't remove it** (user call): raise it a bit and add a loud
  `get_cpu_group` **warning** ("waited Ns for a CPU; N queued — likely oversubscribed") so starvation
  is visible in real time, not a post-hoc autopsy. (Not yet implemented.)
- **`eval_timeout=530 s` left as-is for now** — re-runs show genuinely-slow solutions exist but we
  haven't confirmed which valid solutions legitimately need long budgets; don't want to discard them.
  Still the dominant single-run cost (slow candidate gates the per-gen barrier). Revisit later.
- **`ac1`/`ac2` reserve 2 cores** — reference programs are single-threaded (no mp/threads/joblib), and
  the sandbox caps `ProcessPoolExecutor` to `num_cpus_per_task` anyway, so 2 is likely overkill; but
  the prompt advertises "2 CPUs" and no `ac` runs exist yet. **Verify on the first real `ac` run**
  before dropping to 1. For circle packing, 1 core is correctly sized (single-threaded CPU-bound;
  oversubscribing CPU-bound work doesn't raise throughput — wall ≈ core-seconds / cores).
- **Warm start assessed, deferred as too brittle for now.** Measured cold start (fresh subprocess,
  every candidate): numpy 0.37 s, +scipy 1.55 s, +cvxpy 3.04 s — bigger than the median useful
  compute. But it's largely the *price of isolation* (fresh spawned, killable process). Fork-server
  (fork a pre-imported parent) risks thread/BLAS-after-fork deadlocks; a persistent in-process pool
  leaks state and loses clean timeout-kill. Gain is also tail-gated (near-zero until `eval_timeout`
  is cut). Revisit only after the tail is cut, and then as a carefully-tested fork-server. Note:
  ShinkaEvolve doesn't warm either (re-imports per eval subprocess).

**End-of-day summary**
- ✅ `grade_chunk_size` grade-as-ready pipelining landed (default off / unchanged behavior), verified live.
- ✅ Corrected the mental model of when grading starts (co-batching → simultaneous returns).
- 📝 Captured deferred evaluation decisions: shared-Ray-cluster for parallel runs, keep+warn CPU
  timeout, eval_timeout untouched, `ac` cores to verify, warm start too brittle for now.

---

## 2026-07-25 — Sweep: `max_gen_concurrency` × `grade_chunk_size` (circle_packing_26)

**Question:** for gpt-oss-120b on **2×A100**, `--reasoning-effort medium`, fixed shape
6 parents × 15 = 90 candidates/gen, `n_context=20` — what `(max_gen_concurrency, grade_chunk_size)`
minimizes wall time? Rough answer only, extremes, few runs.

**Method:** 4 runs, sequential (one owns the box), 3 gens each, 25-min safety cap. Metric =
**per-generation wall** (all runs completed gen 0 cleanly; the cap truncated later gens — each gen is
~740 s so 3 gens can't fit in 25 min, expected). `grade_chunk_size=15` (= whole group, no pipelining)
collapses the concurrency axis (only 6 requests), so the meaningful corners are the extremes below.

| grade_chunk_size | max_gen_concurrency | seqs co-batched on GPU | gen-0 wall | valid |
|---|---|---|---|---|
| 15 | 6  | 90 (all at once)   | **739 s** | 53/90 |
| 1  | 90 | 90 (all at once)   | 746 s | 58/90 |
| 1  | 20 | 20 (waves of 20)   | 762 s | 56/90 |
| 1  | 6  | 6  (waves of 6)    | 787 s | 57/90 |

**Verdict — the knobs barely matter for speed on this problem (~6 % spread total):**
- **Generation dominates (~740 s/gen); grading is cheap.** So `grade_chunk_size` — whose only job is
  to overlap grading with generation — buys ~nothing here; `chunk=1/conc=90` was even ~7 s *slower*
  than the no-chunk control (noise).
- **The 2×A100 is already saturated at conc≈20.** Adding sequences (90) doesn't speed each up; cutting
  to 20 doesn't slow them, you just pay a slightly longer wave *tail*. Only **conc=6 genuinely starves
  the GPU** (6 seqs can't fill it + 15 sequential waves) and even that is only +6 %.
- Note: `conc=6` did **not** under-utilize when `chunk=15` (still 90 seqs on the GPU). Only
  `chunk=1 + conc=6` starves it. (Corrected an earlier wrong hunch that low conc always underutilizes.)

**Recommendation:** for fastest wall time, **max out concurrency and don't chunk** — set
`--max-gen-concurrency = groups_per_batch × group_size` (90 here) and leave `--grade-chunk-size` at
default (whole group). Anything `conc ≥ 20` is within 3 % of optimal — not worth tuning further.

**Caveats:**
- **Circle-packing-specific.** Holds whenever *generation ≫ grading*. On `ac`/`erdos` a single eval
  can take many minutes, so grade-as-ready (`chunk=1` + moderate conc) may genuinely cut wall there —
  re-check before assuming. (Not yet run.)
- **`grade_chunk_size=1` still earns its keep for observability** (live `[k/15 graded]` trickle,
  added this session) at only ~3 % cost at conc=20 — a monitoring aid, not a speed lever.
- **Sizing:** ~12–13 min per generation at this shape/effort → a real 3-gen run needs ~40 min; the
  25-min cap was only to bound the sweep.
- **`--reasoning-effort medium` produced 0 empty children** (vs ~70 % `no_code` empties at `high`,
  where completions exhaust the 26 k token budget on hidden reasoning before emitting an answer).

---

## 2026-07-25 — Prompt reordering for prefix caching (env prompt-API split)

**Built**
- Split the `Environment` prompt surface into two zones (`envs/base.py`):
  `problem_intro()` = constant motivation + high-level description (top, parent-independent) and
  `improvement_task()` = rules + the current solution to improve upon, with the **current solution
  rendered LAST**. `get_question()` is kept as `intro + task` for dry-runs.
- `ICLRunner._build_prompt` now weaves the ICL context block **between** the two zones:
  `intro + block + tail` (was `base + block`). Final layout per parent:
  **`[intro] → [past-solutions block] → [rules + current-solution]`**.
- Reordered all envs (circle_packing, ac1/ac2, erdos, toy) to this shape; moved the varying
  `state_ctx` to the end and reworded "above / this" → "below / current".
- Added a contextualizing header before the past-solutions block and renamed the delimiters
  (`context/selection.py`): *"Before proposing a new solution, review the past solutions you've
  already tried below … --- Past solutions you've tried, with their results ---"*.

**Why** — within one parent all children share an identical prompt (already fully prefix-cached).
Across parents in a generation, only the *initial solution* differs; putting it last makes
`intro + context block + rules` a shared prefix, so vLLM re-prefills only the trailing current
solution instead of re-prefilling the rules + the whole `n_context`-solution block for every parent.

**Bugs caught while reviewing the (user's) reordering**
- **erdos:** `code_section` was a plain `'''…'''` string containing `{state_ctx}` (not an f-string)
  → the literal text `{state_ctx}` appeared in the prompt from gen ≥ 1 and the current construction
  was never shown. Fixed (f-string in both branches; also render the current solution at gen 0).
- **ac1:** prose said *"write a search function, `construct_function()`"* (copied from ac2) while
  ac1's entrypoint and its own Rules line are `propose_candidate` → contradictory instruction the
  grader would reject. Fixed to `propose_candidate()`.
- Minor: leading blank lines + a spaces-only line in the ac tails, erdos trailing space — cleaned.

**Verified** — dry-runs + programmatic offset checks, then real **2-gen × 2-parent × 3-child** runs
on all 5 problems. Inspected the real gen-1 `prompt.txt` for each: ordering is
`intro → context header → past solutions → rules → current solution`, no literal `{state_ctx}`,
correct entrypoint names. Results sane (circle 2.541 → 2.611, ac1 100 %/83 % valid, erdos valid
throughout).

**What bit us — simultaneous run launches hang Ray.** Launching several `run_icl.py` at the *same
instant* wedged all but one: each run starts its **own** Ray head that prestarts ~96 workers, and
simultaneous bring-up deadlocks on Ray IPC — stuck mains sit in `unix_stream_data_wait`, the one
healthy run in `ep_poll`. Parallel runs ARE fine once **staggered** (~120 s apart) or run
sequentially — each then gets an isolated cluster (session dir keyed by pid, random ports); the
earlier 3-way run only worked because its launches were ~5 min apart. A code-level fix would cap
`num_cpus` / prestart workers in `sandbox/ray_setup.init_ray`.

---

## 2026-07-25 — Token accounting + `run_sweep.py` coordinated launcher

**Measurement that reset the priorities: this workload is decode-bound, not prefill-bound.**
From `runs/_sweep_c15_g6` (n_context=20, medium, 6×15=90, 739 s/gen):
- gen-1 prompt is **17.2k tokens**, not the ~100k the earlier note assumed (that was n_context=30).
  6 parents × 17.2k ≈ 104k prefill tokens/gen ≈ **30–50 s**, i.e. ~5 % of the generation.
- Aggregate decode ≈ 700–1100 tok/s accounts for essentially all of the remaining wall time, matching
  the earlier batch-scaling numbers. So `--kv-cache-dtype fp8` / `--max-num-batched-tokens` (the
  `notes` file's vLLM items) are worth single-digit percent at n_context=20; they only matter once
  prompts approach ~100k.
- Corollary: since the sweep showed the 2×A100 saturated at conc≈20, **no server flag makes one run
  much faster**. The campaign-level lever is one TP=2 server with **several run_icl.py clients sharing
  it** — vLLM co-batches across experiments, so run B's generation fills the batch while run A grades.

**Measured, and it kills the "prime the prefix" idea for now.** The 2026-07-25 prompt reordering
assumed `intro + context block + rules` is a shared prefix across a generation's parents. It is not:
with `strategy=best, n_context=20` the longest common prefix across gen-1 parents is **500 tok /
2.9 %** of a 17.2k prompt, because `_build_prompt` passes `exclude_id=parent.id` and tie-breaking is
random, so every parent gets a *different* context block. Priming a 500-token head buys nothing.
Unlocking it requires selecting **one context block per generation** instead of per parent — a
semantic change (incompatible with the parent-dependent strategies `per_lineage`/`mmr`), which is
exactly the open question in `notes` line 92. Deferred, not built.

**Built — exact token accounting (`generation/vllm_client.py`, `icl/loop.py`, `results/tracker.py`)**
- `generate()` now returns a `GenResult` (was `list[str]`): `texts`, `reasonings`, `finish_reasons`,
  `prompt_tokens`, `cached_prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `latency`.
- Per generation: `_sum_usage` + a log line — `prompt 104,821 (91% cached) | decode 486,220
  (5402/completion, 657 tok/s) | truncated 2/90`. Persisted to `progress.csv` (new columns incl.
  `wall_seconds`), `summary.json` (`usage`, per-generation + run total), `events.jsonl`
  (`finish_reason`, `reasoning_chars` per candidate).
- **`finish_reason == "length"`** is the per-candidate truncation signal (exact, unlike token counts —
  see caveat below) and now fires a warning. This is what `--max-tokens` tuning was missing.
- `--save-reasoning` (default on) writes `child_NN.reasoning.txt`. Deliberately a *separate* file:
  `recheck_failures.py` re-extracts code from the completion file, so that file must stay raw answer text.

**Three facts the live server taught us (all verified against `localhost:8001`)**
- **`usage` is per *request*, summed over the `n` choices** — exact per-candidate token counts are
  impossible when `n > 1`. Hence per-candidate = `finish_reason` + char counts, per-request = tokens.
- **This server returns empty `reasoning_content`** and silently counts reasoning inside
  `completion_tokens` (a 3-token answer cost 44). Capturing traces needs the server relaunched with
  `--reasoning-parser openai_gptoss`; the plumbing is in place and starts writing files when it is.
- **`usage.prompt_tokens_details.cached_tokens` is always 0** on this build while `/metrics` shows a
  92 % hit rate → the cache signal is scraped from `/metrics` at generation boundaries instead
  (`VLLMClient.cache_counters`). Two caveats baked into the docstrings: those counters are
  **server-global** (they mix concurrent runs) and counted **per sequence**, so `cache_queries ≈
  n × prompt_tokens` — only the *ratio* is meaningful.

**Built — `src/run_sweep.py` + `src/sweeps/*.yaml`: coordinated multi-run launcher**
- One YAML file: `sweep:` (name / max_parallel / stagger / server_max_num_seqs), `common:`, `grid:`
  (cross-product), `runs:` (explicit overrides). Precedence `common` < `grid`/`runs`.
- **Keys are `run_icl.py` long flags verbatim**, validated against the real parser (`run_icl.build_parser`,
  split out for this) — no second vocabulary, nothing to keep in sync. Typos fail *before* launching,
  with a suggestion; `log-path`/`resume-step` are launcher-owned and rejected; bools map to the right
  polarity (`include-code: false` → `--no-include-code`).
- Supervisor: `max_parallel` queue + `stagger` between launches (the Ray-hang mitigation), starts a
  shared `ray start --head` if none is up, children in their own process session so killing the
  supervisor doesn't kill the runs. Preflight warns if the server doesn't serve the requested model
  or if peak concurrency exceeds `server_max_num_seqs`.
- `--status` / `--resume` / `--stop` / `--print-cmds`. Status reconciles the manifest against reality
  and reports **`DIED`** (manifest says running, pid gone) — previously invisible. Table shows
  gens, best, gen wall, tok/s, age.
- Runs land in `runs/<sweep>/<run>/`, so the tracker's `index.csv` becomes a per-sweep index and
  `results/analysis.py` pointed at `runs/<sweep>` sees exactly that sweep.

**Verified** — 44 tests pass (14 new in `tests/test_sweep.py`: grid expansion/naming, flag +
bool-polarity validation, reserved keys, manifest paths, `DIED` reconciliation, half-written summary
tolerance). Live end-to-end: a 3-run `toy` sweep with `max_parallel=2` staggered correctly, queued the
third when the first finished, all exited 0, and the status table + per-sweep `index.csv` came out right.
A 2-gen `toy` ICL run confirmed the token line, `progress.csv` columns, `summary.json.usage`, and
`events.jsonl` fields.

**Deferred**
- Shared per-generation context block (the prerequisite for prefix priming) — semantic change, open
  question. *(Landed the next day as `--no-exclude-parent`, below.)*
- Server relaunch with `--reasoning-parser openai_gptoss` to actually capture reasoning traces.
- `--kv-cache-dtype fp8` / `--max-num-batched-tokens 16384` / sizing `--max-model-len` down from 131k:
  worth doing, but measured as small at n_context=20. **Risk flagged:** at n_context=30 the prompt hit
  ~103k, so 103k + `max_tokens` 26k = 129k is within 2k of the 131k ceiling.

---

## 2026-07-25 — Shared Ray head for concurrent experiments (`init_ray` auto-connect)

**Built** — `sandbox/ray_setup.init_ray` now tries `ray.init(address="auto")` first and only falls
back to a private `ray.init()` if that raises `ConnectionError`. So if a shared head was started
out-of-band (`ray start --head`), every `run_icl.py` **auto-connects** to it; otherwise single runs
behave exactly as before. No `RAY_ADDRESS` env needed (verified both paths: private fallback on a
clean box → random-port GCS; shared head up → connects to `:6379`, and the head persists after the
run exits).

**Why** — this is the fix for the simultaneous-launch hang + N× core oversubscription from the prior
entry. One shared head = one 96-CPU pool, so Ray's own `num_cpus` admission caps *total* grading
across **all** concurrent experiments (each `run_program` is `ray.remote(num_cpus=num_cpus_per_task)`),
and there's no per-run head boot to race. Chose the "start it yourself, like the vLLM server" model
(documented in `src/README.md` + memory) over an env-activation hook — simpler, and the user keeps
explicit control. Ritual: `.venv/bin/ray start --head --disable-usage-stats` once per session, launch
runs, `.venv/bin/ray stop` when done.

**Known caveat (not yet addressed)** — the detached `cpu_scheduler` actor is created once with the
*first* run's `num_cpus_per_task` and reused by name (`"cpu_scheduler"`); it partitions cores into
fixed-size groups. So concurrent **mixed families** (circle/erdos = 1 cpu, ac = 2 cpu) share a
wrong-sized scheduler (functional — Ray admission still bounds total load — but suboptimal pinning /
throughput). Same-family concurrent is optimal. Workaround: `ray stop && ray start` when switching
families. Proper fix (deferred): name the actor `cpu_scheduler_{num_cpus_per_task}` in
`_get_scheduler` + `run_program` so each family self-sizes.

---

## 2026-07-26 — `--exclude-parent` option + gen-0 seed invariant pinned + `docs/PERF_KNOBS.md`

**Built — `exclude_parent_from_context` (CLI `--exclude-parent` / `--no-exclude-parent`, default on)**
- `_build_prompt` now passes `exclude_id = parent.id if cfg.exclude_parent_from_context else None`
  (was unconditionally `parent.id`).
- **Why it matters for speed:** `exclude_id` was the *only* thing making a generation's context blocks
  differ per parent — each parent drops itself, shifting a different solution into the block. Turning
  it off, **with `--context-seed N` to pin tie-breaking**, makes the block byte-identical across
  parents, i.e. a genuine shared prefix vLLM prefills once per generation instead of once per parent.
- **Measured on `toy` (n_context=4, 2 parents):** cross-parent common prefix **50.3 % → 91.6 %** of the
  prompt; the remaining 8 % is the trailing current-solution, exactly the design intent of the
  2026-07-25 reordering. (On circle_packing at n_context=20 the default is only 2.9 %, so the headroom
  there is larger — worth ~5 % of wall at n_context=20, ~15 % at 30.)
- New startup log line states which of the three regimes is active (per-parent / shared-selection-but-
  random-ties / fully identical), since it is otherwise invisible.
- Explicitly a **science trade**, not a free win: it costs prompt diversity between a generation's parents.

**Correction to 2026-07-25's note:** I wrote that a shared block is "incompatible with the
parent-dependent strategies `per_lineage`/`mmr`". Wrong — reading the code, `_greedy_lineage` and
`_mmr_select` compute lineage relationships **among the candidates**, never against the parent.
`exclude_id` is the only parent input to any of the ten strategies, so the option works with all of them.

**Pinned — the seed program is never shown as a "past solution"** (`tests/test_context_pool.py`, 8 tests)
- Already true structurally: `_extend_context_pool` is the only writer and the loop feeds it *graded*
  children only, so a seed (never graded) cannot reach the pool → generation 0 has an empty pool and a
  blank context block. Now locked in by tests, including with `--no-exclude-parent` on (the option must
  not open a back door), plus a counterpart test that later generations *do* inject context.
- Noted: **`context.dedupe_seeds` is now dead code** — exported and unit-tested but called nowhere. It
  became redundant on 2026-07-22 when selection moved from the PUCT buffer to the all-valid pool. Left
  in place (harmless, tested, would be needed again if selection ever reads the buffer).

**Built — `docs/PERF_KNOBS.md`:** one reference table for the vLLM + `run_icl.py` throughput knobs, each
with its measured effect, and a ranked "what to do next". Leads with the framing that matters: the
workload is decode-bound with the GPU saturated at ~20 concurrent sequences, so the only single-run
lever is generating fewer tokens, and the biggest campaign lever is 2 runs sharing one server.

**End-of-day summary**
- ✅ `--exclude-parent` / `--no-exclude-parent` landed; shared-prefix effect measured (50 % → 92 %).
- ✅ Gen-0 "no seed in context" invariant pinned by tests; `dedupe_seeds` identified as dead.
- ✅ Perf knobs consolidated in `docs/PERF_KNOBS.md`; 52 tests pass.

---

## 2026-07-26 — Real reasoning/answer token split + per-candidate decode percentiles

**Bug found: we were reading the wrong field name.** vLLM's `openai_gptoss` reasoning parser emits
`message.reasoning`, **not** `message.reasoning_content` (the qwen/deepseek parsers use the latter).
`vllm_client` only checked `reasoning_content`, so `reasoning` came back `""` and the earlier
conclusion "this server doesn't expose reasoning" was **wrong** — it was exposing it all along, under
another name. Fixed with `_reasoning_of` accepting either (`REASONING_FIELDS`). Lesson recorded: dump
the raw `/v1/chat/completions` JSON before concluding a field is missing; the OpenAI SDK passes
unknown fields straight through, so a `getattr` on the wrong name is indistinguishable from absence.
Corollary: it is still unproven whether `--reasoning-parser openai_gptoss` is strictly *required* —
the pre-relaunch probe checked the same wrong field, so that comparison was invalid. Keep the flag
(it's free and it guarantees the CoT stays out of `content`).

**`reasoning_tokens` was 0 while reasoning was clearly happening.** Cause: this build omits
`usage.completion_tokens_details` entirely, so the server never reports the reasoning/answer split of
`completion_tokens`. Fixed by counting the captured text with the served model's own tokenizer via the
server's **`/tokenize`** endpoint (no local `transformers` dependency):
- `VLLMClient.count_tokens(texts)` — one request per string (the endpoint rejects lists), fired
  concurrently under its own semaphore so it never competes for generation slots. Best-effort:
  falls back to a chars÷4 estimate (marked `~… est`) and warns once.
- `ICLRunner._count_decode_tokens` counts reasoning **and** answer text per candidate, mirrors the
  per-request reasoning total back onto each `GenResult` so generation-level `_sum_usage` reports real
  tokens, and yields per-candidate `reasoning_tokens` / `answer_tokens` / `decode_tokens` for
  `events.jsonl`. If a server ever reports `reasoning_tokens` itself, that value wins and the extra
  work is skipped.
- Derived per generation: `answer_tokens` (= `completion_tokens − reasoning_tokens`, so it absorbs
  per-sequence template/special tokens) and `reasoning_share`. **Verified reconciliation:**
  17,447+8,187 = 25,634 and 12,084+12,431 = 24,515, exactly matching `completion_tokens`.
- **Measured overhead: 0.72 s for 180 calls** (90 candidates × 2) on real completion texts, i.e.
  ~0.1 % of a 740 s generation. CPU-only on the API server, off the GPU, overlapped with grading.
- The chars÷4 estimate it replaced was **~23 % low** (10,982 est vs 13,527 real); chars-per-token
  ranges 3.2–3.9 on this content, so the heuristic was biased, not just noisy.

**Per-candidate decode percentiles (`decode_p50/p90/p99/max`)** — new `icl.loop._percentiles`
(nearest-rank, no interpolation: these are token counts). Logged per generation next to the configured
cap, and written to `progress.csv` + `summary.json.decode_percentiles`. Deliberately **not** inside
`usage`, whose fields are summed into run totals — summing percentiles is meaningless.

**Why it matters (the point of the whole exercise):** a vLLM request returns only when its *slowest*
sequence finishes, so the tail sizes `--max-tokens`, not the mean. Measured in one 8-candidate
generation: decode ranged **1,009 → 6,747 tokens (6.7×)**, and `tokens_per_completion` (mean 3,204)
sat *below* `decode_p50` (3,602) — the distribution is left-skewed, so averages actively mislead here.

**Also confirmed live:** reasoning is **49–70 % of all decode** at `--reasoning-effort medium`, so the
`--reasoning-effort low` A/B is the highest-value speed experiment available. The truncation warning
fired for real (`1/6 completions hit the max_tokens=8000 cap`). Reasoning traces show the model
explicitly consuming the ICL block ("From prior solutions best sum 2.488826 … The target 2.636
better"), which is direct evidence the context is being reasoned over rather than ignored.

**Verified** — `runs/test_reasoning` (circle_packing_26, 2 gens × 2 parents × 4): reasoning + answer
files written per candidate with clean separation (no CoT in the answer file), both generations
reconciling exactly, percentiles in `progress.csv`. 54 tests pass (2 new: nearest-rank percentiles,
reasoning field-name fallback).

**Gotcha worth remembering** — with a shared Ray head up, runs must use the env that *started* it:
the head came from `src/.venv` (Python 3.12.12) while conda `phd-r2` is 3.12.11, and Ray refuses to
attach across a patch-version mismatch. Worse, `init_ray` only falls back to a private cluster on
`ConnectionError`, so the version mismatch (a `RuntimeError`) kills the run instead of degrading.
Widening that `except` is an open suggestion.

---

## 2026-07-26 (later) — Baseline plumbing: Best-of-N parent source, replicate seeds, campaign sweep files

**Built — two flags that Experiment 0 needs and the harness did not have**
- `--parent-source {puct,initial}` (`PUCTSampler.sample_initial_states`). `initial` returns fresh seed
  states every generation, so **Best-of-N** = `--parent-source initial --n-context 0`: no past
  experience via the prompt *and* none via parent selection. The buffer is still *written* during
  grading (best-so-far, context pool, `events.jsonl` stay correct) — it is only never *read* to pick a
  parent. Mirrors the branch `sample_states` already took on an empty buffer, including the AC random
  construction, so `_last_sampled_*` stays consistent for the tracker.
- `--seed` → `ICLConfig.seed` → `PUCTSampler(rng_seed=…)`. PUCT selection is a deterministic score
  sort; the sampler's *only* stochastic surface is the AC problems' random initial construction, which
  used a fresh unseeded `default_rng()` per call. Now one seeded generator per run. Also becomes the
  default for `--context-seed`. **It does not make a run bit-reproducible** — replicate-to-replicate
  variation comes from vLLM sampling at temperature 1.0, which is unseeded on purpose (a fixed request
  seed would make Best-of-N's identical per-parent prompts return identical children).
- `run_sweep.py` naming: a `grid` over `problem` no longer produces `erdos_p-erdos_s-1` — `problem` is
  already the prefix. `run_icl.py` auto log-path labels a Best-of-N run `bon` instead of inheriting the
  unused `--context-strategy` default.
- 5 sweep files: `baselines_{gptoss,qwen}.yaml` (50 runs each: 5 problems × 5 seeds × 2 baselines),
  `calibrate_{gptoss,qwen}.yaml` (2 runs × 3 gens), `smoke_sweep.yaml` (4 toy runs, ~2 min, to test
  the launcher itself). Each carries its cluster's vLLM command and the reasoning it encodes.
- 65 tests pass (11 new in `tests/test_baselines.py` + 2 sweep-naming tests). Best-of-N verified
  end-to-end on `toy`: generation 1's prompt is still the 222-token seed prompt, not the evolved
  parent, while the buffer keeps growing.

**Found — the cost of the requested design, and what actually drives it**
- Measured aggregate decode on 2×A100 for gpt-oss-120b: **~500 tok/s** (368k tokens in a 739 s,
  90-candidate generation). Cost is then purely tokens/candidate: at the measured *medium*-effort
  3.5k that's 15 h per 7,500-candidate run (~31 days for 50 runs); if `high` effort averages ~20k —
  consistent with the earlier "~70 % no_code at a 26k cap" observation — it is **~7 months**. Hence
  the calibration sweeps: 3 generations settle it before the campaign commits.
- **Why TTT-Discover's 26,000 is not our 26,000.** `TwoPhaseTokenCompleter` (their
  `tinker_utils/completers.py`): `phase1_max_tokens=26000` bounds **prompt + thinking**, and when
  phase 1 runs out *without stopping* they re-prompt with `"... okay, I am out of thinking tokens. I
  need to send my final message now."` + `<|end|><|start|>assistant<|channel|>final<|message|>` and
  let it answer inside a 32,768 window. So their runs routinely truncate thinking and **always get an
  answer**. Qwen reproduces this natively (`thinking_token_budget` forces `</think>`); gpt-oss has no
  equivalent, so a long-thinking candidate just returns nothing after burning the full budget. If
  calibration shows high truncation on gpt-oss, porting the two-phase completer is the fix.
- **CPU-family footgun, avoided by pinning:** the shared Ray head's `cpu_scheduler` is a detached
  actor created by the *first* run and partitions cores into fixed-size groups, so a queue mixing
  1-cpu (circle_packing, erdos) and 2-cpu (ac1, ac2) problems silently mis-sizes one family. The sweep
  files set `num-cpus-per-task: 2` for everything: one head stays valid for all 5 problems, at the
  price of some grading parallelism on the 1-cpu problems (irrelevant while decode-bound).

**Measured — context length vs `n_context`** (2,247 saved prompts across 39 runs, tokenized with the
served model's own tokenizer). `prompt ≈ base + n × per_solution`, base 0.2–4.9k by problem;
per-solution 1.2k (ac1) / 1.4k (erdos) / 1.7k (circle_packing) / 2.9k (ac2). The slope is **not**
constant within a run: on circle_packing it drifts 835 → 2,224 tokens/solution from gen 1 to gen 29 as
programs elaborate, so n=30 means 27k of prompt early and ~71k late (max seen 104k of a 131k window).
Late-run, `random` costs 2,537 tokens/solution against `contrastive`'s 1,796. Worst-case sizing for
the ICL runs: 4k/solution for cp/erdos/ac1 and ~8k for ac2 (the latter projected from cp's drift —
only gen-1 data exists for ac2).

---

## 2026-07-28 — Bosch login-node cgroup throttling: 11.5× slow evals, 56% false timeouts; LSF job scripts

**Problem** — `runs/ctx_qwen/cp26_n10_s1_random` (Bosch, `_meta.host = rng-dl01-login1`) recorded 673
`eval_timeout`s out of 1200 candidates (56%) against the registry's 530 s cap for `circle_packing_26`.

**Diagnosis** — paired cross-machine re-grade with `python -m sandbox.regrade`: 20 candidates
(10 SUCCESS + 10 timed-out) re-run on the INESC box `vouga`. **Bosch is ~11.5× slower per candidate**
(p50; range 5.7–17.7×). Of 8 sampled timeouts, **6 completed on `vouga`**, scoring 2.6197, 2.6113,
2.5967, 2.5717, 2.2695, 1.9101 — three of them within 2% of the best score the entire 1200-candidate
run ever found (2.6360). So the cap was discarding near-best candidates, which corrupts the PUCT
buffer's `Q(s)` and rank prior `P(s)`, not merely wasting compute.

**Root cause (confirmed)** — a cgroup on the login node caps the user slice at **5.0 cores of CPU
time** while exposing 64. `python -m sandbox.ray_doctor` on Bosch:
`/sys/fs/cgroup/user.slice/user-120321.slice/cpu.max` = 5 cores, and `cpu.stat` reports
`nr_throttled=1438673`, 5,148,273 s throttled. The quota predicts up to 12.8× at 64-way concurrency;
we measured 11.5×. **Nothing upstream reveals it**: a quota is invisible to `sched_getaffinity`
(still 64), so Ray hands out 64 one-core groups and `queue_seconds` stays ~0. The only symptom is
that every eval runs slow.

**Ruled out by experiment, do not re-test** — scipy/numpy skew (identical 1.18.0 / 2.5.1 both boxes);
contention on the grading box (paired re-grade at load 163 vs 63 → median 1.15×, and only on sub-10 s
evals where process-start and scipy-import dominate); CPU arch / BLAS kernel (Bosch EPYC 7543 →
OpenBLAS `Haswell`, vouga Xeon 6330 → `SkylakeX`; forcing `OPENBLAS_CORETYPE=Haswell` on vouga gave
p50 **1.03×**); program nondeterminism (3× repeat, 9/10 with spread exactly 0). `ray_doctor` §6 shows
the EPYC is **faster** per core than the reference (`slsqp 0.53×`) — the silicon is fine.

**Built** — `src/jobs/` with `icl_sweep.bsub` (CPU-only LSF driver job), `vllm_server.bsub` (GPU
server job that publishes its node to `jobs/vllm_host.txt`), and `README.md` covering queue
discovery and login-node → worker port-forwarding. Design notes:

* **No `-gpu` line on the driver job.** The site's queues are GPU-flavoured (`inter_v100`,
  `batch_h200`) but a queue does not force a GPU on you — `-n 32` with no `-gpu` is a 32-core CPU
  reservation. The driver reaches the server over the network by hostname, which the runs already
  did (`vllm_base_url = http://rng-dl01-w26n05:8001/v1`); only the *driver* was on the login node.
* **The Ray head is started by the job with `--num-cpus=$LSB_DJOB_NUMPROC`.** `run_sweep.py`'s
  `ensure_ray_head()` runs a bare `ray start --head`, which reads the **node's** core count rather
  than the LSF reservation — on a shared compute node that reproduces the login-node failure exactly.
  The job then passes `--ray-head require` so a failed head is loud instead of silently unsized.
* **`RAY_TMPDIR` is set per job** (`/tmp/ray_$USER_$LSB_JOBID`) with a job-derived GCS port, so two
  jobs on one node cannot collide. Verified against ray 2.56.1: `get_ray_address_file(None)` resolves
  under `$RAY_TMPDIR/ray`, so `init_ray`'s `address="auto"` still finds the job's own head.
* The job rewrites `vllm-base-url` in a **scratch copy** of the sweep yaml from `jobs/vllm_host.txt`,
  since the server's node changes on every resubmit, and leaves the tracked file alone.

**Interim fix if still on the login node** — `ray stop && ray start --head --num-cpus=5`. Costs
nothing in throughput (the 5-core quota binds either way: 5×1.0 = 64×1/12.8) but evals then run at
true speed (~30 s, not 250–540 s) and stop hitting the cap. Also clears the leaked CPU group
`ray_doctor` flagged.

**Open, unrelated to the quota** — `envs/circle_packing.py:53` rejects on
`dist < radii[i] + radii[j] - 1e-12`. Optimal packings have *tangent* circles, so the better the
solution the likelier it trips on float noise. Grading one program 3× on one machine gave
`['SUCCESS', 'invalid_result', 'SUCCESS']`, the rejection on a **6.6e-12** overlap. Raising the
tolerance to ~1e-9 would fix it but breaks score comparability with recorded runs — deliberate
decision not yet made. Relatedly, scores are not bit-portable: 5/10 re-graded candidates differed
across boxes (one by −10.58%) while being deterministic on each.

---

## 2026-08-03 — `--resume` verified against the artifacts (it was trusting one field, and that field lies)

**Problem** — a sweep hit LLM request timeouts; the damaged run folders were deleted by hand, and
`python run_sweep.py --resume <dir>` then reported those runs **complete** and skipped them.

**Diagnosis** — `--resume` (and the supervisor's pending filter) read exactly one thing:
`summary.json`'s `status`. Reproduced with a synthetic sweep — three separate ways that field lies:

1. **Deleted data, surviving summary.** Delete `generations/` + `buffer/` and the `"status":
   "complete"` left at the run root still reads as complete: the run is skipped forever. (Deleting the
   *whole* run dir did relaunch it, which is why the failure looked intermittent.)
2. **Resume rewrote the summary with only part of the run.** `ExperimentTracker.__init__` built empty
   books on a resume — `_per_gen = []`, totals 0, `best = None`, `_sol_seq = 0` — and `_write_summary`
   truncates. A run resumed at generation 12 of 15 therefore ended with a summary describing **3**
   generations and `status: complete`: the run's own record of generations 0–11 was destroyed, the next
   `--resume` would rewind to 3, and `sol_000001…` were rewritten on top of kept solutions.
3. **A generation the LLM never answered still counts.** `icl.loop._run_group` catches a failed
   request, logs a warning, calls `record_group(..., [], [])` and returns — so a generation whose
   groups all timed out is recorded as an ordinary finished generation and resume walks straight past
   it. Per the decision on this: if the LLM was unreachable, that generation is corrupt.

Evidence of (2) already on disk: `runs/ac1_gptoss/puct_s3/progress.csv` has **21 rows for 15
generations** (9,0,1,2,10,3,… — two processes interleaved into one run dir), and `gen_0009`'s mtime
(03:14) precedes `gen_0000`'s (03:35). One of them was a manual resume, the other a sweep relaunch;
its `events.jsonl`/`summary.json` happen to be self-consistent, so its analysis stands.

**Built** — `src/results/resume.py`. Every number comes from artifacts that are written **once**:

* `inspect_run()` walks generations from 0 and stops at the first that is not whole:
  `generations/gen_NNNN/meta.json` present and parseable (it is written last, by `end_generation`),
  every parent group non-empty, `len(children) == group_size`, group count == `groups_per_batch`.
  Then `buffer/context_pool.jsonl` must hold `sum(valid_candidates)` lines for the surviving prefix
  (a short pool is reloaded as-is, so resuming on it silently shrinks every later prompt), and the
  step's `buffer/puct_sampler_step_NNNNNN.json` must exist and parse. `complete` means the artifacts
  back all `num_generations`; nothing reads `status` to make a decision.
* `rewind(run_dir, keep)` moves everything from generation `keep` on into `stale_<timestamp>/`
  (nothing deleted): generation dirs, their solutions, orphan snapshots, torn `*.tmp.*` writes, and
  the tails of `events.jsonl` / `progress.csv` / `solutions/manifest.jsonl` / `context_pool.jsonl`.
* `prior_state(run_dir, keep)` rebuilds per-generation stats, usage, totals, failure types, best /
  worst_valid, the next solution id and `state_id -> sol` from the `meta.json` files plus
  `manifest.jsonl` — `ExperimentTracker(..., resume_step=N)` loads it, so a resumed run keeps ONE
  cumulative summary, its original `started_at`, and a `resumes: [...]` provenance list.

**Wired in** — `run_sweep.py --resume` now prints a decision per run (`complete … skipping` /
`resume at generation N/M` / `start over`), rewinds first, and rebuilds `--resume-step` from scratch
(a stale one used to survive into a run that had to start over). `_launch` rewinds to whatever step
its own command line claims, so a relaunch can no longer append onto the previous attempt.
`--status` shows VERIFIED generations, a `DAMAGED` state for "summary says complete, files disagree",
and spells out each run's damage under the table. `run_icl.py --resume-step` accepts `auto` (use the
verified point) and validates an explicit N against the snapshots instead of dying inside the sampler;
`_open_context_pool` now refuses a resume whose pool file is gone instead of truncating it.

**Verified** — `tests/test_resume.py` (18 cases), new sweep-level and tracker-level cases, and
`tests/smoke_icl.py`, whose stub was two interfaces out of date (`GenResult`, `cache_counters`,
`count_tokens`) and which now also runs a **resume leg** end to end. Plus a scripted end-to-end of the
reported scenario: a run whose last generation loses its group to a `TimeoutError` finishes with
`status: complete`, is caught as `resume at generation 2/3`, is rewound, and comes out verified
complete with no duplicate rows, events or solution ids.

**Not changed** — `_run_group` still swallows LLM failures and records an empty group (a run survives
a server blip and keeps its earlier generations). That is now *detected* rather than trusted, but a
generation of pure timeouts is still written; the resume path is what refuses to build on it.

**Same-day revision — a sweep resume restarts whole runs, it does not continue them.** The first cut
of `--resume` continued each incomplete run from its last verified generation. Decision: don't. A
continued run's generations are a mixture of two processes, with the interruption's cause (dead server,
throttled box, evicted job) sitting somewhere inside the search it produced and nothing in the results
saying where — for an experiment whose whole output is a comparison between arms, that is the wrong
trade against saved GPU hours. `plan_resume` now calls `rewind(run_dir, 0)` on anything not verifiably
complete and never passes `--resume-step`; the whole old dir goes to `stale_<timestamp>/`, and the
decision line says how many verified generations the restart discards (e.g. `n10_s3_random: restarting
from generation 0 (discarding 12 verified generation(s))`, ~7 h of grading). `--resume --print-cmds`
shows every such line without moving anything, so the cost is visible before committing.

The mid-run machinery stays and is still tested — `run_icl.py --resume-step N|auto` continues one run by
hand (verified point, rewind of the tail, cumulative summary via `prior_state`). It is now the explicit
per-run escape hatch rather than what a sweep does silently.

**Follow-up — `--continue-run`, the deliberate per-run mid-run resume.** `--resume` restarting whole
runs is right for a sweep but wrong for one long run that is 12 generations in: redoing those costs ~7 h
of grading. The mid-run path already existed on `run_icl.py --resume-step N|auto`, but using it meant
retyping ~20 flags by hand — the same manual-relaunch move that put two processes in
`runs/ac1_gptoss/puct_s3`. So `run_sweep.py --continue-run runs/<sweep>/<run> [--from-generation N|auto]`
now reuses the command line the manifest already records for that run, rewinds from the chosen
generation, queues that run ALONE (`supervise(..., only={name})`) and writes the sweep's full manifest
back so `--status` keeps working. `--from-generation` is refused unless that step's PUCT snapshot
survived (the error lists what did), `0` restarts just that run, and a complete run needs an explicit
generation before it will redo anything. `--resume` + `--continue-run` and `--from-generation` without
`--continue-run` are argparse errors, not silent precedence.

Also added `_check_log_path`: both planners now refuse an entry whose manifest `log_path` and command
`--log-path` disagree, which would otherwise verify one dir and relaunch into another.

---

## 2026-08-04 — TERM_MEMLIMIT post-mortem: bound the eval workers, measure the ceiling, keep the logs

**What happened.** Job 12669131 (interactive, 122 slots on `rng-dl01-w24c02`) was killed by LSF at
23:31 on Aug 3 after 28h45m and 14 generations. `TERM_MEMLIMIT`, nothing else: run time 28h against a
7-day limit, CPU efficiency 45%, zero swap. `MAX MEM 488 G` matches `rusage[mem=4096] x 122 slots` to
the byte, while `AVG MEM 78.4 G` says the memory was never needed — a 6x ramp across the run that
nobody was watching, reconstructed afterwards from `bhist` because nothing in-process logged it.

Three separate failures, fixed separately.

**1. The eval workers were never recycled.** `_get_exec_fn` built the `run_program` remote with Ray's
default `max_calls=0`, i.e. reuse a worker for the life of the job. Everything a worker touches
accumulates — unpickled `result_construction` arrays, numpy/BLAS arenas, glibc fragmentation. None of
it is a leak per-eval and none of it plateaus; ~120 workers over a multi-day run is a monotonic RSS
ramp, which is exactly the shape of 78 G -> 488 G. Now `MAX_CALLS = 1`: a worker's footprint is bounded
to one candidate. Cost is one interpreter start (~1-2 s) per eval, amortised against a whole sandboxed
program run (eval p50 is tens to hundreds of seconds), so it is noise here — that is the condition to
re-check if it ever stops holding.

Worth recording what this did *not* turn out to be. The leading external hypothesis was orphaned grader
subprocesses from `_safe_grade`'s `asyncio.wait_for`. It is the wrong file: `_safe_grade`
(`envs/base.py`) runs `_run_verification` in a `ThreadPoolExecutor` and spawns no process, its timeout
is `EnvConfig.timeout = 8000 s` against `eval_timeout = 1000 s` so it fires essentially never, and the
sandbox path that *does* spawn already sets `start_new_session=True` and does `os.killpg` TERM->KILL
plus a `pkill -P` sweep (`_kill_process_tree`). Implementing that advice would have been a no-op.

**2. Nothing measured the number LSF kills on.** This site runs `LSB_RESOURCE_ENFORCE="cpu memory gpu"`
with `LSF_PROCESS_TRACKING=Y`: LSF sums the process tree's RSS itself and kills on the total, and no
cgroup file carries that figure. So `memory.max` reads `max`, Ray falls back to `/proc/meminfo` and
sizes for the whole 1.5 T node, and Ray's own memory monitor watches that same wrong denominator and
can never fire first. New `sandbox/memwatch.py` computes the same sum LSF does and the loop logs it
every generation (`gen N memory | job rss X across N procs | Y% of the ZG ceiling`), alongside a new
per-candidate `peak_rss_mb` (from `RUSAGE_CHILDREN.ru_maxrss`, per-eval only because `MAX_CALLS = 1`)
with `rss_*` percentiles in `progress.csv` and `meta.json`. Reading the pair is the point: job climbing
while per-eval stays flat is accumulation in the long-lived processes, not a greedy candidate.

`memwatch` deliberately does *not* walk only our descendants. `ray start --head` daemonises, so the
raylet, GCS and every eval worker are reparented away from the driver — a process-tree walk misses
precisely the processes under suspicion. Under LSF it matches `LSB_JOBID` in `/proc/*/environ`, which is
the set LSF accounts; off LSF it falls back to our tree plus Ray components we own. It also sums RSS
without discounting shared pages, matching LSF's arithmetic rather than the machine's truth, because the
goal is predicting its kill decision.

At `--memory-stop-fraction` (default 0.85) the run stops at the next generation boundary instead of
being killed mid-generation: the sampler and tracker have just been flushed there, so the run stays
resumable, which a `TERM_MEMLIMIT` landing wherever it lands does not. The summary is written
`stopped_memory`, never `complete`.

**3. The one artifact that could explain a kill was always deleted.** Ray's session logs live under
`$RAY_TMPDIR` in the compute node's `/tmp`, the node is reclaimed when the job ends, and
`icl_sweep.bsub`'s cleanup ran `rm -rf "$RAY_TMPDIR"`. `ray_head.archive_logs()` now tars them into the
sweep dir — component logs (`raylet.*` carries worker deaths and object-store spilling) always, plus
the newest 400 per-worker logs. Wired into `RayHead.teardown()` *and* into `supervise`'s signal
handler, which is the path that actually matters: `TERM_MEMLIMIT` arrives as SIGINT, SIGTERM, SIGKILL
10 s apart, so the signal path writes uncompressed to fit in that window.

**Found while testing, and fixed because `MAX_CALLS = 1` makes it worse.** `/tmp/ray` on guadiana held
5.8 G across 78,803 worker logs (~75 KB each) from sessions dating to Jul 22, with `/` at 85% — nothing
reaps them, and `plan_head` gives every sweep its own temp dir. `ray_head.remove_temp_dir()` now cleans
up after teardown, guarded to refuse anything not named `ray-<uid>-<host>-<tag>` so the shared default
`/tmp/ray` (which may belong to a co-tenant) is never touched. `max_calls=1` shifts roughly the same
log bytes into many more files rather than growing the total, but the file count is what the cap and the
cleanup are for.

**Still open: no per-eval memory ceiling.** `ray.remote(memory=TASK_MEMORY)` is scheduler bookkeeping,
not enforcement, and the sandbox subprocess has no `RLIMIT_AS`, so one candidate allocating a huge array
can still take the job down. Deliberately not guessing the limit: `peak_rss_mb` is now recorded so the
cap can be read off the real distribution (~96 MB for a trivial candidate; the tail is what matters).
Set it once a real run has reported a few generations of `rss_p99`/`rss_max`.

**Same-day follow-up — the memory guard had to be made shared-head-aware.** The pieces above were
written as if a run owned its box. It does not: `run_sweep.py` starts one Ray head and every run
attaches to it, and two of the three changes behaved badly under that.

Measured on guadiana with a head up and two drivers sampling independently: each reported 1.46 G, of
which **1.39 G (95%) was the shared head and its workers**. Four concurrent runs would log 5.85 G of
"their" memory for a real 1.68 G. `job_rss_gb` is still the right number for the stop decision — it is
what the batch system kills on — but it is not per-run, so `sample()` now also reports `own_rss_gb`
(this driver's tree alone), and the `progress.csv` comment says outright that `job_rss_gb` is identical
across concurrent runs. Reading the three together localises a ramp: `job` climbing while `own` and
`rss_max` stay flat is accumulation in the shared eval workers.

The stop was worse than merely mis-attributed. All N runs measure the same total against the same
ceiling, so they would all cross 85% at the same generation boundary and all stop — losing a whole
sweep where shedding one run frees enough for the rest. `memwatch.claim_shed()` arbitrates through an
`O_CREAT | O_EXCL` claim file in the sweep dir: the first run over the threshold stops, siblings see a
fresh claim and continue (the memory it is about to release is precisely their relief), and after a
cooldown the next run over the threshold may shed too, so a still-growing job sheds one at a time
rather than all at once or only ever one. Filesystem rather than a Ray actor because a claim must
outlive the claiming run and the runs need not share a namespace. Unwritable sweep dir fails toward
stopping, since one run too few is cheaper than a `TERM_MEMLIMIT`.

Two things left standing as known, pre-existing hazards rather than new ones. `RayHead.teardown()` only
fires when `self.owned`, so a second sweep that *attaches* to the first sweep's head never tears down —
and the owning sweep finishing will `stop_head` the cluster the second is still using. That predates
this work (`af65efd`); `remove_temp_dir` now also deletes the session logs in that window, which is a
slightly larger blast radius in an already-broken case, not a new one. And `MAX_CALLS = 1` multiplies
worker *process churn* by the number of concurrent runs: worker startup happens before
`get_cpu_group()`, so it is unpinned CPU competing with pinned evals. It lands in the `grade_seconds`
gap rather than in `eval_seconds`, so the headline metric is unaffected, but the 400-worker-log archive
cap is now shared across all runs in a sweep and gives correspondingly less per-run history.

**Follow-up 2 — the two dials compose.** `--continue-run` on its own touches only the runs it names,
which is wrong when the rest of the sweep is also unfinished: continuing one run then meant the other
runs sat idle until a second command. `plan_resume`/`plan_continue_run` are now one planner,
`plan_runs(manifest, continue_at={name: "auto"|"N"}, restart_others=bool)`, and `--continue-run` is
repeatable and composes with `--resume`:

    python run_sweep.py --resume runs/<sweep> --continue-run deep:8 --continue-run other

Named runs continue at that generation, every other incomplete run still restarts from 0, complete runs
are still skipped, and the queue is supervised as one thing (`only=` the set the planner actually
queued, so nothing else can slip in). `--continue-run` takes a run dir OR — with `--resume` naming the
sweep — a bare run name, with `:N` as the general way to say the generation (`--from-generation` stays
as sugar for the single-run case, and is an error alongside `:N` or several runs). Cross-sweep paths,
duplicate names and unknown run names are all refused before anything moves.

**Same-day follow-up 2 — the log capture was dead on the exact configuration it was for.** Decision:
stop guessing at causes, run again with the instrumentation and let the next crash say what it is. That
makes the archive the load-bearing piece, so it got checked properly, and two things were wrong.

`RayHead.save_logs()` was gated on `self.owned`. But `jobs/icl_sweep.bsub` starts the head itself and
passes `--ray-head require`, so `owned` is False there and the sweep archived **nothing** — on precisely
the job script a Bosch run uses. Archiving is read-only, so it is now gated on `temp_dir` instead, the
attach branch learns the head's temp dir from `RAY_TMPDIR` (and says so, or warns that it cannot), and
`teardown()` archives *before* the ownership gate. Stopping and deleting still require ownership.

The emergency dump could not have finished either. Untruncated it was 762 MB, because the valuable
files are the big ones: `raylet.out` 296 MB, `gcs_server.out` 263 MB, `monitor.log` 89 MB. LSF gives
~10 s between SIGINT and SIGKILL. `archive_logs(tail_bytes=...)` now keeps only the last N bytes of any
oversized file, seeking and declaring the shorter size so the member is a genuine tail (verified
byte-identical on a static file; a first check against a live `raylet.out` "failed" only because the
running raylet was still appending). The signal path uses 8 MB/file uncompressed: **762 MB -> 87 MB,
19.2 s -> 1.0 s**. Truncated members are renamed `*.tail` so nobody reads one as a whole log.

What is armed for the next run: per-generation `job`/`own` RSS and per-candidate `rss_*`, a clean
resumable stop at 85% of the ceiling with sibling arbitration, and a Ray log archive on both the clean
and the signal path. What is NOT: any per-eval memory *enforcement*. `RLIMIT_DATA` is still unbuilt, the
watchdog only samples at generation boundaries (~2 h apart at this problem's pace, so a fast
mid-generation allocation is unguarded), and if `memwatch.ceiling()` returns None inside the job the
self-stop is silently inert — worth running `python -c "from sandbox.memwatch import ceiling;
print(ceiling())"` on the compute node once before trusting it.

**Follow-up 3 — a sweep that launches nothing must not report "12 ok".** On Bosch, `--resume
runs/ac2_qwen_b --continue-run n10_s2_random --from-generation 14` planned all 12 runs correctly, then
printed a status table and exited with `done: 12 ok, 0 failed` having launched none of them.

`supervise`'s queue filter still carried a pre-existing condition, `e.get("returncode") != 0`. That
return code is one number remembered from a past process, and "exited 0" is not "did all its
generations": a manifest whose entries carry 0 (a run LSF killed at a generation boundary, a relaunch
that exited early, a sweep whose runs finished before their dirs were deleted) empties the queue
completely — reproduced with a 12-entry synthetic sweep, `pending = 0`. Completeness is now decided in
exactly one place, `plan_queue()`, and only from the artifacts (`results.resume`); the return code is
not consulted at all.

Two reporting fixes so this cannot be silent again: `supervise` prints the queue it is about to run
(`queue: 12 run(s) to launch: ...`, plus which runs it skipped as verifiably complete and why), says
`nothing to launch` outright when the queue is empty, and the closing line counts what THIS invocation
did (`done: N launched, N ok, N failed`) instead of `len(entries) - failed`, which is what called a
sweep that launched nothing "12 ok".

**Follow-up 4 — the launch loop was missing from three commits, and no test noticed.** On Bosch the
sweep printed `queue: 12 run(s) to launch: ...` and then `done: 0 launched`. Cause: the commits at
`0bc5585`, `e06c985` and `d78e81e` were assembled by splitting a working tree that also held an
unrelated in-flight memory-watchdog stream, and the script that stripped that stream out cut from a
comment inside `supervise`'s `shutdown()` to "the next `_write_manifest(...)` line" — which sat far
below. That deleted `sys.exit(130)`, both `signal.signal` registrations, the `while pending or live:`
header and the process-reaping block, leaving the launch block INSIDE `shutdown()`: dead code unless a
signal arrives. The working tree was never mangled, so every local run and all 109 tests passed.

Why the tests passed: nothing in the suite ever ran `supervise`'s loop. `tests/test_sweep.py` now has
`test_supervise_actually_launches_its_queue`, which gives it a `/bin/sh -c "echo ran > marker"` entry
and asserts the marker file exists, the manifest records the pid and returncode, and the output says
`launched r (pid ...)` / `done: 1 launched`; plus a failing-run case for the reaping path. Both fail
against the mangled file.

The reconstruction now verifies structure rather than just compiling: the AST must show a `while` loop
in `supervise` containing the `_launch` and `poll` calls, and `shutdown` must still contain
`sys.exit(130)` and NOT `_launch`. It also checks cross-module: every `ray_head.X` the committed file
calls has to exist in the COMMITTED `sandbox/ray_head.py` — `teardown` had picked up calls to
`archive_logs`/`remove_temp_dir`, which live only in the uncommitted stream, so the committed tree would
have raised AttributeError on teardown.

---

## 2026-08-09 — Stop producing runs that resume cannot trust (and stop lying about them)

Follow-on to the 2026-08-03 entry. That one made `--resume` verify a run against its artifacts; this
one goes after the thing that *created* untrustworthy artifacts in the first place, plus three
smaller footguns in the launcher. Audit prompted by "is this actually ready to run on Bosch".

**The audit's finding.** `icl.loop._run_group` caught every exception — LLM *and* grading — recorded
that parent group with no candidates, and let the loop walk on. So a single transient failure produced
a run that finished all its generations while every generation after the failure was conditioned on a
buffer and a context pool missing that group's children: the configured experiment on paper, a
different one in fact. `results.resume` then (correctly) refused to trust anything past it, so a run
with 10 generations on disk reported `good_generations = 2`, `complete = False`, and `--resume` threw
all ten away. Reproduced against synthetic run dirs before changing anything.

**Built**
- `_generate_waiting` (`icl/loop.py`): a layer above the OpenAI client's own `max_retries`, for the
  failures those cannot cover — the vLLM job being requeued, a node reboot, a restart to change flags.
  Backs off 5s → 60s, re-probes `/health` so the log distinguishes a dead server from a sick one, and
  **the generation does not advance while it waits**. New `--llm-max-wait` (default 3600s, 0 = wait
  indefinitely).
- `GenerationAborted` / `LLMUnavailable`: a group that cannot be completed now abandons the
  generation instead of recording it hollow. Raised only after every group has settled
  (`asyncio.gather(..., return_exceptions=True)`) so no sibling group is left generating into a
  directory the process is abandoning, and always **before** `end_generation`, so no `meta.json` is
  written — which is exactly what makes the stop resumable at the last complete generation.
- Grading failures are now their own error. A dead Ray worker used to surface as *"the LLM was
  unreachable for them"*, and retrying it would have paid for the decode twice.
- `guard_live_runs` (`run_sweep.py`): a launch or resume that would rewind a run whose process is
  still alive asks first, and **refuses outright with no terminal to ask on** (every batch job).
  `--force` overrides. Needed because `plan_runs`/`plan_queue` consulted `_alive` nowhere.
- PID identity: `_alive` now compares `/proc/<pid>/stat` field 22 against a `pid_start` recorded at
  launch. A manifest outlives the job that wrote it, and on a busy node a dead run's pid belongs to
  something else by the time anyone reads it — `--stop` would have signalled that stranger's group.
- `--stop` halts, rather than advancing, the sweep (see below).
- Snapshot verdicts cached by `(path, mtime_ns, size)` in `results.resume`.

**Design decisions & why**
- **Abort at the point of damage, not at resume time.** If a compromised generation compromises the
  run — it does — then the moment to act is that generation, not twelve generations later when someone
  tries to resume. The old order spent the rest of the run's budget producing data the resume logic
  had already decided was worthless. As a side effect the strict prefix rule in `results.resume` stops
  having to adjudicate anything: a run that aborts on damage leaves a short, honest, undamaged prefix.
- **Wait, don't skip.** A requeued server costs wall clock; a skipped group costs the comparison.
- **`--stop` writes a marker file, not a manifest field.** It only SIGTERMed the live runs, and the
  supervisor reaped them and immediately filled the freed slots — so `--stop` on a 12-run sweep at
  `max_parallel 2` killed two runs and started two more. `<sweep_dir>/.halted` goes down *first*
  (signalling before halting loses the race), `supervise` drops its queue on it, and a deliberate
  relaunch clears it. A field inside `sweep.json` would race with the supervisor's own writes.
- **Report what is discarded, not what verifies.** `--resume` printed `discarding 2 verified
  generation(s)` while moving ten aside. `RunProgress.generations_on_disk` is the number that
  belongs in that sentence.

**What bit us / measured**
- A run that completed 10/10 generations, damaged at gen 2, was reported `2/10` and re-run from
  scratch by `--resume`; `--continue-run auto` continued at 2 (discarding 8); `--continue-run r:10`
  was rejected outright. There was no non-manual way to keep it.
- `--status` re-parsed every buffer snapshot of every run on its 60s timer, inside the same loop that
  reaps and launches runs: 268 MB re-read per render on a synthetic 12-run × 15-generation sweep
  (0.94 s warm on local disk, and these run dirs are routinely on NFS). Snapshots are immutable once
  written, so the re-parse could only ever reach the same verdict — hence the cache.
- Checked and cleared: PUCT cannot return fewer parents than `groups_per_batch` (the initial states
  are never evicted — `flush` keeps parentless states, `_finalize_and_save` keeps initials), so the
  "N groups on disk, M configured" damage check has no false-positive path through normal search.
  Simulated six generations with a single productive parent to confirm.

**Tests**
- New `tests/test_llm_loss.py` (5): a transient failure is waited out and leaves no damage; a 404 is
  not waited on; giving up raises rather than recording an empty group; a lost generation is
  abandoned and `inspect_run` puts the resume point at the previous one with `status: aborted`; a
  grading failure reports as grading and is not re-generated.
- `tests/test_sweep.py` +6: recycled-pid rejection, the live-run refusal and its `--force` escape, a
  stale pid ignored, `--stop` halting a queue, halt/clear idempotence, honest discard counts.
- `tests/test_resume.py` +1: snapshot verdicts cached, and a rewritten file misses the cache.
- 123 pass.

**Not done (deliberately deferred):** the batch-job side. `jobs/icl_sweep.bsub` still has no way to
pass `--resume` / `--continue-run`, so resubmitting it after a wall-clock kill takes the plain-launch
path and restarts every partial run from generation 0. That is the next thing to fix for Bosch.

---

## 2026-08-09 (later) — Choosing the head's core count by hand (`sweep.ray.num_cpus` / `--ray-num-cpus`)

`plan_head` detected everything and offered no way to say "use less than that". The cases are real:
sharing a compute node with another job of your own, leaving headroom for a server on the same host,
or reproducing a run made on a smaller box. Three spellings, one meaning: `sweep.ray.num_cpus` in the
sweep file, `run_sweep.py --ray-num-cpus N` (which overrides it, both in `--print-cmds` and in the head
actually started — one `_ray_cfg()` helper feeds both, so the plan you read is the head you get), and
`run_icl.py --ray-num-cpus N` for the private cluster a lone run boots when no head is up.

**It is the FINAL `--num-cpus`, not another budget.** `reserve_cpus` is not subtracted on top and the
detected budget is not consulted: "let Ray see 16" has to mean 16 or it does not answer the question
being asked. The one adjustment kept is rounding down to a whole multiple of `num-cpus-per-task`,
because `cpu_scheduler` drops the trailing partial group and a `--num-cpus` it cannot back is capacity
Ray will admit against and then strand.

**Two different ceilings, two different failures, so both are warned about separately.** Above the
affinity mask (`num_cpus > logical CPUs`) the run *starves*: `cpu_scheduler` chops the mask, not
`--num-cpus`, so it can only ever hand out `affinity // tpt` groups and the surplus tasks spin in
`get_cpu_group()` until they fail as `cpu_starvation`. Between the physical cores and the logical CPUs
nothing starves — evals just share cores at roughly half speed while Ray reports full occupancy, which
silently moves runtimes and timeout rates. Neither is refused: an explicit override wins, as
`memory_gb` already does.

**Memory follows the request, not the grant.** In the no-cgroup-limit branch the heap is a fair share
of `MemTotal` computed from the core budget; taking 16 of 128 cores and still claiming the whole
node's RAM is precisely the co-tenant-hostile case that branch exists to prevent, so the share is now
computed from the override when there is one.

**Where it deliberately does nothing.** `--num-cpus` is fixed at `ray start` and a connecting client
cannot change it, so on `--ray-head require`/`skip` (and in `init_ray`'s attach path) the value is
reported as ignored, with the `ray start --head --num-cpus=N` command to use instead. Silently
accepting it would let someone size a sweep carefully and never learn the number was dropped.

**Found while doing it:** the `sweep.ray` block was never key-validated. `plan_head` reads it with
`cfg.get()`, so `num_cpu:` or `reserve_cpu:` was accepted and ignored — the sweep then ran at a size
nobody chose, indistinguishable from not having set it. `build_specs` now rejects unknown keys there
against `RAY_SETTINGS`, like every other section.

**Tests.** New `tests/test_ray_head.py` (9) — the first tests this module has had — running against a
faked 64-core/128-thread/256 G box so the assertions mean the same thing on a laptop, on guadiana and
inside an LSF job: the override is honoured exactly with no reserve applied, it beats `reserve_cpus`,
it rounds to whole slots, below one slot it raises, each of the two ceilings produces its own warning,
the memory share follows it, and `memory_gb` still wins. `tests/test_sweep.py` +2 (flag beats yaml
without mutating the manifest; `sweep.ray` typos rejected). 134 pass.

**Same-day follow-up — the grading-side twin, found by asking "what else can OOM or crash".** The
change above stops a run when the model is unreachable. The same silence existed one layer down, and
it was worse, because it survived every check `results.resume` makes.

**A run in which every candidate FAILS TO GRADE verifies as `complete`.** Demonstrated on a synthetic
run dir: 15 generations, full groups, full complement of children, `correctness: 0` throughout →
`good_generations 15/15`, `complete: True`, `damage: []`, zero valid solutions ever produced.
`--resume` skips it and `--status` shows it green. Every check in that module is *structural*, and a
broken evaluator is structurally indistinguishable from a hard problem: the candidates came back,
they just all failed.

There is a mechanism that produces exactly this, and it is not hypothetical. `release_cpu_group` sits
in a `finally` (`sandbox_reward_evaluator.py`), which covers exceptions but **not the worker process
being killed** — Ray's memory monitor, the OOM killer, node pressure, i.e. precisely the conditions of
the 2026-08-04 memory ramp. `CpuScheduler` is a *detached* actor holding a fixed deque built once per
host, with no reaper, so every killed eval permanently removes one CPU group for the life of the Ray
head — which the sweep owns across all its runs, not per run. With `num-cpus-per-task: 1` on a 32-slot
job that is 32 losses from total starvation, after which `get_cpu_group` times out for everything and
every candidate is recorded as a failure. Still open: the scheduler should hand out *leases* with an
owner and a TTL and reclaim groups whose holder died, instead of trusting the borrower to return one.

**Built**
- `--max-empty-generations` (default 3, 0 disables): N consecutive generations with no valid candidate
  stops the run, `status: stopped_no_yield`, with the generation's dominant `failure_type`s in the log
  line — `cpu_starvation` vs `timeout` vs `invalid_result` is what separates the causes. Above 1 on
  purpose: an early generation can legitimately come back empty before the buffer has anything good.
- `results.resume` now refuses `complete` for a run whose whole history produced zero valid solutions,
  and says so in `damage`. Belt and braces — the loop-side stop cannot help the run dirs already on
  disk, or a run launched with the stop disabled.
- `_check_token_budget` in `run_sweep.check_server` (warning only, matching that function's contract).

**Why the token-budget warning.** `max-context-tokens: 94000` + `max-tokens: 34000` = **exactly** the
`--max-model-len: 128000` the sweeps' own server command uses. Two things then have nowhere to go:
`max_context_tokens` bounds only the CONTEXT BLOCK — `problem_intro()` and the rules/current-solution
tail (which renders the parent's whole program) are added on top — and `build_context_block` trims by a
**chars/4 estimate**, which under-counts code. Overflow is a 400, and since the LLM change classifies
400 as permanent, a 400 now stops the run. The warning reads `max_model_len` off the model card
`/v1/models` already returns and fires when under 8000 tokens are left for intro + tail + template.

**Considered and rejected: stripping completion text from `_gen_results`.** It holds every
completion's full text for a generation purely to feed `_sum_usage`, which only needs the counters —
~300 MB in the ac1 Best-of-N arms, which are still `75 x 16` in ONE generation (the cp26 files were
reshaped to `6 x 16 x 25`; `sweeps/ac1_*.yaml` and the three `ac1_gptoss.yaml` copies under the problem
folders were not). Against a ceiling that was 488 G, 300 MB is noise. The real problem with that shape
is that `--memory-stop-fraction` is only checked at a generation boundary, so a single-generation run
has no memory brake at all and no partial recovery — reshaping those arms fixes the brake and the
buffer together, and is the change to make if it ever matters.

**Also noted, not fixed:** the context pool is unbounded (every valid solution, in RAM and mirrored to
JSONL, re-read wholly on resume) and AC states carry `result_construction` with
`construction_length_limits = (1000, 100000)`, so one state can be ~3 MB of Python floats; and the
sandbox subprocess still has no `RLIMIT_AS` (open since 2026-08-04 — `peak_rss_mb` is being recorded so
the cap can be read off a real distribution).

**Tests** — `tests/test_llm_loss.py` +3 (the stop fires, it can be disabled, a recovered streak is
forgiven), `tests/test_resume.py` +2 (a barren run is never complete; a run with any yield is judged
normally), `tests/test_sweep.py` +3 (the budget warning, its silence with headroom, and its silence
when the server reports no window). 142 pass.

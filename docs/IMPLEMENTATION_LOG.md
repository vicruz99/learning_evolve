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

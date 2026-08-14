# learning-evolve ICL harness

Test-time **in-context-learning (ICL)** harness for open-ended discovery. It reuses
TTT-Discover's search machinery — a solution **buffer** + **PUCT** parent selection — and its
**sandboxed evaluators** for the math benchmarks, but replaces the RL head (tinker generation +
GRPO/LoRA training) with a **frozen-model, in-context** generate-and-score loop that talks to a
local **vLLM OpenAI-compatible** server.

The relevant TTT-Discover code is **vendored** here (not imported) so the package is free of
`tinker` / `torch` / `transformers`. Dependency footprint: `ray + numpy + scipy + cvxpy + openai`.

## Layout

- `puct/`       — shared search/buffer: `State`, `PUCTSampler` (reused by ICL, later RL/SFT).
- `sandbox/`    — sandboxed code execution: reward evaluators + ray `CpuScheduler` + `init_ray`.
- `envs/`       — problem definitions (erdos, circle_packing, ac1/ac2) + slim `Environment` base
                  + `registry`. Prompts and initial solutions match TTT-Discover verbatim.
- `generation/` — vLLM OpenAI-compatible client.
- `context/`    — ICL context-selection strategies (10 presets / 4 engines; see `../docs/strategies/`).
- `icl/`        — the ICL search loop + config; `run_icl.py` entrypoint.

## Provenance

Vendored from `../discover` (TTT-Discover, MIT). Files kept close to upstream; only imports were
repointed and the tinker-coupled `Experience` class / `step()` glue removed. See the plan at
`.claude/plans/` for the exact vendoring boundary.

## Setup

```bash
# Python 3.11 or 3.12 (not 3.13 — ray/cvxpy wheels)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

**1. Start a local vLLM server** (raise `--max-model-len` to fit the ICL context):

```bash
cd projects/phd/R2/LLMs/local/vllm_provider/
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/scratch/vicstorage \
vllm serve openai/gpt-oss-120b \
    --async-scheduling \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 131000 \
    --max-num-seqs 256 \
    --port 8001
```

**2. Run ICL** (point `--vllm-base-url` at the server's port):

```bash
python run_icl.py --problem circle_packing_26 --context-strategy best --n-context 30 \
    --groups-per-batch 5 --group-size 12 --num-generations 30 \
    --model openai/gpt-oss-120b --vllm-base-url http://localhost:8001/v1
```

`--dry-run` builds and prints one assembled prompt (base question + context block) and exits — no
server/Ray needed. A copy-paste log of the actual experiment commands lives in `docs/ICL_RUNS.md`.

### Running several experiments at once (shared Ray server)

By default each `run_icl.py` starts its **own** private Ray cluster. Launching several at the same
instant is then a problem: each one grabs all CPU cores (oversubscribing the box), and their Ray
heads can deadlock while booting simultaneously. To run experiments concurrently, start **one shared
Ray head first** — same idea as the vLLM server: launch it once, every run connects to it.

```bash
cd .../src
.venv/bin/ray start --head --disable-usage-stats   # once per session, BEFORE launching runs
# ... now launch as many run_icl.py as you like, even all at once ...
.venv/bin/ray stop                                 # when you're done (optional)
```

Nothing else to do — `run_icl.py` **auto-detects** the shared head and connects to it (via
`address="auto"` in `sandbox/ray_setup.init_ray`; no `RAY_ADDRESS` to export). If no head is running
it silently falls back to a private cluster, so single runs work exactly as before. With one shared
head all experiments draw from a single 96-CPU pool, so Ray caps *total* grading across every run
(no core oversubscription) and simultaneous launches no longer race.

**Caveat — mixing problem families.** The shared CPU scheduler is sized by the *first* run's
cpus-per-task (`circle_packing`/`erdos`/`toy` = 1, `ac1`/`ac2` = 2). Running the *same* family
concurrently is optimal. If you switch between a 1-cpu and a 2-cpu family on the same head, restart
it (`.venv/bin/ray stop && .venv/bin/ray start --head`) so the scheduler re-sizes.

### Sweeps: `run_sweep.py` (several coordinated runs from one file)

For anything beyond one or two runs, drive them from a **sweep file** instead of separate shells:
shared settings in one place, per-run overrides, staggered launches, a bounded queue, and a single
status table. It starts the shared Ray head for you if none is up.

```bash
tmux new -s sweep                                  # the supervisor must stay alive (see below)
python run_sweep.py sweeps/ctx_strategies.yaml
```

```yaml
# sweeps/ctx_strategies.yaml — keys are run_icl.py long flags without the leading `--`
sweep:
  name: ctx_strategies       # -> runs/ctx_strategies/<run>/  (and that dir's own index.csv)
  max_parallel: 2            # runs in flight at once
  stagger: 120               # seconds between launches
  server_max_num_seqs: 256   # optional: only used to warn about oversubscribing the server

common:                      # applies to every run
  problem: circle_packing_26
  groups-per-batch: 6
  group-size: 15
  num-generations: 30
  reasoning-effort: medium
  vllm-base-url: http://localhost:8001/v1

grid:                        # cross-product; run names derive from the varying keys
  context-strategy: [best, random, best_worst, contrastive]

runs:                        # optional explicit entries, each overriding `common`
  - name: cp26_best_n30
    context-strategy: best
    n-context: 30
```

| Command | What it does |
|---|---|
| `run_sweep.py FILE` | Expand, preflight-check, launch, and supervise the queue. |
| `run_sweep.py FILE --print-cmds` | Print the exact `run_icl.py` commands; launch nothing. |
| `run_sweep.py --status DIR` | Status table (works any time, even after the supervisor exits). |
| `run_sweep.py --resume DIR` | **Restart** every run that is not complete, from its first generation. |
| `run_sweep.py --resume DIR --print-cmds` | Show what a resume would decide and launch nothing (also moves nothing). |
| `run_sweep.py --continue-run DIR/RUN` | Continue **one** run mid-run, at the last generation its files can back; the sweep's other runs are left alone. |
| `run_sweep.py --continue-run DIR/RUN --from-generation N` | Same, at a generation you name (`0` restarts just that run). |
| `run_sweep.py --resume DIR --continue-run RUN[:N]` | Resume the whole sweep, but let the named run(s) continue mid-run instead of restarting. Repeatable. |
| `run_sweep.py --stop DIR` | `SIGTERM` every live run of the sweep. |

**What "complete" means** — never `summary.json`'s `status` field: that survives the deletion of
everything it describes, and an older resume could rewrite it with only part of the run.
`results/resume.py` counts a generation as done only when its `generations/gen_NNNN/meta.json` is
present *and* every parent group in it recorded a full `group_size` of candidates (a group with none
means `icl.loop` swallowed a failed LLM request), the context pool holds the solutions those
generations produced, and the step's PUCT snapshot loads. Inspect one run dir on its own with
`python -m results.resume <run_dir>`.

**A sweep resume is whole-run granular.** Any run that is not verifiably complete starts over: its
whole dir is moved to `<run>/stale_<timestamp>/` (nothing is deleted) and it is relaunched from
generation 0. Continuing mid-run is cheaper, but it makes a run's generations a mixture of two
processes with the interruption's cause — dead server, throttled box, evicted job — sitting somewhere
inside the search it produced, and nothing in the results saying where. `--resume --print-cmds` prints
how many verified generations each restart discards before you commit to it.

**When that cost is too high, exempt the named run(s).** `--continue-run` reuses the exact command line
the sweep recorded for a run (no flags to retype), picks up at the last generation its files can back,
and moves generations from there on to `stale_<timestamp>/`. It composes with `--resume`, which is how
you continue one expensive run *and* keep the rest of the sweep going in the same queue:

```bash
# just that run; the sweep's other unfinished runs are not touched or launched
python run_sweep.py --continue-run runs/ac1_gptoss_rest/n10_s3_random --print-cmds   # decide, move nothing
python run_sweep.py --continue-run runs/ac1_gptoss_rest/n10_s3_random                # 12/15 -> continue at 12
python run_sweep.py --continue-run runs/ac1_gptoss_rest/n10_s3_random --from-generation 7

# the whole sweep, but n10_s3_random continues instead of restarting (run NAMES here, `:N` optional)
python run_sweep.py --resume runs/ac1_gptoss_rest --continue-run n10_s3_random
python run_sweep.py --resume runs/ac1_gptoss_rest --continue-run n10_s3_random:8 \
                                                  --continue-run n10_s3_best
```

With `--resume`, every run not named continues to follow the restart rule (incomplete → generation 0,
complete → skipped), and all queued runs share one supervisor, `max_parallel` and `stagger`.

`--from-generation N` must name a generation whose PUCT snapshot survived (`buffer/puct_sampler_step_<N>.json`);
if it did not, the error lists the steps that are available. `0` restarts that one run. The continued
run keeps one cumulative `summary.json` — totals, best-so-far and solution numbering carry on — and
records the resume in `summary.json`'s `resumes[]` and `config.json`'s `_meta.resumes`, so a run built
by more than one process says so. `run_icl.py --resume-step N|auto` does the same for a run that no
sweep manifest knows about.

```
run                  pid       state     gens   best    gen wall  tok/s  updated
cp26_cs-best         41207     running   7/30   2.6312  742s      658    12s ago
cp26_cs-random       41455     running   6/30   2.6109  751s      641    31s ago
cp26_cs-best_worst   -         complete  30/30  2.6350  738s      -      2h ago
cp26_cs-contrastive  -         DIED      3/30   2.5904  744s      -      1h ago
```

`DIED` means the manifest says the run should be alive but its pid is gone — the failure mode that
was previously invisible. Overrides: `--max-parallel`, `--stagger`, `--refresh`, `--sweep-dir`,
`--ray-head {auto,require,skip}`.

**Notes**
- Keys are validated against `run_icl.py`'s real parser, so a typo (`n-contexts`) or an
  inexpressible bool fails *before* anything launches, with a suggestion. `log-path` and
  `resume-step` are owned by the launcher and rejected.
- Booleans take `true`/`false` and map to the right flag (`include-code: false` →
  `--no-include-code`).
- Runs are started in their own process session, so killing the supervisor (or losing the terminal)
  does **not** kill them — but the queue stops advancing, which is why long sweeps want `tmux`.
- Each run writes to `runs/<sweep>/<run>/`, so `results/analysis.py` pointed at `runs/<sweep>` sees
  exactly that sweep's runs.

### Options (`python run_icl.py --help` for the full list)

**Problem / output**
- `--problem` *(required)* — one of the registered problems (`circle_packing_26/32`, `ac1`, `ac2`, `erdos_min_overlap`, `toy_ee`).
- `--log-path` — output dir (default `runs/<problem>_<strategy>_n<ctx>_g<gs>x<gpb>_<timestamp>`).
- `--num-generations` (50) — number of search generations.
- `--resume-step N|auto` — continue an interrupted run: point `--log-path` at the existing run dir. `auto` uses the last generation that run's own files can back (see `results/resume.py`); an explicit `N` is checked against the buffer snapshots (`buffer/puct_sampler_step_<NNNNNN>.json`) and refused with the available steps if it cannot be loaded. Either way the run reloads that snapshot plus `buffer/context_pool.jsonl`, and anything after the resume point is moved to `<log-path>/stale_<timestamp>/` so the relaunch does not write on top of the attempt it replaces. Totals, best-so-far, solution numbering and `summary.json` continue rather than restarting.

**Model / server**
- `--model` (`openai/gpt-oss-120b`), `--vllm-base-url` (`http://localhost:8000/v1`).
- `--reasoning-effort` (`high`; `none` to disable — for non-gpt-oss models).
- `--thinking-token-budget` (default `None` = uncapped) — Qwen3 only: cap on reasoning tokens; vLLM forces `</think>` once hit. Needs the server launched with `--reasoning-parser qwen3`.
- `--no-thinking` — Qwen3 only: disable thinking entirely (passes `enable_thinking=false`); just add the flag, no value.
- `--temperature` (1.0).
- `--max-tokens` (26000) — max tokens the model may **generate per completion** (reasoning + answer combined); a candidate that needs more is truncated.
- `--max-gen-concurrency` (8) — max generation requests in flight to the vLLM server at once. Effective generation parallelism is `min(groups_per_batch, max_gen_concurrency)`, so keep it ≥ `groups_per_batch` to fire all parents' requests together.

**Search shape** — `groups_per_batch × group_size` candidates per generation
- `--groups-per-batch` (8, = parents/gen), `--group-size` (64, = children/parent).

**PUCT**
- `--puct-c` (1.0) — exploration coefficient `c` in the PUCT score `Q + c·scale·P·√(1+T)/(1+n)`; higher = more exploration of under-visited states.
- `--max-buffer-size` (1000) — max states kept in the search buffer.
- `--topk-children` (2) — children per parent retained in the buffer on flush.
- `--parent-source` (`puct`) — **which solution the prompt hands the model as "the current solution to improve upon"**. The buffer is written in every mode — best-so-far, the context pool and `events.jsonl` stay correct — the modes differ only in what is *read* out of it and what reaches the prompt.

  | value | parent given to the model | what it isolates |
  |---|---|---|
  | `puct` | PUCT-selected from the buffer | TTT-Discover's search (the default) |
  | `initial` | always the problem's seed | **Best-of-N**; with `--n-context 0`, the no-past-experience baseline |
  | `best` | always the buffer's best-so-far | **greedy hill-climbing** — same prompt shape as `puct`, so the gap between them is exactly what PUCT's exploration term (under-visited states, lineage spreading) buys |
  | `none` | **no solution at all** | the prompt's "improve upon this" framing is removed and only the objective + target remain, so past experience reaches the model **through the context block and nothing else** — the mode for measuring a context strategy on its own. With `--n-context 0` it is a from-scratch zero-shot arm. |

  `none` is a prompt setting, not a search variant: children are attributed to the seed (as in `initial`), and the evaluator still receives that state, so the constructions the sandbox pre-imports (`height_sequence_1`, `initial_h_values`) are unchanged. With `best`, every slot of a generation gets the same parent, so a generation adds `topk_children` states to the buffer rather than `groups_per_batch × topk_children` (best-so-far always survives; the context pool is unaffected).

  Run-directory naming follows: `initial` → `bon`, `puct` → the bare strategy name, `best`/`none` → `<source>_<strategy>`.

**Reproducibility**
- `--seed N` — replicate seed, recorded in `config.json`. Seeds the sampler's only stochastic surface (the random initial construction of `ac1`/`ac2`) and, unless `--context-seed` is given, the `random` context strategy. It does **not** make a run bit-reproducible: PUCT selection is a deterministic score sort, and replicate-to-replicate variation comes from vLLM sampling at `temperature 1.0`, which is deliberately left unseeded (a fixed request seed would make Best-of-N's identical per-parent prompts return identical children).

**Context selection** (which past solutions enter the prompt; see `../docs/strategies/`)
- `--context-strategy` (`best`) — `random`, `best`, `recent`, `biggest_jump`, `best_worst`, `best_jump`, `per_lineage`, `best_diverse`, `informative`, `contrastive`. Use `--n-context 0` for the **no-ICL baseline**.
- `--n-context` (32) — number of past solutions in context (**the main hyperparameter**).
- `--max-context-tokens` — hard cap on the context block (chars/4 heuristic; trims lowest-ranked first).
- Strategy knobs (read only by the strategies that use them):
  - `--mix-fraction` (0.5) — fraction of `n_context` filled from the primary ("best") pool; the remaining `1 − x` comes from the secondary pool (worst / biggest-jump / low-scoring). Used by `best_worst`, `best_jump`, `per_lineage`, `contrastive`.
  - `--mmr-lambda` (0.7) — MMR quality↔diversity trade-off (1 = quality only, 0 = spread only). Used by `best_diverse`, `informative`, `contrastive`.
  - `--jump-alpha` (0.5) — `informative` only: how much the MMR quality term weights absolute value (`alpha`) vs. improvement-over-parent/"jump" (`1 − alpha`).
  - `--context-seed` — seed for the `random` strategy and for equal-score tie-breaking (reproducibility).
- `--exclude-parent` (default on) / `--no-exclude-parent` — whether a parent is dropped from its own context block. On, each parent never sees itself listed as a past solution (it is already rendered once as the current solution in the prompt tail) — and because each parent drops a *different* solution, the block differs per parent. Off **plus `--context-seed N`** makes the block byte-identical for every parent in a generation, so vLLM prefills it once per generation instead of once per parent (~5 % at `n_context=20`, more at 30) at the cost of prompt diversity between parents. See `../docs/PERF_KNOBS.md`.

> The **seed program is never shown as a past solution**: the context pool only ever receives *graded*
> solutions, so generation 0 has an empty pool and a blank context block regardless of these flags
> (locked in by `tests/test_context_pool.py`). From generation 1 on, everything valid is fair game.

**Rendering** (orthogonal to selection) — `--include-code`/`--no-include-code`, `--include-strategy` (show each solution's `<strategy>` block; `--no-include-code --include-strategy` = strategy-only).

**Eval / misc**
- `--eval-timeout`, `--num-cpus-per-task`, `--grade-timeout` (8000).
- `--save-completions` (on) / `--no-save-completions` — a *completion* is one candidate's full raw LLM output text (reasoning + `<strategy>` + code block) before parsing; saving keeps them per candidate for inspection, `--no-save-completions` skips them for smaller runs.
- `--save-reasoning` (on) / `--no-save-reasoning` — write each candidate's `reasoning_content` to `child_NN.reasoning.txt` (separate from the completion file, which must stay the raw answer text). Requires the server to expose reasoning separately — launch vLLM with `--reasoning-parser openai_gptoss` for gpt-oss, else the field comes back empty and only the token *counts* are recorded.

### Token accounting

Every generation logs exactly where the compute went, and the same numbers land in `progress.csv`
(new columns), `summary.json` (`usage`, per generation and run-total), and `events.jsonl`
(`finish_reason`, `reasoning_chars` per candidate):

```
gen 3 tokens | prompt 104,821 (91% cached) | decode 486,220 (5402/completion, 657 tok/s) | truncated 2/90
```

- **`truncated`** counts completions with `finish_reason == "length"` — they burned the whole
  `--max-tokens` budget, usually emit no code block, and gate their request's return. A warning fires
  whenever it is non-zero; it is the signal for tuning `--max-tokens` / `--reasoning-effort`.
- **`% cached`** is the prefix-cache hit rate, scraped from the server's `/metrics` at generation
  boundaries — some vLLM builds report `usage.cached_tokens = 0` even with caching on. Two caveats:
  the counters are **server-global** (they mix runs sharing one server) and counted **per sequence**,
  so only the ratio is meaningful — never divide `cache_hits` by `prompt_tokens`.
- **`wall_seconds`** per generation is also recorded, so `tok/s` and per-generation cost are
  reconstructable after the fact.
- `--log-level` (`INFO`), `--dry-run`.
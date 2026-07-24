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

### Options (`python run_icl.py --help` for the full list)

**Problem / output**
- `--problem` *(required)* — one of the registered problems (`circle_packing_26/32`, `ac1`, `ac2`, `erdos_min_overlap`, `toy_ee`).
- `--log-path` — output dir (default `runs/<problem>_<strategy>_n<ctx>_g<gs>x<gpb>_<timestamp>`).
- `--num-generations` (50) — number of search generations.
- `--resume-step N` — resume an interrupted run: point `--log-path` at the existing run dir and set `N` to the generation to restart from (the sampler reloads that step's buffer snapshot, `buffer/puct_sampler_step_<NNNNNN>.json`, and continues).

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

**Context selection** (which past solutions enter the prompt; see `../docs/strategies/`)
- `--context-strategy` (`best`) — `random`, `best`, `recent`, `biggest_jump`, `best_worst`, `best_jump`, `per_lineage`, `best_diverse`, `informative`, `contrastive`. Use `--n-context 0` for the **no-ICL baseline**.
- `--n-context` (32) — number of past solutions in context (**the main hyperparameter**).
- `--max-context-tokens` — hard cap on the context block (chars/4 heuristic; trims lowest-ranked first).
- Strategy knobs (read only by the strategies that use them):
  - `--mix-fraction` (0.5) — fraction of `n_context` filled from the primary ("best") pool; the remaining `1 − x` comes from the secondary pool (worst / biggest-jump / low-scoring). Used by `best_worst`, `best_jump`, `per_lineage`, `contrastive`.
  - `--mmr-lambda` (0.7) — MMR quality↔diversity trade-off (1 = quality only, 0 = spread only). Used by `best_diverse`, `informative`, `contrastive`.
  - `--jump-alpha` (0.5) — `informative` only: how much the MMR quality term weights absolute value (`alpha`) vs. improvement-over-parent/"jump" (`1 − alpha`).
  - `--context-seed` — seed for the `random` strategy (reproducibility).

**Rendering** (orthogonal to selection) — `--include-code`/`--no-include-code`, `--include-strategy` (show each solution's `<strategy>` block; `--no-include-code --include-strategy` = strategy-only).

**Eval / misc**
- `--eval-timeout`, `--num-cpus-per-task`, `--grade-timeout` (8000).
- `--save-completions` (on) / `--no-save-completions` — a *completion* is one candidate's full raw LLM output text (reasoning + `<strategy>` + code block) before parsing; saving keeps them per candidate for inspection, `--no-save-completions` skips them for smaller runs.
- `--log-level` (`INFO`), `--dry-run`.
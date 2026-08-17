# TTT-Discover math benchmarks for ShinkaEvolve

These examples port the five math-discovery problems used by **TTT-Discover**
(`discover/examples/`) into the ShinkaEvolve evolutionary-search framework, so the two
systems can be compared on the **same problems** with the **same initial solution** and the
**same domain tips** (see the project `PROJECT_CONTEXT.md` §4.5 and §5).

Problems: `circle_packing_26`, `circle_packing_32`, `erdos_min_overlap`, `ac1`, `ac2`.

## Parity principle

The **initial solution** (seed program + seed data construction) and the **domain
tips/tricks** shown in the prompt are kept **identical** to TTT-Discover. Only *harness
mechanics* differ, because they are framework-specific:

- TTT-Discover asks the model to return code between ` ```python ` fences, forbids
  lambdas/closures, names a specific entrypoint, etc. ShinkaEvolve instead uses
  EVOLVE-BLOCK markers + diff patches and appends its own format instructions automatically.
- So each `run_evo.py` `task_sys_msg` carries the TTT **domain** guidance verbatim (problem
  statement, shown evaluator, "Consider…" bullets / literature blurbs, discretization,
  targets, construction hints) but drops the TTT sandbox rules.

## How the TTT "search-with-budget" seed is reproduced

In TTT-Discover the model writes a **search program** that runs for ~1000s, repeatedly
calling an injected evaluator and (optionally) starting from an injected construction. We
reproduce this faithfully:

- The **EVOLVE-BLOCK** of each `initial.py` holds the exact TTT seed program (or a minimal
  seed where TTT provides none — see per-problem notes).
- The injected pieces — the evaluator (`evaluate_sequence`) and the seed construction
  (`height_sequence_1` / `initial_h_values`) — live **outside** the EVOLVE-BLOCK as fixed,
  read-only code. Shinka shows fixed code to the LLM as context, matching what TTT injects.
- A fixed entrypoint wrapper calls the evolved function with a time budget read from the
  `TTT_BUDGET_S` env var (default 1000), and returns the shape the evaluator expects.

Each example is **fully self-contained**: the evolved `initial.py` is executed standalone in
per-generation results dirs, so the evaluator + construction are embedded directly, and
`evaluate.py` carries its own copy of the verifier. This mirrors TTT's sandbox injection and
keeps scoring byte-for-byte identical.

## Scoring direction

Shinka maximizes `combined_score`. For **minimize** objectives (erdos C₅, ac1 upper bound)
we mirror TTT's reward shaping `combined_score = 1/(1e-8 + metric)` and keep the true metric
in `private`/`public`. ac2 (maximize lower bound) and circle packing (maximize sum of radii)
use the metric directly.

| Problem | Direction | `combined_score` | Target (TTT) |
|---|---|---|---|
| circle_packing_26 | maximize sum radii | sum of radii | 2.636 |
| circle_packing_32 | maximize sum radii | sum of radii | 2.940 |
| erdos_min_overlap | minimize C₅ | `1/(1e-8+C₅)` | 0.3808 (record 0.38092) |
| ac1 | minimize upper bound | `1/(1e-8+bound)` | 1.5030 |
| ac2 | maximize lower bound | bound | 0.97 |

## Per-problem seed notes

- **circle_packing_26 / _32** — TTT starts **cold** (no seed program, no construction; only
  the target in the prompt). The EVOLVE-BLOCK holds a bare `run_packing()` returning a trivial
  valid grid packing, so Shinka must climb from ~zero just like TTT. Prompt tips mirror TTT's
  circle-packing "Consider…" bullets and target exactly.
- **erdos_min_overlap** — TTT gives a random construction (`initial_h_values`) but **no
  algorithm**. The EVOLVE-BLOCK seed simply returns that construction; Shinka evolves the
  optimizer. NOTE: TTT regenerates the construction with an *unseeded* RNG each rollout; we
  pin seed **12345** — the same seed and RNG call order as the ICL runs
  (`src/envs/erdos_min_overlap.py::ErdosMinOverlapEnv`), verified to produce an identical
  81-point construction (C₅ = 0.49399).
- **ac1** — TTT seed algorithm is the LP-reweighting `propose_candidate` search
  (`discover/examples/ac_inequalities/prompt.py::example_ae_program_random_init`). Ported
  verbatim into the EVOLVE-BLOCK.
- **ac2** — TTT seed algorithm is the ThetaEvolve 4-phase Adam optimizer
  (`discover/examples/ac_inequalities/prompt.py::thetaevolve_initial_program_prev_init`).
  Ported verbatim.

## Running

Use the dedicated venv at `ShinkaEvolve/.venv` (has `shinka` installed editable, plus
`scipy`, `cvxpy`, `tqdm` for the evolved programs).

```bash
source ShinkaEvolve/.venv/bin/activate
cd examples/ttt_discover_math/<problem>

# Single-program seed evaluation (sanity check). TTT_BUDGET_S caps the internal search.
TTT_BUDGET_S=60 python evaluate.py --program_path initial.py --results_dir results/manual_eval

# Evolution: dev profile (short budget) or full profile (~1000s, matches TTT)
python run_evo.py --config_path shinka_dev.yaml
python run_evo.py --config_path shinka_full.yaml
```

## Local-qwen baseline runs vs the ICL experiments (`shinka_qwen.yaml`)

`ac1/`, `ac2/` and `erdos_min_overlap/` each carry a `shinka_qwen.yaml` profile that
baselines ShinkaEvolve against the ICL sweeps (`src/sweeps/*/qwen/*.yaml`):

- **Budget parity:** one ICL run tries 6 parents × 16 children × 25 generations = **2400**
  solutions; in Shinka one generation = one tried solution, so `num_generations: 2400`.
  Everything search-related (islands, archive, parent selection, patch types, retries) stays
  at the full-profile / Shinka-default values.
- **Timeout parity:** `job_time: "00:18:20"` — the same 1100 s hard kill the ICL sandbox uses
  (`src/envs/registry.py` `eval_timeout`), with `eval_budget_s: 1000` as the internal search
  budget the prompt promises (`TTT_BUDGET_S`).
- **LLM parity:** single model `local/Qwen/Qwen3.6-27B-FP8@<per-problem ICL server>`,
  `temperature 1.0`, `max_tokens 34000`, and `extra_body.thinking_token_budget: 12000`
  (the vLLM-enforced reasoning cap the ICL runs use; needs a server started with
  `--reasoning-parser qwen3`, vLLM ≥ 0.19). The `extra_body` passthrough is a local patch to
  `shinka/llm/kwargs.py` + `shinka/llm/llm.py`.
- **Embeddings ON (Shinka default):** the duplicate gate (`code_embed_sim_threshold 0.99`,
  `max_novelty_attempts 3`) uses `nomic-embed-text` served by the CPU ollama server instead
  of the stock OpenAI `text-embedding-3-small` (no API key available; GPUs kept free). The
  model isn't pinned in the yaml — pass
  `--embedding_model local/nomic-embed-text:latest@http://localhost:11434/v1` (see
  `RUNS.md`; note nomic scores near-identical code at ~0.93 cosine, so the 0.99 gate only
  catches near-byte-identical proposals). If an embed call fails Shinka logs an error
  (`grep -i "error.*embedding"`) and skips the gate; the run survives — verified against a
  dead port — so grep the logs after launch.
- **Meta scratchpad:** off in `shinka_qwen.yaml`; `shinka_qwen_meta.yaml` is the identical
  config with the scratchpad ON at Shinka's default cadence (`meta_rec_interval: 10`, same
  local model and sampling settings) — one extra run per problem for the with/without
  comparison. LLM novelty judge and prompt evolution stay off in both (Shinka defaults).
  Mutation + (in the meta variant) scratchpad calls are the only LLM traffic; the true call
  count can be read from `gen_*/attempts/**/metadata.json`.
- **Env:** `export LOCAL_OPENAI_API_KEY=local; export SHINKA_PRICING_MODE=offline`.
- **Replicates / resume:** `run_evo.py --results_dir results/<problem>_qwen_r<N>` gives each
  replicate its own dir; re-running with the same dir resumes from the program database.
  **The run matrix and launch commands live in `RUNS.md`** (12 runs: r1–r3 + one
  meta-scratchpad arm per problem, all local against `localhost:8001`). `src/jobs/*.bsub`
  are the equivalent Bosch launchers if the campaign ever moves back to the cluster.

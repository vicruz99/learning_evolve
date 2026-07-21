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
- `context/`    — ICL context-selection strategies (v1: n best past solutions).
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

Start a local vLLM server (see `../R2/LLMs/local/vllm_provider/run_model.md`; raise
`--max-model-len` to fit the ICL context), then:

```bash
python run_icl.py --problem ac2 --n-context 32 --num-generations 20
```


Experiment knobs at launch

--context-strategy {best,recent} · --n-context · --group-size (solutions sampled/gen) · --groups-per-batch (parents/gen) · --num-generations — plus --save-completions/--no-save-completions and --dry-run.

┌─────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│           File / dir            │                                                     What it gives you                                                     │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ summary.json                    │ best-so-far, worst valid, #succeeded / #failed, unique solutions, and the same per generation — exactly your ask          │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ progress.csv                    │ one row per generation (plot-ready: gen_best, best_so_far, success rate, buffer size, PUCT T)                             │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ solutions/sol_NNNNNN.py         │ every valid proposed solution as its own runnable .py, de-fenced, with a header (gen, parent ref, score, entrypoint) +    │
│                                 │ manifest.jsonl                                                                                                            │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ generations/gen_XXXX/meta.json  │ the parent→children view for that generation, each child referencing its sol_ file                                        │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ generations/gen_XXXX/parent_SS/ │ the exact prompt.txt sent + full raw child_CC.txt completions (incl. reasoning)                                           │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ events.jsonl                    │ one line per candidate (valid and failed) for programmatic analysis                                                       │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ buffer/                         │ authoritative PUCT snapshots                                                                                              │
├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ runs/index.csv                  │ one row per experiment — a cross-run leaderboard
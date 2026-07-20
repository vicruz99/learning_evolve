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

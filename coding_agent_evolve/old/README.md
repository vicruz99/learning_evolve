# coding_agent_evolve — the coding-agent arm of the study

The ICL sweeps in `../src/` hold the search fixed and vary the prompt. This folder holds the
*other* arm: the same tasks handed to **Claude Code** as the harness, with **Qwen3.6-27B** behind
it (or, for the early Erdős runs, Claude itself).

| dir | what |
|---|---|
| `gpumode/` | The frozen TriMul grading harness (`evaluate.py`, `trimul/`), the published kernel as `test/candidate.py`, block-size `variants/`, and the agent task folders: `b200_task/` (rng-dl01 B200) and `h100_task/` (Marvin H100). `make_run.sh` in each materialises a run folder *outside* the repo. |
| `local_model/` | Claude Code ↔ local Qwen plumbing: `serve_qwen.sh` (vLLM flags incl. tool calling), `litellm_qwen*.yaml` (Anthropic→OpenAI shim), `env.sh`, `run_guard*.json` (containment), `check_gpu.sh`, `run_agent.sh` (interactive/headless launcher), the pinned froggeric chat template, and `jobs/marvin_agent.bsub` (headless LSF job for Marvin). Start with its README. |
| `experiments/` | The AC1 / AC2 / Erdős coding-agent arm: two prompt variants per problem (`prompt_plain.md` vs `prompt_evo.md`, differing only in the evolutionary search framework), the ICL grading function as `eval.py`, and the ICL initial constructions as `.npy`. `llm_relay.py` keeps bnbcode pointed at a vLLM that moves. Start with `OPERATOR_NOTES.md`. |
| `erdos/` | Prompts and evaluator for the Erdős minimum-overlap agent runs (`runs_description.md`). |
| `misc/` | Prompt drafts not used by any run: a Gaussian-moments proof prompt, an RTX-3090 TriMul variant. |

Run folders (`~/agent_runs/…`, `/scratch/…/kernel_runs/…`) are never inside the repo — an agent
working there would inherit `CLAUDE.md`, the docs, and find the published solution next door.

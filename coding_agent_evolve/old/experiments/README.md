# experiments/ — AC1, AC2 and Erdős handed to a coding agent

The ICL sweeps in `../../src/` hold the search fixed and vary the prompt. This folder is the same
three problems handed to **bnbcode** with **Qwen3.6-27B** behind it: the model is held fixed and the
*harness* changes.

```
AC1/  AC2/  Erdos/       prompt_plain.md · prompt_evo.md · eval.py · initial construction (.npy)
llm_relay.py             a loopback address that survives a vLLM moving between GPU nodes
opencode.json            per-run bnbcode config; copied into every run folder
OPERATOR_NOTES.md        venv, node, bnbcode config, launch recipe, ICL-parity numbers
```

Two prompts per problem, differing in **one block only**: `prompt_evo.md` adds the evolutionary
search framework (diverse portfolio, registry of approach families, BLOCKED routes, `LEDGER.md`);
`prompt_plain.md` says only *"find the best solution you can"*. Everything else is identical.
Keeping it that way is the whole comparison — including bnbcode's config, which must not change
between arms.

Each `eval.py` is the ICL environment's grading function copied byte-for-byte, and each `.npy` is
that environment's own seeded initial construction, so a score here is directly comparable to a
score from a sweep.

**Read `OPERATOR_NOTES.md` before launching anything** — it has the launch recipe (§6) and three
things that will otherwise cost you an afternoon: the `bsub` project ID is mandatory, the login node
silently gives you 5 cores, and gpu6 serves on port 8002.

Run folders go in `~/agent_runs/<problem>_<variant>_s<seed>/`, **never inside this repo** — an agent
working here would read `../../src/envs/` and find the grading code and the reference solutions.

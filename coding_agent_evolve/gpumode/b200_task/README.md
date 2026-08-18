# TriMul / B200 — the coding-agent arm

The same task the `trimul_b200` ICL runs solve (`src/envs/kernel_trimul.py`,
`sweeps/trimul_*_qwen_b200.yaml`), handed to Claude Code driven by the local Qwen
instead of to a single-shot generation loop.

`INITIAL_PROMPT.md` opens with `TRIMUL_PROMPT` and `_HW_RULE_B200` **verbatim** from the ICL
environment, so both arms are told the same thing about the task and the card. Everything
after `## How you work` is agent-specific — ledger, champion promotion, approach families —
and has no ICL counterpart. That asymmetry is the experiment, not an accident: regenerate
the shared part from the source files if the ICL prompt ever changes.

## Design choices worth knowing

**No seed kernel.** `TrimulB200Env.create_initial_state` returns `code=""` and
`value=-1e6` — TTT-Discover's choice, meaning "you have nothing yet". The agent starts the
same way. The published TTT-Discover kernel is deliberately kept out of the run folder:
an agent that found it would copy rather than derive it.

**B200 scores pool with nothing.** Not with `trimul_h100` (TTT-Discover's faithful
baseline, the arm that compares to their 1161 µs), not with `trimul_a100`. B200 is where
GPUs are actually available on Bosch — 20 hosts against `batch_h100`'s 5 — not where the
comparable number lives. And no published B200 number exists at all, so the reference
kernel's own B200 leaderboard score is this card's only yardstick. Measure it once,
with `--repeats 5`, before believing any agent result:

```bash
"$KPY" ../coding_agent_evolve/gpumode/evaluate.py \
    src/gpumode_local/reference/trimul_best.py --mode leaderboard --repeats 5
```

(Run that from the repo, not from the agent's folder — it grades the published kernel,
which the agent must never see.)

**The grading venv must be cu128.** `sm_100` gencode only comes from CUDA 12.8, so the
cu126 build cannot run on a B200 at all — and it does not fail cleanly, it produces a wall
of ptxas errors that read like bad candidates. `src/gpumode_local/reference/README.md` has
the creation steps for `~/venvs/kernel-eval`.

**One card is not enough.** vLLM holds a GPU and the grader needs one; sharing makes every
timing meaningless. Ask for two and pin the server with `GPUS=`.

## Use

```bash
./make_run.sh ~/agent_runs/trimul_b200_qwen1
```

Copies the prompt and the frozen harness into a folder outside the repo, seeds an empty
`LEDGER.md`, and refuses to write inside the repo. Then launch it with
`../../local_model/run_agent.sh`.

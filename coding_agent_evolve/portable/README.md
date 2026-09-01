# Portable AC1/AC2 coding-agent kit

Everything needed to run the AC1 / AC2 optimisation tasks against **bnbcode** and
**Claude Code** on a machine that is not the Bosch cluster — in particular against a hosted
model such as Kimi, which cannot be reached from inside Bosch.

It is a deliberately small subset of the Bosch campaign: **two arms, both unaided.** No
visible-output guard, no external nudger, no keepalive, no Stop hook. Those exist to make a
*small* model keep working; a strong model should not need them, and testing without them
is the cleaner experiment. If a strong model still stops early, that is the interesting
result — and it is only visible if nothing is propping the run up.

```
tasks/AC1, tasks/AC2   prompt templates + eval.py + the seed construction
config/                harness configuration for both arms
bin/mkcell             build one run folder
bin/runcell            run it
bin/procsample         the process sampler (see "Why the sampler exists")
analysis/stopclass.py  classify every stop
analysis/score.py      re-grade a finished run with its own eval.py
PROMPT_DEVIATIONS.md   exactly how these prompts differ from the ones Bosch ran
```

## The task

Both tasks optimise a bound on an autocorrelation inequality by searching over step
functions. The agent gets `eval.py` (the real grading function, which it must not modify),
a seed construction, and a time budget.

| | starts at | reference to beat | direction |
|---|---|---|---|
| **AC2** | 0.6667 | **0.97** (AlphaEvolve) | higher is better |
| **AC1** | 2.0000 | **1.5030** (AlphaEvolve) | lower is better |

Best Bosch pilot result so far: **0.9242** on AC2. Nothing has beaten the reference yet.

Two prompt variants, run as separate cells:

- **`evo`** — prescribes an evolutionary search *over approaches*: keep a diverse portfolio,
  maintain a ledger of approach families, mark blocked routes, measure everything.
- **`plain`** — the same task, environment, and rules, with the entire strategy section
  replaced by "Find the best solution you can!!". This is the contrast that tests whether
  the scaffolding in the prompt is doing any work.

## Setup

### 1. Keys

```bash
export KIMI_API_KEY=sk-...
```

### 2. bnbcode arm

Put `config/opencode.kimi.jsonc` at `~/.config/opencode/opencode.jsonc`.

**Read the comments in that file before you run anything.** The one that matters:
`"temperature": true` is a boolean *capability* flag, not a value. A custom provider that
does not declare it sends **no temperature at all**, silently, no matter what
`agent.build.temperature` says. That defect ran undetected through an entire pilot here —
every "temperature 0.6" number in it was fiction. `tool_call: true` is the same shape of
trap: without it the model is never offered tools.

bnbcode stores session history in PostgreSQL. It must be reachable when `stopclass.py`
runs; point `BNB_PSQL` at whatever runs `psql` against it.

### 3. Claude Code arm

No shim needed: Moonshot exposes an Anthropic-compatible endpoint, so `runcell` sets
`ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` and Claude Code talks to Kimi directly.
**Verify the host against your key's region** — `api.moonshot.ai` (international) vs
`api.moonshot.cn` (mainland). A key issued for one 401s against the other.

Settings come from `config/claude_settings.json`, passed with `--settings` so they sit
*outside* the folder the agent can edit. Two things in there are load-bearing:

- `defaultMode: acceptEdits` **plus** an explicit `Bash` allow. Plain `acceptEdits`
  auto-accepts edits but still prompts on every Bash call, which wedges an unattended run
  on its first command. Measured, not assumed.
- `claudeMdExcludes`. Without it, any `CLAUDE.md` above the run folder is injected into the
  agent's context and silently becomes part of the experimental condition — different at
  every site, and invisible in the results.

## Running

```bash
export KIMI_API_KEY=sk-...
export RUNS_ROOT=$HOME/agent_runs          # where cells are created
export CELL_CORES=16 CELL_HOURS=18         # must match what the box really gives
export AGENT_PYTHON=/path/to/venv/bin/python

bin/mkcell ac2_evo_kimi_bnb_s1 AC2 evo bnb
bin/runcell $RUNS_ROOT/ac2_evo_kimi_bnb_s1 bnb 18

bin/mkcell ac2_evo_kimi_cc_s1  AC2 evo cc
bin/runcell $RUNS_ROOT/ac2_evo_kimi_cc_s1  cc  18
```

`CELL_CORES` and `CELL_HOURS` are written *into the prompt*. Set them to the truth. An
agent told it has 32 cores on a 16-core box oversubscribes, and its own candidate timings —
which is how it decides what works — stop meaning anything.

The full matrix is 2 harnesses × 2 prompts × 2 seeds = 8 cells per task. Seeds are
independent repetitions: bnbcode's temperature is 0.6, so a single run of a cell is one
sample, not a measurement.

## Why the sampler exists

`procsample` writes a snapshot every 15 s of which processes are actually working inside the
run folder, with CPU%.

A turn that makes no tool call **ends the agent loop** — there is no waiting state. But an
agent that launched a 20-minute evaluation and ended its turn is behaving *correctly*, and
one that ended its turn with nothing running is not. Transcripts cannot tell those apart
after the fact, and in the pilot 44 correct waits were counted as failures because of it.

The refinement that matters, and the reason `stopclass.py` looks at the *next* call too: a
live process is not sufficient. If the agent launched work and then stopped paying
attention to it, the run is just as dead. So a stop counts as legitimate waiting only when
a worker was alive **and** the agent's next tool call actually reads that work.

```
mid-thought    reasoning, no visible text                        -> a stop
waiting        worker alive AND the agent came back and checked  -> NOT a stop
abandoned      worker alive but the agent never checked on it    -> a stop
signoff-idle   visible text, nothing running                     -> a stop
truncation     finish = length / max_tokens                      -> a stop
empty          neither text nor reasoning                        -> a stop
```

## Analysis

```bash
python3 analysis/stopclass.py $RUNS_ROOT/ac2_evo_kimi_bnb_s1     # stop taxonomy
python3 analysis/score.py     $RUNS_ROOT/ac2_evo_kimi_bnb_s1     # re-graded best
```

Run `stopclass.py` **on the machine that executed the cell**, before archiving: it caches
turns to `run/turns.json`, and bnbcode's session store may not be reachable from wherever
the analysis eventually happens. Do not delete that cache to force a rebuild unless you are
on that machine — off-node the rebuild silently returns nothing.

A second reason it matters: bnbcode keys its session store by run **directory**, and the
store outlives the process. Re-running a cell in the same folder therefore reads back the
previous attempt's turns interleaved with the new ones. `stopclass.py` dates each attempt
from the sampler's first sample and drops anything older; a restarted cell here came back
with 79 turns, 71 of them from the attempt killed an hour earlier.

`score.py` re-grades every `.npy` the run produced using the run's own `eval.py`. Do not
take the agent's word for its score — pilot runs had ledger entries claiming improvements
that did not survive a re-grade.

## What to report per cell

- best re-graded score, and the time it was first reached
- wall-clock and turns to the first unresumed stop
- the stop taxonomy above
- tool calls per hour, candidates evaluated per hour
- compactions per hour
- whether 0.97 (AC2) / 1.5030 (AC1) was ever passed

Report both seeds of a cell, always. Never a single run.

# Runs

Two agent runs on the same TriMul task and the same harness, differing only in
the starting point (and the GPU, so they can run concurrently without
contaminating each other's timings).

**The run folders live outside this repo**, at `/scratch/vicstorage/kernel_runs/`,
so an agent working in one cannot inherit this repo's `CLAUDE.md` (which orders
it to read `docs/PROJECT_CONTEXT.md` and `docs/EXPERIMENT_PLAN.md` — both of
which discuss this task), reuse this project's auto-memory, or stumble on the
published kernel in a sibling directory.

| run | starting point | seed score | GPU | question |
|---|---|---|---|---|
| `run1` | the TTT-Discover published Triton kernel | ~2450 µs | 0 | can the agent improve on an already-optimized kernel? |
| `run2` | the naive PyTorch reference | ~21500 µs | 1 | can the agent rediscover the optimization path from scratch? |

## Launching

```bash
cd /scratch/vicstorage/kernel_runs/run2
claude --settings /home/guests2/vic/work/projects/phd/learning_evolve/coding_agent_evolve/gpumode/run_guard.json
```

The guard lives here, outside the run folder, so the agent cannot edit the rules
that constrain it. It denies reads of `discover/`, `gpumode/test/`,
`gpumode/variants/`, this repo's `docs/` and `CLAUDE.md`, and the auto-memory
directory; it also disables auto-memory and excludes the repo `CLAUDE.md`.

Do not pass `--add-dir`, and avoid `--dangerously-skip-permissions` — the
working-directory scoping does much of the containment work.

Containment is not airtight: bubblewrap is not installed on this machine, so
Claude Code's sandbox cannot run and Bash can still read any file the user can.
It is enough for incidental contamination, not a determined search.

## Notes

- Each folder is self-contained: `INITIAL_PROMPT.md`, `evaluate.py`, the frozen
  `trimul/` harness, and its seed as `starting_point.py`.
- Score = geometric mean over the 7 benchmark shapes, lower is better:
  `python evaluate.py <file> --gpu N --mode leaderboard`.
- `evaluate.py` keys its Triton cache on the folder path, so the two runs never
  share compiled artifacts.
- `run2` deliberately does **not** contain the fast kernel — an agent that found
  it would copy rather than derive it. `run1` does, because it is the seed.

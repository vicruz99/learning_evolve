# What differs from the prompts the Bosch campaign ran

The task text is the experimental condition, so it is copied byte-for-byte except where a
sentence describes *this cluster* rather than *the problem*. Those sentences are
placeholders, rendered by `bin/mkcell`. `eval.py` and `height_sequence_1.npy` are
byte-identical — see `tasks/CHECKSUMS.txt`.

## Placeholders

| placeholder | rendered from | why it cannot be a constant |
|---|---|---|
| `{{VENUE}}` | `CELL_VENUE` | "an LSF interactive job on the `rng-dl01` cluster" is false anywhere else |
| `{{CORES}}` / `{{PARALLEL}}` | `CELL_CORES` | the claimed budget must equal the real one, or the agent oversubscribes and its own timings stop meaning anything |
| `{{HOURS}}` | `CELL_HOURS` | the prompt states the time budget; it must match the run's |
| `{{PYTHON}}` / `{{PYTHON_BLOCK}}` | `AGENT_PYTHON` / `CELL_PYTHON_BLOCK` | hardcoded `/home/crv1pi/venvs/agent-eval/bin/python` |
| `{{START_SCORE}}` | the task | see below |

## Two defects fixed here, both found while building this kit

**1. AC2 told the agent its starting construction scores 2.0. It scores 0.6667.**

The AC2 prompt was forked from AC1 and inherited AC1's number in the sentence describing
`height_sequence_1.npy`, while its own *Objective* section correctly says 0.6667. So the
prompt contradicted itself: one section named the real baseline, another named AC1's.
Verified by running the shipped `eval.py` on the shipped file:

```
AC2  SCORE = 0.6666666666666639   (higher is better)
AC1  SCORE = 1.9999999999999944   (lower is better)
```

Here `{{START_SCORE}}` renders per task, so the two sections agree.

**2. `AC1/prompt_plain.md` claimed 32 cores and 16 candidates in flight** where the other
three prompts claimed 24 and 12. `mkcell` on the cluster normalises the *first* of those
two numbers but not the second, so an AC1 plain cell would have said "use at most 16 CPU
cores" and, four lines later, "with 32 cores … you can hold 16 candidates in flight". Both
numbers are derived from `CELL_CORES` here.

**Impact on the running Bosch campaign:** none for AC2, which is what is running. Every
AC2 cell carries the same erroneous sentence, so cells stay comparable with each other, and
an agent that runs `eval.py` once — all of them do — sees the true value immediately. AC1
has not started; fix `experiments/AC1/prompt_plain.md` there before it does.

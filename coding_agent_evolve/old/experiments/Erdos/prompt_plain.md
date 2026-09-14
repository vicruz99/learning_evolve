# Minimizing the Erdős Overlap Integral

## The Problem

You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \max_k \int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as `n_points` samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

## Your Objective

**Lower C₅ values are better** — they provide tighter upper bounds on the Erdős constant.

The starting construction supplied to you scores **0.49399**. The published record is
**C₅ ≤ 0.38092**, and the milestone to beat is **0.38080**. But the milestone is not a stopping
condition: your true goal is the absolute minimum value you can reach. Do not stop iterating
because you hit a number. You have 24h to find the best possible solution you can!!!!

## How to work

Find the best solution you can!!

## Environment, Compute & Disk

- **Where you are**: an LSF *interactive* job on the `rng-dl01` cluster. `nproc` reports how many
  cores it was given. There is no GPU and you do not need one.
- **Your own budget**: use at most **24 CPU cores** at any one time, whatever `nproc` says. The rest
  belong to the session hosting you — saturating the node will stall your own tool calls.
- **Per-candidate limit**: any single candidate program you run **must return within 1000 seconds**
  and may use **exactly 1 CPU core**. Anything still running at **1100 s** is killed and scores
  nothing. Pin your children (`taskset -c`, or `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`) — an unpinned numpy/scipy build will happily grab
  every core on the node and make your own timings meaningless.
- With 24 cores and 1 core per candidate you can hold **24 candidates in flight at once**. Use that.
- **Python**: `/home/crv1pi/venvs/agent-eval/bin/python` — Python 3.12 with numpy 2.5.1,
  scipy 1.18.0 and cvxpy 1.9.2 (solvers: CLARABEL, SCS, ECOS, ECOS_BB, SCIPY, HIGHS, OSQP).
  Use that interpreter, not a bare `python`. **Do not create another virtual environment** and do
  not try to install packages: this job has no outbound network, so every install attempt is a
  wasted turn. If you genuinely need a library that is not there, say so in your final report
  instead of working around it.
- **Disk**: write everything into your working directory. Keep the total under ~2 GB.

## How your solution is scored

`eval.py` in your working directory holds the grading function, byte-for-byte the one that will be
used to score you. **Do not modify it.** It exposes:

```python
verify_c5_solution(h_values, c5_achieved, n_points) -> float   # raises on an invalid solution
```

Note what it checks: `h` must be 1-D of exactly `n_points` finite entries, every entry in [0, 1],
and the C₅ value you *report* must match the value it *computes* to within 1e-4. It renormalizes
`h` so that `sum(h) == n_points / 2`, and rejects the solution if that renormalization pushes any
entry outside [0, 1].

It also runs as a script:

```bash
/home/crv1pi/venvs/agent-eval/bin/python eval.py candidate.npy
```

which loads a saved `h`, validates it, and prints the computed C₅.

`initial_h_values.npy` in your working directory is the initial construction the ICL arm starts
from (n = 81 samples, C₅ = 0.49399). You may start your search from it, or from anywhere else —
you are encouraged to explore other starting points to avoid getting stuck in a local optimum.

## Rules

- Every candidate program must define **`run(seed=42, budget_s=1000, **kwargs)`** returning
  `(h_values, c5_bound, n_points)`. That is the entrypoint that gets invoked.
- `run` must return within its budget. A program that has not returned by 1100 s is killed and
  scores nothing.
- Allowed libraries: numpy, scipy, cvxpy (with the solvers listed above), math, and the rest of the
  standard library.
- Use `print()` inside your candidates to log progress, intermediate bounds and timing — you will
  read that output back.

## What to keep

Keep the work, not just the answer. I need to see how the score moved and to be able to re-run
anything you tried.

**Put everything you produce in a folder called `run/`**, and create it on your first write. That
keeps your output separate from the files that were already here: `INITIAL_PROMPT.md`, `eval.py` and
the input `.npy` stay at the top level — read them there, do not move or copy them.

```
run/best.npy               the best construction found so far
run/best.py                the program that produced it
run/BEST.md                half a page: the C₅, which attempt reached it, and why it works
run/attempts/001_name.py   every candidate you actually ran, numbered in the order you ran it
run/attempts/001_name.log  its stdout, and the C₅ `eval.py` gave it
run/LEDGER.md              one line per candidate: number, one phrase, C₅, kept or rejected
```

- **`run/best.*` is a copy, never a work in progress.** Replace it only when a candidate has been scored
  by `eval.py` *and* beats the current best's own freshly re-scored value. Never edit it in place: if
  the best is also the file you are editing, you cannot tell whether you improved it.
- **Number attempts in the order you ran them, and never renumber or delete one.** A failed candidate
  is data. The record of a direction that did not work is what stops you walking back into it later.
- **`run/LEDGER.md` is one line per candidate, not a paragraph.** It should be readable end to end in a
  minute and show the C₅ moving. Anything longer belongs in `run/BEST.md` or in that attempt's own
  `.log`.
- **Logging inside a candidate**: print the current best and the elapsed time periodically — often
  enough that I can see whether it was still improving when the budget ran out, rarely enough that
  the log stays readable. Every few seconds, not every iteration.

## No Web Access

You have **no web access**. Do not attempt to search the web or fetch a URL — the tools are
disabled and every attempt is a wasted turn. Everything must come from your own reasoning and
from experiments you run on this machine.

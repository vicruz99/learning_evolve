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

## Search Strategy

Treat this as a search over *approaches*, not as a single program you polish. Do not get stuck
fine-tuning one idea that has stalled in a local optimum. Run several diverse approaches in
parallel, so that there is genuine exploration of new ideas as well as exploitation of the
promising ones.

- **Diverse initial solutions.** Begin with a genuinely diverse portfolio: substantially different
  formulations, not restatements of one idea. Different search algorithms, different optimization
  models, different discretizations, different objective relaxations.
- **Registry of approach families.** Maintain an explicit registry in `run/LEDGER.md`, grouping
  candidates by the *mathematical idea* they use, not by superficial wording. If several candidates
  converge onto one family, deliberately redirect effort into an underexplored formulation.
- **Independence before cross-pollination.** Let independent branches develop far enough to expose
  their real strengths and gaps before you merge ideas across them. Merging too early collapses the
  portfolio into a single family.
- **Creative and novel streams.** Standard techniques are a good baseline, but keep distinct
  branches dedicated to unconventional approaches you have not seen used for this problem.
- **Critical analysis.** Be critical of your own proposals. For every candidate, say exactly *why*
  it worked or failed, and feed that into the next generation. A failed candidate is search signal,
  not waste — record its concrete failure cause (invalid solution / too slow / numerically
  unstable / converged to a worse optimum).
- **Blocked routes.** When an approach stalls, mark it BLOCKED in the ledger together with the
  evidence that blocked it (a measurement, a resource limit, repeated verified regressions).
  Reopen it only when you have a genuinely new mechanism that addresses that evidence.
- **Measure, do not assume.** Every claim about why something is better or worse needs a scored
  number behind it. An unscored candidate is not progress.

## Bookkeeping

**Put everything you produce in a folder called `run/`**, and create it on your first write. The
files already here (`INITIAL_PROMPT.md`, `eval.py`, the input `.npy`) stay at the top level — read
them there, do not move them.

- **`run/best.npy` / `run/best.py`** — the current champion construction and the program that produced it.
  Replace them only when a candidate has been scored by `eval.py` and genuinely beats the
  champion's own freshly re-scored value. Never edit the champion in place.
- **`run/LEDGER.md`** — one row per candidate: approach family, one-line description, valid yes/no,
  score, delta vs champion, verdict. Keep the candidate programs in `run/attempts/`.
- **`run/NOTES.md`** — a running narrative of what you learned about the problem itself, separate from
  the per-candidate ledger.

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
- Give every candidate a short docstring at the top summarizing its algorithm.

## Reporting

When you are done, report: the best C₅ reached, the program and construction that produced it,
which approaches you tried, and what blocked the ones that failed. Leave the working directory in a
state where I can reproduce your best result by running one command.

## No Web Access

You have **no web access**. Do not attempt to search the web or fetch a URL — the tools are
disabled and every attempt is a wasted turn. Everything must come from your own reasoning and
from experiments you run on this machine.

# Maximizing the Autocorrelation Inequality Lower Bound (AC2)

## The Problem

Act as an expert software developer and inequality specialist specializing in creating step
functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that **maximizes**
the following evaluation function:

```python
import numpy as np
def evaluate_sequence(sequence: list[float]) -> float:
    # Verify that the input is a list
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")

    # Reject empty lists
    if not sequence:
        raise ValueError("Empty sequence")

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Check if sum of sequence will be too close to zero
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    
    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    # --- Security Checks ---

    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound
```

A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encoraged to explore.
```latex
Their procedure is a coarse-to-fine optimization of the score. It starts with a stochastic global search that repeatedly perturbs the current best candidate and keeps the perturbation whenever it improves (Q), with the perturbation scale gradually reduced over time. Once a good basin is found, they switch to a deterministic local improvement step, performing projected gradient ascent (move in the gradient direction and project back to the feasible region). To reach higher resolution, they lift a good low-resolution solution to a higher-dimensional one by simply repeating its entries and then rerun the local refinement. Iterating this explore–refine–upscale cycle yields their final high-resolution maximizer and the improved lower bound.
```

## Your Objective

You are optimizing a **lower bound** (higher is better).

The starting construction supplied to you scores **0.6667**. The target is **0.97** — reaching or
passing it is a strong result. But the target is a milestone, not a stopping condition: your true
goal is the highest value you can reach. Do not stop iterating because you hit a number. You have {{HOURS}}h to find the best possible solution you can!!!!

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

{{VENUE}}
- **Your own budget**: use at most **{{CORES}} CPU cores** at any one time, whatever `nproc` says. The rest
  belong to the session hosting you — saturating the node will stall your own tool calls.
- **Per-candidate limit**: any single candidate program you run **must return within 1000 seconds**
  and may use **at most 2 CPU cores**. Anything still running at **1100 s** is killed and scores
  nothing. Pin your children (`taskset -c`, or `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2`) — an unpinned numpy/scipy build will happily grab
  every core on the node and make your own timings meaningless.
- With {{CORES}} cores and 2 cores per candidate you can hold **{{PARALLEL}} candidates in flight at once**. Use that.
{{PYTHON_BLOCK}}
- **Disk**: write everything into your working directory. Keep the total under ~2 GB.

## How your solution is scored

`eval.py` in your working directory holds the grading function, byte-for-byte the one that will be
used to score you. **Do not modify it.** It exposes:

```python
evaluate_sequence(sequence: list[float]) -> float   # the score; np.inf means invalid
```

It also runs as a script:

```bash
{{PYTHON}} eval.py candidate.npy
```

which loads a saved sequence, validates it, and prints the score.

`height_sequence_1.npy` in your working directory is the initial construction the ICL arm starts
from (6520 entries, all equal to 0.22733602246716966; it scores {{START_SCORE}}). You may start your search
from it, or from any other starting point you like — you are encouraged to explore other starting
points to avoid getting stuck in a local minimum.

## Rules

- Every candidate program must define **`construct_function(seed=42, budget_s=1000, **kwargs)`**
  returning the best `list[float]` it found. That is the entrypoint that gets invoked.
- The returned sequence must contain only finite, non-negative numbers.
- `construct_function` must return within its budget. A program that has not returned by 1100 s is
  killed and scores nothing.
- Larger sequences (1000s of entries) often have better attack surface, but sequences with hundreds
  of thousands of entries may be too slow to search within the budget.
- Import `evaluate_sequence` from `eval.py`. You may call it as often as you like.
- Allowed libraries: numpy, scipy, cvxpy (with the solvers listed above), math, and the rest of the
  standard library.
- Use `print()` inside your candidates to log progress, intermediate bounds and timing — you will
  read that output back.
- Give every candidate a short docstring at the top summarizing its algorithm.

## Reporting

When you are done, report: the best score reached, the program and construction that produced it,
which approaches you tried, and what blocked the ones that failed. Leave the working directory in a
state where I can reproduce your best result by running one command.

## No Web Access

You have **no web access**. Do not attempt to search the web or fetch a URL — the tools are
disabled and every attempt is a wasted turn. Everything must come from your own reasoning and
from experiments you run on this machine.

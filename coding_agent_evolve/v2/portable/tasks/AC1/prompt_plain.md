# Minimizing the Autocorrelation Inequality Upper Bound (AC1)

## The Problem

Act as an expert software developer and inequality specialist specializing in creating step
functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that **minimizes**
the following evaluation function:

```python
import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # --- Security Checks ---

    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))
```

A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.
```latex
Starting from a nonnegative step function $f=(a_0,\dots,a_{n-1})$ normalized so that $\sum_j a_j=\sqrt{2n}$, set $M=\|f*f\|_\infty$. Next compute $g_0=(b_0,\dots,b_{n-1})$ by solving a linear program, i.e.\ maximizing $\sum_j b_j$ subject to $b_j\ge0$ and $\|f*g_0\|_\infty\le M$; as is standard, the optimum is attained at an extreme point determined by an active set of binding inequalities, here corresponding to important constraints where the convolution bound $(f*g_0)(x)\le M$ is tight and limiting. Rescale $g_0$ to match the normalization, $g=\frac{\sqrt{2n}}{\sum_j b_j}g_0$, and update $f\leftarrow (1-t)f+t g$ for a small $t>0$. Repeating this step produces a sequence with nonincreasing $\|f*f\|_\infty$, and the iteration is continued until it stabilizes.
```

## Your Objective

You are optimizing an **upper bound** (lower is better).

The starting construction supplied to you scores **2.0000**. The target is **1.5030** — reaching or
passing it is a strong result. But the target is a milestone, not a stopping condition: your true
goal is the lowest value you can reach. Do not stop iterating because you hit a number. You have {{HOURS}}h to find the best possible solution you can!!!!

## How to work

Find the best solution you can!!

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

- Every candidate program must define **`propose_candidate(seed=42, budget_s=1000, **kwargs)`**
  returning the best `list[float]` it found. That is the entrypoint that gets invoked.
- The returned sequence must contain only finite, non-negative numbers.
- `propose_candidate` must return within its budget. A program that has not returned by 1100 s is
  killed and scores nothing.
- Larger sequences (1000s of entries) often have better attack surface, but sequences with hundreds
  of thousands of entries may be too slow to search within the budget.
- Import `evaluate_sequence` from `eval.py`. You may call it as often as you like.
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
run/BEST.md                half a page: the score, which attempt reached it, and why it works
run/attempts/001_name.py   every candidate you actually ran, numbered in the order you ran it
run/attempts/001_name.log  its stdout, and the score `eval.py` gave it
run/LEDGER.md              one line per candidate: number, one phrase, score, kept or rejected
```

- **`run/best.*` is a copy, never a work in progress.** Replace it only when a candidate has been scored
  by `eval.py` *and* beats the current best's own freshly re-scored value. Never edit it in place: if
  the best is also the file you are editing, you cannot tell whether you improved it.
- **Number attempts in the order you ran them, and never renumber or delete one.** A failed candidate
  is data. The record of a direction that did not work is what stops you walking back into it later.
- **`run/LEDGER.md` is one line per candidate, not a paragraph.** It should be readable end to end in a
  minute and show the score moving. Anything longer belongs in `run/BEST.md` or in that attempt's own
  `.log`.
- **Logging inside a candidate**: print the current best and the elapsed time periodically — often
  enough that I can see whether it was still improving when the budget ran out, rarely enough that
  the log stays readable. Every few seconds, not every iteration.

## No Web Access

You have **no web access**. Do not attempt to search the web or fetch a URL — the tools are
disabled and every attempt is a wasted turn. Everything must come from your own reasoning and
from experiments you run on this machine.

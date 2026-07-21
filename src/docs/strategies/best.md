# `best` — top-n by quality

**Engine:** topk · **Knobs:** `n_context`

## Definition
The `n_context` highest-scoring solutions in the buffer (by normalized `value`, higher = better),
excluding the current parent.

## Algorithm
1. Candidates = buffered states with a value, minus the current parent.
2. Sort by `value` descending; take the first `n_context`.

## Intuition
Pure exploitation: show the model the current frontier and ask it to push further. Simple and strong,
but it can be **redundant** — the top-n are often near-duplicates (e.g. several restarts of the same
SLSQP packing that converged to the same layout), so effective diversity of the context can be low.
`best_diverse` and `per_lineage` are the fixes for that.

## When to use
The natural default and the main quality baseline. Compare against `best_diverse`/`per_lineage` to
measure how much redundancy in the top-n is costing you.

# `recent` — most recently generated

**Engine:** topk · **Knobs:** `n_context`

## Definition
The `n_context` most recently created solutions, by `timestep` (newest first), excluding the current
parent.

## Algorithm
1. Candidates = buffered states with a value, minus the current parent.
2. Sort by `timestep` descending; take the first `n_context`. Seeds (`timestep = -1`) rank last.

## Intuition
Temporal conditioning: "here's what was just tried." Biases the model toward continuing the current
line of exploration rather than reverting to the all-time best. Useful for probing whether *recency*
carries signal that raw quality doesn't — and as a foil to `best`.

## When to use
Ablation against `best`/`random` to isolate the value of recency. Note the buffer is pruned to the
top-2 children per parent each generation, so "recent" is recency among *surviving* solutions.

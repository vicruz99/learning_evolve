# `best_jump` — x% best + (1−x)% biggest improvement

**Engine:** mix · **Knobs:** `n_context`, `mix_fraction` (x)

## Definition
A two-block context: the top `x·n_context` by quality, plus the `(1−x)·n_context` solutions with the
largest improvement over their parent (`jump`), shown as a separate block.

## Algorithm
1. `n_pos = round(x · n_context)`, `n_neg = n_context − n_pos`.
2. Positives = top `n_pos` by `value`.
3. Secondary = top `n_neg` by `jump` among the remaining candidates.

## Intuition
Combine *where we are* (the frontier) with *what recently moved the needle* (high-improvement edits).
The frontier says "match this level"; the jumps say "here are changes that produced gains." Same
machinery as `best_worst`, but the second pool is improvement instead of failure — so it's the
constructive counterpart.

## When to use
When you suspect the model benefits from seeing productive *moves* alongside the best states, without
committing to the full MMR blend of `informative`.

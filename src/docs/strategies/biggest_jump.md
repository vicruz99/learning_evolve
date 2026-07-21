# `biggest_jump` — largest improvement over parent

**Engine:** topk · **Knobs:** `n_context`

## Definition
The `n_context` solutions with the largest **improvement over their immediate parent**,
`jump = value − parent_values[0]` (seeds → 0), highest first.

## Algorithm
1. Candidates = buffered states with a value, minus the current parent.
2. Sort by `jump` descending; take the first `n_context`.

## Intuition
High-information experiences. A solution that jumped a lot from its parent captures *what change
mattered* — a more instructive signal than an absolute best that may sit on a long plateau. Note this
ranks by delta, not level, so a big jump from a bad parent to a mediocre child can outrank the
frontier; that's intentional (it surfaces productive *moves*).

## When to use
When you want the context to teach *edits that help* rather than *end states*. Also the pure form of
the improvement axis that `informative` blends with quality.

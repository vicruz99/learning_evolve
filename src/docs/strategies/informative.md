# `informative` — MMR over quality + improvement

**Engine:** mmr · **Knobs:** `n_context`, `mmr_lambda` (λ), `jump_alpha` (α)

## Definition
`best_diverse` with a richer notion of "quality": a blend of absolute score and improvement-over-parent,
so high-information leaps outrank incrementally-tuned frontier clones — while still keeping the picks
spread across the tree.

## Algorithm
1. Normalize `value` and `jump` (`= value − parent_values[0]`) separately to `[0,1]`.
2. Blended quality `q = α·value_norm + (1−α)·jump_norm`.
3. Run MMR (as in `best_diverse`) using `q` as the quality term:
   pick `argmax λ·q(c) − (1−λ)·max sim(c, picked)` greedily until `n_context`.

## Intuition
Absolute quality alone can fill the context with frontier solutions that differ only by tiny tuning.
The `jump` term re-elevates solutions that *made a real gain* over where they started, so the model
sees both strong end-states and the moves that got there — and the MMR term keeps them diverse. This
is the most expressive strategy: `α` trades level vs. gain, `λ` trades quality vs. diversity. `α=1`
recovers `best_diverse`; `α=1, λ=1` recovers `best`.

## When to use
When you believe the model learns most from *high-improvement, non-redundant* examples. Has the most
knobs, so tune `α` and `λ` deliberately (start `α=0.5, λ=0.7`).

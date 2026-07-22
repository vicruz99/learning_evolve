# `random` — random baseline

**Engine:** topk · **Knobs:** `n_context`, `context_seed`

## Definition
Uniformly sample `n_context` solutions from the context pool (excluding the current parent), ignoring
quality, recency, and structure entirely. Seeded by `context_seed` for reproducibility (unset → a fresh
draw each call).

## Algorithm
1. Candidates = valid pool solutions with a value, minus the current parent.
2. Shuffle with the RNG (`context_seed`); take the first `n_context`.

## Intuition
The control condition. Any "smart" strategy has to beat *showing the model arbitrary past work*. If a
sophisticated selector doesn't outperform `random`, its structure isn't buying anything — so this is
the reference every other strategy is measured against.

## When to use
As the baseline arm in every comparison. Not intended for actual discovery performance.

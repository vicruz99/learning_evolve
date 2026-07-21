# `best_worst` — x% best + (1−x)% worst

**Engine:** mix · **Knobs:** `n_context`, `mix_fraction` (x)

## Definition
A two-block context: the top `x·n_context` by quality, plus the bottom `(1−x)·n_context` by quality
(lowest-scoring valid solutions), shown as a separate "for contrast" block.

## Algorithm
1. `n_pos = round(x · n_context)`, `n_neg = n_context − n_pos`.
2. Positives = top `n_pos` by `value`.
3. Negatives = lowest `n_neg` by `value` among the remaining candidates.

## Intuition
The ICL analogue of the **negative signal** RL gets for free. Showing the model both what works and
what doesn't gives it a gradient to reason about, instead of only good examples to imitate. Cheap (no
distance metric). "Worst" means lowest-scoring *valid* solutions — the buffer holds no broken code
(see `_concepts.md`). For a spectrum of negatives (low **and** mid) rather than just the bottom, use
`contrastive`.

## When to use
First test of "do negative examples help in-context?" Sweep `mix_fraction` to trade off how much of
the budget goes to contrast.

# `contrastive` — diverse best + stratified low→mid negatives

**Engine:** mmr · **Knobs:** `n_context`, `mix_fraction` (x), `mmr_lambda` (λ)

## Definition
A two-block context: a lineage-diverse set of the best solutions (MMR), plus a set of negatives that
**span the low-to-mid reward range** (not just the bottom), each block lineage-diverse.

## Algorithm
1. `n_pos = round(x · n_context)`, `n_neg = n_context − n_pos`.
2. Positives = MMR over quality (as in `best_diverse`), `n_pos` picks.
3. Negative pool = valid solutions scoring **below the positives' minimum**.
4. Negatives = stratify that pool into `n_neg` value-quantile bins; from each bin take one
   representative, preferring the one most lineage-distant from those already chosen. This yields a
   *ladder* from clearly-bad up to just-below-frontier rather than `n_neg` copies of the single worst.

## Intuition
The full contrastive picture: diverse successes **and** a spectrum of shortfalls. Compared to
`best_worst` (which takes only the very bottom), the stratified negatives show *degrees* of failure —
"this is terrible, this is mediocre, this almost worked" — which is a more informative contrast for
in-context reasoning. Both blocks are lineage-diverse so neither collapses onto one idea. Negatives are
low-scoring *valid* solutions (the buffer stores no broken code; see `_concepts.md`).

## When to use
The richest negative-signal arm — use when comparing how much a *graded* set of failures adds over
best-only (`best_diverse`) or bottom-only (`best_worst`) contrast.

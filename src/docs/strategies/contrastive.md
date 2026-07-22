# `contrastive` — diverse best + diverse worst negatives

**Engine:** mmr · **Knobs:** `n_context`, `mix_fraction` (x), `mmr_lambda` (λ)

## Definition
A two-block context: a lineage-diverse set of the best solutions (MMR), plus a lineage-diverse set of
the **genuinely worst** solutions (MMR again, run on inverted quality). Both blocks spread across the
tree so neither collapses onto a single idea.

## Algorithm
1. `n_pos = round(x · n_context)`, `n_neg = n_context − n_pos`.
2. Positives = MMR over quality (as in `best_diverse`), `n_pos` picks.
3. Negative pool = valid solutions scoring **below the positives' minimum**.
4. Negatives = MMR over that pool with the quality term **inverted** (`−value`, min-max normalized), so
   the same greedy rule now prefers the *lowest*-scoring solutions while still penalizing lineage
   redundancy. This picks the worst attempts, spread across lineages — it does **not** try to cover the
   mid-tier.

## Intuition
Same hard-negative insight as `best_worst` — show the model what clearly fails, not near-misses — but
with the diversity guard applied to **both** blocks. Where `best_worst` takes the bottom `n_neg` by raw
value (which can all come from one bad branch, i.e. near-duplicate failures), `contrastive` uses MMR to
spread the negatives across distinct lineages, so the failures shown are genuinely *different* ways of
being bad. Negatives are low-scoring *valid* solutions (a failed rollout is never graded valid; see
`_concepts.md`).

## When to use
The diversity-aware negative arm — use when `best_worst` helped but its negatives looked like copies of
the same failure, or to compare diverse-worst vs bottom-n-worst contrast. `mmr_lambda` controls how hard
both blocks push for spread (λ=1 → pure quality/pure-worst, λ=0 → pure lineage spread).

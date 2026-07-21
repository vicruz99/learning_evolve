# `per_lineage` — best (and worst) with same-lineage skipping

**Engine:** per_lineage · **Knobs:** `n_context`, `mix_fraction` (x)

## Definition
Iterate solutions in preference order and take one only if it is **not on the same root-to-node path**
as one already taken (i.e. not an ancestor/descendant of a pick). Applied to the best pool, and — if
`x < 1` — independently to the worst pool for the remaining slots.

## Algorithm
1. `n_pos = round(x · n_context)`, `n_neg = n_context − n_pos`.
2. Positives: sort candidates by `value` descending; walk the list, skipping any candidate that is on
   the same lineage path as an already-picked one, until `n_pos` are collected.
3. Negatives (if `n_neg > 0`): same greedy skip, but over the remaining candidates sorted by `value`
   ascending (worst first).

## Intuition
A **hard** diversity rule — the exact procedure you asked for: take the best, then keep walking down
the ranking skipping anything from the same lineage until you find a genuinely different branch.
Directly kills the "child that barely changed its parent" redundancy, because once you pick a node its
whole ancestral line is excluded. Siblings/cousins are *not* excluded (they're not on the same path);
if you also want siblings suppressed, use `best_diverse` (soft tree-distance) instead.

This is the hard-threshold limit of `best_diverse`'s MMR: binary same-lineage exclusion instead of a
soft similarity penalty.

## When to use
When you want *one representative per branch* with a crisp, parameter-free notion of "same idea." Set
`mix_fraction = 1.0` for best-per-lineage only; lower it to also include lineage-diverse low-scorers.

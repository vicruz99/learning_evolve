# `best_diverse` — MMR over quality

**Engine:** mmr · **Knobs:** `n_context`, `mmr_lambda` (λ)

## Definition
Maximal Marginal Relevance selection: greedily pick high-quality solutions that are **far apart in the
search tree**, so the context isn't several near-copies of the same idea.

## Algorithm
1. Normalize `quality = value` to `[0,1]` over the candidates.
2. First pick = highest quality.
3. Each subsequent pick maximizes `λ·quality(c) − (1−λ)·max sim(c, picked)`, where
   `sim = 1/(1 + tree_distance)` (parent–child closest, then siblings, then across branches).
4. Repeat until `n_context` chosen.

## Intuition
`best` shows the frontier but often redundantly; `best_diverse` keeps quality high while spreading the
picks across the tree, so the model sees *different good ideas* rather than variations of one. The dial
`λ` interpolates: `λ=1` = `best` (quality only), `λ=0` = maximum spread, default `0.7` (quality-led,
but a same-parent near-clone gets docked and loses to a slightly-worse solution from another branch).
Because the penalty is *max* similarity to the already-picked set, once one member of a lineage is in,
its close relatives are strongly suppressed. Similarity is pluggable (lineage now; embeddings later)
and needs no embedding model. See MMR in `_concepts.md`.

## When to use
The recommended upgrade over `best` whenever the top-n tend to collapse onto one layout/idea. Sweep
`λ` to find the quality/diversity sweet spot.

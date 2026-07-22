# Context-selection strategies — shared concepts

In the ICL experiments the frozen model is conditioned on a block of **past solutions** injected into
the prompt. A *context-selection strategy* answers two questions:

1. **Selection** — *which* buffered solutions to show.
2. **Rendering** — *what* of each solution to show (code, `<strategy>` reasoning, or both).

These are orthogonal: any strategy can be rendered with any combination of the render flags.

All strategies live in `context/selection.py`, registered in `STRATEGIES`, and are driven by
`ICLConfig` (`--context-strategy` and the knobs below). Each has its own doc in this folder.

---

## The state buffer is a tree

Every buffered `State` records its full ancestor chain (`state.parents`, most-recent-first, and the
matching `state.parent_values`). So the archive is a **search tree**: seeds are roots, and each new
valid solution hangs off the parent it was generated from. Two structural notions drive the
diversity-aware strategies (helpers in `context/lineage.py`):

- **Same lineage (path relationship).** `a` and `b` are on the same root-to-node path iff one is an
  ancestor of the other. Siblings/cousins are *not* the same lineage (they merely share an ancestor).
  Used by `per_lineage` as a hard skip.
- **Tree-edge distance.** Number of edges between two nodes via their most-recent common ancestor:
  parent–child = 1, siblings = 2, cousins = 4, … We turn it into a soft similarity
  `sim = 1 / (1 + dist)` (1.0 = same node, → 0 as they get far apart). Used by the MMR strategies.

Why lineage as a proxy for redundancy: a child that barely changed its parent, or several children of
the same parent, tend to be near-duplicates. Being far apart in the tree makes two solutions *likely*
(not guaranteed) to be genuinely different — and it's free, needing no embeddings.

## Ranking keys (`context/ranking.py`)

Scalar signals, higher = more preferred:

- **quality** — the buffered `value` (already normalized so higher is better; minimize envs store the
  negated raw score).
- **jump** — improvement over the immediate parent, `value − parent_values[0]` (seeds → 0). "How much
  did this solution add on top of where it started."
- **recent** — creation `timestep` (newest first).

## MMR (Maximal Marginal Relevance)

A greedy way to trade quality against redundancy. Pick solutions one at a time; each step maximizes

```
lam * quality(c)  −  (1 − lam) * max_similarity(c, already_picked)
```

`lam ∈ [0,1]` is the dial: `lam=1` → pure quality (= `best`), `lam=0` → pure spread. **MMR does not
require embeddings** — `similarity` is pluggable; here it is the lineage tree-distance similarity
above. `quality` is min-max normalized to `[0,1]` so `lam` is meaningful.

## Shared knobs (`ICLConfig` / CLI)

| knob | flag | used by | meaning |
|---|---|---|---|
| `n_context` | `--n-context` | all | number of past solutions in context (the main hyperparameter) |
| `mix_fraction` (x) | `--mix-fraction` | `best_worst`, `best_jump`, `per_lineage`, `contrastive` | fraction of `n_context` taken from the primary/"best" pool; the rest from the secondary pool |
| `mmr_lambda` (λ) | `--mmr-lambda` | `best_diverse`, `informative`, `contrastive` | quality↔diversity trade-off |
| `jump_alpha` (α) | `--jump-alpha` | `informative` | blend value(α) vs jump(1−α) in the MMR quality term |
| `context_seed` | `--context-seed` | `random` | reproducibility |
| `max_context_tokens` | `--max-context-tokens` | all (rendering) | approx cap on block size (chars/4); trims lowest-ranked first |

## Rendering flags (orthogonal; apply to every strategy)

The prompt template asks the model to emit its plan between `<strategy>…</strategy>` before the code.
We capture that text on each `State` (`state.strategy`), so context can show reasoning, code, or both:

| flag | flag | effect |
|---|---|---|
| `include_code` | `--include-code` / `--no-include-code` | show each solution's code (default **on**) |
| `include_strategy` | `--include-strategy` | show each solution's `<strategy>` reasoning (default **off**) |

"Strategy-only" = `--no-include-code --include-strategy`. Guard: if strategy display is requested but a
solution has no `<strategy>` text, it falls back to showing code so a solution is never rendered empty.

## What the strategies select over: the context pool (not the PUCT search buffer)

There are **two** distinct pools, and context selection uses the second:

- The **PUCT search buffer** (`sampler._states`) is what *parents* are sampled from. It is pruned:
  `topk_children` keeps only the best few children per parent, and `max_buffer_size` caps the total,
  biased toward the highest scorers. Good for search — but it holds almost no low-scoring solutions.
- The **context pool** (built in `icl/loop.py`, mirrored to `buffer/context_pool.jsonl`) holds **every
  valid solution graded in previous generations**, unpruned. All context-selection strategies operate
  over *this* pool.

This split is deliberate: drawing context from the pruned search buffer would leave `best_worst` /
`contrastive` with no genuine low-scorers to use as negatives — their "worst" would just be the
least-good survivor. Selecting over the full valid-solution pool is what makes the negative signal real.
The pool is still a lineage tree (every `State` carries its ancestry), so all the tree/MMR machinery
above applies unchanged.

## Note on "worst" / failures

The pool stores only **valid** solutions — a failed rollout is never graded valid, so it never enters
the pool (it only advances PUCT visit counts). So "worst" and the contrastive negatives mean
**lowest-scoring valid** solutions, not broken code. Injecting genuine failures (code + error message)
would need a separate failure log; that is intentionally deferred.

## Tie-breaking among equal scores

Many solutions share a score (e.g. lots of zero-scoring attempts). To avoid always surfacing the same
few — an artifact of stable sorting on ties — candidates are **pre-shuffled** before ranking, then
sorted by score with a stable sort. Equal scores therefore come out in random order, while the order of
*distinct* scores is untouched. So repeated calls resample which of the tied solutions appear (relevant
for the "worst" blocks especially). Set `context_seed` to make this reproducible; leave it unset for a
fresh draw each call.

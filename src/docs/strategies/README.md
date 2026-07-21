# ICL context-selection strategies

How past solutions are chosen and rendered into the prompt for the ICL discovery runs. Start with
[`_concepts.md`](./_concepts.md) for the tree/lineage, ranking keys, MMR, and rendering flags shared
by everything below.

Select one with `--context-strategy <name>`; tune with the knobs noted in each doc.

| strategy | axis it probes | one-line |
|---|---|---|
| [`random`](./random.md) | baseline | random sample of past solutions |
| [`best`](./best.md) | quality | the top-`n` by score |
| [`recent`](./recent.md) | recency | the most recently generated |
| [`biggest_jump`](./biggest_jump.md) | improvement | biggest gain over parent |
| [`best_worst`](./best_worst.md) | quality + failures | x% best, (1−x)% worst |
| [`best_jump`](./best_jump.md) | quality + improvement | x% best, (1−x)% biggest-jump |
| [`per_lineage`](./per_lineage.md) | quality + hard diversity | best (and worst) with same-lineage skipping |
| [`best_diverse`](./best_diverse.md) | quality + soft diversity | MMR over quality |
| [`informative`](./informative.md) | quality + improvement + diversity | MMR over quality+jump |
| [`contrastive`](./contrastive.md) | quality + failures + diversity | diverse best + stratified low→mid negatives |

These are ten *presets* over four engines (`topk`, `mix`, `per_lineage`, `mmr`); several differ only
by parameters. See the table in `context/selection.py` for the engine mapping.

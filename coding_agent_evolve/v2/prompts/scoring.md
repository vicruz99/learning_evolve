**Official scoring.** The only score that counts is the one printed by

```bash
{{ PYTHON }} eval.py <candidate>.npy
```

run from your working directory. Every such call is recorded by the host: the construction is
snapshotted, its {{ P.metric_word }} is logged, and the best one is kept automatically — you do not have to
copy, promote or register anything, and nothing you write elsewhere is used for the result. A
construction that was never scored with that command does not exist as far as the result is
concerned, so score every candidate you consider worth keeping, as soon as it is ready.
{% if MAX_EVALS %}You have at most **{{ MAX_EVALS }} official evaluations**; the host stops the run when they are used up.
{% endif %}Calls to `{{ P.eval_function }}` imported from `eval.py` inside your own programs are free and
unrecorded — use them as much as you like inside a search. `eval.py`, `_official_evaluator.py` and
`_adrs_track.json` are read-only; do not modify, move or replace them.

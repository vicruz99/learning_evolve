**Official scoring — this is very important.** The only score that counts is the one printed by

```bash
{{ PYTHON }} eval.py <candidate>.npy
```

run from your working directory. **Every time you want to know how good a candidate is — to test it,
compare it, decide whether to keep it, or pick a direction — run that command on it.** That is how
you evaluate solutions in this task; there is no other way that counts. Every such call is recorded
by the host: the construction is snapshotted, its {{ P.metric_word }} is logged, and the best one is kept
automatically — you do not have to copy, promote or register anything, and nothing you write
elsewhere is used for the result. A construction that was never scored with that command does not
exist as far as the result is concerned.

Do not build your own scorer, re-implement the formula, or trust a number printed by your own
program as the value of a candidate: the official evaluator is already highly optimized, a homemade
version is not faster, and only the official number is real. When your program has produced a
candidate, save it as `.npy` and score it officially **right away**, before you judge it or build on
it. Scoring many candidates is cheap and expected; leaving a promising candidate unscored is the
one mistake you cannot recover from.
{% if MAX_EVALS %}You have at most **{{ MAX_EVALS }} official evaluations**; the host stops the run when they are used up.
{% endif %}Inside a candidate program's own search loop you may import `{{ P.eval_function }}` from `eval.py` to
steer the search (those inner calls are not recorded) — but the moment that program has a result,
the result is scored with the official command, nothing else. `eval.py`, `_official_evaluator.py`
and `_adrs_track.json` are read-only; do not modify, move or replace them.

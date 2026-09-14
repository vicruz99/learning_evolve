# GPU-mode kernel evaluation (local)

Self-contained harness for scoring a GPU-mode kernel submission on a local GPU.
Nothing here imports the TTT-Discover repo.

## Quickstart

```bash
python evaluate.py test/candidate.py                 # official score
python evaluate.py my_kernel.py --mode test          # correctness only, fast
python evaluate.py my_kernel.py --gpu 1 --repeats 3  # pin a GPU, measure noise
python evaluate.py my_kernel.py --json out.json      # machine-readable results
```

Exit code is 0 only if everything passed. The score is printed as
`SCORE (geom of N benchmarks): <us>`; lower is better.

## Layout

| path | what it is |
|---|---|
| `evaluate.py` | the runner — the only thing you invoke |
| `trimul/` | the task: `task.yml` plus the frozen harness (`eval.py`, `reference.py`, `task.py`, `utils.py`) and a baseline `submission.py` |
| `test/candidate.py` | the TTT-Discover published kernel, byte-identical to `discover/results/kernel-engineering/trimul.py` |
| `variants/` | block-size retunings of that kernel, used to check local-vs-Modal ranking transfer |
| `regime_test.py` | a *failed* experiment, kept as a documented negative (see its docstring) |

`test/trimul/` is a byte-identical duplicate of `trimul/`. `evaluate.py` resolves
`--task trimul` to `trimul/` first (`find_task_dir` checks `HERE/<task>` before
`HERE/test/<task>`), so the duplicate is dead weight: it is **deliberately not
committed** while everything else here is, and is safe to delete locally.

This harness is what `src/envs/kernel_trimul.py` shells out to for the `trimul_a100` /
`trimul_h100` ICL problems — it is a dependency of those runs, not just a manual tool.
See `src/gpumode_local/reference/README.md` for the reference scores per GPU and for
setting `TRIMUL_EVAL_PYTHON` / `TRIMUL_EVALUATE_PY` on a machine that is not guadiana.

## Modes

| mode | what runs | produces a score |
|---|---|---|
| `test` | all 18 `tests:` shapes, correctness only | no |
| `benchmark` | the 7 `benchmarks:` shapes, correctness checked once up front | yes |
| `leaderboard` (default) | `tests:` first, and only if all pass, the timed benchmarks with correctness re-checked every rep | yes — this is the ranked path |

## What the score is

The geometric mean of the per-shape mean runtimes, in microseconds. This
reproduces `libkernelbot.submission.compute_score` with `ranking_by: geom`, and
the inputs handed to `eval.py` (case strings, file set, timeouts) have been
verified byte-identical to what the official runner produces.

## Measurement caveats

- **Use an idle GPU.** A shared card makes timings meaningless; `eval.py`'s own
  convergence rule (relative error < 0.1%) can never be met on one, so every
  benchmark burns its full rep budget and is noisy anyway.
- **~2% resolution floor.** Run-to-run spread is ~0.3-1% on an idle card within
  a session, but drifts ~1.2% across sessions on an identical file. Use
  `--repeats` before believing a small win.
- **A100 numbers here run ~2% above Modal**, concentrated in the small shapes
  (launch overhead), and variant *rankings* transfer (6/6 pairwise orderings
  agreed across the two machines). Don't expect rankings to transfer across
  GPU architectures — occupancy and shared-memory limits differ (a
  `BLOCK_H=128, BLOCK_K=64` config needs 180 KB and dies on A100's 166 KB).

## Harness quirk: the mask is never exercised

`eval.py`'s case-file regex leaves `nomask: False` as the *string* `"False"`,
which is truthy, so `generate_input` hands out an all-ones mask on every shape.
The published kernel reads `cfg.get("nomask", True)` against a `config` dict
that never contains that key, so it skips masking entirely and still passes.
Feeding a real `nomask=False` bool makes it fail with ~1.8M mismatched
elements. Any kernel you evolve here inherits that free pass — don't assume the
masked path works.

## Requirements

`torch`, `triton`, `pyyaml`, `numpy` (see `requirements.txt`). Known-good: torch 2.7.1
(cu126 or cu128 — measured identical), triton 3.3.1, Python 3.13, matching the
official harness image. A working env is at
`/scratch/vicstorage/learning_evolve/.venv`.

## Adding a task

Drop a folder next to `evaluate.py` (or under `tasks/`) containing `task.yml`
and the files it references, then `--task <name>`. Python (`lang: py`) tasks
only.

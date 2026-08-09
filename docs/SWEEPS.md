# Running sweeps (`run_sweep.py`)

A **sweep** is one YAML file expanded into N `run_icl.py` processes, launched by a supervisor that
enforces `max_parallel`, staggers starts, owns one shared Ray head, and keeps a manifest
(`runs/<sweep>/sweep.json`) of what it launched.

Two rules that explain most of the behaviour:

- **The artifacts decide state, not the manifest.** "Complete" means the run dir verifies as complete
  (`results.resume`), never that `summary.json` says so or that a process exited 0.
- **One run = one process = one lineage.** `--resume` therefore *restarts* partial runs rather than
  splicing a second process onto them. Continuing mid-run is a separate, deliberate flag.

Run the supervisor under `tmux`: it must stay alive to enforce `max_parallel`, but runs are started
in their own process session, so killing it does **not** kill them.

## The commands

```bash
python run_sweep.py sweeps/ctx_qwen.yaml              # launch (foreground supervisor)
python run_sweep.py sweeps/ctx_qwen.yaml --print-cmds # expand + print; launch nothing
python run_sweep.py --status runs/ctx_qwen            # per-run table: generations, status, pid
python run_sweep.py --stop   runs/ctx_qwen            # drop the queue + SIGTERM every live run
python run_sweep.py --resume runs/ctx_qwen            # restart everything not complete, from gen 0
python run_sweep.py --continue-run runs/ctx_qwen/n10_s2   # continue ONE run where it stopped
```

`--print-cmds` composes with `--resume`/`--continue-run` and moves nothing, so it is always safe to
look first — it prints the exact commands, the launch/skip decision per run, how much a restart would
discard, and how the Ray head would be sized on this box.

### `--resume` — the sweep-level restart

Restarts every run that does not verify as complete, each **from generation 0**; complete runs are
skipped. The old dir is moved to `<run>/stale_<timestamp>/`, never deleted. Use it after a crash, a
wall-clock kill, or a machine going away.

### `--continue-run` — the deliberate mid-run continue

Reuses the command line the manifest recorded for that run and rewinds to a chosen generation, so a
run 12 generations deep does not redo ~7 h of grading.

```bash
python run_sweep.py --continue-run runs/ctx_qwen/n10_s2                  # auto: last verifiable gen
python run_sweep.py --continue-run runs/ctx_qwen/n10_s2 --from-generation 7
python run_sweep.py --resume runs/ctx_qwen --continue-run n10_s2:7       # ...and the rest of the sweep
python run_sweep.py --resume runs/ctx_qwen --continue-run a:8 --continue-run b   # repeatable
```

- `RUN:N` and `--from-generation N` are the same thing; the flag is sugar for the single-run case and
  is an error alongside `:N` or several runs.
- With `--resume` naming the sweep, a bare run **name** works; on its own, pass the run **directory**.
- `N` is checked against the run's PUCT snapshots and refused if that one did not survive (the error
  lists what did). `0` restarts just that run. A *complete* run needs an explicit generation before it
  will redo anything.
- Generations from `N` on move to `<run>/stale_<timestamp>/`.

### `--stop`

Writes a halt marker, drops the queue and SIGTERMs every live run. A supervisor watching that sweep
stops launching. Relaunching the sweep is what clears the marker.

### Other flags

`--max-parallel` / `--stagger` / `--refresh` override the sweep file; `--sweep-dir` relocates the
output; `--force` skips the "a run of this sweep is still alive" refusal (which otherwise asks, and
refuses outright with no terminal to ask on).

## Ray

The launcher starts and sizes the head itself (`--ray-head auto`, default) and stops it when the queue
drains — no `ray start` or `OMP_NUM_THREADS` by hand.

```bash
--ray-head auto|require|skip   # auto: start one if none is up; require: attach, fail if none
                               # (what jobs/icl_sweep.bsub uses); skip: leave Ray alone
--ray-num-cpus 16              # cores the head may use, instead of detecting them
```

`--ray-num-cpus` (or `sweep.ray.num_cpus`) **is** the final `--num-cpus`: `reserve_cpus` is not
subtracted on top and the rest of the allocation is left idle. Use it to share a node, leave room for
a server on the same host, or reproduce a run made on a smaller box. It only applies where a head is
actually started (`auto`); elsewhere it is reported as ignored. See `docs/BOSCH_CLUSTER.md` for LSF.

## Use cases

| Situation | Command |
|---|---|
| Check a sweep file / this box before committing to it | `--print-cmds` |
| Job hit its wall clock; most runs are partial | `--resume runs/<sweep>` |
| One long run died 12 generations in; the rest are fine | `--resume runs/<sweep> --continue-run <run>` |
| Continue at a specific generation (later ones look wrong) | `--continue-run <run>:7` |
| Redo one run from scratch, leave the sweep alone | `--continue-run <run>:0` |
| "Did anything actually launch?" | `--status runs/<sweep>` |
| Stop everything now | `--stop runs/<sweep>` |
| Share the machine with another job | `--ray-num-cpus N` |

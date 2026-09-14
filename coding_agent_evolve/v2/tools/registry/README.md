# `registry` — candidate-solution registry for coding-agent runs

One stdlib-only CLI (`registry.py`, installed as `~/.local/bin/registry`) that gives a coding agent a
place to **submit** every candidate solution it produces, to **consult** what it has tried, and that
gives the operator a faithful record of **how the solutions evolved**. It replaces the hand-written
`run/LEDGER.md` / `run/best.npy` convention, whose failure modes in the 2026-09 AC2 campaign motivated
every field below (unpromoted bests, ledger numbers that were bounds not scores, three champion values
in one file, a ledger truncated to 0 bytes, lineage only in prose).

Two levels:

| level | command | who | what |
|---|---|---|---|
| **problem** | `registry problem add NAME …` | operator, once | evaluation script + auxiliary files + direction + timeout + baseline, stored in `$REGISTRY_HOME/problems/NAME/` (default `~/.local/share/registry`) |
| **table** | `registry init --problem NAME` | operator, per run folder | copies the problem's files into the run folder, creates `run/registry/registry.sqlite`, evaluates the baseline as candidate **id 0** |

Then the agent uses `submit` / `list` / `show` / `best` / `lineage` from anywhere inside the run folder.

## Install

```bash
bash tools/registry/install.sh          # copies registry.py to ~/.local/bin/registry, registers AC1/AC2/Erdos
registry problem list
```

It installs a *copy*, not a symlink: editing the repo while cells run cannot change the tool they
use. Every run records the installed copy's sha256 in its config (`registry config`).

## Operator: create a table

```bash
cd ~/agent_runs/<cell>
registry init --problem AC2 --baseline-expect 0.6667          # fails if the seed does not score that
registry init --problem AC2 --eval-policy every-5             # see policies below
```

`init` copies `eval.py` and the seed into the run folder (agents keep importing `eval.py` as before),
creates `run/registry/{registry.sqlite,artifacts/,programs/,eval_logs/}`, inserts and evaluates the
baseline as id 0, and writes `run/best.<ext>` + `run/registry/best.json`.

## Agent: submit and consult

```bash
registry best                                  # verified best so far
registry list --last 20                        # chronological table; * = best, ? = unverified, ! = claimed≠verified
registry list --grep gauss                     # did I already try this idea?
registry submit --desc "coarse-to-fine, n=4096" --artifact run/attempts/c2f.npy \
    --program run/attempts/c2f.py --claimed 0.8731 --runtime 412 \
    --config '{"n":4096,"seed":3,"lr":0.02}' --parents 12 --tag c2f --sources "AlphaEvolve app. B"
registry submit --desc "run crashed: TypeError in refine()" --status failed --program run/attempts/x.py
registry submit --kind note --desc "flat-top bound: C<=0.9529 for q=8" --tag bounds
registry show 17          # every field, evaluator log path, artifact_drift check
registry lineage 17       # ancestors via --parents, and children
```

`submit` prints fixed `key=value` lines (or `--json`):

```
id=17
status=done
kind=candidate
claimed_score=0.8731
score=0.872498
eval=ok (3.2s)            # or: skipped (policy=…) | invalid (INVALID: …) | timeout (…) | error (…)
best=0.872810 (id=12)
delta=-0.000312
new_best=no               # yes → extra line  promoted=run/best.npy
warning=claimed differs from verified by 6.0e-04   # only when both present and they disagree
duplicate_of=9            # only when the artifact or program sha256 is already registered
artifact=run/registry/artifacts/000017.npy
```

Rules the tool enforces:
- **`score` is written only by the evaluator.** The agent's number is `claimed_score`. `best`,
  `delta` and the `*` marker use `score` only.
- **The registry owns copies.** Artifacts go to `run/registry/artifacts/<id>.<ext>`, programs to
  `run/registry/programs/<id>.py`; the evaluator runs on the copy; `show` reports `artifact_drift`
  if the original changed. Artifacts above `--max-copy-mb` (default 64) are referenced by path+sha256.
- **`run/best.<ext>` and `run/best.py` are maintained by the registry** (atomic replace on a verified
  new best). Agents must not write them.
- **Notes carry no score.** `--kind note` rejects `--claimed`/`--artifact`; bounds and analyses can
  be recorded without polluting the score column.
- **No edit, no delete.** Rows are append-only; `--parents` must reference existing ids.

## Evaluator hook and policies

The problem's `--eval` command runs with `{artifact}` / `{program}` substituted by the stored copy,
`cwd` = run folder, under the problem's timeout. Stdout is parsed with the problem's `--score-regex`
(default matches `SCORE = x`, `C5 = x`, `SCORE (geom of N benchmarks): x`, last match wins). Outcomes:
exit 0 + finite number → `ok`; non-zero exit, an `INVALID…` line, `inf`/`nan` → `invalid` (status
becomes `invalid`); timeout → `timeout`; no score line → `error`. Full output in `run/registry/eval_logs/<id>.log`.

| `--eval-policy` at init | when the evaluator runs at `submit` |
|---|---|
| `on-claimed-best` (default) | only when `--claimed` strictly beats the verified best; no claim → `skipped` (`?` in list) |
| `always` | every done candidate (cheap evaluators) |
| `every-N` | as on-claimed-best, plus every Nth done submission |
| `never` | never at submit; verify later with `registry eval` |

`registry eval ID` re-runs one; `registry eval --pending [--limit N] [--timeout S]` sweeps all done
candidates without a verified score, highest claim first — for the operator or a keepalive cycle,
off the agent's critical path. Exact AC2 grading is O(n²) (minutes at n=200k), which is why the
default policy is not `always`.

## Operator: reading the evolution

- `registry list --all --sort id` — the chronological record; `--sort score` — leaderboard.
- `registry lineage ID` — parent chain with scores.
- `registry export [--csv] [PATH]` — one row per candidate (JSONL default) for `grade_all.py`-style
  analysis; or open `run/registry/registry.sqlite` read-only after the job ends
  (`sqlite3.connect("file:…?mode=ro", uri=True)`).
- Every row keeps its artifact copy, so any point of the history can be re-graded.

## Schema

```sql
config(key, value)  -- problem, direction, eval_cmd, eval_timeout_s, eval_policy, eval_every_n,
                    -- eval_counter, score_regex, artifact_ext, max_copy_mb, journal_mode, tool_sha256, …
candidates(id, created_at, status{done,failed,timeout,invalid}, kind{candidate,note}, description, tag,
           claimed_score, score, eval_status{ok,invalid,timeout,error,skipped}, eval_error, eval_runtime_s,
           runtime_s, config JSON, artifact, artifact_sha256, artifact_bytes, program, program_sha256,
           parents JSON, sources, error)
```

SQLite in WAL mode with `busy_timeout=60000`, short `BEGIN IMMEDIATE` transactions and a retry wrapper;
`init` probes WAL from a child process and falls back to `journal_mode=DELETE` (recorded in config)
if the filesystem refuses shared memory. Tested with 24 concurrent submitters. Do not read the
database from another host while the job is writing it (WAL is single-host); use `export` inside the
job or read after it ends.

## Suggested prompt block

Replaces the `run/best.*` / `run/LEDGER.md` / `run/BEST.md` instructions. Put the *same* text in the
plain and evo variants so they still differ in exactly one block.

```markdown
## Candidate registry (mandatory)
This run folder has a registry (`registry help`). Start every session with `registry best` and
`registry list --last 20`. Register **every** candidate you run, including failures:

    registry submit --desc "one line: the idea" --artifact run/attempts/NNN.npy --program run/attempts/NNN.py \
        --claimed <your measured score> --runtime <seconds> --config '{"n": ..., "seed": ...}' \
        --parents <ids it builds on> --tag <approach family> [--sources "paper / idea origin"]
    registry submit --desc "..." --status failed|timeout --error "..." --program run/attempts/NNN.py
    registry submit --kind note --desc "bound / analysis, not a candidate"

`score` is written only by the official evaluator; your own number is `claimed_score`. `run/best.npy`
and `run/best.py` are maintained by the registry — never write them. Before trying an idea, check
`registry list --grep TERM`; use `registry show ID` and `registry lineage ID` to revisit earlier work.
```

## Tests

```bash
python3 tools/registry/test_registry.py -v      # toy evaluator; policies, invalid/timeout, notes, lineage, 24-way concurrency
cd ~/agent_runs/some_dir && bsub < tools/registry/wekatest.bsub   # compute node: WAL on WekaFS, 24 concurrent submits, n=200k eval
cd ~/agent_runs/some_dir && bsub < tools/registry/dryrun.bsub     # compute node: full AC2 dry run against the real eval.py
```

Run the LSF scripts on a compute node, not the login node: the login node is cgroup-capped and often
at load average 100+, where a sub-second `eval.py` call takes minutes. Both write to
`~/agent_runs/registry_{wekatest,dryrun}` and can be deleted afterwards. Verified 2026-09-07 on
w24c02: all checks pass; the exact AC2 evaluator took 2.4 s at n=200 000 there.

## Files

```
tools/registry/registry.py        the CLI (stdlib only; shebang = the agent-eval venv python)
tools/registry/install.sh         install copy + register AC1/AC2/Erdos problems
tools/registry/test_registry.py   unittest suite
tools/registry/README.md          this file
tools/registry/wekatest.bsub      LSF verification: WAL + concurrency + n=200k eval on a compute node
tools/registry/dryrun.bsub        LSF verification: full AC2 dry run
```

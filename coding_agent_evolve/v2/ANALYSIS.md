# v2 results and transcripts — where everything is

Hand-off for an analysis session on `rng-dl01` (`ssh cluster`). Written 2026-09-16.
Companion to `GUADIANA.md` (the Kimi port). Everything here is **read-only work**; do not
touch a cell that is still running.

`$S` below means
`/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/vicruz`.

---

## 0. Read this first

- **All run data lives on scratch.** `~/agent_runs/v2/<campaign>` is a symlink into
  `$S/agent_runs_v2/<campaign>`. Home has a ~100 GB quota that weka reports as
  `[Errno 28] No space left on device`; it killed a whole wave on 2026-09-15. Never write
  analysis output to `$HOME`. `mkcell` now refuses to create cells there.
- **`q38ac_r3` is still running** (as of 2026-09-16). Report health only for it; do not draw
  conclusions from its endpoints until every cell has spent its 18 h budget.
- **The finished campaigns are `q38ac_r2`, `q36ac2_r2` and `q36ac2_r3`.** The `q38ac` /
  `q36ac2` (no suffix) campaigns are the aborted 2026-09-08 pilots, all `.hold`, run under
  different config — **excluded from analysis**.
- Score direction: **AC1 is minimised** (ICL best 1.50444, published 1.50287).
  **AC2 is maximised** (ICL best 0.96137, AlphaEvolve 0.9610).

## 1. Run data

```
$S/agent_runs_v2/
  q38ac_r2/     24 cells   Qwen3.8, AC1+AC2, 3 prompts x 2 harnesses x 2 seeds   FINISHED
  q36ac2_r2/     6 cells   Qwen3.6, AC2, 3 prompts x 2 harnesses, seed 1         FINISHED
  q36ac2_r3/     2 cells   Qwen3.6 reruns                                        FINISHED
  q38ac_r3/     15 cells   Qwen3.8 reruns                                        RUNNING
  q38ac_r3_aborted_20260915/   the first r3 attempt, superseded — ignore
  rsm1 rsm2 rsm3 rsm36 smoke*  smoke tests (0.4 h budgets) — ignore
  _analysis/    RUNS_LEDGER.md, runs_ledger.json
  _transcripts/ rendered transcripts + INDEX.md
```

### What a cell contains, and which files lie

| file | trust | notes |
|---|---|---|
| `score.json` | **authoritative** | host re-grade of `best.npy` after the run |
| `best.json` | **authoritative** | best official eval, with the snapshot it came from |
| `iterations.jsonl` | **authoritative** | one line per *official* `eval.py` call: `t`, `score`, `elapsed_s`, `eval_s`, `snap` |
| `submissions/NNN.npy` | authoritative | the graded snapshot for iteration NNN |
| `clock.json` | authoritative | `active_s` = real budget consumed across launches. **Align on this, never wall time** |
| `session.json` | authoritative | `outcome` (`time_limit` = ran its full budget) |
| `cell.json` | careful | the config, but it was **re-rendered** during r2; a running cell keeps its launch-time copy |
| `events.jsonl` | **lies for bnbcode** | goes silent at the first compaction, because bnbcode continues in successor sessions the driver never sees. Fine for Claude Code |
| `pgbackup.sql` | ground truth for bnbcode | the whole session store; `part.time_created` is the real activity clock |
| `driver.log`, `lsf.out/err` | authoritative | launch boundaries, turn ends, `TERM_*` reasons |
| `workspace/` | careful | reaped at the end; `.npy` files here are candidates the agent never officially scored |

## 2. The run-health ledger — start here

`analysis/ledger.py` already classifies every finished cell.

```bash
V2=~/work/learning_evolve/coding_agent_evolve/v2
$V2/analysis/ledger.py --campaigns q38ac_r2 q36ac2_r2 q36ac2_r3 --out $S/agent_runs_v2/_analysis
```

Output: `RUNS_LEDGER.md` (human) and `runs_ledger.json` (machine), already generated
2026-09-14 for the r2 campaigns. Per cell it gives `class` ∈ {clean, resumed, wedged,
never_ran}, `active_h`, **`healthy_h` and `healthy_until`**, evals and best *inside the
healthy window*, and flags. `--event-rows` adds the postgres row counts (slow, ~2 min).

**Use `healthy_until` in every time plot**: a wedged cell must be drawn solid up to it and
dashed after, because the driver kept burning budget while the agent was dead.

Headline for r2: 11 clean, 3 resumed, 6 wedged, 10 never dispatched.

## 3. Transcripts

Rendered ones (r2 only, 2026-09-12):

```
$S/agent_runs_v2/_transcripts/INDEX.md          table: cell, evals, best, link
$S/agent_runs_v2/_transcripts/q38ac_r2/*.txt    41 files
$S/agent_runs_v2/_transcripts/q36ac2_r2/*.txt   17 files
```

`*.txt` is the conversation; `*.thinking.txt` keeps the chain of thought.

To render more (r3, once it finishes):

```bash
$V2/bin/render_transcripts q36ac2_r3 q38ac_r3          # whole campaigns, writes into _transcripts
$V2/bin/transcript  <cell_dir> --thoughts              # one cell, from events.jsonl (Claude Code)
$V2/bin/db_transcript <cell_dir> --thoughts [--host N] # one cell, from the bnbcode session DB
```

**For bnbcode cells always use `db_transcript`, not `transcript`.** `events.jsonl` stops at
the first compaction, so an `events.jsonl` transcript silently truncates the run. Five r2
cells show `session database unavailable` in INDEX.md because their node-local `/tmp/v2pg`
is gone; their `pgbackup.sql` survives, so restore it into a temporary
`bin/bnbcode-pg-node` on a **compute node** and re-run `db_transcript`.

## 4. Notebooks and plots

```
$S/v2_analysis/                     (symlinked as $V2/notebooks/v2_analysis)
  v2_interim_analysis.ipynb         built at the 7.4 h mark, has a hand-written KNOWN fault dict
  v2_final_health.ipynb             the health notebook
  mknb*.py                          the scripts that BUILD the notebooks — edit these, not the .ipynb
  collect.py collect2.py ...        data loaders (iterations, .npy walk, liveness)
  run_nb_short.bsub run_nb*.bsub    executors
```

Workflow: edit `mknb*.py` → regenerate the notebook → execute on a compute node:

```bash
cd $S/v2_analysis && bsub < run_nb_short.bsub     # 30 min, queue short, writes .ipynb + .html
```

Python is `$S/venvs/analysis/bin/python` (3.12.11, numpy 2.5.2, pandas 3.0.5, nbconvert).
**Not** `~/venvs/analysis`, which is empty. Never execute notebooks on a login node, and
keep `OMP_NUM_THREADS=1` as the bsub does.

**The interim notebook needs updating**: replace its hand-written `KNOWN` dict and its
"last eval < 2 h ago" liveness test with the ledger's `class` / `healthy_until`.

## 5. Caveats that change conclusions

- **Official eval counts undercount the search.** The prompt did not require `eval.py` for
  every candidate. `.npy` written vs officially scored: 340/53, 618/288, 19845/2156. Only
  the `many` prompt kept the ratio near 1:1. **Rank on distinct scores or tokens, not rows.**
  In-process evaluations are unrecorded everywhere.
- **One cell is score-contaminated**: `q38ac_r2/ac2_evo_cc_rxhigh_s1` read a sibling's best
  score out of the process table at 04:14 and searched toward it. Keep it for harness/cost
  analysis, **exclude it from the plain/evo/many comparison**.
- **There is no clean seed pair anywhere in r2.** Every prompt or harness contrast is n = 1
  per condition. Say so explicitly.
- **The two campaigns differ in more than the model**: `continual_work` true (q38) vs false
  (q36), `min_evals` 250 vs 300, 32 vs 24 cores, 262k vs 180k context.
- **CPU allocation is double what the prompt states.** LSF slots are physical cores and both
  hyperthreads land in the cgroup, so a 32-slot cell sees `nproc` = 64 while the prompt says
  "use at most 32 CPU cores". Possible per-cell throughput confound; disclose it.
- **`q38ac_r3` cells are resumed runs.** Several carry multiple launches and restored state
  after the 2026-09-15 quota outage; one q36 evo cell took a watchdog restart at 09:36 that
  interrupted live work.
- Wall-clock start times differ by up to 18 h across cells. Align on `clock.json`.

## 6. Older, non-v2 results

```
$S/runs/              ICL campaign runs (19)
$S/runs_guadiana/     imported Guadiana runs (12)
$S/shinka_math_results/
$S/home_offload/agent_runs_v1/   the pre-v2 August campaigns
$S/home_offload/v2.home.bak/     duplicate of the aborted 2026-09-08 pilots — ignore
```

## 7. House rules

- Filter the Proxy4Server banner out of any parsed `ssh cluster` output.
- `scp`/`sftp` are broken here; use `ssh cluster 'cat > path' < file` and **verify the md5**.
- Grading is CPU-heavy: run it under `bsub`, never on a login node.
- Do not re-grade or re-render inside a cell that is still in LSF.

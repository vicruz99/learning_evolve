# Running the v2 cells on Guadiana against Kimi

Hand-off for a Claude Code session on Guadiana (outside Bosch, so it can reach hosted models
such as Kimi, and cannot reach anything on rng-dl01). Written 2026-09-14 from the state of the
rng-dl01 checkout; everything below was read from that code, not remembered.

Goal: the **same experimental condition** as the `q38ac_r2`/`q38ac_r3` and `q36ac2_r2`/`_r3`
campaigns (AC1/AC2 × {plain, evo, many} × {bnbcode, Claude Code} × seeds, 18 h per cell) with
Kimi as the model, laid out on disk so that `analysis/ledger.py` and the notebooks on rng-dl01 can
ingest the cells unchanged afterwards.

## 0. Do not start from `v2/portable/`

`portable/` is the July/August kit. Its `tasks/AC*/prompt_{evo,plain}.md` are the *pre-v2* prompts
(60 differing lines against today's rendered AC2 evo prompt, no `many` variant, no `run/` layout
rules, no tracker). Using them would not be "the same prompts". What is still worth reading there:

- `portable/config/opencode.kimi.jsonc` -- the exact bnbcode provider block for Kimi and the
  `temperature: true` / `tool_call: true` / `reasoning: true` capability-flag trap.
- `portable/config/claude_settings.json` and `portable/bin/runcell` -- the Claude-Code-to-Moonshot
  wiring (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, region host).
- `portable/README.md` -- why the sampler exists and the stop taxonomy.

## 1. What "the same prompt" is, concretely

A cell's prompt is `workspace/INITIAL_PROMPT.md`, rendered by `bin/mkcell` (`driver/mkcell.py`,
needs `jinja2` + `pyyaml`, see `env/requirements.txt`) from

    prompts/base.md.j2  +  problems/<P>/prompt.md.j2  +  prompts/method_<plain|evo|many>.md  +  prompts/scoring.md

The three method files are the **only** thing that differs between plain/evo/many; `PROMPT_DIFF.md`
in every cell folder is the diff against plain. Placeholders and where their values come from:

| placeholder | source | rng-dl01 value (q38 / q36 campaigns) |
|---|---|---|
| `CORES`, `PARALLEL` (= cores // `cpus_per_candidate`) | yaml `cores`, `problems/<P>/meta.yaml` | 32 / 24 cores; AC1, AC2 use 2 cpus per candidate |
| `HOURS` | yaml `hours` | 18 |
| `PYTHON` | yaml `python` (interpreter with numpy the agent calls `eval.py` with) | `~/venvs/agent-eval/bin/python` |
| `MIN_EVALS`, `MAX_EVALS`, `EVALS_PER_HOUR` | yaml `min_evals`, `max_evals` | 250 / 300, 0 (no cap) |
| `BUDGET_S`, `KILL_S` | `meta.yaml` `budget_s`/`kill_s` | 1000 / 1100 s, all problems |
| `P.start_score`, `P.target`, titles | `meta.yaml` | AC2 0.6667 -> 0.97, AC1 2.0 -> 1.5030 |

Two things in the prompt describe the machine and must be **true on Guadiana**, not copied:

1. `prompts/base.md.j2` line 15: "an LSF batch job on the `rng-dl01` cluster. `nproc` reports how
   many cores it was given." Replace that one sentence with the truth about Guadiana. Change
   nothing else in the prompt files.
2. `cores` (and hence the `PARALLEL` number four lines later) must be what the box really gives
   the cell. An agent told 32 cores on a 16-core box oversubscribes and its own candidate timings
   stop meaning anything. Pick a value, keep it fixed across all Kimi cells, and record it.

Start a Kimi campaign yaml as a copy of `campaigns/qwen38_ac_r3.yaml` (the q38 settings:
`continual_work: true`, `min_evals: 250`) or `campaigns/qwen36_ac2_r3.yaml` (`continual_work:
false`, `min_evals: 300`, AC2 only). Say in the yaml header which one it copies; the two differ in
more than the model, so the comparison partner is fixed by that choice. Keep `hours: 18`,
`stall_hours: 2`, `guard_cap: 8`, `patience: 3`, `keep_going: true`, `nudge_min_gap_s: 300`,
`temperature 0.6 / top_p 0.95` (bnbcode).

## 2. What is cluster-only (do not port literally) vs. portable

Cluster-only, replace with a plain runner script on Guadiana:

- `bin/submit` (bsub), `bin/launch` (LSF job body: relay, LiteLLM, node-local postgres,
  pgbackup dump loop, `reap_workspace`), `bin/status`, `bin/health`, `bin/restart_cells`,
  `bin/retune_*`, `relay/llm_relay.py` (finds vLLM servers by LSF job name), `config/litellm_*.yaml`
  and `config/cc_sampling_hook.py` (LiteLLM shim between Claude Code and vLLM; Moonshot speaks the
  Anthropic API natively so no shim is needed), `tools/registry` (shelved).

Portable as-is or with the edits in section 3:

- `driver/run_cell.py <cell_dir>` -- the turn loop, budget watcher (`clock.json`), nudge policy,
  per-turn **stall watchdog** (`stall_hours`, restarts the harness with a fresh session after 2 h
  without an official eval, a new file under the workspace, a tool call or a live worker).
  One external input: it reads `run/procsample.jsonl`, written by `analysis/procsample <cell> 15`,
  which `bin/launch` starts in the background. Start it in your runner too.
- `driver/harness_bnbcode.py` (bnbcode over ACP, needs the `agent-client-protocol` SDK and the
  `bnbcode` binary on PATH), `driver/harness_claude.py` (`claude -p` headless, prompt on stdin,
  `--output-format stream-json`, `--continue` per nudge, `--settings <cell>/claude_settings.json`).
  rng-dl01 ran Claude Code 2.1.234; older versions do not read the prompt from stdin.
- `tracker/` -- `eval_wrapper.py` is installed as `workspace/eval.py` in front of the real grader
  (`_official_evaluator.py`) and records every official evaluation to `<cell>/iterations.jsonl`
  and `<cell>/submissions/NNN.npy`; it kills a grade after `grader_timeout_s` (= `kill_s`, 1100 s,
  written to `_adrs_track.json`). `tracker/score.py <cell>` re-grades the best snapshot after the
  run and writes `score.json` (`--python` or `AGENT_PYTHON`; its default is an rng-dl01 path).
- `prompts/`, `problems/`, `config/opencode.base.json` (bnbcode permission block),
  `config/claude_guard.json` / `claude_guard_stop.json` (Claude Code settings; see 3c),
  `hooks/cc_stop_hook.py`, `analysis/ledger.py`, `analysis/stopclass.py`, `analysis/grade_all.py`.

## 3. Edits needed for Kimi (all small, all in `driver/mkcell.py` and the two harnesses)

a. **Model family.** `mkcell.py` derives everything from `model_family` (`qwen3.8`/`qwen3.6`):
   the bnbcode provider name `vllm38`/`vllm36` with `baseURL http://127.0.0.1:<relay_port>/v1`,
   the `chat_template_kwargs` reasoning-effort option, the LiteLLM yaml, and the Claude alias
   `cc_model = f"{model_family}-{reasoning}"`. Add a `kimi` family:
   - bnbcode provider block = the one in `portable/config/opencode.kimi.jsonc`
     (`baseURL https://api.moonshot.ai/v1` or `.cn` -- a key for one 401s on the other,
     `apiKey "{env:KIMI_API_KEY}"`, `temperature: true, reasoning: true, tool_call: true`,
     `limit.context/output` from the model card), no `chat_template_kwargs`, no LiteLLM yaml.
   - `harness_bnbcode.py`: `self.model_id = f"vllm{fam}/{model}"` -> `f"kimi/{model}"`.
   - `harness_claude.py::env()`: `ANTHROPIC_BASE_URL` = Moonshot's Anthropic-compatible endpoint,
     `ANTHROPIC_AUTH_TOKEN = $KIMI_API_KEY` (never `sk-local`), `ANTHROPIC_MODEL` and the four
     `ANTHROPIC_DEFAULT_*`/`SMALL_FAST` aliases = the Kimi model id. Re-check `MAX_THINKING_TOKENS=0`
     (set for the Qwen/LiteLLM path) against what Moonshot expects for kimi-k2-thinking.
   - `REASONING = ("low","medium","xhigh","off")` is a Qwen chat-template knob; it is also baked
     into the cell name (`ac2_evo_bnb_r<reasoning>_s1`). Keep `reasoning: xhigh` for the name and
     treat it as a no-op for Kimi, and say so in the results.

b. **bnbcode needs PostgreSQL** (session store, with `pg_trgm`). `bin/bnbcode-pg-node` is the
   per-host userspace pgserver helper the cluster uses (`start`, `env`, `dump`, `psql`); it expects
   the `pgserver` build under the bnbcode venv. Any reachable Postgres works: export
   `BNBCODE_DATABASE_URL` before `run_cell.py`. `harness_bnbcode.py` also calls
   `bin/bnbcode-pg-node psql` to find the newest session for a resume; if you do not resume
   (recommended on Guadiana: a dead cell is rerun, not restored) that path is never taken.

c. **`config/claude_guard*.json` hard-code `/home/crv1pi/...`** in the deny rules and
   `claudeMdExcludes`. Rewrite them for Guadiana's home: deny Read/Edit/Write on the repo, `~/.ssh`,
   `~/.config`; keep `defaultMode: acceptEdits` **plus** the explicit `Bash` allow (plain acceptEdits
   prompts on every Bash call and wedges an unattended run), keep `claudeMdExcludes` (otherwise any
   `CLAUDE.md` above the run folder becomes part of the condition), keep the `curl`/`wget`/`pip
   install`/`git push` denies. Note the real Kimi key is in the Claude Code process environment,
   so the agent's Bash can print it; use a key with a spend cap and do not leave it in any file
   the agent can read.

d. **Sampling parity.** On rng-dl01 the Claude arm was forced to temperature 0.6 / top_p 0.95 by
   the LiteLLM hook. Claude Code exposes no sampling knobs, so against Moonshot the Claude arm runs
   at the provider default. Disclose it; do not try to fake it.

e. `tracker/score.py --python` and the yaml `python` must point at a Guadiana interpreter with
   numpy/scipy (the agent's candidate programs and `eval.py` run under it).

## 4. Runner on Guadiana (what `bin/launch` does, minus LSF)

Per cell, in order: `bin/mkcell --cells campaigns/kimi_ac.yaml --campaign kimi_ac_r1 [--only REGEX]
[--seeds 1 2]` (renders `<runs_root>/<campaign>/<cell>/`), then in the background under
`nohup`/tmux with the cell's cores pinned (`taskset`) so cells do not steal from each other:

    analysis/procsample <cell_dir> 15  &          # writes run/procsample.jsonl (stall watchdog input)
    python driver/run_cell.py <cell_dir>          # writes driver.log, events.jsonl, session.json, clock.json, iterations.jsonl
    python tracker/score.py  <cell_dir>           # writes score.json; then touch <cell_dir>/.done

`bin/launch` shows the order and the environment it sets (`HF_HUB_OFFLINE`, `PATH`, the bnbcode
guard variables come from `harness_bnbcode.py` itself). Do not edit `driver/*.py` or the runner
while cells are running: bash reads scripts by byte offset and a running cell keeps its `cell.json`,
so the change would apply to some cells and not others (this corrupted 26 cells on the cluster once).

Before the first 18 h cell, run a **smoke cell**: `--set hours=0.4 --set min_evals=3 --set
min_hours=0.2 --set stall_hours=0.08` and check (1) tool calls happen in both harnesses (bnbcode
without `tool_call: true` is never offered tools and looks like a model that refuses), (2)
`iterations.jsonl` gets rows with real scores, (3) `score.json` appears and matches `best.json`,
(4) for bnbcode the temperature is on the wire (a logging proxy in front of `baseURL`; the cluster's
`tests/wirecheck.sh` does this for the relay), (5) the driver's nudge after the first turn end is
answered.

## 5. What must survive for the rng-dl01 analysis

Keep the cell layout exactly: `<campaign>/<cell>/{cell.json, PROMPT_DIFF.md, driver.log,
events.jsonl, session.json, clock.json, iterations.jsonl, submissions/, best.npy, best.json,
score.json, .done, run/procsample.jsonl, workspace/}`. `analysis/ledger.py --campaigns <c>` builds
the health verdict (clean / resumed / wedged / never_ran) from `driver.log` turn ends,
`iterations.jsonl`, `.npy` mtimes under `workspace/` and `events.jsonl` (Claude only); it needs
those files, not the bnbcode database. `lsf.out` will be absent: the ledger reads launch times from
`clock.json`/`driver.log` when it is.

Bringing results back: pack with the **campaign directory as the archive root**, never a `runs/`
root (an earlier Guadiana zip merged into the live runs tree on rng-dl01 because of that), and
extract on rng-dl01 into `~/agent_runs/v2/<campaign>` via a staging directory. Then
`analysis/ledger.py --campaigns q38ac_r2 q38ac_r3 kimi_ac_r1 ...`. `workspace/run/**` can be
large (candidate programs, npy); keep it, the ledger's `.npy` clock reads it, but drop nothing
else.

## 6. Comparability disclosures to write down before starting

model and endpoint (Kimi id, region host, date), Claude-arm sampling at provider default, `cores`
and the venue sentence, context/output limits used (`bnbcode.context_tokens/output_tokens`,
`claude.max_context_tokens/max_output_tokens` -- set from the Kimi model card, not the Qwen values),
`continual_work` choice (q38 true / q36 false), `min_evals`, Claude Code and bnbcode versions,
`reasoning` no-op. Two seeds per cell; the rng-dl01 campaigns have no clean seed pair, so the
Guadiana runs are the first place a within-cell variance estimate can come from.

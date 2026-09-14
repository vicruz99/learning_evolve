# v2 — clean coding-agent experiment framework (Qwen3.8 · bnbcode + Claude Code · AC1/AC2/Erdős)

Self-contained successor to `../experiments/` + `~/bin/{mkcell,runcell,submitcells,bnb-nudge}`.
Nothing here imports from those; what was worth keeping was copied in (see *Inherited*). Run
folders live in `~/agent_runs/v2/<campaign>/<cell>/`, never inside the repo.

Running these cells off-cluster against a hosted model (Kimi on Guadiana): see `GUADIANA.md`.

## The experiment

Qwen3.8-27B-FP8 (vLLM, agentic servers launched separately) × {bnbcode over ACP, Claude Code
headless} × {AC1, AC2, Erdős} × three prompts that differ in **exactly one block**:

| prompt | method block |
|---|---|
| `plain` | "Find the best solution you can." |
| `evo`   | evolutionary search framework: diverse portfolio, family registry, BLOCKED routes, ledger |
| `many`  | evo + an explicit volume target: N officially scored candidates, time-boxed, machine kept full |

Configurable per cell (`cells.yaml`, every key described in `CONFIG.md`): cores, hours, per-candidate
timeout (`budget_s`/`kill_s`), reasoning effort (`low|medium|xhigh|off`),
minimum official evaluations / hours before the host lets the run end, eval cap, and every
bnbcode knob (continual work, visible-output guard, output-token and tool-output limits).

## Solution registry = Tim's tracker (the agent records nothing)

`tracker/eval_wrapper.py` is installed as `workspace/eval.py` in front of the real grader
(`_official_evaluator.py`). Every `python eval.py candidate.npy` snapshots the candidate to
`<cell>/submissions/NNN.npy`, appends `{i, elapsed_s, score, snap}` to `<cell>/iterations.jsonl`
(`score: null` for an invalid construction), replaces `best.npy`/`best.json` on improvement and
writes `STOP` at the eval cap. `from eval import evaluate_sequence` is re-exported and unrecorded.
The three files are read-only and edit-denied in both harnesses. After the run
`tracker/score.py` re-grades the tracker's best snapshot with the pristine grader — nothing the
agent copied or wrote by hand is ever the result. (Ported from `arxiv_graph/baselines/adrs_acp/`.)

## Driver

`driver/run_cell.py` (port of `adrs_acp/run.py::run_session`): the agent's turn end is the stop
signal; a 1 Hz watcher owns the hard budget; `policy.py` decides whether to nudge
(`[host] N min remaining; K official evaluations (minimum M)` + `prompts/continuation.md`).
First stop always continued; while `evals < min_evals` or `elapsed < min_hours` or nothing better
than the start is scored, continue immediately; afterwards throttle to one nudge per
`nudge_min_gap_s` (`keep_going: false` lets the run end instead). Turns that never reached the
model (fault signature in stderr, or error in < 60 s) are infrastructure: back off, restart, never
a model stop. `patience` barren turns → fresh session in the same workspace (context reset, disk
kept). `give_up` barren turns → `outcome: refused` (default 0 = never).

Backends: `harness_bnbcode.py` (`bnbcode acp`, SDK `agent-client-protocol`; per-cell config via
`workspace/opencode.json` **and** `OPENCODE_CONFIG_CONTENT`; env `BNBCODE_VISIBLE_OUTPUT_GUARD[_CAP]`,
`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`) and `harness_claude.py` (`claude -p … --output-format
stream-json`, `--continue` per nudge because headless Claude Code exits after one Stop-hook block;
reasoning chosen by pointing `ANTHROPIC_MODEL` at the LiteLLM alias `qwen3.8-<effort>`).

### Relaunch = resume (added 2026-09-08 after the quota outage)

A cell can be killed (quota outage, node reboot, `bkill`, LSF walltime) and resubmitted onto the
same folder; `bin/health --resubmit-dead` does this automatically. What survives, by design:

| what | where | mechanism |
|---|---|---|
| budget | `<cell>/clock.json` | `driver/clock.py`: the budget is **active** time accumulated across launches, saved every 30 s; a relaunch gets only the remaining hours and `session.json` records `launches` |
| official scores, snapshots, best | `<cell>/iterations.jsonl`, `submissions/`, `best.*` | tracker files are append-only on the shared filesystem; `t0` stays the first start |
| agent files | `<cell>/workspace/run/` | never touched |
| Claude Code conversation | `<cell>/.cc/projects/*/*.jsonl` | `harness_claude.start()` finds the transcript and passes `--continue` on the first prompt |
| bnbcode conversation | `<cell>/pgbackup.sql` | `bin/launch` dumps the per-cell node-local postgres every 2 min (and at exit) with `bin/bnbcode-pg-node dump`; on relaunch it wipes the node's stale `/tmp` copy, restores the dump, and `harness_bnbcode.start()` adopts the newest session row via ACP `session/load` |

What the agent is told: if the conversation was restored, only the host line plus `RESUME_RESTORED`
(interrupted by infrastructure, files and scores intact, candidates mid-evaluation were not scored,
continue). If it could not be restored (no dump, `--continue` refused), the full initial prompt plus
`RESUME_NOTE` (read `run/` first, continue from the best result). `driver.log` says which. What is
lost: candidates being evaluated at the kill, and up to 2 min of bnbcode conversation / 30 s of clock.
Before this the first campaign's relaunches were all cold starts with a fresh 18 h clock; see the
2026-09-08 session notes.

## Run it

```bash
bin/mkcell --campaign c1                    # render every cell in cells.yaml (36 by default)
bin/mkcell --campaign smoke --only 'ac2_plain_bnb' --seeds 1 --set hours=0.5 --set min_evals=3 --set min_hours=0.2
bin/submit c1 [--only REGEX] [--dry]        # bsub -q batch_cpu -P BH-000557-01 -n <cores> ... exec bin/launch <cell>
bin/status c1                               # LSF state, evals, best, last activity, outcome
```
`bin/launch` (what the job runs): one relay per cell (`relay/llm_relay.py`, follows the vLLM jobs and
probes tool calling; port = campaign hash + cell index, server list rotated per cell so concurrent
cells spread over the GPUs but each stays sticky to one server) → one LiteLLM per Claude cell
(`<cell>/litellm.yaml`) → node-local
postgres for bnbcode (`~/bin/bnbcode-pg-node`) → `driver/run_cell.py` → `tracker/score.py` → reap the
relay/LiteLLM if this was the last v2 cell on the host (detached children keep the LSF job RUN).

Deploy edits with rsync (temp file + rename), never by overwriting `bin/launch` in place while jobs run
(bash reads scripts by byte offset; the campaign corrupted 26 running cells that way).

Per cell: `cell.json` (resolved config), `PROMPT_DIFF.md` (vs plain), `driver.log`, `events.jsonl`
(raw harness stream), `session.json` (turns, nudges, outcome), `iterations.jsonl`, `submissions/`,
`best.*`, `score.json`, `workspace/` (the agent's cwd; its own files go under `workspace/run/`).

Tests: `~/venvs/agent-eval/bin/python -m unittest tests/test_tracker.py`,
`~/venvs/v2-driver/bin/python -m unittest tests/test_policy.py`, `tests/acp_handshake.py <cell>`,
`tests/wirecheck.sh <bnbcode cell>` (logproxy: what reaches vLLM from both harnesses).

## Inherited (do not re-derive)

| here | from | note |
|---|---|---|
| `problems/*/eval.py`, `*.npy` | `experiments/{AC1,AC2,Erdos}` | byte-identical to the ICL graders (md5 in `portable/tasks/CHECKSUMS.txt`) |
| `problems/*/prompt.md.j2`, `prompts/method_evo.md` | `portable/tasks/*/prompt_*.md`, `experiments/Erdos/` | AC1 "32 cores" and AC2 "scores 2.0" defects fixed at the source; cores/hours/python templated |
| `relay/llm_relay.py` | `experiments/` | + fixed `host:port` tokens; wildcard candidates shuffled per process so per-cell relays spread over servers |
| `config/litellm_qwen38.yaml`, `cc_sampling_hook.py` | `experiments/litellm_claude38.yaml`, `cc_sampling_hook.py` | + reasoning aliases / `chat_template_kwargs` |
| `config/claude_guard*.json`, `config/opencode.base.json` | `experiments/` | + denies for the tracker files and `chmod` |
| `hooks/cc_stop_hook.py` | `experiments/hooks/` | nudge text matches the scoring rule |
| `tracker/` | `arxiv_graph/baselines/adrs_acp/tracker.py`, `score.py` | `.npy` in / float out / exception → `score: null`; state in the host dir; `TRACKER_HOST=1` for host scoring |
| `driver/run_cell.py`, `policy.py` | `adrs_acp/run.py` | harness-agnostic; minima + keep_going |
| `analysis/stopclass.py`, `procsample`, `grade_all.py` | `portable/analysis`, `agent_continuity_report/analysis` | unchanged |

## Settled by the 2026-09 campaign — not re-measured here

Qwen3.6 stalls 25 %/turn in bnbcode and dies in minutes; Qwen3.8 works for hours in both harnesses
(0.7–0.9 %/turn). The visible-output guard only rescues text-free replies — it is not a
keep-working device. bnbcode's continual work re-injects the whole prompt (68 % of context) and
was what kept early runs alive; a 133-char external nudge does the same at 1/123 the cost. A
single `finish=length` truncation at 32k ended a cell — hence `output_tokens: 65536` here. 22 of 62
AC2 cells sat at the 0.8825 attractor; no plain-vs-evo effect was detectable at two seeds. Full
text: `../agent_continuity_report/{REPORT.md,CAMPAIGN_ANALYSIS_2026-09-02.md,MECHANISMS.md}`.

## Known traps (verified while building this)

- The ACP SDK does **not** pass `os.environ` to the child: pass `env=` explicitly or bnbcode dies
  with `PgClient: Failed to connect`; its default stderr is an undrained pipe — give it a file.
- A relay of ours can hold a port without answering (the campaign's :9003 on w24c03). `launch`
  kills a non-answering relay that belongs to us before starting; v2 ports are per cell (9300+/4300+).
- Qwen3.8's chat template accepts `reasoning_effort ∈ {low, medium, xhigh}` only (default xhigh);
  `high` → 400. Keep `MAX_THINKING_TOKENS=0` so Claude Code never sends a `thinking` budget that
  LiteLLM would bucket into `high`.
- The vLLM servers must be launched with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`;
  the relay's tool probe silently skips servers that were not.
- Grade on a batch node, never on the login node (cgroup caps the slice at ~5 cores).

## Hardening after the r2 campaign (2026-09-14)

Every r2 cell reached its 18 h budget, but 6 of 11 bnbcode cells spent most of it inside one ACP
prompt that never returned (a `ContextOverflowError` raised inside the session service, or the
visible-output guard re-prompting thousands of times). Verdicts per cell: `analysis/ledger.py` ->
`runs/_analysis/RUNS_LEDGER.md`. What changed, in order of importance:

- **Stall watchdog** (`driver/run_cell.py`, cell key `stall_hours`, default 2): a per-turn task
  watches five clocks -- `iterations.jsonl`, the newest file under the workspace, live workers in
  `run/procsample.jsonl`, the harness's last tool call, and (Claude Code only) `events.jsonl`. When
  all are older than `stall_hours` it cancels the prompt, `restart()`s the harness and re-sends the
  initial prompt with the resume note (turn `stop_reason=stall_restart`). Text-only output is not
  activity on purpose. Verified with a fake harness whose prompt hangs.
- **`guard_cap: 8`** in the campaign yamls (was 1000000).
- **`bin/launch` does not restore a runaway store**: before restoring `pgbackup.sql` it counts the
  COPY blocks and refuses only the wedge signature -- more than `PGBK_EVENT_PER_PART_MAX` (50) `event`
  rows per message `part` (healthy 2-4, wedged 1500-2600) or more than `PGBK_MESSAGES_MAX` (50000)
  messages (the guard storm had 223 k). Size is deliberately not the test: a healthy 15 h store
  reached 445 MB. Everything else is restored whatever its size; stores above 500 MB are dumped
  every 10 min instead of every 2.
- **Claude Code prompt on stdin** (`harness_claude.py`): the host nudge used to be an argv, readable
  in the process table by every cell on the host; one r2 cell read a sibling's best score from it.
- **Walltime = hours + 4 h** (`bin/submit`) so the host grade after the driver is never cut by LSF;
  `tracker/score.py` timeout 3 h (`SCORE_TIMEOUT_S`), keeping the agent-reported score on timeout.
- **Official grading capped** at `grader_timeout_s` = the problem's `kill_s` (1100 s), written to
  `_adrs_track.json` by `tracker.py` and enforced by `eval_wrapper.py`. The candidate program always
  had that budget; the grader did not, and r2 agents fed it sequences of 10^7-10^8 entries that
  took up to 6.5 h to grade for no gain (40 of 13490 official calls). Such a call now scores
  nothing and prints why. Drop r2 rows with `eval_s > 1100` when comparing with r3.

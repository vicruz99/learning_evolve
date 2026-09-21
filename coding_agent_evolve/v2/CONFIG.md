# cells.yaml — every key, what it does, where it lands

`bin/mkcell --campaign NAME` merges `defaults` with each entry under `cells:`, multiplies by
`seeds`, and renders one folder per cell. Any default can be overridden per cell or with
`--set key=value` (nested: `--set bnbcode.output_tokens=32000`).

## Top level

| key | meaning |
|---|---|
| `runs_root` | where campaigns live (`~/agent_runs/v2`). A *campaign* is one folder of cells rendered together; `bin/submit` turns each cell into one LSF job, so cells run in parallel as far as `batch_cpu` allows. |

## Placement and servers

| key | meaning | lands in |
|---|---|---|
| `model` | served model name the relay insists on and the harness requests | relay `--model`, `opencode.json`, LiteLLM upstream |
| `model_family` | `qwen3.8` (default) or `qwen3.6`: selects `config/litellm_<family>.yaml`, the Claude alias prefix (`qwen3.6-xhigh`), the bnbcode provider id, and whether `reasoning_effort` is sent (Qwen3.6 has none) | mkcell |
| `relay_jobs` | where the vLLM servers are: comma list of `NAME:PORT`, NAME = your LSF job name (`gpu10`), `*` (every RUN job of yours) or a fixed host/IP (`rng-dl01-w26n16`). Only servers that serve `model` and pass a real tool-call probe are used; re-resolved on every new connection, so servers may move. | relay `--jobs` |
| `balance` | rotate the `relay_jobs` preference order by cell index, so concurrent cells spread over the servers while each cell stays on one server (vLLM prefix cache). Default true |
| `relay_port_base` / `litellm_port_base` | optional pins. By default every cell gets its own relay and LiteLLM on `base + cell index`, base derived from the campaign name, so campaigns do not collide |
| `budget_s` / `kill_s` | per-candidate time budget and hard kill the prompt states; defaults from `problems/<P>/meta.yaml` (1000 / 1100) |
| `python` | interpreter the agent must use; has numpy 2.5.1 / scipy 1.18.0 / cvxpy 1.9.2, same pins as the ICL grader | prompt `{{PYTHON}}`, host scorer |
| `cores` | CPU slots requested from LSF **and** the budget stated in the prompt | `bsub -n`, prompt |
| `mem_mb` | LSF `rusage[mem=]`; the 1 GB default kills AC2 evaluations | `bsub -R` |
| `hours` | wall budget: stated in the prompt, enforced by the driver (agent killed at `hours`), LSF walltime = hours + 30 min | prompt `{{HOURS}}`, driver, `bsub -W` |
| `seeds` | replicate indices; nothing else changes between seeds (sampling is stochastic on the server) | cell name `_sN` |

| `site` | `rngdl01` (default) or `marvin`: names the cluster in the prompt, skips the rng-dl01 "/home quota" guard on Marvin (home is the only filesystem there), and selects the submitter (`bin/submit` = bsub, `bin/submit_marvin` = subbin). 2026-09-21 |
| `host_type` | Marvin only: `subbin -h` (`amd` = 64-core Genoa nodes; `intel` = 48 cores and 1.9x slower on the graders' `np.convolve`) |
| `wall_extra_h` | LSF walltime = `hours` + this (default 4). A paused run (LLM outage, `bin/pause`) waits inside the job, so a 48 h budget gets 24 |

## Problem-level (in `problems/<P>/meta.yaml`, not per cell)

`cpus_per_candidate` (AC1/AC2: 2, Erdős: 1, from `src/envs/registry.py`) sets the per-candidate
core limit the prompt states and `PARALLEL = cores // cpus_per_candidate`. It is a prompt rule,
not enforced by `eval.py` (single-threaded numpy); the agent's candidate programs are what use cores.

## Reasoning

| key | meaning |
|---|---|
| `reasoning` | `low` / `medium` / `xhigh` (Qwen3.8 template values; default of the template is `xhigh`) or `off` (thinking disabled). bnbcode: `options.chat_template_kwargs` on the model; Claude Code: LiteLLM alias `qwen3.8-<effort>`. Qwen3.6 has no effort levels: `low/medium/xhigh` all mean "thinking on". |

## Host driver (keeps the agent working)

The agent's turn ending (no more tool calls) is the stop signal. The driver then sends a *nudge*
into the same conversation: `[host] N min remaining; K official evaluations so far (minimum M); best ...`
followed by `prompts/continuation.md`.

| key | meaning |
|---|---|
| `min_evals` | official `python eval.py` calls the run must reach; until then every stop is nudged immediately |
| `min_hours` | wall time the run must reach; until then every stop is nudged immediately |
| `max_evals` | tracker cap on official evaluations; reaching it writes `STOP` and ends the run. 0 = unlimited |
| `keep_going` | after both minima are met (and something better than the start is scored): `true` keeps nudging until `hours`, `false` lets the run end at the next stop |
| `nudge_min_gap_s` | throttle after the minima are met: at most one nudge per this many seconds; the driver waits out the remainder before nudging (Tim's `CONTINUE_MIN_GAP_S`, but waiting instead of ending) |
| `hours` (resume) | the budget is ACTIVE time across relaunches (`clock.json`); a resubmitted cell resumes its conversation and gets only the remaining hours -- see README "Relaunch = resume" |
| `patience` | consecutive turns with zero tool calls before a fresh session in the same workspace (context reset, files kept, prompt + resume note). 0 = never. From the campaign's `bnb-nudge`, not Tim's code |
| pause / resume | not a key: the driver polls the relay's `/_relay/status` every minute; no live upstream for 5 min, or a `<cell>/PAUSE` file (`bin/pause <campaign> on|off`), stops the harness cleanly, stops the budget clock (`clock.json` `paused_s`, `pauses`) and resumes with the conversation restored when an upstream answers again. Turns with API errors and no tool call are infrastructure, not agent stops. 2026-09-21 |
| `give_up` | consecutive turns with zero tool calls before recording `outcome: refused`. 0 = never |

Turns that never reached the model (error in < 60 s, fault signature in stderr, dead process) are
infrastructure: back-off 30 s / 120 s, restart, give up after 20 in a row; never counted as barren.

## bnbcode

| key | bnbcode default | meaning / where |
|---|---|---|
| `continual_work` | true | bnbcode's built-in continuation mode (re-injects the whole prompt, ~16 KB, each time). `continual_work.enabled` in `opencode.json` |
| `visible_output_guard` | off | when a reply has neither text nor tool call, bnbcode injects "[Your previous response had no visible output...]" and continues. Env `BNBCODE_VISIBLE_OUTPUT_GUARD=1` |
| `guard_cap` | 8 | how many times per session the guard may fire (cumulative). Env `BNBCODE_VISIBLE_OUTPUT_GUARD_CAP` |
| `output_tokens` | 32000 (hard `min` with `limit.output`) | `max_tokens` per request = thinking + answer; raising it needs the env `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`, which the driver sets |
| `context_tokens` | none (must be declared) | `limit.context`; must not exceed the server's `--max-model-len` (262144 on the Qwen3.8 servers, 180000 on the Qwen3.6 one) |
| `tool_output.max_lines/max_bytes` | 2000 / 51200 | tool results longer than this are cut and saved to disk; `tool_output` in `opencode.json` |
| `compaction_context_limit` | 250000 | token count at which bnbcode compacts by force (`COMPACTION_FORCE_TOKENS`). Independent of this, it always compacts when input+output+reasoning ≥ `context_tokens − output_tokens` |
| `compaction_notes` | true | reminders at 100k, then every 50k tokens, asking the agent to write handoff notes for the next compaction |
| `temperature` / `top_p` | 1.0 / model default | sampling; pinned to 0.6 / 0.95 (Qwen's thinking-mode recommendation), same in both harnesses |

## Claude Code

| key | Claude Code default | meaning / where |
|---|---|---|
| `max_output_tokens` | undocumented for a non-Anthropic model id | `CLAUDE_CODE_MAX_OUTPUT_TOKENS` |
| `max_context_tokens` | undocumented | `CLAUDE_CODE_MAX_CONTEXT_TOKENS`; the window Claude Code assumes |
| `autocompact` | `auto` | `--autocompact <auto|tokens>`; the campaign used `160k` |
| `cliff` / `cliff_threshold` / `cliff_keep_recent` | off / 110000 / 3 | CliffCompaction proxy (`~/venvs/ccproxy/bin/cliff serve`) per cell in front of LiteLLM; Claude Code gets `ANTHROPIC_BASE_URL` = cliff and `DISABLE_AUTO_COMPACT=1`, so compaction is cliff-style in both harnesses (bnbcode has it built in; its threshold is `compaction_context_limit`). 2026-09-21 |
| `stop_hook` | off | `hooks/cc_stop_hook.py` runs whenever Claude Code wants to end its turn and answers `block` + nudge, so the agent continues in-turn until the deadline or `give_up` blocks without a new tool call. Headless Claude Code honours one block then exits, so the driver's own nudge (`--continue`) sits on top |

Sampling for Claude Code is pinned to 0.6 / 0.95 by `config/cc_sampling_hook.py` in LiteLLM.

# Claude Code driven by the local Qwen

The coding-agent arm of the ICL study: the same tasks as `../erdos/` and `../gpumode/`,
but with **Claude Code as the harness** and **Qwen3.6-27B as the model behind it** —
so the agent scaffold is held fixed and only the model changes, the same way the ICL
sweeps hold the search fixed and vary the prompt.

**If you just want to run it, go to [The runbook](#the-runbook).** Everything before it is
reference: what each piece is and why it is configured the way it is.

Claude Code never learns it is not talking to Anthropic: it speaks the Anthropic Messages
API to a LiteLLM proxy, which translates to the OpenAI API your vLLM server already serves.

```
claude  --(Anthropic Messages, /v1/messages)-->  LiteLLM :4000
                                                     |
                                    (OpenAI, /v1/chat/completions)
                                                     v
                                            vLLM :8001  (Qwen3.6-27B)
```

## Reference: one-time setup

```bash
python3 -m venv /scratch/vicstorage/venvs/ccproxy
/scratch/vicstorage/venvs/ccproxy/bin/pip install 'litellm[proxy]'
# litellm 1.97.0 (the newest on our index) does not import against fastapi 0.141+ --
# `ImportError: cannot import name 'get_flat_dependant'`, which surfaces confusingly as
# `ModuleNotFoundError: No module named 'proxy_server'`. Pin both back:
/scratch/vicstorage/venvs/ccproxy/bin/pip install 'fastapi==0.140.0' 'sse-starlette==2.1.3'
```

Verified working on guadiana 2026-08-18 with litellm 1.97.0 + fastapi 0.140.0 +
sse-starlette 2.1.3 on Python 3.13.

The `claude` CLI is the native install (`~/.local/share/claude/versions/<v>`, a
self-contained binary — not npm). To put it on another machine, copy that directory or
run `curl -fsSL https://claude.ai/install.sh | bash`. **Pin the version** so the model
arm does not silently change harness mid-study.

## Reference: the server, and why tool calling is the whole game

This is the one non-obvious requirement. Claude Code does everything through tools —
Read, Bash, Edit, Task — so a server that emits tool calls as plain text gives you an
agent that narrates what it would do and never does it. The ICL sweeps do not need this,
so the standard `vllm serve` line in `src/jobs/vllm_server.bsub` **does not have it**.

```bash
./serve_qwen.sh          # foreground; own tmux pane
```

The two flags that matter, on top of the usual ICL server line:

```
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

`qwen3_xml` matches this checkpoint's chat template, which emits
`<tool_call><function=NAME><parameter=P>…`. If a vLLM version mismatch makes it drop
calls, `qwen3_coder` parses the same shape. Verify before going further — you want a
`tool_calls` array, not a `<tool_call>` string sitting in `content`:

```bash
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3.6-27B",
  "messages":[{"role":"user","content":"List the files in /tmp"}],
  "tools":[{"type":"function","function":{"name":"bash","description":"run a shell command",
    "parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}],
  "tool_choice":"auto"}' | python -m json.tool
```

If you forget the flags, the failure is at least unambiguous — it propagates all the way
back through the proxy into the Claude Code session:

```
API Error: 400 litellm.BadRequestError: Hosted_vllmException -
{"error":{"message":"\"auto\" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set","type":"BadRequestError","code":400}}
```

(You will also see a harmless `[claude-code:unrecognized_model]` line on startup. Claude
Code does not know `qwen3.6-27b`; that is the point.)

Keep `--enable-prefix-caching`. Claude Code resends a long, stable prefix (system prompt +
tool schemas, measured at ~3k here) on every single turn; without prefix caching you pay to
prefill it hundreds of times per run.

## Reference: the proxy

```bash
/scratch/vicstorage/venvs/ccproxy/bin/litellm --config litellm_qwen.yaml --port 4000
```

`run_agent.sh` starts this for you if it is not already up. Three things it must get
right, from the gateway protocol contract:

* **Stream.** Claude Code consumes SSE as it arrives; a proxy that buffers whole
  responses stalls the client.
* **Relay keep-alive pings.** Claude Code counts every byte and aborts a stream that goes
  silent for 300 s. Long Qwen thinking pauses have nothing else to send.
* **Forward error bodies unmodified.** Claude Code's auto-retry matches on the upstream's
  error *wording*; a proxy that wraps errors in its own envelope breaks the recovery path
  even with the status code preserved.

## Reference: the run folder and its guard

```bash
./run_agent.sh /scratch/vicstorage/agent_runs/erdos_qwen1        # interactive
./run_agent.sh /scratch/vicstorage/agent_runs/erdos_qwen1 -p     # headless -> agent.jsonl
```

The run directory is a self-contained task folder — `INITIAL_PROMPT.md`, its evaluator,
its seed — **kept outside this repo**, exactly as `../gpumode/run_description.md`
describes. An agent working inside the repo would inherit `CLAUDE.md` (which orders it to
read `docs/PROJECT_CONTEXT.md` and `docs/EXPERIMENT_PLAN.md`, both of which discuss these
very tasks), reuse this project's auto-memory, and possibly find the published solution
in a sibling directory. For the Erdős task, seed the folder with
`../erdos/erdos_prompt_noweb.md` as `INITIAL_PROMPT.md` and `../erdos/eval.py`.

`run_guard.json` is passed with `--settings` from *outside* the run folder, so the agent
cannot edit the rules that constrain it. It denies the docs, the vendored `discover/`
repo, this folder, and the auto-memory directory, and disables auto-memory. Containment
is not airtight — bubblewrap is not installed, so Bash can still read any file you can.
It stops incidental contamination, not a determined search.

Each run gets its own `CLAUDE_CONFIG_DIR` (`<run_dir>/.cc`), so no history, memory or
config leaks between runs.

## What does not work behind a local model

* **WebSearch** is an Anthropic server-side tool: it is executed by the API, not the
  client, so it simply does not exist here. **WebFetch**'s domain safety check calls
  `api.anthropic.com` directly, ignoring `ANTHROPIC_BASE_URL`. `run_guard.json` denies
  both, and `../erdos/erdos_prompt_noweb.md` is the prompt variant with the web-search
  section replaced — the tracked `erdos_prompt.md` requires a `research_log.md` and web
  citations, which is unsatisfiable here. (It also makes the Qwen arm a cleaner
  comparison against the ICL runs, which have no web access either.)
* **Fast mode** probes `api.anthropic.com` directly and will report a connectivity error
  on a network without egress. Harmless.
* Subagents, skills and hooks all work — they are client-side.

## The two budgets you will actually want to tune

### How much Qwen is allowed to think

**Currently unconstrained, on purpose.** The model is agentic and decides for itself when
to stop reasoning; the ICL arm caps it (`thinking-token-budget: 11000` in the trimul
sweeps) because a single-shot generation has no other way to bound cost, which is not the
situation here.

`MAX_THINKING_TOKENS` is not the knob either way. It controls what Claude Code *asks* for,
and `drop_params: true` discards the ask before it reaches vLLM.

The real knob, if you ever want it, is vLLM's per-request `thinking_token_budget`,
commented out in `litellm_qwen.yaml` under `extra_body` because Claude Code has no way to
send it. vLLM tracks the `<think>` section and forces the end token once the budget is
spent; verified through the full chain, a budget of 24 cut reasoning at 89 characters.
Two things before you turn it on: the cut is **hard**, so a small budget leaves the model
mid-sentence and the "answer" is just the rest of its reasoning with the tags gone; and
`chat_template_kwargs: {"enable_thinking": false}` is the cheaper way to get zero, though
this checkpoint implements that by injecting an empty `<think></think>` block.

### How often Claude Code compacts, and what the 100k floor is

The 100k floor is **the smallest value the variable accepts**, not a compaction period.
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` takes 100,000-1,000,000 and Claude Code caps it at the
model's window. It is a *window size*, so it cannot express a ceiling below 100k at all —
which is why `env.sh` does not use it.

The variable that can is `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, which declares the window Claude
Code should assume for a model ID it does not recognise. It applies directly here because
`qwen3.6-27b` neither starts with `claude-` nor contains `[1m]`. `env.sh` sets **88000**.

Compaction then fires at **window minus a fixed 29.4k buffer**. Measured with
`claude -p "/context"` at three settings — the buffer does not scale:

| declared window | free space | autocompact buffer | messages compact at |
|---|---|---|---|
| 88k | 55.6k | 29.4k (33.4%) | ~58k |
| 130k | 97.6k | 29.4k (22.6%) | ~100k |
| 200k | 167.6k | 29.4k (14.7%) | ~170k |

At 88k the session reported `Tokens: 3k / 88k`, with the system prompt at ~1.6k and skills
at ~1.4k — **~3k, not the 20-25k** an earlier version of this file guessed.

To turn compaction off entirely, set `DISABLE_COMPACT`. Do not: the session will hit
`Prompt is too long` and stop instead.

For the study, compaction is not plumbing — it is the harness's own context-truncation
strategy, doing the same job as `src/context/` in the ICL loop. Log how often it fires.

### Can the context overrun `--max-model-len`?

It could, badly, and the fix is not a compaction setting — it is the **tokenizer**.

Claude Code enforces its ceiling against whatever the proxy's `/v1/messages/count_tokens`
returns. LiteLLM defaults to tiktoken for a non-Anthropic model, and tiktoken is not
Qwen's tokenizer. Measured drift on the same text:

| content | tiktoken (what Claude Code believed) | Qwen actual | undercount |
|---|---:|---:|---:|
| Triton code + prose | 12,128 | 13,329 | 10% |
| benchmark tables, digit-heavy | 34,206 | 51,099 | **49%** |

The second row is not a contrived case — it is what a TriMul agent's context actually fills
with, since `eval.py` emits per-shape `mean / std / err / best / worst` in nanoseconds. An
88k ceiling counted 49% low is **131k real tokens**, past the server's 130k
`--max-model-len`, before a single output token.

`model_info.custom_tokenizer` in `litellm_qwen.yaml` points the counter at the real
tokenizer and closes the gap: the same two texts now count 13,328 and 51,096 against vLLM's
13,329 and 51,099. With that in place 88k means 88k, and 88k + the 16k output cap leaves
~26k of headroom under 130k.

It needs network the first time (HuggingFace hub) and caches to `~/.cache/huggingface`, so
on Bosch start the proxy once with the p4s proxy module loaded. **If you cannot give it
network, raise `--max-model-len` instead** — this checkpoint is documented at ~260k, and
200k would absorb even the 49% case. Do not simply trust the ceiling with a tiktoken
counter behind it.

Do not count on Claude Code recovering if it does overrun. Its retry path matches on the
*upstream's* error wording, and LiteLLM wraps vLLM's error in its own envelope:

```
litellm.BadRequestError: Hosted_vllmException - {"error":{"message":"max_tokens=200000
cannot be greater than max_model_len=max_total_tokens=130000 ..."}}
```

That may well defeat the matcher, in which case the overrun surfaces as a hard API error
mid-run rather than as a compaction. The proactive window is the protection.

## The chat template

`froggeric/Qwen-Fixed-Chat-Templates` (covers Qwen 3.5 / 3.6 / 3.8) is worth taking
seriously. Three of its claims check out against the official template shipped with this
checkpoint, though only one of them currently bites on this stack:

| claim | verdict here |
|---|---|
| tool-call args as a JSON string crash the template | **Real, but masked.** Rendering `arguments` as the JSON string the OpenAI spec sends fails with `TypeError: Can only get item pairs from a mapping` — the template does `arguments\|items` with no string handling. vLLM 0.26.0 parses the string to a dict before templating, so it never surfaces. It would surface on llama.cpp, LM Studio, or a vLLM that stops normalising. |
| historical mutation destroys prefix caching | **Real.** The template strips `<think>` from any assistant message older than the last user query. So each new turn retroactively deletes the previous turn's reasoning, and the render diverges from that point on — a per-turn cache miss on the tail, proportional to the reasoning block dropped. Capping `thinking_token_budget` bounds the damage; `chat_template_kwargs: {"preserve_thinking": true}` removes it entirely, at the cost of keeping every reasoning block in a 130k window forever. |
| empty `<think></think>` causes premature aborts | **Construction confirmed, effect not measured.** `enable_thinking: false` does inject exactly that block here. A single-turn probe answered fine; the claim is about rate over many turns, which we have not tested. |
| graded reasoning effort (`<\|think_low\|>` … `<\|think_xhigh\|>`) | **Genuinely absent here.** This template has only the binary `enable_thinking` switch — no effort tags at all, and no `xhigh` default to fix. The fixed templates would *add* per-prompt effort steering, which is a real gain for an agent loop where some turns deserve more thought than others. |

### Which file, and what it actually changes

**Take the version from `archive/qwen3.6/`, not the repo root.** The root
`chat_template.jinja` is `qwen3.8-froggeric-v22.1` — the Qwen3.8 template, wrong model. The
3.6 line stops at **v19**, and v19 does *not* carry the `<|think_low|>`…`<|think_xhigh|>`
inline tags; those are a v21/v22 (3.8) feature. So the effort-steering upside is **not**
available to this checkpoint. What v19 does fix is real, measured with
`compare_templates.py`:

```
============ /scratch/vicstorage/qwen/chat_template.jinja          (official)
  stringified tool args : FAILS -> TypeError: Can only get item pairs from a mapping.
  prefix stability      : 52/127 chars of the turn-1 render survive
  old reasoning kept    : False
============ qwen3.6_chat_template-v19.jinja                       (fixed)
  stringified tool args : RENDERS
  prefix stability      : 108/126 chars of the turn-1 render survive
  old reasoning kept    : True
```

Read the last line as a trade, not a free win: v19 keeps every historical reasoning block
in the prompt. That is what makes the prefix stable, and with reasoning unconstrained it
also means the 88k window fills faster and compaction fires sooner. Pass
`chat_template_kwargs: {"preserve_thinking": false}` to get the old stripping behaviour
back if the window turns out to be the binding constraint.

```bash
./fetch_chat_template.sh                                   # writes qwen3.6_chat_template-v19.jinja
python compare_templates.py /scratch/vicstorage/qwen/chat_template.jinja \
                            qwen3.6_chat_template-v19.jinja
CHAT_TEMPLATE=$PWD/qwen3.6_chat_template-v19.jinja ./serve_qwen.sh
```

`serve_qwen.sh` passes it as `--chat-template` rather than editing `tokenizer_config.json`
in the checkpoint, so the model stays pristine and the template is a logged, switchable
part of the run config. Treat it as a variable held fixed across arms, not a fix applied
once and forgotten: it changes every rendered prompt, so runs before and after are not
comparable.

## Things that will bite

**Context.** Handled by `CLAUDE_CODE_MAX_CONTEXT_TOKENS=88000`; see *How often Claude Code
compacts* above. The server's `--max-model-len` is 130k, so the 88k ceiling leaves headroom
rather than racing it.

**Betas.** Claude Code sends its full Anthropic capability set to any `ANTHROPIC_BASE_URL`
gateway, including body fields (`context_management`, `output_config`, `strict` /
`defer_loading` on tool schemas) that a non-Anthropic upstream rejects with a hard 400.
`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` in `env.sh` turns them off. This set grows with
each Claude Code release — a version bump can reintroduce a field Qwen's endpoint rejects,
which is a second reason to pin the version.

**Thinking.** Claude Code treats a model name it does not recognise as a current model and
sends `thinking: {"type":"adaptive"}`. It auto-retries without the field after the first
rejection, so it self-heals, but that is one wasted request per conversation and it
interacts badly with `--reasoning-parser qwen3`. `env.sh` sets `MAX_THINKING_TOKENS=0`. Note that this only stops Claude Code *asking* for
extended thinking — `--reasoning-parser qwen3` still returns Qwen's own reasoning, which
LiteLLM maps into an Anthropic `thinking` block (with a null signature, which Claude Code
tolerates). That reasoning comes out of the same output budget: in a smoke test, a 128-token
cap was consumed entirely by reasoning before a single visible character. Budget
`CLAUDE_CODE_MAX_OUTPUT_TOKENS` generously or tool calls get truncated mid-argument.
Get a run working first, then try raising it — a reasoning model in an agent loop is
plausibly a big part of what makes this arm competitive, so it is worth revisiting rather
than leaving off by default.

**GPU contention on the TriMul task.** vLLM holds the card and `../gpumode/evaluate.py`
grades on a local GPU. Sharing one card makes every timing meaningless (`gpumode/README.md`
§Measurement caveats). Give vLLM its own cards with `GPUS=0,1 ./serve_qwen.sh` and leave
one free for grading. The Erdős task grades on CPU, so one GPU is enough there.

## The runbook

Nothing here uses `bsub` for the run itself — only to get the interactive allocation. Three
tmux panes on one node. Everything below assumes `cd` into this directory.

### Step 0 — get a node (Bosch only)

Never on a login node: a cgroup caps the user slice at 5 cores while exposing 64, invisibly
to every tool, and evals run ~12x slower (`docs/BOSCH_CLUSTER.md` §1).

```bash
bsub -Is -q batch_b200 -gpu "num=2" -J ccagent -P BH-000557-01 \
     -G rb_bd_dlp_rng-dl01_cr_AIQ_employees \
     -n 32 -R "span[hosts=1] rusage[mem=65536]" -M 65536MB -W 24:00 /bin/bash
```

**Two GPUs, not one.** vLLM holds one for the whole run and the grader needs a clean card;
sharing makes every TriMul timing meaningless. Pass `-M` explicitly — LSF's default memory
limit is 1 GB and will kill the server.

Then, on the compute node, turn on outbound network. Needed for the one-time installs, and
for the proxy's first start (it downloads Qwen's tokenizer):

```bash
source /fs/applications/modules/current/init/bash
module load proxy4server-access/2.0 && sleep 1
source /fs/applications/p4s-access/2.0/ActivateP4S.sh -a
```

On guadiana, skip this step entirely — the cards are already there and `GPUS=1` steers vLLM
off card 0.

### Step 1 — one-time, per machine

```bash
# The proxy goes in the PROJECT venv -- it does not need one of its own. Verified with a
# dry run against src/.venv: 62 pure additions, aiohttp 3.14.1 -> 3.14.3 (ray needs
# >=3.13.3, so it is fine), and nothing downgraded or removed.
uv pip install --python src/.venv/bin/python \
    'litellm[proxy]==1.97.0' 'fastapi==0.140.0' 'sse-starlette==2.1.3'

# the fixed chat template (v19 — the 3.6 line; the repo root file is Qwen3.8)
./fetch_chat_template.sh
python compare_templates.py /scratch/vicstorage/qwen/chat_template.jinja \
                            qwen3.6_chat_template-v19.jinja

# the grading interpreter, for the TriMul task only. MUST be the cu128 build: sm_100
# gencode comes only from CUDA 12.8, and a cu126 venv on a B200 fails as a wall of ptxas
# errors that read like bad candidates. Full steps in gpumode_local/reference/README.md.
python3 -m venv ~/venvs/kernel-eval
~/venvs/kernel-eval/bin/pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128
~/venvs/kernel-eval/bin/pip install pyyaml numpy
```

**Both pins are load-bearing, in opposite directions.** `fastapi` must be `<0.141`:
`get_flat_dependant` was removed in 0.141, and litellm 1.97.0 still imports it, so it dies
with a `ModuleNotFoundError: No module named 'proxy_server'` that names the wrong thing
entirely. It must also be `>=0.136.3`, because that is litellm's own declared floor and uv
refuses to resolve below it. 0.140.0 is the version that satisfies both — an earlier version
of this file said 0.115.12, which works but violates litellm's constraint and makes the
resolver reject the whole install.

Then verify the grading stack on the card the scheduler gave you:

```bash
KPY=~/venvs/kernel-eval/bin/python ./check_gpu.sh
```

It prints torch/triton/card/arch-list/shared-memory, fails loudly if the interpreter has no
kernels for this capability, and then **compiles and runs a real Triton kernel** — because
the arch list alone is not proof, and this is what the grader actually does. Expect on a
Bosch B200 node (measured 2026-08-17, recorded in `gpumode_local/reference/README.md`):

```
torch 2.7.1+cu128 | triton 3.3.1 | NVIDIA B200 | sm_100
['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120'] | shared mem 232448
```

A `BUSY` result means the card has no free memory — something else owns it — and says
nothing about the toolchain. Inside an LSF job the scheduler sets `CUDA_VISIBLE_DEVICES` to
your card, so this only comes up on an unscheduled box.

Also install the `claude` CLI once (`curl -fsSL https://claude.ai/install.sh | bash`, or
copy `~/.local/share/claude/versions/<v>` across) and **pin the version** — the set of
capabilities Claude Code sends to a gateway grows with each release, so an upgrade can
reintroduce a body field Qwen's endpoint rejects, and it changes the harness mid-study.

### Step 2 — pane 1: the model server

```bash
GPUS=0 CHAT_TEMPLATE=$PWD/qwen3.6_chat_template-v19.jinja ./serve_qwen.sh
```

When it is serving, **check tool calling from another shell**. This is the one failure that
produces a plausible-looking agent that narrates what it would do and never acts:

```bash
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3.6-27B",
  "messages":[{"role":"user","content":"List files in /tmp"}],
  "tools":[{"type":"function","function":{"name":"bash","description":"run a shell command",
    "parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}],
  "tool_choice":"auto"}' | python -m json.tool | grep -A5 tool_calls
```

A `tool_calls` array is what you want. A `<tool_call>` string sitting in `content`, or a 400
saying `"auto" tool choice requires --enable-auto-tool-choice`, means the flags did not take.

### Step 3 — pane 2: the B200 baseline, once per machine

**Do this before the first agent run.** No published B200 number for this kernel exists, so
the reference kernel's own score on this card is the only yardstick you will have to judge
an agent result against. Run it from the repo — it grades the published TTT-Discover
kernel, which the agent must never see:

```bash
cd ~/projects/phd/learning_evolve
export KPY=~/venvs/kernel-eval/bin/python

"$KPY" coding_agent_evolve/gpumode/evaluate.py \
    src/gpumode_local/reference/trimul_best.py --mode leaderboard --repeats 5
```

No `--gpu` flag, ever: the grader inherits `CUDA_VISIBLE_DEVICES`, which is the card LSF
granted. Naming an index overrides that and can time another job's GPU. Record the score
and the spread in `gpumode_local/reference/README.md`.

### Step 4 — pane 3: the run

```bash
cd ~/projects/phd/learning_evolve/coding_agent_evolve/local_model
export KPY=~/venvs/kernel-eval/bin/python

../gpumode/b200_task/make_run.sh ~/agent_runs/trimul_b200_qwen1
./run_agent.sh ~/agent_runs/trimul_b200_qwen1          # interactive
./run_agent.sh ~/agent_runs/trimul_b200_qwen1 -p       # headless -> agent.jsonl
```

**Interactive is the default and it is a real Claude Code session** — the normal UI opens
in the run folder and `INITIAL_PROMPT.md` is submitted for you, so the run starts on the
task and you watch it work, interrupt it, and steer it like any other session. Drop the
positional argument in `run_agent.sh` if you would rather land in an empty prompt and type.

`-p` is headless: no UI at all, the transcript streams to `agent.jsonl`. Use it for a run
you are not sitting with. Nothing can answer a permission prompt in that mode, which is why
`run_guard.json` sets `defaultMode: acceptEdits`.

Either way `run_agent.sh` health-checks vLLM, starts LiteLLM on :4000 if it is not already
up, runs `check_gpu.sh` for kernel tasks, sources `env.sh`, gives the run its own
`CLAUDE_CONFIG_DIR`, and launches under `run_guard.json` from outside the run folder.

For the Erdős task instead, build the folder by hand — it needs no harness:

```bash
mkdir -p ~/agent_runs/erdos_qwen1 && cd ~/agent_runs/erdos_qwen1
cp ~/projects/phd/learning_evolve/coding_agent_evolve/erdos/erdos_prompt_noweb.md INITIAL_PROMPT.md
cp ~/projects/phd/learning_evolve/coding_agent_evolve/erdos/eval.py .
python3 -m venv .claude_venv                      # the prompt tells the agent this exists
```

### Checks worth running once, in the first session

```bash
claude -p "/context"      # expect `Tokens: ~3k / 88k`. A different denominator means
                          # CLAUDE_CODE_MAX_CONTEXT_TOKENS did not take.
```

And confirm the token counter is the real one — with tiktoken behind it, the 88k ceiling can
be 131k real tokens and overrun the server (see *Can the context overrun `--max-model-len`?*):

```bash
curl -s http://127.0.0.1:4000/v1/messages/count_tokens \
  -H 'x-api-key: sk-local' -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"1234567890 9876543210"}]}'
```

If the proxy could not reach HuggingFace on its first start it silently falls back to
tiktoken. The fallback is not announced, so check rather than assume; if it happened, either
give it network and restart, or raise `--max-model-len` to 200000 and rely on headroom.

### Shutting down

`Ctrl-C` pane 1 for vLLM. The proxy is backgrounded by `run_agent.sh`; find it with
`pgrep -f "litellm.*4000"` and kill that PID. Both die with the LSF job anyway.

## Files

| file | what it is |
|---|---|
| `serve_qwen.sh` | the vLLM line, with the tool-calling flags the ICL server lacks |
| `litellm_qwen.yaml` | the Anthropic→OpenAI translation config |
| `env.sh` | the environment that points `claude` at the proxy |
| `run_agent.sh` | health-checks the server, starts the proxy, launches the agent in a run dir |
| `run_guard.json` | `--settings` guard: denies web tools, the docs, and auto-memory |
| `fetch_chat_template.sh` | pulls the fixed Qwen3.6 chat template (v19) from HuggingFace |
| `compare_templates.py` | renders two templates side by side on the three things that break agent loops |
| `check_gpu.sh` | proves the grading interpreter can compile Triton for the granted card |

The task folders live next door: `../gpumode/b200_task/` (TriMul on a B200) and
`../erdos/` (the Erdős prompt plus `eval.py`).

# Claude Code driven by the local Qwen

The coding-agent arm of the ICL study: the same tasks as `../erdos/` and `../gpumode/`,
but with **Claude Code as the harness** and **Qwen3.6-27B as the model behind it** —
so the agent scaffold is held fixed and only the model changes, the same way the ICL
sweeps hold the search fixed and vary the prompt.

Claude Code never learns it is not talking to Anthropic: it speaks the Anthropic Messages
API to a LiteLLM proxy, which translates to the OpenAI API your vLLM server already serves.

```
claude  --(Anthropic Messages, /v1/messages)-->  LiteLLM :4000
                                                     |
                                    (OpenAI, /v1/chat/completions)
                                                     v
                                            vLLM :8001  (Qwen3.6-27B)
```

## One-time setup

```bash
python3 -m venv /scratch/vicstorage/venvs/ccproxy
/scratch/vicstorage/venvs/ccproxy/bin/pip install 'litellm[proxy]'
# litellm 1.97.0 (the newest on our index) does not import against current fastapi --
# `ImportError: cannot import name 'get_flat_dependant'`, which surfaces confusingly as
# `ModuleNotFoundError: No module named 'proxy_server'`. Pin both back:
/scratch/vicstorage/venvs/ccproxy/bin/pip install 'fastapi==0.115.12' 'sse-starlette==2.1.3'
```

Verified working on guadiana 2026-08-18 with litellm 1.97.0 + fastapi 0.115.12 +
sse-starlette 2.1.3 on Python 3.13.

The `claude` CLI is the native install (`~/.local/share/claude/versions/<v>`, a
self-contained binary — not npm). To put it on another machine, copy that directory or
run `curl -fsSL https://claude.ai/install.sh | bash`. **Pin the version** so the model
arm does not silently change harness mid-study.

## 1. The server, with tool calling on

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
tool schemas ≈ 20-25k tokens) on every single turn; without prefix caching you pay to
prefill it hundreds of times per run.

## 2. The proxy

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

## 3. Run

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

`MAX_THINKING_TOKENS` does **not** do this. It controls what Claude Code *asks* for, and
`drop_params: true` discards the ask before it reaches vLLM. Qwen thinks as much as it likes
regardless.

The real control is vLLM's per-request `thinking_token_budget`, pinned in
`litellm_qwen.yaml` under `extra_body` because Claude Code has no knob to send it:

```yaml
extra_body:
  thinking_token_budget: 2048     # 0 = no thinking at all
```

vLLM tracks the `<think>` section and forces the end token once the budget is spent. It
needs `--reasoning-parser` on the server, which `serve_qwen.sh` sets. Verified through the
full chain (Anthropic `/v1/messages` -> LiteLLM -> vLLM): a budget of 24 cut reasoning at
89 characters and handed control back to visible output.

Two things to know before you turn it down:

* **The cut is hard, not graceful.** At 24 tokens the model was mid-sentence in step 1 of
  its plan, and the "answer" that followed was just the rest of the reasoning with the
  `<think>` tags gone. Budget enough for a complete thought (1-4k) or set 0 for none.
  Values in between mostly buy you incoherence.
* **The other lever is the template.** `chat_template_kwargs: {"enable_thinking": false}`
  turns thinking off through the template instead, which costs nothing at sample time.
  Confirmed working on this checkpoint. But this template implements it by injecting an
  empty `<think>\n\n</think>` block, which is exactly the construction the fixed-template
  project below blames for premature turn aborts. `thinking_token_budget: 0` gets you the
  same place without the empty block.

### How often Claude Code compacts

`CLAUDE_CODE_AUTO_COMPACT_WINDOW` is how full the context may get before compaction. Three
constraints matter here:

* **100,000 is the floor.** You cannot make Claude Code compact earlier. With a 130k
  server window and ~20-25k of system prompt and tool schemas, 100k is nearly the whole
  usable window. If you want more headroom between compactions, raise `--max-model-len`
  (this checkpoint is documented at ~260k) rather than lowering this.
* **Plain integers only** in the environment variable: `100k` reads as `100` and clamps to
  the floor. The `/autocompact 500k` command and the `--autocompact 500k` flag do accept
  suffixes; the environment variable overrides both while it is set.
* **Declare the real window too.** Claude Code does not recognise the ID `qwen3.6-27b` and
  assumes a context window for it. `CLAUDE_CODE_MAX_CONTEXT_TOKENS=130000` corrects that,
  and applies directly here because the ID neither starts with `claude-` nor contains
  `[1m]`. Without it the status line percentage is measured against a window that does not
  exist.

To turn compaction off entirely, set `DISABLE_COMPACT`. Do not, on a 130k window: the
session will hit `Prompt is too long` and stop instead.

For the study, compaction is not plumbing -- it is the harness's own context-truncation
strategy, doing the same job as `src/context/` in the ICL loop. Log how often it fires.

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

If you swap it in, point `--chat-template <file>` at it in `serve_qwen.sh` rather than
editing `tokenizer_config.json` in `/scratch/vicstorage/qwen` — the checkpoint stays
pristine and the template becomes a logged, switchable part of the run config. Treat it as
a variable to hold fixed across arms, not a fix to apply once and forget: it changes the
rendered prompt, so runs before and after are not comparable.

## Things that will bite

**Context.** The server window is 130k. Claude Code's system prompt and tool schemas
already cost ~20-25k of it before the task prompt, and tool results (file reads, script
output) accumulate fast. `CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000` compacts before the
ceiling; without it a long run dies mid-tool-call. Compaction is itself an ICL-relevant
variable — it is the harness's own truncation strategy, and it is doing the same job as
`src/context/` in the ICL loop. Worth logging how often it fires.

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

## Files

| file | what it is |
|---|---|
| `serve_qwen.sh` | the vLLM line, with the tool-calling flags the ICL server lacks |
| `litellm_qwen.yaml` | the Anthropic→OpenAI translation config |
| `env.sh` | the environment that points `claude` at the proxy |
| `run_agent.sh` | health-checks the server, starts the proxy, launches the agent in a run dir |
| `run_guard.json` | `--settings` guard: denies web tools, the docs, and auto-memory |

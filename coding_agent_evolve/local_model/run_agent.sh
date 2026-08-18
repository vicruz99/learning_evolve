#!/usr/bin/env bash
# Run one coding-agent experiment against the local Qwen.
#
#   ./run_agent.sh <run_dir> [-p]
#
#   <run_dir>  a self-contained task folder holding INITIAL_PROMPT.md and its evaluator.
#              Keep it OUTSIDE this repo (e.g. /scratch/vicstorage/agent_runs/erdos_qwen1)
#              so the agent cannot inherit this repo's CLAUDE.md or read docs/ about the task.
#   -p         headless: pipe the prompt in and stream JSON to agent.jsonl.
#              Default is an interactive session you can watch and steer.
#
# Assumes the vLLM server is already up (see serve_qwen.sh for the flags it needs).
# Starts the LiteLLM proxy on demand and leaves it running for later invocations.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?usage: run_agent.sh <run_dir> [-p]}"
MODE="${2:-}"

VLLM_URL=${VLLM_URL:-http://127.0.0.1:8001}
PROXY_PORT=${PROXY_PORT:-4000}
# The proxy lives in the project venv by default -- litellm only ADDS packages there
# (verified: 62 additions, one aiohttp patch bump, no downgrades), so it does not need one
# of its own. Override with LITELLM= if you keep it separate.
_REPO="$(cd "$HERE/../.." && pwd)"
LITELLM=${LITELLM:-$_REPO/src/.venv/bin/litellm}
[ -x "$LITELLM" ] || LITELLM=$HOME/venvs/ccproxy/bin/litellm
PROXY_LOG=${PROXY_LOG:-/tmp/litellm_qwen.log}

[ -d "$RUN_DIR" ] || { echo "no such run dir: $RUN_DIR" >&2; exit 1; }
[ -f "$RUN_DIR/INITIAL_PROMPT.md" ] || { echo "no INITIAL_PROMPT.md in $RUN_DIR" >&2; exit 1; }

# --- 1. the model server must already be serving, with tool calling on ------------------
curl -sf -m 5 "$VLLM_URL/v1/models" >/dev/null \
  || { echo "vLLM not answering at $VLLM_URL -- start serve_qwen.sh first" >&2; exit 1; }

# --- 2. the Anthropic-format shim ------------------------------------------------------
if ! curl -sf -m 3 "http://127.0.0.1:$PROXY_PORT/health/liveliness" >/dev/null 2>&1; then
    echo "[run_agent] starting LiteLLM on :$PROXY_PORT (log: $PROXY_LOG)"
    nohup "$LITELLM" --config "$HERE/litellm_qwen.yaml" --port "$PROXY_PORT" \
        >"$PROXY_LOG" 2>&1 &
    for _ in $(seq 60); do
        curl -sf -m 2 "http://127.0.0.1:$PROXY_PORT/health/liveliness" >/dev/null 2>&1 && break
        sleep 2
    done
    curl -sf -m 2 "http://127.0.0.1:$PROXY_PORT/health/liveliness" >/dev/null 2>&1 \
      || { echo "LiteLLM never came up; see $PROXY_LOG" >&2; exit 1; }
fi
echo "[run_agent] proxy ready on :$PROXY_PORT"

# --- 3. the harness --------------------------------------------------------------------
# shellcheck source=env.sh
source "$HERE/env.sh"
export CLAUDE_CONFIG_DIR="$RUN_DIR/.cc"        # per-run config: no shared history or memory
mkdir -p "$CLAUDE_CONFIG_DIR"

# The kernel tasks grade through a specific interpreter, and a stack without kernels for
# this card fails as a wall of ptxas errors that read like bad candidates. check_gpu.sh
# compiles a real Triton kernel rather than trusting the arch list.
if [ -d "$RUN_DIR/trimul" ]; then
    export KPY=${KPY:-$HOME/venvs/kernel-eval/bin/python}
    "$HERE/check_gpu.sh" || exit 1
fi

cd "$RUN_DIR"
if [ "$MODE" = "-p" ]; then
    # Headless: no UI, the transcript lands in agent.jsonl. Nothing can answer a permission
    # prompt, which is why run_guard.json sets defaultMode acceptEdits.
    claude --settings "$HERE/run_guard.json" \
           --output-format stream-json --verbose \
           -p "$(cat INITIAL_PROMPT.md)" 2>&1 | tee -a agent.jsonl
else
    # Interactive: the normal Claude Code UI opens and the initial prompt is submitted for
    # you, so the run starts on the task and you can watch, interrupt and steer it. Drop the
    # positional argument if you would rather type into an empty session.
    claude --settings "$HERE/run_guard.json" "$(cat INITIAL_PROMPT.md)"
fi

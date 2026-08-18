#!/usr/bin/env bash
# Serve Qwen3.6-27B for the Claude Code harness.
#
# This is the SAME server the ICL runs use, plus the two flags Claude Code cannot
# live without: --enable-auto-tool-choice --tool-call-parser. Claude Code is
# entirely tool-driven (Read / Bash / Edit / Task), so a server that emits tool
# calls as plain text is useless to it -- the agent just narrates and never acts.
#
# --tool-call-parser qwen3_xml matches this checkpoint's chat template, which emits
#   <tool_call>\n<function=NAME>\n<parameter=P>\nvalue\n</parameter>\n</function>\n</tool_call>
# If a vLLM version mismatch makes that parser drop calls, try qwen3_coder (same shape).
#
# Usage:  ./serve_qwen.sh            # foreground, in its own shell/tmux pane
#         PORT=8001 GPUS=0,1 ./serve_qwen.sh
set -euo pipefail

MODEL=${MODEL:-/scratch/vicstorage/qwen}
SERVED_NAME=${SERVED_NAME:-Qwen/Qwen3.6-27B}
VLLM=${VLLM:-/scratch/vicstorage/venvs/vllm026/bin/vllm}
PORT=${PORT:-8001}
TP=${TP:-2}

# On a box where the agent also grades on a GPU (the TriMul task), pin vLLM to a
# subset of cards and leave one free -- a shared card makes every timing garbage.
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="$GPUS"

# Optional replacement chat template (see fetch_chat_template.sh and the README). Passed as
# a flag rather than written into tokenizer_config.json so the checkpoint stays pristine and
# the template is a logged, switchable part of the run config.
TEMPLATE_ARGS=()
[ -n "${CHAT_TEMPLATE:-}" ] && TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")

exec "$VLLM" serve "$MODEL" \
    "${TEMPLATE_ARGS[@]+"${TEMPLATE_ARGS[@]}"}" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.95 \
    --max-model-len 130000 \
    --max-num-seqs 80 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

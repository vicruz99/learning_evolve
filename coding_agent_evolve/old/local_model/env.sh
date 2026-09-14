# Point Claude Code at the local Qwen. Source this in the shell you launch `claude` from:
#     source /path/to/coding_agent_evolve/local_model/env.sh
#
# Setting a credential variable replaces the claude.ai login for the session, so no
# subscription is consumed and no login is needed -- the token is checked by LiteLLM,
# not by Anthropic.

export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_AUTH_TOKEN=sk-local

# Every slot points at the same server. HAIKU/SMALL_FAST are the ones Claude Code uses
# for its own background work (title generation, file summaries); leave them unset and
# it tries to reach a model the proxy has never heard of.
export ANTHROPIC_MODEL=qwen3.6-27b
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.6-27b
export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.6-27b
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.6-27b
export ANTHROPIC_SMALL_FAST_MODEL=qwen3.6-27b

# Claude Code sends its full Anthropic capability set to any ANTHROPIC_BASE_URL gateway.
# A non-Anthropic upstream 400s on the body fields those betas carry (context_management,
# output_config, the strict/defer_loading tool fields). This turns them off.
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1

# No model-discovery probe, no connection warming, no telemetry to a server that isn't there.
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1
export DISABLE_ERROR_REPORTING=1
export DISABLE_AUTOUPDATER=1

# Claude Code treats an unrecognised model name as a current model and sends
# thinking:{"type":"adaptive"}. It auto-retries without it after the first rejection, so
# it self-heals -- but that costs a request per conversation and interacts badly with
# --reasoning-parser qwen3.
#
# This does NOT limit how much Qwen thinks. It only controls what Claude Code *asks* for,
# and drop_params discards the ask before it reaches vLLM. Qwen's actual thinking budget is
# `extra_body.thinking_token_budget` in litellm_qwen.yaml -- see the README.
export MAX_THINKING_TOKENS=0

export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384

# THE context knob. Claude Code does not recognise the ID `qwen3.6-27b`, so it assumes a
# window for it; this declares the one to assume, and applies directly because the ID
# neither starts with `claude-` nor contains `[1m]`. Compaction then happens against this
# number, so the session never sends more than this many tokens to the server.
#
# Deliberately NOT the server's 130k --max-model-len: 88k is the ceiling we want per run.
# CLAUDE_CODE_AUTO_COMPACT_WINDOW is not the way to do this -- it has a hard floor of
# 100000 and is capped at the model's window anyway, so it cannot express a sub-100k
# budget. Leave it unset.
#
# Measured with `claude -p "/context"` at this setting: `Tokens: 3k / 88k`, with a 29.4k
# autocompact buffer, i.e. messages compact around 58k and total context never passes 88k.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=88000

# Keep config, history and memory out of ~/.claude so one run cannot inherit another's
# state. run_agent.sh overrides this per run directory.
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$PWD/.cc}"

#!/usr/bin/env bash
# wirecheck.sh <cell_dir> -- prove what reaches vLLM from both harnesses, on the login node.
# Puts bnbcode_findings/scripts/logproxy.py between each harness and a tool-capable Qwen3.8
# server and prints the captured request bodies' model / max_tokens / temperature / top_p /
# chat_template_kwargs. Uses ports 9101 (logproxy) and 4101 (LiteLLM).
set -u
CELL="$(cd "${1:?usage: wirecheck.sh <bnbcode cell_dir>}" && pwd)"
V2="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP=/tmp/v2wire.$$; mkdir -p $CAP
export PATH=$HOME/.local/bin:$HOME/bin:$PATH XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p $XDG_RUNTIME_DIR
eval "$(~/bin/bnbcode-pg-node env)"; export BNBCODE_DATABASE_URL
MODEL=Qwen/Qwen3.8-27B-FP8

# 1. a tool-capable upstream
UP=""
for h in $(bjobs -w -noheader 2>/dev/null | awk '/gpu/{print $6}' | sed 's/^[0-9]*\*//' | sort -u); do
  for p in 8001 8002; do
    r=$(timeout 40 curl -sS --noproxy '*' http://$h:$p/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"weather in Berlin?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":400}" 2>/dev/null)
    if echo "$r" | grep -q '"tool_calls"'; then UP="$h:$p"; break 2; fi
  done
done
[ -n "$UP" ] || { echo "no tool-capable $MODEL server found"; exit 2; }
echo "upstream: $UP"

# 2. logproxy
python3 ~/work/learning_evolve/coding_agent_evolve/bnbcode_findings/scripts/logproxy.py --port 9101 --upstream "$UP" --log $CAP/wire.jsonl > $CAP/logproxy.out 2>&1 &
LP=$!; sleep 2

# 3. bnbcode through the logproxy, with the cell's config (baseURL swapped)
CFG=$(python3 -c "import json,sys; c=json.load(open('$CELL/workspace/opencode.json')); c['provider']['vllm38']['options']['baseURL']='http://127.0.0.1:9101/v1'; print(json.dumps(c))")
export OPENCODE_CONFIG_CONTENT="$CFG" OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=65536 BNBCODE_VISIBLE_OUTPUT_GUARD=1
mkdir -p $CAP/bnbws && cd $CAP/bnbws
timeout 300 bnbcode run --agent build -m "vllm38/$MODEL" --format json "Reply with the single word OK and nothing else. Do not use any tool." > $CAP/bnb_events.json 2> $CAP/bnb_err.txt
echo "bnbcode rc=$? events=$(wc -l < $CAP/bnb_events.json) err=$(tail -c 300 $CAP/bnb_err.txt)"
cd /

# 4. LiteLLM through the logproxy
sed 's#http://127.0.0.1:9203/v1#http://127.0.0.1:9101/v1#' "$V2/config/litellm_qwen38.yaml" > $CAP/litellm.yaml
( cd "$V2/config" && PYTHONPATH="$V2/config" ~/venvs/ccproxy/bin/litellm --config $CAP/litellm.yaml --port 4101 > $CAP/litellm.log 2>&1 & echo $! > $CAP/litellm.pid )
for i in $(seq 150); do curl -sf --noproxy '*' --max-time 3 http://127.0.0.1:4101/health/liveliness >/dev/null 2>&1 && break; sleep 2; done
echo "litellm up after ~$((i*2))s: $(tail -2 $CAP/litellm.log | tr '\n' ' ' | head -c 200)"
for alias in qwen3.8-low qwen3.8-off; do
  r=$(curl -s --max-time 120 --noproxy '*' -X POST http://127.0.0.1:4101/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-local' \
     -d "{\"model\":\"$alias\",\"max_tokens\":200,\"temperature\":1.0,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}]}")
  echo "litellm $alias -> $(echo "$r" | head -c 200)"
done
# an Anthropic-shaped request, as Claude Code would send it
r=$(curl -s --max-time 120 --noproxy '*' -X POST http://127.0.0.1:4101/v1/messages -H 'Content-Type: application/json' -H 'x-api-key: sk-local' -H 'anthropic-version: 2023-06-01' \
   -d '{"model":"qwen3.8-medium","max_tokens":200,"temperature":1.0,"messages":[{"role":"user","content":"Reply with the single word OK."}]}')
echo "anthropic qwen3.8-medium -> $(echo "$r" | head -c 200)"

# 5. what reached vLLM
echo "=== captured requests"
python3 - $CAP/wire.jsonl <<'PY'
import json, sys
for line in open(sys.argv[1]):
    e = json.loads(line)
    b = e.get("body") or e.get("request") or e.get("json") or {}
    if isinstance(b, str):
        try: b = json.loads(b)
        except Exception: b = {}
    if not isinstance(b, dict) or "messages" not in b:
        print(f"#{e.get('seq')} {e.get('path', e.get('url', '?'))} (non-chat) keys={list(e)[:8]}"); continue
    print(f"#{e.get('seq')} model={b.get('model')} max_tokens={b.get('max_tokens') or b.get('max_completion_tokens')} temp={b.get('temperature')} top_p={b.get('top_p')} "
          f"ctk={b.get('chat_template_kwargs')} reasoning_effort={b.get('reasoning_effort')} tools={len(b.get('tools') or [])} stream={b.get('stream')} n_msgs={len(b['messages'])}")
PY
kill $LP $(cat $CAP/litellm.pid) 2>/dev/null; sleep 1; pkill -f "litellm --config $CAP/litellm.yaml" 2>/dev/null
echo "--- litellm.log tail"; tail -15 $CAP/litellm.log
echo "capture dir: $CAP"

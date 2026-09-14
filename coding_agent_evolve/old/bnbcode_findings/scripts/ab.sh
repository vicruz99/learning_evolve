#!/bin/bash
# A/B: does pinning temperature change how often bnbcode's model stops calling tools?
#
#   ab.sh setup <force_temp|none>     put the logging proxy on :9001, optionally pinning temp
#   ab.sh run   <label> <runs> <secs> N sequential headless runs of the real prompt
#   ab.sh report <label>              per-run turn / stall counts
set -u
E=$HOME/work/learning_evolve/coding_agent_evolve/experiments
SC=$HOME/work/learning_evolve/coding_agent_evolve/bnbcode_findings/scripts
CAP=/tmp/bnbtest/capture; mkdir -p $CAP
export PATH=$HOME/.local/bin:$PATH
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)

case "${1:?}" in
setup)
  T=${2:-none}
  pkill -f "[l]ogproxy.py"; pkill -f "[l]lm_relay.py"; sleep 2
  nohup python3 $E/llm_relay.py --jobs gpu2:8001,gpu6:8002 --listen-port 9002 \
        </dev/null >>$CAP/relay9002.log 2>&1 & disown 2>/dev/null
  sleep 5
  ARG=""; [ "$T" != none ] && ARG="--force-temperature $T"
  nohup python3 $SC/logproxy.py --port 9001 --upstream 127.0.0.1:9002 --log $CAP/ab.jsonl $ARG \
        </dev/null >>$CAP/proxy.log 2>&1 & disown 2>/dev/null
  sleep 3
  echo -n "  $(hostname -s): temp=$T  proxy "
  (exec 3<>/dev/tcp/127.0.0.1/9001) 2>/dev/null && echo -n "up  " || echo -n "DOWN  "
  curl -sS --noproxy '*' --max-time 20 http://127.0.0.1:9001/v1/models | head -c 40; echo
  ;;
run)
  LABEL=${2:?}; N=${3:-4}; SECS=${4:-480}
  eval "$($HOME/bin/bnbcode-pg-node env)" 2>/dev/null; export BNBCODE_DATABASE_URL
  nohup bash -c '
    for i in $(seq 1 '"$N"'); do
      D=/tmp/bnbtest/ab_'"$LABEL"'_$i
      rm -rf $D; mkdir -p $D; cd $D
      cp '"$E"'/AC2/eval.py '"$E"'/AC2/*.npy '"$E"'/opencode.json .
      cp $HOME/agent_runs/ac2_plain_s2/INITIAL_PROMPT.md .
      timeout '"$SECS"' bnbcode run --agent build --format json "$(cat INITIAL_PROMPT.md)" \
          > events.json 2> stderr.txt
      echo $? > exitcode
    done
    touch /tmp/bnbtest/ab_'"$LABEL"'_DONE' </dev/null >/dev/null 2>&1 &
  disown 2>/dev/null
  echo "  $(hostname -s): launched $N runs labelled $LABEL (${SECS}s each)"
  ;;
report)
  LABEL=${2:?}
  [ -f /tmp/bnbtest/ab_${LABEL}_DONE ] && echo "  ($LABEL: all runs finished)" || echo "  ($LABEL: still running)"
  python3 - "$LABEL" <<'PY'
import glob, json, os, re, sys
lab = sys.argv[1]
dirs = sorted(glob.glob('/tmp/bnbtest/ab_%s_[0-9]*' % lab),
              key=lambda d: int(re.search(r'_(\d+)$', d).group(1)))
print("  %-6s %-9s %-7s %-7s %-8s %s" % ("run", "turns", "tools", "texts", "exit", "last text"))
tot_turns = tot_tools = 0
for d in dirs:
    ev = os.path.join(d, 'events.json')
    if not os.path.exists(ev):
        continue
    tools = texts = steps = 0
    last = ""
    for line in open(ev, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get('type')
        if t == 'tool_use':
            tools += 1
        elif t == 'text':
            texts += 1
            last = (e.get('text') or e.get('properties', {}).get('text') or last)
        elif t == 'step_start':
            steps += 1
    code = open(os.path.join(d, 'exitcode')).read().strip() if os.path.exists(os.path.join(d, 'exitcode')) else '-'
    tot_turns += steps; tot_tools += tools
    print("  %-6s %-9d %-7d %-7d %-8s %s"
          % (os.path.basename(d).split('_')[-1], steps, tools, texts, code,
             (last or '')[:60].replace('\n', ' ')))
print("  TOTAL  turns=%d  tool_calls=%d  over %d runs" % (tot_turns, tot_tools, len(dirs)))
print("  exit 124 = hit the wall-clock cap while still working (good);"
      " exit 0 = the model ended its turn (stopped)")
PY
  ;;
esac

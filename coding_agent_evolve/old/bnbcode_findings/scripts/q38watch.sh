#!/bin/bash
# Liveness for a Qwen3.8 cell WITHOUT relying on postgres: bnbcode stops persisting parts
# during a long multi-step turn, so the DB goes stale while the agent is working fine.
# File mtimes and relay connections are the real heartbeat.
set -u
NAME=${1:?}
R=$HOME/agent_runs/$NAME
LOG=$HOME/agent_runs/relay9003.$(hostname -s).log
echo "===== $NAME @ $(date +%H:%M:%S) ====="
if [ -f "$R/exitcode" ]; then
  echo "  EXITED code=$(cat $R/exitcode) after $(( $(cat $R/.ended_at) - $(cat $R/.started_at) ))s"
else
  echo "  running $(( ($(date +%s) - $(cat $R/.started_at)) / 60 )) min"
fi
echo "  LLM requests so far : $(grep -c 'connection ->' "$LOG" 2>/dev/null)"
echo "  last LLM request    : $(grep 'connection ->' "$LOG" 2>/dev/null | tail -1 | sed 's/.*\[llm-relay \([0-9:]*\)\].*/\1/')"
echo "  events.json         : $(wc -l < $R/events.json 2>/dev/null) lines, tool_use=$(grep -c tool_use $R/events.json 2>/dev/null)"
echo "  run/attempts        : $(ls $R/run/attempts 2>/dev/null | wc -l) files"
echo "  newest file in run/ : $(ls -t $R/run 2>/dev/null | head -1)  at $(stat -c %y $R/run/$(ls -t $R/run 2>/dev/null | head -1) 2>/dev/null | cut -c12-19)"
echo "  scratch /tmp/opencode: $(ls /tmp/opencode 2>/dev/null | wc -l) files, newest $(ls -t /tmp/opencode 2>/dev/null | head -1) at $(stat -c %y /tmp/opencode/$(ls -t /tmp/opencode 2>/dev/null | head -1) 2>/dev/null | cut -c12-19)"
echo "  ledger              : $([ -f $R/run/LEDGER.md ] && wc -l < $R/run/LEDGER.md || echo none) lines"
[ -f "$R/run/LEDGER.md" ] && tail -3 "$R/run/LEDGER.md" | sed 's/^/    /'

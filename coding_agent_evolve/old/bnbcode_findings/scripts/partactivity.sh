#!/bin/bash
# Part-level activity for a run. A single assistant message can contain many steps, so
# message timestamps go stale during a long multi-step turn; parts are the real heartbeat.
set -u
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
PG="$HOME/bin/bnbcode-pg-node"
D="${1:?usage: partactivity.sh <dir-substring>}"

SID=$("$PG" psql -t -A -c "select id from session where directory like '%$D%' order by id desc limit 1;" 2>/dev/null | tr -d ' ')
echo "session: $SID    now: $(date +%H:%M:%S)"

echo "--- part heartbeat ---"
"$PG" psql -t -A -F'|' -c "
select '  parts: ' || count(*)
    || '   newest part: ' || to_char(to_timestamp(max(time_created)/1000),'HH24:MI:SS')
    || '   oldest: ' || to_char(to_timestamp(min(time_created)/1000),'HH24:MI:SS')
from part where session_id='$SID';" 2>/dev/null

echo "--- the last 12 parts, in order ---"
"$PG" psql -t -A -F'|' -c "
select to_char(to_timestamp(time_created/1000),'HH24:MI:SS') || '  ' ||
       rpad(coalesce(data->>'type','?'),12) ||
       rpad(coalesce(data->>'tool',''),14) ||
       coalesce(data->'state'->>'status','') || ' ' ||
       left(regexp_replace(coalesce(data->'state'->'input'->>'command', data->>'text', ''),'\s+',' ','g'), 70)
from part where session_id='$SID' order by time_created desc limit 12;" 2>/dev/null

echo "--- any tool still marked running? ---"
"$PG" psql -t -A -c "
select '  ' || coalesce(data->>'tool','?') || ' status=' || coalesce(data->'state'->>'status','?')
    || ' started ' || to_char(to_timestamp(((data->'state'->'time'->>'start')::bigint)/1000),'HH24:MI:SS')
from part where session_id='$SID' and data->>'type'='tool'
  and coalesce(data->'state'->>'status','') <> 'completed';" 2>/dev/null

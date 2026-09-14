#!/bin/bash
# Progress + stall report for one bnbcode run folder. Read-only.
#   bash watchruns.sh <dir-substring>          e.g. ac2_plain_s2
set -u
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
PG="$HOME/bin/bnbcode-pg-node"
D="${1:?usage: watchruns.sh <run-folder-substring>}"

echo "===== $D  on $(hostname -s)  at $(date '+%H:%M:%S') ====="
"$PG" status >/dev/null 2>&1 || { echo "  no postgres on this node"; exit 0; }

SID=$("$PG" psql -t -A -c "select id from session where directory like '%$D%' order by id desc limit 1;" 2>/dev/null)
if [ -z "$SID" ]; then echo "  no session yet for $D on this node"; exit 0; fi
echo "  session $SID"

"$PG" psql -t -A -F'|' -c "
select '  started      ' || to_char(to_timestamp(min(time_created)/1000),'HH24:MI:SS')
       || '   last activity ' || to_char(to_timestamp(max(time_created)/1000),'HH24:MI:SS')
       || '   (' || round(extract(epoch from (now()-to_timestamp(max(time_created)/1000)))/60) || ' min ago)'
from message where session_id='$SID';" 2>/dev/null

"$PG" psql -t -A -F'|' -c "
with a as (select m.id,
             (m.data->'tokens'->>'input')::bigint as inp,
             exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='tool') as acted
           from message m where m.session_id='$SID' and m.data->>'role'='assistant'
             and coalesce(m.data->>'mode','')='build')
select '  assistant turns ' || count(*)
       || '   stalls ' || count(*) filter (where not acted)
       || '   (' || coalesce(round(100.0*count(*) filter (where not acted)/nullif(count(*),0))::text,'-') || '%)'
       || '   context now ' || coalesce(max(inp)::text,'-')
from a;" 2>/dev/null

echo "  --- stalls by context bucket ---"
"$PG" psql -t -A -F'|' -c "
with a as (select (m.data->'tokens'->>'input')::bigint as inp,
             exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='tool') as acted
           from message m where m.session_id='$SID' and m.data->>'role'='assistant'
             and coalesce(m.data->>'mode','')='build' and (m.data->'tokens'->>'input')::bigint>0)
select '    ' || lpad(((inp/20000)*20000)::text,6) || '-' || lpad(((inp/20000)*20000+19999)::text,6)
       || '   turns ' || lpad(count(*)::text,4)
       || '   stalls ' || lpad((count(*) filter (where not acted))::text,4)
       || '   ' || lpad(round(100.0*count(*) filter (where not acted)/count(*))::text,3) || '%'
from a group by (inp/20000) order by 1;" 2>/dev/null

echo "  --- tools used ---"
# group by the tool name, not by the formatted line: an aggregate inside the GROUP BY
# expression is a hard error, and with stderr discarded it silently prints nothing.
"$PG" psql -t -A -F'|' -c "
select '    ' || rpad(t,14) || n from (
  select data->>'tool' as t, count(*) as n
  from part where session_id='$SID' and data->>'type'='tool'
  group by 1) z order by n desc;" 2>/dev/null

echo "  --- THE CHECK: any continual-work injection left? ---"
"$PG" psql -t -A -c "
select '    injections in context: ' || count(*) ||
       '   (expected 0 with continual_work disabled)'
from part where session_id='$SID' and data->>'type'='text'
  and data->>'text' like '%continual-work-north-star%';" 2>/dev/null

"$PG" psql -t -A -c "
select '    continual-work state: ' || coalesce(metadata->'sessionContinualWork'->>'mode','none set')
from session where id='$SID';" 2>/dev/null

echo "  --- context composition so far ---"
"$PG" psql -t -A -F'|' -c "
with p as (
  select case when d.t='reasoning' then 'reasoning'
              when d.t='tool' then 'tool call+result'
              when d.t='text' and m.data->>'role'='assistant' then 'assistant text'
              when d.t='text' then 'user/injected'
              else d.t end as k,
         length(coalesce(pt.data->>'text','')) +
         length(coalesce((pt.data->'state'->'input')::text,'')) +
         length(coalesce(pt.data->'state'->>'output','')) as chars
  from part pt join message m on m.id=pt.message_id
       cross join lateral (select pt.data->>'type' as t) d
  where pt.session_id='$SID' and d.t in ('reasoning','tool','text'))
select '    ' || rpad(k,18) || lpad(to_char(sum(chars),'999G999G999'),12) || ' chars  '
       || lpad(round(100.0*sum(chars)/sum(sum(chars)) over ())::text,3) || '%'
from p group by k order by sum(chars) desc;" 2>/dev/null

echo "  --- output on disk ---"
R="$HOME/agent_runs/$D"
echo "    attempts: $(ls "$R"/run/attempts 2>/dev/null | wc -l) files   run/: $(ls "$R"/run 2>/dev/null | tr '\n' ' ')"
[ -f "$R/run/LEDGER.md" ] && echo "    LEDGER.md: $(wc -l < "$R/run/LEDGER.md") lines" || echo "    LEDGER.md: not written"

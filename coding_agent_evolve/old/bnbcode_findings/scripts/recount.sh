#!/bin/bash
# Corrected stall rate: only assistant rows that were REAL completed API calls.
# A real call has a finish reason and nonzero token accounting. Rows with finish=null and
# tokens=0 are aborted/incomplete (e.g. the process was killed mid-turn) and must not count.
set -u
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
PG="$HOME/bin/bnbcode-pg-node"

echo "########## $(hostname -s) ##########"
echo "=== per session: real calls, stalls, and how many rows the old query miscounted ==="
"$PG" psql -t -A -F'|' -c "
select rpad(split_part(s.directory,'/',5),16),
       'rows=' || count(*) ||
       '  real_calls=' || count(*) filter (where real) ||
       '  stalls=' || count(*) filter (where real and not acted) ||
       '  rate=' || coalesce(round(100.0*count(*) filter (where real and not acted)
                                   / nullif(count(*) filter (where real),0))::text,'-') || '%' ||
       '  incomplete_rows=' || count(*) filter (where not real)
from (
  select m.session_id,
         (m.data->>'finish' is not null
          and coalesce((m.data->'tokens'->>'output')::bigint,0) > 0) as real,
         exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='tool') as acted
  from message m where m.data->>'role'='assistant' and coalesce(m.data->>'mode','')='build'
) z join session s on s.id=z.session_id
group by s.directory, s.id order by s.id;" 2>/dev/null

echo
echo "=== corrected stall rate by context bucket (real calls only) ==="
"$PG" psql -t -A -F'|' -c "
with a as (
  select (m.data->'tokens'->>'input')::bigint as inp,
         exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='tool') as acted
  from message m
  where m.data->>'role'='assistant' and coalesce(m.data->>'mode','')='build'
    and m.data->>'finish' is not null
    and coalesce((m.data->'tokens'->>'output')::bigint,0) > 0
    and coalesce((m.data->'tokens'->>'input')::bigint,0) > 0)
select lpad(((inp/20000)*20000)::text,6) || '-' || lpad(((inp/20000)*20000+19999)::text,6)
       || '   calls ' || lpad(count(*)::text,4)
       || '   stalls ' || lpad((count(*) filter (where not acted))::text,4)
       || '   ' || lpad(round(100.0*count(*) filter (where not acted)/count(*))::text,3) || '%'
from a group by (inp/20000) order by 1;" 2>/dev/null

echo
echo "=== what the incomplete rows look like (the ones the old query counted as stalls) ==="
"$PG" psql -t -A -F'|' -c "
select coalesce(m.data->>'finish','NULL') as finish,
       coalesce(m.data->'tokens'->>'input','-') as inp,
       coalesce(m.data->'tokens'->>'output','-') as outp,
       count(*)
from message m
where m.data->>'role'='assistant' and coalesce(m.data->>'mode','')='build'
  and not (m.data->>'finish' is not null and coalesce((m.data->'tokens'->>'output')::bigint,0) > 0)
group by 1,2,3 order by 4 desc limit 8;" 2>/dev/null

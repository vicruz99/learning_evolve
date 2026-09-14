#!/bin/bash
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
PG="$HOME/bin/bnbcode-pg-node"
echo "########## $(hostname -s)  $(date '+%H:%M:%S') ##########"
"$PG" status >/dev/null 2>&1 || { echo "no postgres on this node"; exit 0; }

echo "=== node load ==="
echo "  load: $(cut -d' ' -f1-3 /proc/loadavg)   slots: $(nproc)"
ps -eo user,pcpu --no-headers | awk '{c[$1]+=$2} END {for (u in c) printf "  %-10s %7.0f%% CPU\n", u, c[u]}' | sort -k2 -rn | head -4

echo
echo "=== bash calls: plain shell vs runs-python ==="
$PG psql -t -A -F'|' -c "
with p as (
  select ((data->'state'->'time'->>'end')::bigint - (data->'state'->'time'->>'start')::bigint)::numeric as ms,
         coalesce(data->'state'->'input'->>'command','') as cmd
  from part where data->>'type'='tool' and data->>'tool'='bash'
    and data->'state'->'time'->>'end' is not null)
select case when cmd ~ 'venvs/agent-eval|python|eval\.py' then 'runs python' else 'plain shell' end,
       count(*), round(avg(ms)/1000,1) as avg_s,
       round((percentile_cont(0.5) within group (order by ms))::numeric/1000,2) as p50_s,
       round(max(ms)/1000,1) as max_s,
       count(*) filter (where ms > 890000) as killed_at_900s
from p group by 1;" 2>&1 | head -5

echo
echo "=== wall clock burned on killed commands ==="
$PG psql -t -A -F'|' -c "
with p as (select ((data->'state'->'time'->>'end')::bigint - (data->'state'->'time'->>'start')::bigint) as ms
           from part where data->>'type'='tool' and data->>'tool'='bash'
             and data->'state'->'time'->>'end' is not null)
select round(sum(ms) filter (where ms > 890000)/60000.0) as min_killed,
       round(sum(ms)/60000.0) as min_total_bash,
       round(100.0*coalesce(sum(ms) filter (where ms > 890000),0)/nullif(sum(ms),0)) as pct_wasted
from p;" 2>&1 | head -3

echo
echo "=== turns that ended with no tool call ==="
$PG psql -t -A -F'|' -c "
select count(*) filter (where not has_tool) as silent_stops,
       count(*) as assistant_turns
from (select m.id, exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='tool') as has_tool
      from message m where m.data->>'role'='assistant') z;" 2>&1 | head -3

echo
echo "=== of those silent stops, how many came right after a killed command? ==="
$PG psql -t -A -F'|' -c "
with tools as (
  select (m.data->'time'->>'created')::bigint as t,
         ((p.data->'state'->'time'->>'end')::bigint - (p.data->'state'->'time'->>'start')::bigint) as ms
  from part p join message m on m.id=p.message_id
  where p.data->>'type'='tool' and p.data->>'tool'='bash'
    and p.data->'state'->'time'->>'end' is not null),
stops as (
  select (m.data->'time'->>'created')::bigint as t from message m
  where m.data->>'role'='assistant'
    and not exists (select 1 from part p where p.message_id=m.id and p.data->>'type'='tool'))
select count(*) as silent_stops,
       count(*) filter (where exists (select 1 from tools x
              where x.t < s.t and x.t > s.t - 1200000 and x.ms > 890000)) as within_20min_of_a_kill
from stops s;" 2>&1 | head -3

echo
echo "=== background option usage ==="
$PG psql -t -A -F'|' -c "
select coalesce(data->'state'->'input'->>'background','(not set)'), count(*)
from part where data->>'type'='tool' and data->>'tool'='bash' group by 1 order by 2 desc;" 2>&1 | head -4

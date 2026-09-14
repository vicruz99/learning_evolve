#!/bin/bash
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
PG="$HOME/bin/bnbcode-pg-node"
DIR="${1:-ac2_evo_s1}"
OUT=$HOME/.bnbparts.jsonl

echo "=== which json keys does each part type actually carry? ==="
$PG psql -t -A -F'|' -c "
with s as (select id from session where directory like '%$DIR%' order by id desc limit 1)
select p.data->>'type', string_agg(distinct k, ',' order by k)
from part p, s, jsonb_object_keys(p.data) k
where p.session_id=s.id group by 1;" 2>&1 | head -8

$PG psql -t -A -c "
with s as (select id from session where directory like '%$DIR%' order by id desc limit 1)
select row_to_json(x) from (
  select p.data->>'type' as ptype, coalesce(p.data->>'tool','') as tool,
         coalesce(p.data->>'text','') as text,
         coalesce(p.data->>'reasoning','') as reasoning,
         coalesce((p.data->'state'->'input')::text,'') as tool_input,
         coalesce(p.data->'state'->>'output','') as tool_output,
         coalesce(m.data->>'role','') as role
  from part p join message m on m.id=p.message_id, s
  where p.session_id=s.id
) x;" > "$OUT" 2>/dev/null

"$HOME/venvs/ccproxy/bin/python" - "$OUT" "$DIR" <<'PY'
import collections, glob, json, os, sys
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(glob.glob(os.path.join(
    os.environ["HF_HOME"], "hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/tokenizer.json"))[0])
def ntok(s): return len(TOK.encode(s, add_special_tokens=False).ids) if s else 0

cat, cnt = collections.Counter(), collections.Counter()
injected = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line: continue
    try: r = json.loads(line)
    except Exception: continue
    t, role = r.get("ptype"), r.get("role")
    if t == "reasoning":
        # the content sits in 'text' for reasoning parts, not in 'reasoning'
        s = r.get("reasoning") or r.get("text") or ""
        cat["assistant reasoning"] += ntok(s); cnt["assistant reasoning"] += 1
    elif t == "text":
        if role == "assistant":
            cat["assistant text"] += ntok(r["text"]); cnt["assistant text"] += 1
        else:
            n = ntok(r["text"]); cat["user / injected text"] += n
            cnt["user / injected text"] += 1
            injected.append((n, r["text"][:90].replace("\n", " ")))
    elif t == "tool":
        cat["tool CALLS (the request)"] += ntok(r["tool_input"]); cnt["tool CALLS (the request)"] += 1
        cat["tool RESULTS (the output)"] += ntok(r["tool_output"]); cnt["tool RESULTS (the output)"] += 1

tot = sum(cat.values()) or 1
print("\n=== bnbcode: %s ===" % sys.argv[2])
print("%-28s %10s %8s %6s" % ("category", "tokens", "blocks", "share"))
for k, v in cat.most_common():
    print("  %-26s %10s %8d %5.0f%%" % (k, f"{v:,}", cnt[k], 100.0*v/tot))
print("  %-26s %10s" % ("TOTAL", f"{tot:,}"))

print("\nthe injected user-side blocks, largest first:")
for n, s in sorted(injected, reverse=True)[:8]:
    print("  %7d tok  %s" % (n, s))
PY
rm -f "$OUT"

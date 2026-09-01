# -*- coding: utf-8 -*-
"""Classify every stop, using process evidence AND whether the agent monitored the work.

Why this exists. A turn that makes no tool call ENDS the agent loop -- there is no waiting
state. So a live background process does not, on its own, make a stop benign: if the agent
launched something and then stopped paying attention to it, the run is just as dead. The
pilot's evidence that one cell was behaving correctly was never the live process itself, it
was the 456 polling calls around it.

A stop therefore counts as legitimate waiting only when BOTH hold:
  1. a worker was actually running at the moment of the stop  (run/procsample.jsonl), and
  2. the next tool call after the agent resumed reads that work
     (tail/cat a log, ps, jobs, pgrep, ls of an output dir, ...)

Classes:
  mid-thought    reasoning, no visible text                        -> a stop
  waiting        worker alive AND the agent came back and checked  -> NOT a stop
  abandoned      worker alive but the agent never checked on it    -> a stop
  signoff-idle   visible text, nothing running                     -> a stop
  truncation     finish = length / max_tokens                      -> a stop
  empty          neither text nor reasoning                        -> a stop

usage: stopclass.py <run_dir> [<run_dir> ...]
"""
import calendar
import json
import os
import re
import subprocess
import sys
import time as _time

# How to reach bnbcode's session store. bnbcode keeps its history in PostgreSQL; on the
# Bosch cluster that is a userspace server started per node, so the wrapper is what knows
# the socket. Point BNB_PSQL at whatever runs psql against your bnbcode database.
PG = os.environ.get("BNB_PSQL", os.path.expanduser("~/bin/bnbcode-pg-node")).split()
CPU_BUSY = 20.0        # percent; below this a "process" is an idle shell, not real work
NEAR = 30              # seconds either side of a stop that a sample may be taken from

POLL = re.compile(
    r"\b(tail|cat|head|less|grep)\b[^|;]*\.(log|out|err|txt|json|jsonl)\b"
    r"|\bps\b|\bpgrep\b|\bjobs\b|\bnvidia-smi\b"
    r"|\bls\b[^|;]*(run/|attempts|results|out)"
    r"|\bcheck",
    re.I)

SQL = """
select m.time_created,
       coalesce(m.data->>'finish',''),
       coalesce((m.data->'tokens'->>'output')::bigint,0),
       exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='tool'),
       exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='text'
              and coalesce(p.data->>'synthetic','false')<>'true'
              and length(trim(coalesce(p.data->>'text','')))>0),
       exists(select 1 from part p where p.message_id=m.id and p.data->>'type'='reasoning'),
       coalesce((select string_agg(substr(p.data->'state'->>'input',1,300),' ~ ')
                 from part p where p.message_id=m.id and p.data->>'type'='tool'),'')
from message m join session s on s.id=m.session_id
where m.data->>'role'='assistant' and coalesce(m.data->>'mode','')='build'
  and s.directory = '{run}'
order by m.time_created asc;
"""


def samples(run):
    out = []
    path = os.path.join(run, "run", "procsample.jsonl")
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def busy_at(sam, t):
    """Was real work in flight at epoch t? None means we have no evidence either way."""
    best = None
    for r in sam:
        d = abs(r.get("t", 0) - t)
        if d <= NEAR and (best is None or d < best[0]):
            best = (d, r)
    if best is None:
        return None
    return any(p.get("cpu", 0) >= CPU_BUSY for p in best[1].get("procs", []))


def bnb_turns(run):
    """Assistant build turns for one run directory, oldest first."""
    try:
        out = subprocess.run(PG + ["psql", "-t", "-A", "-F", "\x1f", "-c", SQL.format(run=run)],
                             capture_output=True, text=True, timeout=180).stdout
    except Exception as exc:
        sys.stderr.write("psql failed for %s: %s\n" % (run, exc))
        return []
    turns = []
    for line in out.splitlines():
        f = line.split("\x1f")
        if len(f) < 7:
            continue
        try:
            turns.append({"t": int(f[0]) // 1000, "finish": f[1], "out": int(f[2]),
                          "tool": f[3] == "t", "text": f[4] == "t", "reason": f[5] == "t",
                          "cmd": f[6]})
        except ValueError:
            continue
    return turns


def cc_turns(run):
    turns, seen = [], {}
    root = os.path.join(run, ".cc", "projects")
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".jsonl"):
                continue
            for line in open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") != "assistant":
                    continue
                m = r.get("message") or {}
                mid = m.get("id")
                e = seen.get(mid)
                if e is None:
                    ts = r.get("timestamp", "")
                    try:
                        tt = calendar.timegm(_time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
                    except ValueError:
                        tt = 0
                    u = m.get("usage") or {}
                    e = seen[mid] = {"t": tt, "finish": m.get("stop_reason") or "",
                                     "out": u.get("output_tokens", 0), "tool": False,
                                     "text": False, "reason": False, "cmd": ""}
                    turns.append(e)
                content = m.get("content") or []
                for b in (content if not isinstance(content, str) else []):
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        e["tool"] = True
                        e["cmd"] += " " + json.dumps(b.get("input", {}))[:300]
                    elif b.get("type") == "thinking":
                        e["reason"] = True
                    elif b.get("type") == "text" and (b.get("text") or "").strip():
                        e["text"] = True
    turns.sort(key=lambda x: x["t"])
    return turns


ORDER = ("mid-thought", "waiting", "abandoned", "signoff-idle", "truncation", "empty")


def load_turns(run, is_cc):
    """Turns for a run, cached to run/turns.json.

    bnbcode session state lives in a PER-NODE postgres (bnbcode-pg-node keeps a node-local
    PGDATA so concurrent nodes do not collide). A run's history is therefore only readable
    on the node that executed it. Caching to the shared run folder makes the later
    aggregate analysis node-independent -- run this once on-node when a cell finishes.
    """
    cache = os.path.join(run, "run", "turns.json")
    if os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                return cached
        except (OSError, ValueError):
            pass
    turns = cc_turns(run) if is_cc else bnb_turns(run)
    if turns and not is_cc:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(turns, f)
        except OSError:
            pass
    return turns


def classify(run):
    is_cc = os.path.isdir(os.path.join(run, ".cc"))
    turns = load_turns(run, is_cc)
    trunc_word = "max_tokens" if is_cc else "length"
    sam = samples(run)

    real = [t for t in turns if t["finish"] and t["out"] > 0]
    counts = dict((k, 0) for k in ORDER)
    blind = 0
    for i, t in enumerate(real):
        if t["tool"]:
            continue
        if t["finish"] == trunc_word:
            counts["truncation"] += 1
            continue
        if not t["text"]:
            counts["mid-thought" if t["reason"] else "empty"] += 1
            continue
        busy = busy_at(sam, t["t"])
        nxt = next((x for x in real[i + 1:] if x["tool"]), None)
        checked = bool(nxt and POLL.search(nxt["cmd"] or ""))
        if busy is None:
            blind += 1
            counts["signoff-idle"] += 1
        elif busy and checked:
            counts["waiting"] += 1
        elif busy:
            counts["abandoned"] += 1
        else:
            counts["signoff-idle"] += 1

    stops = sum(v for k, v in counts.items() if k != "waiting")
    n = len(real) or 1
    detail = "  ".join("%s=%d" % (k, counts[k]) for k in ORDER if counts[k])
    note = "   [%d sign-offs with no sampler data]" % blind if blind else ""
    print("%-32s %5d calls  %4d true stops (%4.1f%%)   %s%s"
          % (os.path.basename(run), len(real), stops, 100.0 * stops / n, detail, note))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        classify(os.path.abspath(d))

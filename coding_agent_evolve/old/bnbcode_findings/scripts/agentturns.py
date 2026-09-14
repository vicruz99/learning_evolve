# -*- coding: utf-8 -*-
"""List every real agent turn from a proxy capture and show how it ended.

Title-generation calls are excluded by prompt size: they run ~3k prompt tokens with no
tools, while an agent turn carries the 14.5k system prompt plus 39k of tool schema.
"""
import json, re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bnbtest/capture/ab.jsonl"
rows = []
for line in open(path, encoding="utf-8", errors="replace"):
    line = line.strip()
    if line:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

resps = [r for r in rows if r.get("kind") == "response"]
print("total responses captured: %d" % len(resps))

def usage(blob):
    m = re.search(r'"usage":\s*\{[^}]*"prompt_tokens":\s*(\d+)[^}]*"completion_tokens":\s*(\d+)', blob)
    if not m:
        m2 = re.search(r'"prompt_tokens":\s*(\d+)', blob)
        m3 = re.search(r'"completion_tokens":\s*(\d+)', blob)
        return (int(m2.group(1)) if m2 else 0, int(m3.group(1)) if m3 else 0)
    return int(m.group(1)), int(m.group(2))

print("\n  %-4s %-9s %-9s %-11s %-9s %s"
      % ("#", "prompt", "output", "finish", "tool_call?", "kind"))
agent_stops = []
for i, r in enumerate(resps):
    blob = (r.get("tail") or "") + (r.get("head") or "")
    pt, ct = usage(blob)
    fin = "stop" if '"finish_reason":"stop"' in blob else (
          "tool_calls" if '"finish_reason":"tool_calls"' in blob else (
          "length" if '"finish_reason":"length"' in blob else "?"))
    has_tc = '"tool_calls"' in blob
    kind = "AGENT TURN" if pt > 10000 else "title-gen/aux"
    print("  %-4d %-9d %-9d %-11s %-9s %s" % (i, pt, ct, fin, has_tc, kind))
    if kind == "AGENT TURN" and fin == "stop" and not has_tc:
        agent_stops.append((i, r, pt, ct))

print("\n  AGENT turns that ended with `stop` and no tool call: %d" % len(agent_stops))
for i, r, pt, ct in agent_stops[:2]:
    tail = r.get("tail", "")
    # reassemble the text the model actually produced, in order
    content, reasoning = [], []
    for m in re.finditer(r'"delta":\{"(content|reasoning)":"((?:[^"\\]|\\.)*)"', tail):
        try:
            piece = json.loads('"%s"' % m.group(2))
        except Exception:
            piece = m.group(2)
        (content if m.group(1) == "content" else reasoning).append(piece)
    print("\n" + "=" * 74)
    print("  response #%d  prompt=%d output=%d" % (i, pt, ct))
    print("  --- last 900 chars of REASONING ---")
    print("  " + "".join(reasoning)[-900:].replace("\n", "\n  "))
    print("  --- CONTENT emitted (%d chars) ---" % len("".join(content)))
    print("  " + ("".join(content)[-900:] or "(EMPTY — no content at all)").replace("\n", "\n  "))
    print("  --- any function/tool markup in the raw stream? ---")
    for pat in ("<function", "<tool_call", "tool_calls", "</think", "im_end", "<parameter"):
        print("      %-14s %d occurrences" % (pat, tail.count(pat)))

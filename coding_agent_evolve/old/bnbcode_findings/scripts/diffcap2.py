# -*- coding: utf-8 -*-
"""Classify every captured request and report the sampling params bnbcode really sends."""
import collections, hashlib, json, os, sys

CAP = "/tmp/bnbtest/capture"


def load(tag):
    rows = []
    p = os.path.join(CAP, "path_%s.jsonl" % tag)
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


for tag in ("headless", "tui"):
    rows = load(tag)
    reqs = [r for r in rows if r.get("kind") != "response"
            and isinstance(r.get("body"), dict) and "messages" in r["body"]]
    print("\n" + "=" * 76)
    print("PATH %s — %d chat requests" % (tag, len(reqs)))
    print("=" * 76)
    print("  %-4s %-8s %-6s %-9s %-7s %-6s %s"
          % ("#", "bytes", "msgs", "tools", "temp", "top_p", "system-prompt md5 / first user line"))
    for i, r in enumerate(reqs):
        b = r["body"]
        tools = b.get("tools") or []
        sysmsgs = [m for m in b["messages"] if m.get("role") == "system"]
        systext = "\n".join(str(m.get("content", "")) for m in sysmsgs)
        firstuser = ""
        for m in b["messages"]:
            if m.get("role") == "user":
                c = m.get("content")
                firstuser = (c if isinstance(c, str) else json.dumps(c))[:46].replace("\n", " ")
                break
        print("  %-4d %-8d %-6d %-9d %-7s %-6s %s | %s"
              % (i, r.get("n_bytes", 0), len(b["messages"]), len(tools),
                 b.get("temperature"), b.get("top_p"),
                 hashlib.md5(systext.encode()).hexdigest()[:8], firstuser))

    agent = [r for r in reqs if (r["body"].get("tools") or [])]
    print("\n  requests WITH tools (the agent loop): %d of %d" % (len(agent), len(reqs)))
    if agent:
        b = agent[0]["body"]
        tools = b["tools"]
        sysmsgs = [m for m in b["messages"] if m.get("role") == "system"]
        systext = "\n".join(str(m.get("content", "")) for m in sysmsgs)
        print("    sampling actually sent : temperature=%r  top_p=%r  max_tokens=%r"
              % (b.get("temperature"), b.get("top_p"), b.get("max_tokens")))
        print("    tool_choice            : %r" % b.get("tool_choice"))
        print("    system prompt          : %d chars  md5=%s"
              % (len(systext), hashlib.md5(systext.encode()).hexdigest()[:12]))
        print("    tools (%d)              : %s"
              % (len(tools), ", ".join(sorted(t.get("function", {}).get("name", "?") for t in tools))))
        print("    tool schema bytes      : %d" % sum(len(json.dumps(t)) for t in tools))
        # reasoning resend, counted only over agent requests
        seen = tot = 0
        for r in agent:
            for m in r["body"]["messages"]:
                if m.get("role") == "assistant":
                    tot += 1
                    if m.get("reasoning_content") or m.get("reasoning"):
                        seen += 1
        print("    assistant msgs resent  : %d, of which carry reasoning: %d" % (tot, seen))
        keys = collections.Counter()
        for r in agent:
            for k in r["body"]:
                keys[k] += 1
        print("    body keys across agent requests:", dict(keys))
        # save the first agent request for the record
        out = os.path.join(CAP, "agent_request_%s.json" % tag)
        json.dump(agent[0]["body"], open(out, "w"), indent=1)
        print("    saved first agent request ->", out)

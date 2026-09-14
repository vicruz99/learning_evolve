# -*- coding: utf-8 -*-
"""Compare what two captured harness paths actually sent to the model."""
import glob, json, os, sys, hashlib

CAP = "/tmp/bnbtest/capture"


def load(tag):
    p = os.path.join(CAP, "path_%s.jsonl" % tag)
    rows = []
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def summarise(tag):
    rows = load(tag)
    reqs = [r for r in rows if r.get("kind") != "response"
            and isinstance(r.get("body"), dict) and "messages" in r["body"]]
    resps = [r for r in rows if r.get("kind") == "response"]
    print("\n" + "=" * 74)
    print("PATH: %s   (%d proxy records, %d chat requests, %d responses)"
          % (tag, len(rows), len(reqs), len(resps)))
    print("=" * 74)
    if not reqs:
        print("  no chat/completions requests captured")
        return None

    b = reqs[0]["body"]
    sysmsgs = [m for m in b["messages"] if m.get("role") == "system"]
    systext = "\n".join(str(m.get("content", "")) for m in sysmsgs)
    tools = b.get("tools") or []
    print("  --- first request ---")
    print("    model          :", b.get("model"))
    print("    top-level keys :", ", ".join(sorted(b)))
    print("    sampling       : temperature=%s top_p=%s max_tokens=%s stream=%s"
          % (b.get("temperature"), b.get("top_p"), b.get("max_tokens"), b.get("stream")))
    print("    messages       : %d  (roles: %s)"
          % (len(b["messages"]), ",".join(m.get("role", "?") for m in b["messages"])[:120]))
    print("    system prompt  : %d chars across %d message(s)   md5=%s"
          % (len(systext), len(sysmsgs), hashlib.md5(systext.encode()).hexdigest()[:12]))
    print("    tools          : %d  -> %s"
          % (len(tools), ", ".join(sorted(t.get("function", {}).get("name", "?") for t in tools))))
    tl = sum(len(json.dumps(t)) for t in tools)
    print("    tool schema    : %d chars total" % tl)
    print("    tool_choice    :", b.get("tool_choice"))

    # does the harness send the model's own prior reasoning back?
    reasoning_seen = 0
    assistant_msgs = 0
    for r in reqs:
        for m in r["body"]["messages"]:
            if m.get("role") == "assistant":
                assistant_msgs += 1
                if m.get("reasoning_content") or m.get("reasoning") or (
                        isinstance(m.get("content"), str) and "<think>" in m["content"]):
                    reasoning_seen += 1
    print("  --- across all %d requests ---" % len(reqs))
    print("    assistant messages resent : %d" % assistant_msgs)
    print("    ...carrying reasoning     : %d   <-- does the model see its own thinking?"
          % reasoning_seen)
    growth = [len(json.dumps(r["body"])) for r in reqs]
    print("    request size chars        : first %d, last %d" % (growth[0], growth[-1]))
    fins = {}
    for r in resps:
        for h in r.get("finish_hints", []):
            fins[h] = fins.get(h, 0) + 1
    print("    response finish hints     :", fins or "(none seen)")
    return {"sys": systext, "tools": tools, "body": b}


a = summarise("headless")
b = summarise("tui")

if a and b:
    print("\n" + "=" * 74)
    print("DIFF")
    print("=" * 74)
    same_sys = a["sys"] == b["sys"]
    print("  system prompt identical :", same_sys)
    if not same_sys:
        print("    headless %d chars, tui %d chars" % (len(a["sys"]), len(b["sys"])))
        # first divergence
        for i, (x, y) in enumerate(zip(a["sys"], b["sys"])):
            if x != y:
                print("    first difference at char %d:" % i)
                print("      headless: ...%s..." % a["sys"][max(0, i-90):i+110].replace("\n", " "))
                print("      tui     : ...%s..." % b["sys"][max(0, i-90):i+110].replace("\n", " "))
                break
        else:
            longer, shorter = (a, b) if len(a["sys"]) > len(b["sys"]) else (b, a)
            extra = longer["sys"][len(shorter["sys"]):]
            who = "headless" if longer is a else "tui"
            print("    identical prefix; %s has %d extra chars:" % (who, len(extra)))
            print("      %s" % extra[:900].replace("\n", " "))
    na = sorted(t.get("function", {}).get("name") for t in a["tools"])
    nb = sorted(t.get("function", {}).get("name") for t in b["tools"])
    print("  tool sets identical     :", na == nb)
    if na != nb:
        print("    only headless:", sorted(set(na) - set(nb)))
        print("    only tui     :", sorted(set(nb) - set(na)))
    for k in ("temperature", "top_p", "max_tokens", "tool_choice", "stream"):
        if a["body"].get(k) != b["body"].get(k):
            print("  %-22s: headless=%r  tui=%r" % (k, a["body"].get(k), b["body"].get(k)))

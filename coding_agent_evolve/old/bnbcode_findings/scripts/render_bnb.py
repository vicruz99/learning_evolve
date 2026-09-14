# -*- coding: utf-8 -*-
"""Render an exported bnbcode session (.jsonl, one row per part) as readable Markdown.

    python3 render_bnb.py <findings_dir>

Reads transcripts/*.jsonl, writes transcripts/*.md beside them. Long tool inputs and
outputs are truncated in the Markdown; the .jsonl beside it is the faithful record.
"""
import glob, io, json, os, sys

CAP = 3000          # chars per tool input/output in the rendered view


def clip(s, n=CAP):
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n… [%d more chars, see the .jsonl]" % (len(s) - n)


def render(path):
    msgs = []
    order = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        mid = r.get("message_id")
        if mid not in msgs_index:
            msgs_index[mid] = {
                "role": r.get("role"), "finish": r.get("finish"), "mode": r.get("mode"),
                "tokens": r.get("tokens") or {}, "t": r.get("time_created"), "parts": [],
            }
            order.append(mid)
        p = r.get("part")
        if isinstance(p, dict):
            msgs_index[mid]["parts"].append(p)
    return order


out_lines = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "transcripts", "*.jsonl"))):
    if os.path.basename(path).startswith("_"):
        continue
    msgs_index = {}
    order = render(path)
    name = os.path.basename(path)[:-6]
    o = [ "# %s\n" % name,
          "Rendered from `%s.jsonl`, which is the faithful record.\n" % name,
          "Legend: **REASONING** = the model's thinking; **TOOL** = a tool call and its result;",
          "**INJECTED** = text bnbcode's continual-work feature inserted as a user turn.\n" ]
    stalls = 0
    for mid in order:
        m = msgs_index[mid]
        tk = m["tokens"] or {}
        kinds = [p.get("type") for p in m["parts"]]
        has_tool = "tool" in kinds
        if m["role"] == "assistant" and not has_tool:
            stalls += 1
        hdr = "## %s%s" % (m["role"], "" if not m["finish"] else "  ·  finish=`%s`" % m["finish"])
        if tk:
            hdr += "  ·  in %s / out %s" % (tk.get("input", "?"), tk.get("output", "?"))
        if m["role"] == "assistant" and not has_tool:
            hdr += "   ← **NO TOOL CALL (stall)**"
        o.append("\n---\n\n" + hdr + "\n")
        for p in m["parts"]:
            t = p.get("type")
            if t == "reasoning":
                o.append("**REASONING**\n\n> " + clip(p.get("text", "")).replace("\n", "\n> ") + "\n")
            elif t == "text":
                txt = p.get("text", "")
                tag = "INJECTED (continual-work)" if "continual-work-north-star" in txt else (
                    "TEXT" if m["role"] == "assistant" else "USER")
                o.append("**%s**\n\n```\n%s\n```\n" % (tag, clip(txt, 1200)))
            elif t == "tool":
                st = p.get("state") or {}
                o.append("**TOOL** `%s`\n\n```\n%s\n```\n" % (
                    p.get("tool", "?"), clip(json.dumps(st.get("input", {}), indent=1), 1500)))
                if st.get("output"):
                    o.append("_result_\n\n```\n%s\n```\n" % clip(st["output"]))
    md = path[:-6] + ".md"
    io.open(md, "w", encoding="utf-8").write("\n".join(o))
    out_lines.append("  %-58s %4d messages, %3d stalls" % (name, len(order), stalls))

print("rendered:")
print("\n".join(out_lines))

#!/usr/bin/env python3
"""Render a headless Claude Code transcript (--output-format stream-json) as readable text:
what the interactive UI would have shown -- assistant text, tool calls, tool results, final result.

    python3 render_transcript.py RUN_DIR/agent.jsonl  > RUN_DIR/transcript.md
    python3 render_transcript.py agent.jsonl --follow   # rewrite transcript.md every 60 s

Stdlib only. Tool results are truncated (--max-result chars, default 1500) except grader output,
which is kept in full up to 6000 chars because the scores live there.
"""
import argparse
import json
import os
import sys
import time


def _text(c):
    t = c.get("content")
    if isinstance(t, list):
        t = "\n".join(x.get("text", "") for x in t if isinstance(x, dict))
    return t or ""


def render(path, max_result=1500):
    out, n_tools, n_err = [], 0, 0
    out.append(f"# Agent transcript — {os.path.dirname(os.path.abspath(path))}\n")
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type")
        ts = d.get("timestamp", "")[11:19]
        if t == "system" and d.get("subtype") == "init":
            out.append(f"_session {d.get('session_id','')} · model {d.get('model','')} · cwd {d.get('cwd','')}_\n")
        elif t == "assistant":
            for c in d["message"]["content"]:
                if c.get("type") == "text" and c["text"].strip():
                    out.append(f"\n**Claude** `{ts}`\n\n{c['text'].strip()}\n")
                elif c.get("type") == "tool_use":
                    n_tools += 1
                    inp = c.get("input", {})
                    if c["name"] == "Bash":
                        body = f"```bash\n{inp.get('command','')}\n```"
                        if inp.get("description"):
                            body = f"_{inp['description']}_\n{body}"
                    elif c["name"] in ("Write", "Edit", "MultiEdit"):
                        body = f"`{inp.get('file_path','')}`"
                        if c["name"] == "Write":
                            content = inp.get("content", "")
                            body += f" ({len(content.splitlines())} lines)\n```python\n{content[:max_result]}" + ("\n…" if len(content) > max_result else "") + "\n```"
                        else:
                            body += f"\n```diff\n- {inp.get('old_string','')[:400]}\n+ {inp.get('new_string','')[:400]}\n```"
                    elif c["name"] == "Read":
                        body = f"`{inp.get('file_path','')}`"
                    else:
                        body = "```json\n" + json.dumps(inp)[:600] + "\n```"
                    out.append(f"\n> ⚙️ **{c['name']}** `{ts}`\n>\n" + "\n".join("> " + l for l in body.splitlines()) + "\n")
        elif t == "user":
            for c in d["message"]["content"]:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    txt = _text(c)
                    err = c.get("is_error", False)
                    n_err += bool(err)
                    lim = 6000 if "SCORE (geom" in txt or "correctness:" in txt else max_result
                    if len(txt) > lim:
                        txt = txt[:lim] + f"\n… [{len(txt) - lim} more chars]"
                    tag = "❌ result (error)" if err else "result"
                    out.append(f"\n<details><summary>{tag}</summary>\n\n```\n{txt}\n```\n</details>\n")
        elif t == "result":
            out.append(f"\n---\n**RUN ENDED** `{ts}` — {d.get('subtype')} · turns {d.get('num_turns')} · "
                       f"{(d.get('duration_ms') or 0)/60000:.0f} min\n\n{(d.get('result') or '')}\n")
    out.insert(1, f"_tool calls: {n_tools} · errored results: {n_err} · rendered {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--out", help="default: transcript.md next to the jsonl")
    ap.add_argument("--follow", action="store_true", help="re-render every --every seconds until the job ends")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--max-result", type=int, default=1500)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.jsonl)), "transcript.md")
    while True:
        if os.path.exists(a.jsonl):
            tmp = out + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(render(a.jsonl, a.max_result))
            os.replace(tmp, out)
        if not a.follow:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()

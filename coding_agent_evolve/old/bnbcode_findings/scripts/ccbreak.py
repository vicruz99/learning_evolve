# -*- coding: utf-8 -*-
"""What is in the Claude Code arm's context, by category, in Qwen tokens."""
import glob, json, os, collections
from tokenizers import Tokenizer

TOK = Tokenizer.from_file(glob.glob(os.path.expanduser(
    "~/models/hf/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/tokenizer.json"))[0]
    if glob.glob(os.path.expanduser("~/models/hf/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/tokenizer.json"))
    else glob.glob(os.path.join(os.environ["HF_HOME"], "hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/tokenizer.json"))[0])

def ntok(s):
    return len(TOK.encode(s, add_special_tokens=False).ids) if s else 0

root = os.path.expanduser("~/agent_runs/ac2_evo_cc_s1/.cc/projects")
# the live session is the largest file
f = max(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True), key=os.path.getsize)
print("session file:", os.path.basename(f))

cat = collections.Counter()
cnt = collections.Counter()
models = collections.Counter()
usage_in = usage_out = 0
compact_marks = 0
last = None

def blocks(c):
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return [b for b in (c or []) if isinstance(b, dict)]

for line in open(f, encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue
    t = rec.get("type")
    if rec.get("isCompactSummary"):
        compact_marks += 1
    m = rec.get("message") or {}
    if t == "assistant":
        if m.get("model"):
            models[m["model"]] += 1
        u = m.get("usage") or {}
        if m.get("id") != last:                      # usage repeats per block
            usage_in += (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                         + u.get("cache_creation_input_tokens", 0))
            usage_out += u.get("output_tokens", 0)
            last = m.get("id")
        for b in blocks(m.get("content")):
            k = b.get("type")
            if k == "thinking":
                cat["assistant reasoning"] += ntok(b.get("thinking", "")); cnt["assistant reasoning"] += 1
            elif k == "text":
                cat["assistant text"] += ntok(b.get("text", "")); cnt["assistant text"] += 1
            elif k == "tool_use":
                cat["tool CALLS (the request)"] += ntok(json.dumps(b.get("input", {}))); cnt["tool CALLS (the request)"] += 1
    elif t == "user":
        for b in blocks(m.get("content")):
            k = b.get("type")
            if k == "tool_result":
                c = b.get("content")
                s = c if isinstance(c, str) else json.dumps(c)
                cat["tool RESULTS (the output)"] += ntok(s); cnt["tool RESULTS (the output)"] += 1
            elif k == "text":
                cat["user / injected text"] += ntok(b.get("text", "")); cnt["user / injected text"] += 1
    elif t == "attachment":
        cat["attachments"] += ntok(json.dumps(rec.get("attachment", rec))[:200000]); cnt["attachments"] += 1

print("model field on assistant responses:", dict(models))
print("cumulative usage across the run: input %s  output %s  (sum %s)"
      % (f"{usage_in:,}", f"{usage_out:,}", f"{usage_in+usage_out:,}"))
print("compact-summary markers:", compact_marks)

tot = sum(cat.values())
print("\n%-28s %>9s" % ("category", "tokens") if False else
      "\n%-28s %10s %8s %6s" % ("category", "tokens", "blocks", "share"))
for k, v in cat.most_common():
    print("  %-26s %10s %8d %5.0f%%" % (k, f"{v:,}", cnt[k], 100.0 * v / tot))
print("  %-26s %10s" % ("TOTAL", f"{tot:,}"))

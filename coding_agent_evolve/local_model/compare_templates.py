#!/usr/bin/env python3
"""Render the same conversations through two chat templates and diff what matters.

    python compare_templates.py /scratch/vicstorage/qwen/chat_template.jinja \
                               qwen3.6_chat_template-v19.jinja

Checks the three things that actually break an agent loop:

1. Tool-call arguments arriving as a JSON string. The OpenAI spec sends `arguments`
   as a string; a template that does `arguments|items` raises TypeError on it. vLLM
   normalises to a dict first, so this is latent on vLLM and fatal on llama.cpp or
   LM Studio.
2. Prefix stability across turns. A template that retroactively strips <think> from
   older assistant messages changes the rendered prefix every turn, so the KV cache
   misses from that point on. Claude Code resends the whole conversation each turn,
   so this is paid on every request.
3. What enable_thinking=False actually emits.

Needs only jinja2.
"""
import json
import sys

import jinja2


def load(path):
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.filters["tojson"] = lambda v, **k: json.dumps(v)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    return env.from_string(open(path).read())


STRING_ARGS = [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    {"role": "user", "content": "Q2"},
]
TURN1 = [{"role": "user", "content": "Q1"},
         {"role": "assistant", "content": "A1", "reasoning_content": "REASONING ONE"}]
TURN2 = TURN1 + [{"role": "user", "content": "Q2"},
                 {"role": "assistant", "content": "A2", "reasoning_content": "REASONING TWO"}]


def report(path):
    t = load(path)
    print("=" * 12, path)

    try:
        t.render(messages=STRING_ARGS, tools=None, add_generation_prompt=True)
        print("  stringified tool args : RENDERS")
    except Exception as e:
        print(f"  stringified tool args : FAILS -> {type(e).__name__}: {e}")

    r1 = t.render(messages=TURN1, tools=None, add_generation_prompt=True)
    r2 = t.render(messages=TURN2, tools=None, add_generation_prompt=True)
    shared = next((i for i, (a, b) in enumerate(zip(r1, r2)) if a != b),
                  min(len(r1), len(r2)))
    print(f"  prefix stability      : {shared}/{len(r1)} chars of the turn-1 render survive")
    print(f"  old reasoning kept    : {'REASONING ONE' in r2}")

    try:
        out = t.render(messages=[{"role": "user", "content": "Q"}], tools=None,
                       add_generation_prompt=True, enable_thinking=False)
        print(f"  enable_thinking=False : ...{out[-40:]!r}")
    except Exception as e:
        print(f"  enable_thinking=False : FAILS -> {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)

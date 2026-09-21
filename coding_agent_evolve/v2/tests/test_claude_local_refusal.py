#!/usr/bin/env python3
"""Claude Code can refuse a prompt LOCALLY, before any HTTP request is made.

The refusal exists only in the stream-json: a synthetic assistant message
(model "<synthetic>", duration_api_ms 0, text "Prompt is too long") followed by a result
with is_error=true and terminal_reason="blocking_limit" -- while subtype is still "success".
Nothing reaches the relay, LiteLLM, cliff or vLLM, and stderr stays empty, so every proxy
log looks healthy while the cell is dead.

It happens once the conversation Claude Code keeps locally is over its limit. With cliff in
front, Claude Code's own auto-compaction is off and cliff only shrinks what goes upstream,
so the local copy keeps growing. From then on EVERY --continue of that conversation fails
the same way in well under a second.

On 2026-09-21 this killed two cells of m48_q38 (ac1_plain_cc, ac2_evo_cc): 60 identical
retries over 2 h, then outcome=infrastructure. The recovery existed but only fired on the
literal string "No conversation found" in stderr, which this failure never produces.

This test replays the exact recorded event sequence through the harness and asserts the
conversation is retired, so the next prompt drops --continue and starts a fresh session on
the same workspace, keeping every file and score.

    python3 tests/test_claude_local_refusal.py
"""
import asyncio
import os
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "driver"))
import common  # noqa: E402
import harness_claude  # noqa: E402

# what the dead cell actually recorded, trimmed to the fields the harness reads
REPLAY = '''\
    #!/usr/bin/env python3
    import json, sys
    sys.stdin.read()
    for e in [
      {"type":"system","subtype":"init","cwd":".","session_id":"s1","model":"qwen3.8-xhigh"},
      {"type":"assistant","message":{"model":"<synthetic>","role":"assistant",
        "stop_reason":"stop_sequence","content":[{"type":"text","text":"Prompt is too long"}]}},
      {"type":"result","subtype":"success","is_error":True,"duration_api_ms":0,"num_turns":1,
       "terminal_reason":"blocking_limit","session_id":"s1"},
    ]: print(json.dumps(e), flush=True)
    sys.exit(1)
'''


async def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    # the harness prepends $HOME/.local/bin and $HOME/bin to PATH, so an isolated HOME is the
    # only way to make the stub win over a real `claude` installed on the machine.
    os.environ["HOME"] = str(tmp)
    ws = tmp / "workspace"; ws.mkdir()
    cell_dir = tmp / "cell"; cell_dir.mkdir()
    bin_dir = tmp / "bin"; bin_dir.mkdir()
    stub = bin_dir / "claude"
    stub.write_text(textwrap.dedent(REPLAY))
    stub.chmod(0o755)

    settings = cell_dir / "settings.json"; settings.write_text("{}")
    cell = {"workspace": str(ws), "cell_dir": str(cell_dir), "cc_model": "qwen3.8-xhigh",
            "claude_settings": str(settings), "cliff_port": 6000, "litellm_port": 4000,
            "give_up": 1000,
            "claude": {"max_output_tokens": 65536, "max_context_tokens": 160000, "cliff": True}}
    events = common.EventLog(cell_dir / "events.jsonl")

    h = harness_claude.ClaudeHarness(cell, events, deadline_epoch=9e12)
    await h.start()
    h.has_conversation = True          # as if a saved transcript were being --continue'd
    te = await h.prompt("the host nudge that the dead cells kept re-sending")

    assert te.rc == 1, f"expected rc=1, got {te.rc}"
    assert te.tools == 0, f"expected no tool calls, got {te.tools}"
    assert te.is_error(), "a local refusal must count as an errored turn"
    assert te.stop_reason == "success", "the result subtype really is 'success' -- do not gate on it"
    assert te.extra.get("local_refusal") == "Prompt is too long", te.extra
    assert h.conversation_lost is True, "the conversation was not retired: the cell would loop forever"
    assert h.has_conversation is False, "the next prompt would still pass --continue"

    log = (cell_dir / "driver.log").read_text()
    assert "refused the prompt locally" in log, log
    print("PASS: local refusal detected, conversation retired, next prompt starts a fresh session")


if __name__ == "__main__":
    asyncio.run(main())

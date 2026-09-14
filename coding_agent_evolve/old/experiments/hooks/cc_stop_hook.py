#!/usr/bin/env python3
"""Claude Code Stop hook: hold a headless run working until a wall-clock deadline.

Set 2 measures performance, not continuity, so the agent is not allowed to stop early.
This blocks the stop in-session (no restart, no context loss) and feeds a nudge back as a
new instruction.

Two ways it lets go, both deliberate:
  * the deadline passes  -- the run has had its full budget
  * REFUSE_LIMIT consecutive blocks produce no new tool call -- the model is refusing, and
    a wedged run should not burn its remaining hours. Recorded, because "refused after N
    attempts at hour X" is a result.

State lives beside the run so concurrent runs never share it.

env: CC_RUN_DIR (required), CC_DEADLINE_EPOCH (required), CC_REFUSE_LIMIT (default 10)
"""
import json, os, sys, time

REFUSE_LIMIT = int(os.environ.get("CC_REFUSE_LIMIT", "10"))
NUDGE = ("Continue working toward the objective. Keep improving the score and recording "
         "every attempt under run/. Do not stop to report progress -- make the next tool call.")


def tool_calls_in(path):
    """How many tool calls the transcript holds so far."""
    n = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "\"type\":\"tool_use\"" in line or "\"type\": \"tool_use\"" in line:
                    n += 1
    except OSError:
        return None
    return n


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    # stop_hook_active means "this turn began because I blocked the previous stop". It is
    # set on EVERY continuation, so releasing on it unconditionally caps the hook at a
    # single block -- which is what happened: cells ended at 21 minutes of an 18-hour
    # budget with {"blocks": 1, "barren": 0}, the model still working. It is only a
    # runaway when the continuations stop producing work, and the barren counter below
    # already detects exactly that and releases at REFUSE_LIMIT. So the flag is recorded,
    # not obeyed.
    chained = bool(payload.get("stop_hook_active"))

    run_dir = os.environ.get("CC_RUN_DIR") or os.getcwd()
    state_path = os.path.join(run_dir, "run", "stophook.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)

    deadline = float(os.environ.get("CC_DEADLINE_EPOCH", "0"))
    now = time.time()

    state = {"blocks": 0, "barren": 0, "tools": 0, "released": None}
    try:
        with open(state_path) as f:
            state.update(json.load(f))
    except (OSError, ValueError):
        pass

    def release(reason):
        state["released"] = reason
        state["released_at"] = int(now)
        with open(state_path, "w") as f:
            json.dump(state, f)
        # exit 0 with no output = allow the stop
        sys.exit(0)

    if deadline and now >= deadline:
        release("deadline")

    # Did the last nudge actually produce work? Compare tool-call count with last firing.
    tpath = payload.get("transcript_path")
    seen = tool_calls_in(tpath) if tpath else None
    if seen is not None:
        if seen > state.get("tools", 0):
            state["barren"] = 0
        else:
            state["barren"] = state.get("barren", 0) + 1
        state["tools"] = seen
    else:
        state["barren"] = state.get("barren", 0) + 1

    if state["barren"] >= REFUSE_LIMIT:
        release("refused_after_%d" % state["barren"])

    state["blocks"] = state.get("blocks", 0) + 1
    state["chained"] = chained
    with open(state_path, "w") as f:
        json.dump(state, f)

    json.dump({"hookSpecificOutput": {"hookEventName": "Stop",
                                      "decision": "block",
                                      "reason": NUDGE}}, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Host loop for one cell: keep the agent working until the budget, never trusting its bookkeeping.

    run_cell.py <cell_dir>

Ported from arxiv_graph/baselines/adrs_acp/run.py::run_session. The agent's turn ending is the
stop signal; a 1 Hz watcher owns the hard budget; policy.py decides whether to nudge. Turns that
never reached the model (harness faults) are retried with back-off and never counted as stops.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tracker"))

from clock import Clock  # noqa: E402
from common import EventLog, INFRA_FAST_S, TurnEnd, harness_fault, load_cell, log  # noqa: E402
from policy import Rules, State, decide, host_line  # noqa: E402
from tracker import best_record, improvement_banked, read_iterations, set_t0, stop_reason_file, tracker_installed  # noqa: E402

RESUME_NOTE = ("\n\n---\nNOTE: work from an earlier session is already in `run/` -- read your ledger/notes and "
               "the files there first, and continue from the best result found so far rather than starting over. "
               "The host has kept every official score you made; nothing is lost.")
# sent instead of the whole initial prompt when the harness could restore the agent's conversation
RESUME_RESTORED = ("[host] Your session was interrupted by an infrastructure restart -- nothing you did -- and has "
                   "been restored with its history. Every file in the workspace and every official score is intact. "
                   "Any candidate that was still being evaluated at the moment of the interruption was NOT scored; "
                   "re-run those evaluations if they matter. Continue exactly where you left off.")
MAX_INFRA = 20
STALL_CHECK_S = 120
def last_activity(cell_dir: Path, ws: Path, harness, window_s: float) -> float:
    """Newest sign that the agent is doing anything: an official evaluation, a file written anywhere
    under the workspace, a live candidate process (run/procsample.jsonl), the harness's last tool
    call, and for Claude Code the continuous events.jsonl stream (bnbcode's stream goes quiet at its
    first compaction, so it is not used there). Text-only chatter deliberately does not count: the
    2026-09 guard storms produced thousands of text messages while doing nothing."""
    now = time.time(); cands = [0.0]
    paths = [cell_dir / "iterations.jsonl"]
    if getattr(harness, "name", "") == "claude":
        paths.append(cell_dir / "events.jsonl")
    for p in paths:
        try:
            cands.append(p.stat().st_mtime)
        except OSError:
            pass
    cands.append(float(getattr(harness, "last_tool_t", 0.0) or 0.0))
    ps = cell_dir / "run" / "procsample.jsonl"
    try:
        if json.loads(ps.read_text().splitlines()[-1]).get("n", 0) > 0:
            cands.append(ps.stat().st_mtime)
    except Exception:
        pass
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - window_s))
    try:
        out = subprocess.run(["find", str(ws), "-type", "f", "-newermt", since, "-print", "-quit"],
                             capture_output=True, text=True, timeout=300).stdout
        if out.strip():
            cands.append(now)
    except Exception:
        pass
    return max(cands)
async def watch_stall(cell_dir: Path, ws: Path, harness, stall_s: float, turn_t0: float) -> str:
    """Resolves when the agent has shown no activity for stall_s. Every 2026-09 wedge was one ACP
    prompt that never returned; the only other exit from the turn was the 18 h budget."""
    while True:
        await asyncio.sleep(STALL_CHECK_S)
        last = max(turn_t0, last_activity(cell_dir, ws, harness, stall_s))
        idle = time.time() - last
        if idle >= stall_s:
            return f"no official evaluation, file write, live worker or tool call for {idle / 3600:.1f} h"


async def watch_limits(cell_dir: Path, started: float, budget_s: float) -> str:
    while True:
        await asyncio.sleep(1)
        if time.time() - started >= budget_s:
            (cell_dir / "STOP").write_text("time_limit\n")
            return "time_limit"
        reason = stop_reason_file(cell_dir)
        if reason:
            return reason


def make_harness(cell: dict, events: EventLog, deadline: float):
    if cell["harness"] == "bnbcode":
        from harness_bnbcode import BnbcodeHarness
        return BnbcodeHarness(cell, events)
    from harness_claude import ClaudeHarness
    return ClaudeHarness(cell, events, deadline)


async def run(cell_dir: Path) -> dict:
    cell = load_cell(cell_dir)
    ws = Path(cell["workspace"])
    if not tracker_installed(cell_dir):
        raise SystemExit(f"tracker not installed in {ws}; run mkcell first")
    meta = cell["problem_meta"]
    maximize = bool(meta["maximize"]); start_score = float(meta["start_score"])
    budget_s = float(cell["hours"]) * 3600
    rules = Rules(min_evals=int(cell["min_evals"]), min_hours=float(cell["min_hours"]),
                  keep_going=bool(cell.get("keep_going", True)), nudge_min_gap_s=float(cell["nudge_min_gap_s"]))
    patience = int(cell.get("patience", 3))   # 0 = never open a fresh session
    stall_s = float(cell.get("stall_hours", 2.0)) * 3600   # 0 = never restart on inactivity
    give_up = int(cell.get("give_up") or 0)
    initial = (ws / "INITIAL_PROMPT.md").read_text()
    continuation = (Path(cell["v2"]) / "prompts" / "continuation.md").read_text().strip()
    events = EventLog(cell_dir / "events.jsonl")
    clock = Clock(cell_dir, budget_s)
    started = clock.started
    if not clock.relaunch:
        set_t0(cell_dir, started)          # iterations.jsonl elapsed_s stays relative to the FIRST start
    else:
        log(cell_dir, f"relaunch #{clock.launches}: {clock.active_before / 3600:.2f}h of the {cell['hours']}h budget "
                      f"already used, {clock.remaining_at_start / 3600:.2f}h remaining")
    deadline = clock.deadline
    if clock.remaining_at_start <= 60:
        (cell_dir / "STOP").write_text("time_limit\n")
        log(cell_dir, "budget already exhausted at relaunch -- nothing to do")
        events.close()
        return finish(cell_dir, cell, clock, [], 0, 0, "time_limit", "budget exhausted before relaunch")
    harness = make_harness(cell, events, deadline)
    turns: list[dict] = []
    continues = 0; last_nudge_at = started; barren = 0; infra = 0; fresh_sessions = 0
    outcome = "budget"; outcome_detail = None
    log(cell_dir, f"start {cell['name']} harness={cell['harness']} hours={cell['hours']} min_evals={rules.min_evals} "
                  f"min_hours={rules.min_hours} keep_going={rules.keep_going} reasoning={cell['reasoning']} "
                  f"stall_hours={stall_s / 3600:g}")

    async def start_harness() -> bool:
        nonlocal infra
        for attempt in range(MAX_INFRA):
            try:
                await harness.start()
                return True
            except Exception as exc:
                infra += 1
                log(cell_dir, f"harness start failed ({attempt + 1}): {str(exc)[:300]} -- infrastructure, backing off")
                events.write("infra", stage="start", error=str(exc)[:1000])
                await asyncio.sleep(30 if attempt < 5 else 120)
        return False

    if not await start_harness():
        outcome = "infrastructure"; outcome_detail = "harness never started"
        events.close()
        return finish(cell_dir, cell, clock, turns, continues, fresh_sessions, outcome, outcome_detail)

    watcher = asyncio.create_task(watch_limits(cell_dir, started, clock.remaining_at_start))
    beat = asyncio.create_task(clock.heartbeat())

    def resume_text() -> str:
        """What a relaunched/restarted agent is told. If the harness restored the conversation the
        initial prompt is already in it: send only the host line + a short note. Otherwise the agent
        starts cold and needs the whole prompt plus the pointer to its files."""
        evals_now = len(read_iterations(cell_dir)); b = best_record(cell_dir)
        line = host_line(deadline - time.time(), float(cell["hours"]), evals_now, rules.min_evals,
                         int(cell.get("max_evals") or 0), b.get("score") if b else None, meta["metric_word"])
        if getattr(harness, "restored", False):
            log(cell_dir, "conversation restored by the harness -- sending the short resume note")
            return line + "\n\n" + RESUME_RESTORED
        log(cell_dir, "conversation NOT restored -- sending the full initial prompt with the resume note")
        return line + "\n\n" + initial + RESUME_NOTE

    next_text = initial
    # a resubmitted cell (driver died, job relaunched on the same folder): tell the agent so
    if read_iterations(cell_dir) or any((ws / "run").glob("*")):
        log(cell_dir, "existing work found in run/ -- resuming")
        next_text = resume_text()
    fresh = False
    try:
        while True:
            t_turn = time.time()
            prompt_task = asyncio.create_task(harness.prompt(next_text, fresh=fresh) if harness.name == "claude"
                                              else harness.prompt(next_text))
            fresh = False
            staller = asyncio.create_task(watch_stall(cell_dir, ws, harness, stall_s, t_turn)) if stall_s > 0 else None
            waitset = {prompt_task, watcher} | ({staller} if staller else set())
            done, _ = await asyncio.wait(waitset, return_when=asyncio.FIRST_COMPLETED)
            if staller and not staller.done():
                staller.cancel()
            if watcher in done:
                reason = watcher.result()
                log(cell_dir, f"limit reached: {reason} -- terminating agent")
                await harness.terminate()
                if not prompt_task.done():
                    prompt_task.cancel()
                turns.append({"t": t_turn, "stop_reason": reason, "ran_s": round(time.time() - t_turn, 1), "cut": True})
                outcome = reason
                break
            if staller and staller in done and not prompt_task.done():
                # --- inactivity: the prompt is hung (or the agent is idling inside it). Restart the
                #     harness with a fresh conversation; files and official scores are kept. -------
                why = staller.result()
                log(cell_dir, f"STALL: {why} -- restarting the harness with a fresh session (files kept)")
                events.write("stall_restart", why=why, ran_s=round(time.time() - t_turn, 1))
                prompt_task.cancel()
                try:
                    await prompt_task
                except (asyncio.CancelledError, Exception):
                    pass
                turns.append({"t": t_turn, "stop_reason": "stall_restart", "ran_s": round(time.time() - t_turn, 1),
                              "evals": len(read_iterations(cell_dir)), "why": why})
                await harness.restart(); fresh_sessions += 1; barren = 0
                now = time.time(); evals = len(read_iterations(cell_dir)); best = best_record(cell_dir)
                next_text = host_line(deadline - now, float(cell["hours"]), evals, rules.min_evals,
                                      int(cell.get("max_evals") or 0), best.get("score") if best else None,
                                      meta["metric_word"]) + "\n\n" + initial + RESUME_NOTE
                fresh = True
                continue
            try:
                te: TurnEnd = prompt_task.result()
            except Exception as exc:            # a harness bug must not end an 18 h cell
                te = TurnEnd(stop_reason="error", ran_s=time.time() - t_turn, error=f"harness exception: {exc!r}"[:1000])
                events.write("harness_exception", error=repr(exc)[:2000])
                log(cell_dir, f"harness exception treated as infrastructure: {exc!r}"[:300])
            evals = len(read_iterations(cell_dir))
            best = best_record(cell_dir)
            rec = {"t": t_turn, "stop_reason": te.stop_reason, "ran_s": round(te.ran_s, 1), "tools": te.tools,
                   "text_chars": te.text_chars, "error": te.error, "rc": te.rc, "evals": evals,
                   "best": best.get("score") if best else None}
            # --- context overflow: the session itself is wedged (every retry re-sends the same
            #     oversized context); reset it, keep the files ---------------------------------
            blob = (te.error or "") + " " + te.stderr_tail
            if te.is_error() and any(k in blob for k in ("ContextOverflow", "input_tokens", "maximum context length", "reduce the length of the input")):
                rec["overflow"] = True; turns.append(rec)
                log(cell_dir, "context overflow -- starting a fresh session (files kept)")
                events.write("context_overflow", error=(te.error or "")[:500])
                await harness.restart(); fresh_sessions += 1; barren = 0
                now = time.time()
                next_text = host_line(deadline - now, float(cell["hours"]), evals, rules.min_evals,
                                      int(cell.get("max_evals") or 0), rec["best"], meta["metric_word"]) + "\n\n" + initial + RESUME_NOTE
                fresh = True
                continue
            # --- infrastructure, not a model stop -------------------------------------------
            fault = harness_fault(blob)
            if te.is_error() and (te.ran_s < INFRA_FAST_S or fault or not harness.alive()):
                infra += 1
                rec["infra"] = fault or f"errored in {te.ran_s:.0f}s"
                turns.append(rec)
                log(cell_dir, f"turn failed before reaching the model ({rec['infra']}) [{infra}/{MAX_INFRA}]")
                if infra >= MAX_INFRA:
                    outcome = "infrastructure"; outcome_detail = rec["infra"]
                    break
                await asyncio.sleep(30 if infra < 5 else 120)
                if not harness.alive():
                    if not await start_harness():
                        outcome = "infrastructure"; outcome_detail = "harness restart failed"; break
                    fresh_sessions += 1
                    next_text = resume_text()
                    fresh = not getattr(harness, "restored", False)
                elif getattr(harness, "conversation_lost", False):
                    # Claude Code could not --continue (transcript unusable): start over with the files
                    harness.conversation_lost = False
                    next_text = initial + RESUME_NOTE
                    fresh = True
                else:
                    next_text = continuation
                continue
            infra = 0
            # --- barren / refusal accounting --------------------------------------------------
            barren = barren + 1 if te.tools == 0 else 0
            rec["barren"] = barren
            turns.append(rec)
            if give_up and barren >= give_up:
                outcome = "refused"; outcome_detail = f"{barren} consecutive turns without a tool call"
                log(cell_dir, f"GIVING UP: {outcome_detail}")
                break
            # --- policy ---------------------------------------------------------------------
            now = time.time()
            banked = improvement_banked(cell_dir, start_score, maximize)
            st = State(elapsed_s=clock.elapsed(), hard_budget_s=budget_s, evals=evals, banked=banked,
                       continues=continues, since_last_nudge_s=now - last_nudge_at, stop_reason=stop_reason_file(cell_dir))
            go, why = decide(st, rules)
            log(cell_dir, f"turn end stop={te.stop_reason} ran={te.ran_s:.0f}s tools={te.tools} evals={evals} "
                          f"best={rec['best']} barren={barren} -> {'CONTINUE' if go else 'END'} ({why})")
            events.write("decision", go=go, why=why, evals=evals, banked=banked, continues=continues)
            if not go:
                outcome = "ended" if not st.stop_reason else st.stop_reason; outcome_detail = why
                break
            if why.startswith("WAIT"):
                wait_s = min(float(why.split()[1].rstrip("s")), max(0.0, deadline - now - 5))
                if wait_s > 0:
                    events.write("throttle", wait_s=wait_s)
                    await asyncio.sleep(wait_s)
                now = time.time()
                if now >= deadline or stop_reason_file(cell_dir):
                    outcome = stop_reason_file(cell_dir) or "time_limit"; outcome_detail = "budget reached during throttle"
                    break
                evals = len(read_iterations(cell_dir)); best = best_record(cell_dir); rec["best"] = best.get("score") if best else None
            continues += 1; last_nudge_at = now
            line = host_line(deadline - now, float(cell["hours"]), evals, rules.min_evals,
                             int(cell.get("max_evals") or 0), rec["best"], meta["metric_word"])
            if patience > 0 and barren >= patience:
                # the session is wedged (reasoning-only turns); reset context, keep the disk
                log(cell_dir, f"{barren} barren turns -- starting a fresh session")
                await harness.restart()
                fresh_sessions += 1; barren = 0
                next_text = line + "\n\n" + initial + RESUME_NOTE
                fresh = True
            else:
                next_text = line + "\n\n" + continuation
    finally:
        for t in (watcher, beat):
            if not t.done():
                t.cancel()
        clock.save()
        await harness.terminate()
        events.close()
    return finish(cell_dir, cell, clock, turns, continues, fresh_sessions, outcome, outcome_detail)


def finish(cell_dir: Path, cell: dict, clock: Clock, turns: list, continues: int, fresh_sessions: int,
           outcome: str, detail) -> dict:
    clock.save()
    rows = read_iterations(cell_dir)
    best = best_record(cell_dir)
    record = {
        "cell": cell["name"], "harness": cell["harness"], "prompt": cell["prompt"], "problem": cell["problem"],
        "reasoning": cell["reasoning"], "seed": cell["seed"], "started": clock.started,
        "first_started": clock.first_started, "launches": clock.launches,
        "elapsed_s": round(clock.elapsed(), 1), "hours_budget": cell["hours"],
        "outcome": outcome, "outcome_detail": detail, "continues": continues, "fresh_sessions": fresh_sessions,
        "n_turns": len(turns), "n_evals": len(rows), "n_valid_evals": sum(1 for r in rows if r.get("score") is not None),
        "best": best, "cutoff": stop_reason_file(cell_dir), "turns": turns,
    }
    (cell_dir / "session.json").write_text(json.dumps(record, indent=2) + "\n")
    log(cell_dir, f"finished: outcome={outcome} evals={len(rows)} best={best.get('score') if best else None} "
                  f"turns={len(turns)} continues={continues}")
    return record


def main() -> int:
    cell_dir = Path(sys.argv[1]).resolve()
    rec = asyncio.run(run(cell_dir))
    return 0 if rec["outcome"] not in ("infrastructure",) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""ledger.py -- per-cell run-health ledger for finished v2 campaigns. Read-only, stdlib only.

    ledger.py [--campaigns q38ac_r2 q36ac2_r2] [--out DIR] [--event-rows] [--no-live]

Every dispatched cell in these campaigns ran its full active budget (clock.json), so "did the
driver finish" says nothing. The ledger answers "how long was the AGENT working": it takes
three clocks per cell -- the last official evaluation (iterations.jsonl), the last .npy the
agent wrote anywhere under workspace/ (problem seed files excluded), and for Claude Code
cells the last write to the continuous events.jsonl stream (bnbcode's stream goes silent at
the first cliff-compaction, so it is NOT used there), plus the last driver turn the agent
ended itself (stop=end_turn/success; a turn that died in acp_error is not activity) -- and
calls the cell healthy up to the latest of them. A cell whose budget ended more than SILENCE_H after that point is `wedged`
(all six 2026-09 cases are bnbcode ACP prompts that never returned). Launch history comes
from lsf.out ("Started at" / "Terminated at" / TERM_*), restarts from driver.log.

Cells that LSF still lists as RUN/PEND are classified `running`, not clean/resumed/wedged:
their budget has not ended, so the silence test is meaningless for them and a cell that is
merely mid-turn would otherwise read as wedged. Their healthy_until/healthy_h are still
computed, so a partial run can be plotted and truncated like any other. --no-live skips the
bjobs query (useful off-cluster, or to re-classify a campaign after it has fully landed).

Two facts come from transcripts, not from anything this script parses, and are hard-coded in
ANNOTATIONS below with their evidence. Writes runs_ledger.json and RUNS_LEDGER.md to --out.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from pathlib import Path

ROOT = Path.home() / "agent_runs/v2"
V2 = Path(__file__).resolve().parents[1]
SILENCE_H = 1.5          # budget ending this long after the last activity = wedged
LATE_DISPATCH_H = 2.0    # first start this long after the campaign's first cell = late

# (campaign, cell) -> annotation. Established by reading transcripts on 2026-09-09/13.
ANNOTATIONS = {
    ("q38ac_r2", "ac2_evo_cc_rxhigh_s1"): {
        "flags": ["contaminated"],
        "note": "At 04:14 on 09-09 it read the sibling ac2_many_cc_rxhigh_s1 host nudge (\"best official "
                "score so far 0.8785\") from the process table -- harness_claude.py passes the nudge as an "
                "argv -- and searched toward that number for hours (241 'sister' mentions). Its 0.93107 is "
                "not an independent result; exclude from the plain/evo/many comparison."},
    ("q38ac_r2", "ac1_plain_bnb_rxhigh_s1"): {
        "flags": ["agent_stall"],
        "note": "Harness fine, agent stalled: from ~03:41 to ~14:00 on 09-09 the transcript is only "
                "`ps aux | grep sota3` and `sleep 100; tail -1 <log>` polls of three background optimisers; "
                "it recovered and scored 99 candidates in hour 16. Not 18 h of search."},
    ("q36ac2_r2", "ac2_many_bnb_rxhigh_s1"): {
        "healthy_until": "2026-09-09 11:35",
        "note": "The .npy writes until 17:05 came from orphaned background optimisers; the agent's single "
                "ACP turn from 10:36 never returned and the session store entered a message/event runaway "
                "from ~11:48 (223k message rows, 0.9M event rows). Last official evaluation 11:35."},
    ("q38ac_r3", "ac2_evo_bnb_rxhigh_s2"): {
        "flags": ["repetition_loop"],
        "note": "Ran its full 18 h and scored 0.88254, but the tail is not search. Its last official "
                "evaluation was 2026-09-17 05:34 and tool activity stopped later that morning; by 12:12 the "
                "session store held 9 `text` parts in an hour and NOTHING else -- five byte-identical copies "
                "of 'Check node load (uptime). If load < 100, score cbinom_n3400000.npy officially ...'. The "
                "stated precondition was satisfied (bhosts showed r1m 2.0, ut 17% on a 256-CPU host), so this "
                "is an agent repetition loop, not waiting. The stall watchdog stayed silent because those "
                "`text` parts keep store_activity_t fresh. Count the productive window, not the 18 h."},
    ("q38ac_r3", "ac1_evo_bnb_rxhigh_s2"): {
        "flags": ["mem_limit_killed"],
        "note": "Killed TERM_MEMLIMIT 2026-09-17 08:19 at exactly 262144 MB -- the submit default of "
                "8192 MB/slot x 32 slots -- after 10.8 h and 325 evals. A MEMLIMIT kill stops clock.json, so "
                "the budget survived; resubmitted with mem_limit_mb 12288 (384 GB) and resumed at 7.19 h "
                "remaining, restoring a 145 MB session DB with 110 sessions. Its two earlier launches ended "
                "in the outage above (launch `exit 3`, no budget lost)."},
    ("q38ac_r3", "ac2_many_bnb_rxhigh_s2"): {
        "flags": ["stall_recovered_x2"],
        "note": "The clearest evidence of what the r3 watchdog is worth. Wedged TWICE (store silent while "
                "bnbcode burned >100% CPU) and recovered both times: STALL 05:17 -> evals 130->131 with a "
                "real 1143 s / 55-tool turn; STALL 09:18 -> evals 275->287. It then went on to the campaign's "
                "best AC2 score. In r2 each of those two events would have ended the run. ~4 h of its budget "
                "went to the two 2 h detection windows."},
    ("q38ac_r2", "ac2_plain_cc_rxhigh_s1"): {
        "note": "Official evaluations stop at 7.6 h by the agent's choice: it moved to huge-n 'box root' "
                "candidates whose OFFICIAL grading takes 6.5 h each (eval_s=23228 for n=10M) and kept "
                "polling/wrapping up to the end. LSF walltime killed the launch script during "
                "reap_workspace after the driver had finished -> score.json/.done were never written."},
}

CAMPAIGN_NOTES = {
    "q38ac_r3": (
        "**Infrastructure, not agent behaviour, dominates this campaign's gaps.** Read `wedged` and any "
        "short `healthy_h` here against these three facts before concluding anything about the agents.\n\n"
        "1. **Server outage 2026-09-16 14:41 -> 22:26 (7.75 h).** The private Qwen3.8 server `v2q38` exited "
        "(SIGTERM, LSF logged it 'Done successfully') and every cell had been rendered with a SINGLE relay "
        "upstream, so all 10 running cells sat at `no upstream` for the whole window. `driver/clock.py` has "
        "no outage accounting, so the 18 h budget was SPENT, not paused: each affected cell's `active_h` "
        "overstates its real search by up to 7.75 h. Cells dispatching during the window died at launch "
        "(`exit 3`, relay never answered) -- that path stops the clock, so those cells lost a dispatch but "
        "no budget. Relay configs were then changed to a fallback list, which is why most cells show "
        "multiple launches and class `resumed`.\n"
        "2. **The bnbcode wedge is now auto-recovered.** Cells run with `stall_hours=2` and the watchdog "
        "reads the postgres session store, so the failure that killed 6 of 11 bnbcode cells in r2 costs "
        "~2 h and continues instead of ending the run. `STALLx<n>` in the health digest counts them.\n"
        "3. **The watchdog is blind to repetition loops.** `store_activity_t` counts ANY part, including "
        "bare `text`, so an agent re-emitting the same plan line keeps the clock fresh with no tool call "
        "and no eval. See the per-cell note on ac2_evo_bnb_rxhigh_s2."),
}

LSF_DATE = "%a %b %d %H:%M:%S %Y"
DRV = re.compile(r"^\[driver (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")


def live_cells() -> dict:
    """(campaign, cell) -> LSF state, for cells LSF still lists. Cells are submitted with job
    name "<campaign>.<cell>" (see bin/submit), which is the only reliable link between a run
    directory and a running job -- a cell mid-budget has no .done and a finished one may have
    none either (q38ac_r2/ac2_plain_cc_rxhigh_s1 lost its .done to an LSF walltime kill)."""
    try:
        out = subprocess.run(["bjobs", "-noheader", "-o", "job_name:60 stat:8"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    live = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or "." not in parts[0]:
            continue
        name, stat = parts
        if stat in ("RUN", "PEND", "PSUSP", "USUSP", "SSUSP"):
            camp, _, cell = name.partition(".")
            live[(camp, cell)] = stat
    return live


def ts(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S" if len(s) > 16 else "%Y-%m-%d %H:%M"))


def fmt(t) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def read_rows(p: Path) -> list:
    rows = []
    try:
        for l in p.read_text().splitlines():
            l = l.strip()
            if l:
                try:
                    rows.append(json.loads(l))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def lsf_launches(p: Path) -> list:
    """One record per LSF report block in lsf.out: start, end, term (None = completed)."""
    out, cur = [], {}
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return out
    for l in text.splitlines():
        l = l.strip()
        if l.startswith("Started at "):
            cur = {"start": time.mktime(time.strptime(l[11:].strip(), LSF_DATE)), "end": None, "term": None}
        elif l.startswith("Terminated at ") and cur:
            cur["end"] = time.mktime(time.strptime(l[14:].strip(), LSF_DATE))
        elif l.startswith("TERM_") and cur:
            cur["term"] = l.split(":")[0]
        elif l.startswith("Successfully completed") and cur:
            cur["term"] = None
        elif l.startswith("Max Memory") and cur:
            try:
                cur["max_mem_mb"] = int(l.split(":")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif l.startswith("Resource usage summary") and cur:
            out.append(cur); cur = {}
    if cur:
        out.append(cur)
    return out


def driver_facts(p: Path) -> dict:
    f = {"turn_ends": [], "limit_reached": None, "fresh_sessions": 0, "restored": 0, "not_restored": 0,
         "relaunches": 0, "turn_failed": 0, "harness_exception": 0, "starts": [], "last_turn_end": None,
         "last_good_turn_end": None, "turn_ends_per_launch": []}
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return f
    per = 0
    for l in text.splitlines():
        m = DRV.match(l)
        if not m:
            continue
        t, msg = ts(m.group(1)), m.group(2)
        if msg.startswith("start "):
            f["starts"].append(t)
            if len(f["starts"]) > 1:
                f["turn_ends_per_launch"].append(per); per = 0
        elif msg.startswith("turn end "):
            f["turn_ends"].append(t); per += 1
            m2 = re.search(r"stop=(\S+)", msg)
            if m2 and m2.group(1) in ("end_turn", "success"):
                f["last_good_turn_end"] = t   # a turn the agent finished itself = activity; acp_error/error is not
        elif msg.startswith("limit reached"):
            f["limit_reached"] = t
        elif "context overflow -- starting a fresh session" in msg or "barren turns -- starting a fresh session" in msg:
            f["fresh_sessions"] += 1
        elif "conversation restored by the harness" in msg:
            f["restored"] += 1
        elif "conversation NOT restored" in msg:
            f["not_restored"] += 1
        elif msg.startswith("relaunch #"):
            f["relaunches"] += 1
        elif msg.startswith("turn failed"):
            f["turn_failed"] += 1
        elif "harness exception" in msg:
            f["harness_exception"] += 1
    f["turn_ends_per_launch"].append(per)
    f["last_turn_end"] = f["turn_ends"][-1] if f["turn_ends"] else None
    return f


def npy_walk(ws: Path, exclude: set) -> tuple:
    n, last = 0, None
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__")]
        for fn in files:
            if fn.endswith(".npy") and fn not in exclude:
                try:
                    mt = (Path(root) / fn).stat().st_mtime
                except OSError:
                    continue
                n += 1
                if last is None or mt > last:
                    last = mt
    return n, last


def event_rows(p: Path) -> dict:
    """Rows per session-store table in a pg_dump (COPY blocks). Streams; ~10 s per GB."""
    counts, table = {}, None
    try:
        with open(p, "rb") as fh:
            for line in fh:
                if table is None:
                    if line.startswith(b"COPY public."):
                        table = line.split()[1].decode()[len("public."):]
                        counts[table] = 0
                elif line == b"\\.\n":
                    table = None
                else:
                    counts[table] += 1
    except OSError:
        return {}
    return {k: v for k, v in counts.items() if k in ("event", "message", "part", "session")}


def problem_seed_files() -> set:
    s = set()
    for d in (V2 / "problems").glob("*"):
        for f in d.glob("*.npy"):
            s.add(f.name)
    s.add("problem_meta.npy")
    return s


def ledger_cell(camp: str, d: Path, camp_t0: float, want_rows: bool, prev_rows: dict | None = None,
                live: str | None = None) -> dict:
    cfg = jload(d / "cell.json") or {}
    rec = {"campaign": camp, "cell": d.name, "problem": cfg.get("problem"), "harness": cfg.get("harness"),
           "prompt": cfg.get("prompt"), "seed": cfg.get("seed"), "model": cfg.get("model"),
           "hours_budget": cfg.get("hours"), "dispatched": (d / "driver.log").is_file(), "flags": [], "note": ""}
    ann = ANNOTATIONS.get((camp, d.name), {})
    if not rec["dispatched"]:
        rec.update({"class": "never_ran", "launches": 0})
        return rec
    sess = jload(d / "session.json") or {}
    clock = jload(d / "clock.json") or {}
    track = jload(d / "workspace" / "_adrs_track.json") or {}
    maximize = bool(track.get("maximize", cfg.get("problem") != "AC1"))
    rows = read_rows(d / "iterations.jsonl")
    drv = driver_facts(d / "driver.log")
    launches = lsf_launches(d / "lsf.out")
    if live:
        # lsf_launches only closes a record at "Resource usage summary", which LSF writes when
        # the job ends -- so the in-flight launch has no block yet and is invisible here. Without
        # it the live launch contributes nothing to active_between() and healthy_h collapses
        # towards zero. clock.json's last_started is the driver's own record of that launch.
        ls = clock.get("last_started")
        if ls and not any(abs(L["start"] - ls) < 60 for L in launches):
            launches.append({"start": ls, "end": None, "term": None})
    t0 = clock.get("first_started") or sess.get("first_started") or (launches[0]["start"] if launches else None)
    t_end = drv["limit_reached"] or clock.get("saved") or (launches[-1]["end"] if launches and launches[-1]["end"] else None)
    if launches and launches[-1]["end"] is None:
        launches[-1]["end"] = t_end
    seeds = problem_seed_files()
    n_npy, last_npy = npy_walk(d / "workspace", seeds)
    last_eval = rows[-1]["t"] if rows else None
    ev = d / "events.jsonl"
    last_event = ev.stat().st_mtime if (ev.is_file() and cfg.get("harness") == "claude") else None
    clocks = {"last_eval": last_eval, "last_npy": last_npy, "last_event_claude": last_event,
              "last_turn_end": drv["last_turn_end"], "last_good_turn_end": drv["last_good_turn_end"]}
    cand = [x for x in (last_eval, last_npy, last_event, drv["last_good_turn_end"]) if x]
    healthy_until = min(t_end, max(cand)) if cand and t_end else t_end
    if "healthy_until" in ann:
        healthy_until = ts(ann["healthy_until"])
    silence_h = (t_end - healthy_until) / 3600 if t_end and healthy_until else 0.0

    def active_between(a: float, b: float) -> float:
        tot = 0.0
        for L in launches:
            s, e = L["start"], L["end"] or b
            tot += max(0.0, min(e, b) - max(s, a))
        return tot / 3600
    active_h = (clock.get("active_s") or 0) / 3600
    if t0 and healthy_until and launches:
        # Wall time in the launch windows is NOT active time: the windows also cover relay setup,
        # pg restore and the post-driver host grade, so summing them can exceed the 18 h budget
        # (three q38ac_r3 cells reported 20.0-20.6 h). clock.json keeps only an aggregate
        # active_s, so apportion it across the windows instead of reporting wall time.
        wall_healthy = active_between(t0, healthy_until)
        wall_total = active_between(t0, t_end) if t_end else 0.0
        if wall_total > 0 and active_h:
            healthy_h = active_h * min(1.0, wall_healthy / wall_total)
        else:
            healthy_h = min(wall_healthy, active_h) if active_h else wall_healthy
    else:
        healthy_h = active_h

    win = [r for r in rows if r.get("t", 0) <= (healthy_until or 0) + 60] if healthy_until else rows
    scored = [r["score"] for r in win if r.get("score") is not None]
    best_in_window = (max if maximize else min)(scored) if scored else None
    all_scored = [r["score"] for r in rows if r.get("score") is not None]
    best_overall = (max if maximize else min)(all_scored) if all_scored else None
    # launch-level facts
    n_launch = len(launches) or int(sess.get("launches") or 1)
    dead_relaunch = False
    for i, L in enumerate(launches[1:], start=2):
        ev_in = sum(1 for r in rows if L["start"] <= r.get("t", 0) <= (L["end"] or 1e18))
        te_in = sum(1 for t in drv["turn_ends"] if L["start"] <= t <= (L["end"] or 1e18))
        if ev_in == 0 and te_in == 0:
            dead_relaunch = True
    cell_json_mtime = (d / "cell.json").stat().st_mtime
    cfg_changed = any(launches[i - 1]["start"] < cell_json_mtime < launches[i]["start"] for i in range(1, len(launches)))

    flags = list(ann.get("flags", []))
    if not (d / "score.json").is_file(): flags.append("score_json_missing")
    if not (d / ".done").is_file(): flags.append("done_missing")
    if t0 and camp_t0 and (t0 - camp_t0) / 3600 > LATE_DISPATCH_H: flags.append("late_dispatch")
    if cfg_changed: flags.append("config_changed_between_launches")
    if dead_relaunch: flags.append("relaunch_dead")
    if drv["fresh_sessions"] >= 5: flags.append("many_fresh_sessions")

    if live:
        klass = "running"          # budget not spent yet -- the silence test does not apply
        flags.append(f"lsf_{live.lower()}")
    elif silence_h > SILENCE_H:
        klass = "wedged"
    elif n_launch > 1:
        klass = "resumed"
    else:
        klass = "clean"
    rec.update({
        "class": klass, "flags": flags, "note": ann.get("note", ""), "live": live,
        "outcome": sess.get("outcome"), "active_h": round(active_h, 2), "budget_h": (clock.get("budget_s") or 0) / 3600,
        "first_started": fmt(t0), "budget_end": fmt(t_end), "wall_span_h": round((t_end - t0) / 3600, 1) if t0 and t_end else None,
        "launches": n_launch,
        "launch_history": [{"start": fmt(L["start"]), "end": fmt(L["end"]), "term": L["term"] or "completed",
                            "max_mem_mb": L.get("max_mem_mb")} for L in launches],
        "restored": drv["restored"], "not_restored": drv["not_restored"],
        "clocks": {k: fmt(v) for k, v in clocks.items()},
        "healthy_until": fmt(healthy_until), "silence_h": round(silence_h, 2), "healthy_h": round(healthy_h, 2),
        "healthy_frac": round(healthy_h / active_h, 3) if active_h else None,
        "evals": len(rows), "evals_valid": len(all_scored), "distinct_scores": len({round(s, 9) for s in all_scored}),
        "evals_in_window": len(win), "distinct_scores_in_window": len({round(s, 9) for s in scored}),
        "best_in_window": best_in_window, "best_overall": best_overall, "maximize": maximize,
        "score_json": (jload(d / "score.json") or {}).get("score"),
        "npy_count": n_npy, "turn_ends": len(drv["turn_ends"]), "turn_ends_per_launch": drv["turn_ends_per_launch"],
        "fresh_sessions": drv["fresh_sessions"], "turn_failed": drv["turn_failed"],
        "context_overflows_stderr": (d / "stderr.txt").read_text(errors="replace").count("ContextOverflowError") if (d / "stderr.txt").is_file() else 0,
        "pgbackup_mb": round((d / "pgbackup.sql").stat().st_size / 1e6) if (d / "pgbackup.sql").is_file() else None,
    })
    if want_rows and (d / "pgbackup.sql").is_file():
        rec["store_rows"] = event_rows(d / "pgbackup.sql")
    elif prev_rows:
        rec["store_rows"] = prev_rows   # carried over from the previous ledger (pgbackup.sql is final)
    return rec


def markdown(recs: list) -> str:
    order = {"clean": 0, "resumed": 1, "running": 2, "wedged": 3, "never_ran": 4}
    lines = [f"# v2 run-health ledger -- generated {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Classes: clean = one launch, active to the end; resumed = killed externally and relaunched with restored "
             f"state, active to the end; wedged = budget ended > {SILENCE_H} h after the agent's last activity "
             f"(analyse only up to `healthy_until`); running = still in LSF, budget not yet spent, figures must mark it in-flight; never_ran = rendered, never dispatched. AC1 is minimised, AC2 maximised.", ""]
    for camp in sorted({r["campaign"] for r in recs}):
        lines += [f"## {camp}", ""]
        if camp in CAMPAIGN_NOTES:
            lines += [CAMPAIGN_NOTES[camp], ""]
        lines += [
                  "| class | cell | launches | active h | healthy h | healthy_until | evals (distinct) | best in window | flags |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted([r for r in recs if r["campaign"] == camp], key=lambda r: (order[r["class"]], r["cell"])):
            if r["class"] == "never_ran":
                lines.append(f"| never_ran | `{r['cell']}` | 0 | - | - | - | - | - | |")
                continue
            b = f"{r['best_in_window']:.5f}" if r["best_in_window"] is not None else "-"
            lines.append(f"| {r['class']} | `{r['cell']}` | {r['launches']} | {r['active_h']} | {r['healthy_h']} | "
                         f"{r['healthy_until'][5:]} | {r['evals_in_window']} ({r['distinct_scores_in_window']}) | {b} | "
                         f"{', '.join(r['flags'])} |")
        lines.append("")
    lines += ["## Launch history and notes", ""]
    for r in recs:
        if r["class"] == "never_ran":
            continue
        hist = "; ".join(f"{L['start'][5:]} -> {L['end'][5:]} ({L['term']}{', ' + str(L['max_mem_mb']) + ' MB' if L.get('max_mem_mb') else ''})"
                         for L in r["launch_history"])
        lines.append(f"- **{r['campaign']}/{r['cell']}** [{r['class']}]: {hist}. Clocks: last eval {r['clocks']['last_eval'][5:]}, "
                     f"last .npy {r['clocks']['last_npy'][5:]}, last turn end {r['clocks']['last_turn_end'][5:]}"
                     + (f", last stream write {r['clocks']['last_event_claude'][5:]}" if r['clocks']['last_event_claude'] != '-' else "")
                     + f". Fresh sessions {r['fresh_sessions']}, stderr overflows {r['context_overflows_stderr']}, "
                     f".npy written {r['npy_count']} vs {r['evals']} official rows."
                     + (f" Store rows: {r['store_rows']}." if r.get("store_rows") else "")
                     + (f" {r['note']}" if r["note"] else ""))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaigns", nargs="+", default=["q38ac_r2", "q36ac2_r2"])
    ap.add_argument("--out", default=str(ROOT / "_analysis"))
    ap.add_argument("--event-rows", action="store_true", help="also count session-store rows in pgbackup.sql (slow)")
    ap.add_argument("--no-live", action="store_true", help="skip the bjobs query that marks in-flight cells `running`")
    a = ap.parse_args()
    recs = []
    live = {} if a.no_live else live_cells()
    if live: print(f"LSF still has {len(live)} cell(s) in flight: "
                   + ", ".join(f"{c}/{n} [{st}]" for (c, n), st in sorted(live.items())))
    prev = {}
    old = jload(Path(a.out) / "runs_ledger.json")
    if old:
        prev = {(r["campaign"], r["cell"]): r.get("store_rows") for r in old.get("cells", [])}
    for camp in a.campaigns:
        cells = sorted(p for p in (ROOT / camp).iterdir() if (p / "cell.json").is_file())
        starts = []
        for d in cells:
            c = jload(d / "clock.json")
            if c and c.get("first_started"):
                starts.append(c["first_started"])
        camp_t0 = min(starts) if starts else None
        for d in cells:
            recs.append(ledger_cell(camp, d, camp_t0, a.event_rows, prev.get((camp, d.name)),
                                    live.get((camp, d.name))))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "runs_ledger.json").write_text(json.dumps({"generated": time.time(), "silence_h": SILENCE_H, "cells": recs}, indent=1) + "\n")
    (out / "RUNS_LEDGER.md").write_text(markdown(recs))
    from collections import Counter
    print(Counter(r["class"] for r in recs))
    for r in recs:
        if r["class"] != "never_ran":
            print(f"{r['class']:9} {r['campaign']}/{r['cell']:26} healthy={r['healthy_h']:5.1f}h/{r['active_h']:.1f} until {r['healthy_until']} "
                  f"evals={r['evals_in_window']:4d} best={r['best_in_window']} flags={r['flags']}")
    print("wrote", out / "runs_ledger.json", "and", out / "RUNS_LEDGER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

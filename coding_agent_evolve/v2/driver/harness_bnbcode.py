"""bnbcode over ACP (Agent Client Protocol), ported from adrs_acp/run.py + acp_smoke/client.py.

One long-lived `bnbcode acp` process per session; `prompt()` returns when the agent ends its turn.
`restart(fresh=True)` starts a new session in the same workspace (context reset, disk kept).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process, text_block
from acp.schema import (AllowedOutcome, ClientCapabilities, FileSystemCapabilities, Implementation,
                        ReadTextFileResponse, RequestPermissionResponse, WriteTextFileResponse)

from common import EventLog, TurnEnd, log

STALL_FOLLOWUP = ("Your previous turn was cut off by a provider stream stall. "
                  "Continue the task from where you stopped; do not restart from scratch.")


def _pick_permission(options) -> str:
    by_kind: dict[str, str] = {}
    for opt in options or []:
        kind = getattr(opt, "kind", None)
        oid = getattr(opt, "option_id", None) or getattr(opt, "optionId", None)
        if kind and oid:
            by_kind.setdefault(kind, oid)
    for kind in ("allow_always", "allow_once"):
        if kind in by_kind:
            return by_kind[kind]
    raise RequestError.internal_error("agent offered no allow option")


class Client:
    """ACP client: auto-allows permission requests (the config's deny rules never reach us),
    serves fs reads/writes, refuses terminals, and streams every update to the event log."""

    def __init__(self, events: EventLog):
        self.events = events
        self.tools = 0
        self.text_chars = 0
        self.tool_ids: set[str] = set()
        self.last_tool_t = 0.0      # wall time of the last NEW tool call seen (the driver's stall watchdog reads it)
        self.replaying = False      # True while session/load replays history: do not log or count it

    def reset_turn(self):
        self.tools = 0
        self.text_chars = 0
        self.tool_ids = set()

    async def request_permission(self, session_id, tool_call, options, **kw):
        oid = _pick_permission(options)
        self.events.write("permission", option=oid, title=getattr(tool_call, "title", None))
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=oid))

    async def session_update(self, session_id, update, **kw) -> None:
        if self.replaying:
            return
        kind = getattr(update, "session_update", None)
        content = getattr(update, "content", None)
        text = getattr(content, "text", None) if content is not None else None
        rec: dict[str, Any] = {"update": kind}
        if kind == "agent_message_chunk" and isinstance(text, str):
            self.text_chars += len(text)
            rec["text"] = text
        elif kind == "agent_thought_chunk" and isinstance(text, str):
            rec["thought"] = text
        tid = getattr(update, "tool_call_id", None)
        if tid:
            rec.update(tool_call_id=tid, title=getattr(update, "title", None),
                       status=getattr(update, "status", None), tool_kind=getattr(update, "kind", None))
            if tid not in self.tool_ids:
                self.tool_ids.add(tid)
                self.tools += 1
                self.last_tool_t = time.time()
        self.events.write("acp", **rec)

    async def write_text_file(self, session_id, path, content, **kw):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content, encoding="utf-8")
        return WriteTextFileResponse()

    async def read_text_file(self, session_id, path, line=None, limit=None, **kw):
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        if line is None and limit is None:
            return ReadTextFileResponse(content=raw)
        lines = raw.splitlines(keepends=True)
        start = max((line or 1) - 1, 0)
        end = start + limit if limit is not None else None
        return ReadTextFileResponse(content="".join(lines[start:end]))

    async def create_terminal(self, *a, **kw): raise RequestError.method_not_found("terminal/create")
    async def terminal_output(self, *a, **kw): raise RequestError.method_not_found("terminal/output")
    async def release_terminal(self, *a, **kw): raise RequestError.method_not_found("terminal/release")
    async def wait_for_terminal_exit(self, *a, **kw): raise RequestError.method_not_found("terminal/wait_for_exit")
    async def kill_terminal(self, *a, **kw): raise RequestError.method_not_found("terminal/kill")
    async def create_elicitation(self, *a, **kw): raise RequestError.method_not_found("elicitation/create")
    async def complete_elicitation(self, *a, **kw): return None
    async def ext_method(self, method, params): raise RequestError.method_not_found(method)
    async def ext_notification(self, method, params): return None
    def on_connect(self, conn): return None


class BnbcodeHarness:
    name = "bnbcode"
    @property
    def last_tool_t(self) -> float:
        return float(getattr(self.client, "last_tool_t", 0.0) or 0.0)

    def __init__(self, cell: dict, events: EventLog):
        self.cell = cell
        self.events = events
        self.ws = Path(cell["workspace"])
        self.cell_dir = Path(cell["cell_dir"])
        fam = cell.get("model_family", "qwen3.8")
        self.model_id = f"vllm{fam.replace('qwen', '').replace('.', '')}/{cell['model']}"
        self.stack: AsyncExitStack | None = None
        self.conn = None
        self.proc = None
        self.session_id: str | None = None
        self.client = Client(events)
        self.sessions = 0
        self.restored = False     # True when start() adopted an earlier session of this workspace
        self._orphans: set[str] = set()   # empty sessions created by session/new and then abandoned for an adopted one
        self._stderr = None

    def env(self) -> dict:
        b = self.cell["bnbcode"]
        env = dict(os.environ)
        env["PATH"] = f"{Path.home() / '.local/bin'}:{Path.home() / 'bin'}:" + env.get("PATH", "")
        if b.get("visible_output_guard"):
            env["BNBCODE_VISIBLE_OUTPUT_GUARD"] = "1"
            env["BNBCODE_VISIBLE_OUTPUT_GUARD_CAP"] = str(int(b.get("guard_cap", 8)))
        else:
            env.pop("BNBCODE_VISIBLE_OUTPUT_GUARD", None)
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(int(b["output_tokens"]))
        cfg_path = Path(self.cell["opencode_config"])
        env["OPENCODE_CONFIG_CONTENT"] = cfg_path.read_text()   # highest-precedence layer, out of the agent's reach
        env.setdefault("XDG_RUNTIME_DIR", f"/tmp/xdg-{os.getuid()}")
        env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def store_activity_t(self) -> float:
        """Wall time (s) of the newest message part in the node-local session store, across ALL
        sessions of this workspace -- including successor sessions whose ACP updates never reach us.
        The stall watchdog reads it; cached for 60 s so the 2-min check costs one psql per call."""
        import subprocess
        now = time.time()
        cache = getattr(self, "_store_t_cache", (0.0, 0.0))
        if now - cache[1] < 60:
            return cache[0]
        dirs = ", ".join(f"'{d}'" for d in {str(self.ws), str(self.ws.resolve())})
        q = (f"SELECT coalesce(max(p.time_created), 0) FROM part p JOIN session s ON s.id = p.session_id "
             f"WHERE s.directory IN ({dirs})")
        t = cache[0]
        try:
            out = subprocess.run([str(Path(self.cell["v2"]) / "bin" / "bnbcode-pg-node"), "psql", "-t", "-A", "-c", q],
                                 capture_output=True, text=True, timeout=30, env=self.env()).stdout.strip()
            ms = float(out.splitlines()[-1].strip()) if out else 0.0
            if ms > 0:
                t = ms / 1000.0
        except Exception:
            pass
        self._store_t_cache = (t, now)
        return t

    def latest_session_id(self, exclude: str | None = None) -> str | None:
        """Newest bnbcode session for this workspace, from the node-local postgres. bnbcode's
        cliff-compaction continues the work in a SUCCESSOR session with a new id; prompting the
        old id gets an instant "maximum context length" error and its updates go dark."""
        import subprocess
        # never fall back onto an abandoned empty session: it is the NEWEST row right after a relaunch,
        # and the pre-prompt re-check switched back to it once (rsm2 smoke, 2026-09-08)
        skip = set(self._orphans) | ({exclude} if exclude else set())
        ex = (" AND id NOT IN (" + ", ".join(f"'{i}'" for i in sorted(skip)) + ")") if skip else ""
        # bnbcode records process.cwd(), which is the REAL path -- the run dirs are symlinks into
        # scratch, so match both spellings (a relaunch found "no earlier session" otherwise, 2026-09-08)
        dirs = ", ".join(f"'{d}'" for d in {str(self.ws), str(self.ws.resolve())})
        q = f"SELECT id FROM session WHERE directory IN ({dirs}){ex} ORDER BY time_created DESC LIMIT 1"
        try:
            out = subprocess.run([str(Path(self.cell["v2"]) / "bin" / "bnbcode-pg-node"), "psql", "-t", "-A", "-c", q],
                                 capture_output=True, text=True, timeout=30, env=self.env()).stdout.strip()
        except Exception as exc:
            self.events.write("adopt_error", error=str(exc)[:300]); return None
        return out.splitlines()[-1].strip() if out else None

    async def adopt_latest(self, reason: str, exclude: str | None = None) -> bool:
        """Switch to the newest session for the workspace via session/load. True if switched."""
        latest = self.latest_session_id(exclude=exclude)
        if not latest or latest == self.session_id or self.conn is None:
            return False
        self.client.replaying = True
        try:
            try:
                await self.conn._conn.send_request("session/load", {"sessionId": latest, "cwd": str(self.ws), "mcpServers": []})
                how = "session/load"
            except Exception as exc1:
                await self.conn._conn.send_request("session/resume", {"sessionId": latest, "cwd": str(self.ws), "mcpServers": []})
                how = f"session/resume (load failed: {str(exc1)[:80]})"
        except Exception as exc:
            self.events.write("adopt_error", latest=latest, error=str(exc)[:300])
            log(self.cell_dir, f"could not adopt successor session {latest}: {str(exc)[:160]}")
            return False
        finally:
            self.client.replaying = False
        old = self.session_id; self.session_id = latest
        self.events.write("adopt_session", old=old, new=latest, how=how, reason=reason)
        log(self.cell_dir, f"adopted successor session {latest} (was {old}) via {how} [{reason}]")
        return True

    async def start(self) -> dict:
        # The SDK does NOT inherit os.environ (a child spawned with env=None saw only HOME, PATH,
        # USER -- and died with "PgClient: Failed to connect"). Pass the full env explicitly, and
        # give stderr a file: the SDK default is an undrained PIPE.
        self.stack = AsyncExitStack()
        self._stderr = open(self.cell_dir / "stderr.txt", "ab")
        self.conn, self.proc = await self.stack.enter_async_context(
            spawn_agent_process(self.client, "bnbcode", "acp", cwd=str(self.ws), env=self.env(),
                                transport_kwargs={"stderr": self._stderr}))
        init = await self.conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(fs=FileSystemCapabilities(read_text_file=True, write_text_file=True), terminal=False),
            client_info=Implementation(name="coding-agent-evolve-v2", version="0.1"))
        raw = await self.conn._conn.send_request("session/new", {"cwd": str(self.ws), "mcpServers": []})
        self.session_id = raw["sessionId"] if isinstance(raw, dict) else raw.session_id
        self.restored = False
        prior_work = (self.ws / "run").exists() or (self.cell_dir / "iterations.jsonl").exists()
        if not getattr(self, "_fresh_requested", False) and prior_work:
            # a relaunched cell (or a harness restarted after a crash): continue the agent's latest
            # (compacted) session rather than starting cold. The session DB is node-local; bin/launch
            # restores it from <cell>/pgbackup.sql, so this works on any node. The session just
            # created is itself the newest row: exclude it.
            orphan = self.session_id
            self.restored = await self.adopt_latest("relaunch", exclude=orphan)
            if self.restored:
                self._orphans.add(orphan)
            else:
                log(self.cell_dir, "no earlier session found in the session DB -- starting cold (files kept)")
        model_set = None
        try:
            await self.conn._conn.send_request("session/set_model", {"sessionId": self.session_id, "modelId": self.model_id})
            model_set = "session/set_model"
        except Exception:
            try:
                await self.conn._conn.send_request("session/set_config_option",
                                                   {"sessionId": self.session_id, "configId": "model", "value": self.model_id})
                model_set = "session/set_config_option"
            except Exception as exc:  # the workspace opencode.json also names the model, so not fatal
                model_set = f"failed: {exc}"
        mode_set = None
        try:
            await self.conn._conn.send_request("session/set_mode", {"sessionId": self.session_id, "modeId": "build"})
            mode_set = "build"
        except Exception as exc:
            mode_set = f"not set: {str(exc)[:120]}"
        self.sessions += 1
        info = {"protocol_version": getattr(init, "protocol_version", None), "session_id": self.session_id,
                "model": self.model_id, "model_set": model_set, "mode_set": mode_set, "session_no": self.sessions,
                "pid": getattr(self.proc, "pid", None)}
        self.events.write("session_start", **info)
        log(self.cell_dir, f"bnbcode ACP session {self.session_id} (model_set={model_set}, mode={mode_set})")
        return info

    async def prompt(self, text: str) -> TurnEnd:
        self.client.reset_turn()
        t0 = time.time()
        await self.adopt_latest("pre-prompt")
        self.events.write("prompt", chars=len(text), head=text[:200], session=self.session_id)
        payload = text
        last_exc = None
        adopted_once = False
        for attempt in range(5):
            try:
                resp = await self.conn.prompt(session_id=self.session_id, prompt=[text_block(payload)])
                stop = getattr(resp, "stop_reason", None) or "end_turn"
                te = TurnEnd(stop_reason=str(stop), ran_s=time.time() - t0, tools=self.client.tools,
                             text_chars=self.client.text_chars)
                self.events.write("turn_end", **te.__dict__)
                return te
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if ("maximum context length" in msg or "ContextOverflow" in msg or "input_tokens" in msg) and not adopted_once:
                    adopted_once = True
                    if await self.adopt_latest("context error"):
                        continue          # the successor session has the compacted context; retry there
                if "stalled" in msg.lower() and attempt < 3:
                    self.events.write("stall_retry", attempt=attempt + 1, error=str(exc)[:300])
                    payload = STALL_FOLLOWUP
                    continue
                break
        alive = self.proc is not None and getattr(self.proc, "returncode", None) is None
        te = TurnEnd(stop_reason="acp_error", ran_s=time.time() - t0, tools=self.client.tools,
                     text_chars=self.client.text_chars, error=str(last_exc)[:1000],
                     extra={"process_alive": alive})
        self.events.write("turn_end", **te.__dict__)
        return te

    def alive(self) -> bool:
        return self.proc is not None and getattr(self.proc, "returncode", None) is None

    async def terminate(self) -> None:
        if self.proc is not None:
            for meth in ("terminate", "kill"):
                try:
                    getattr(self.proc, meth)()
                    break
                except (ProcessLookupError, AttributeError):
                    continue
        if self.stack is not None:
            try:
                await asyncio.wait_for(self.stack.aclose(), timeout=15)
            except Exception:
                pass
        self.stack = None; self.conn = None; self.proc = None; self.session_id = None
        try:
            self._stderr.close()
        except Exception:
            pass

    async def restart(self) -> dict:
        await self.terminate()
        self._fresh_requested = True
        try:
            return await self.start()
        finally:
            self._fresh_requested = False

"""Claude Code, headless. One `claude -p` process per turn; `--continue` resumes the conversation
on the next prompt (a new process each time, because headless Claude Code honours one Stop-hook
block and then exits rc=0 regardless of CLAUDE_CODE_STOP_HOOK_BLOCK_CAP -- campaign finding).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from common import EventLog, TurnEnd, log, tail


class ClaudeHarness:
    name = "claude"

    def __init__(self, cell: dict, events: EventLog, deadline_epoch: float):
        self.cell = cell
        self.events = events
        self.last_tool_t = 0.0    # wall time of the last tool_use block (the driver's stall watchdog reads it)
        self.ws = Path(cell["workspace"])
        self.cell_dir = Path(cell["cell_dir"])
        self.deadline = deadline_epoch
        self.proc: asyncio.subprocess.Process | None = None
        self.has_conversation = False
        self.restored = False            # True when start() found a transcript to --continue
        self.conversation_lost = False   # set when --continue was refused; the driver re-sends the initial prompt
        self._fresh = False
        self.sessions = 0
        self.stderr_path = self.cell_dir / "stderr.txt"
        self.config_dir = self.cell_dir / ".cc"

    def env(self) -> dict:
        c = self.cell["claude"]
        env = dict(os.environ)
        env["PATH"] = f"{Path.home() / '.local/bin'}:{Path.home() / 'bin'}:" + env.get("PATH", "")
        alias = self.cell["cc_model"]
        # claude.cliff: Claude Code talks to a per-cell CliffCompaction proxy (bin/launch starts it in
        # front of LiteLLM) and its own auto-compaction is disabled, so compaction is cliff-style in
        # both harnesses (bnbcode has it built in). 2026-09-21.
        use_cliff = bool(c.get("cliff"))
        base_port = self.cell["cliff_port"] if use_cliff else self.cell["litellm_port"]
        env.update({
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{base_port}",
            "ANTHROPIC_AUTH_TOKEN": "sk-local",
            "ANTHROPIC_MODEL": alias, "ANTHROPIC_DEFAULT_OPUS_MODEL": alias,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": alias, "ANTHROPIC_DEFAULT_HAIKU_MODEL": alias,
            "ANTHROPIC_SMALL_FAST_MODEL": alias,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(int(c["max_output_tokens"])),
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(int(c["max_context_tokens"])),
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_AUTOUPDATER": "1",
            # Only what Claude Code ASKS for; LiteLLM drops it. Without this an unrecognised model id
            # makes it send thinking:{type:adaptive}, get a 400, and retry -- and LiteLLM 1.97 would
            # bucket a real budget into "high", which the Qwen3.8 template rejects.
            "MAX_THINKING_TOKENS": "0",
            "CLAUDE_CONFIG_DIR": str(self.config_dir),
            "CC_RUN_DIR": str(self.cell_dir),
            "CC_DEADLINE_EPOCH": str(int(self.deadline)),
            "CC_REFUSE_LIMIT": str(int(self.cell.get("give_up") or 1000000)),
            "CLAUDE_CODE_STOP_HOOK_BLOCK_CAP": "100000",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        })
        if use_cliff:
            env["DISABLE_AUTO_COMPACT"] = "1"
        env["no_proxy"] = (env.get("no_proxy", "") + ",127.0.0.1,localhost").lstrip(",")
        env["NO_PROXY"] = env["no_proxy"]
        return env

    def saved_transcripts(self) -> list[Path]:
        """Claude Code keeps one JSONL per conversation under <config>/projects/<encoded cwd>/. The
        config dir is inside the cell on the shared filesystem, so a relaunch on any node can
        --continue the conversation a killed job left behind."""
        return sorted(self.config_dir.glob("projects/*/*.jsonl"), key=lambda q: q.stat().st_mtime)

    async def start(self) -> dict:
        self.config_dir.mkdir(exist_ok=True)
        self.sessions += 1
        self.restored = False
        saved = [] if self._fresh else self.saved_transcripts()
        if saved:
            self.has_conversation = True
            self.restored = True
            log(self.cell_dir, f"claude: {len(saved)} saved transcript(s), newest {saved[-1].name} "
                               f"({saved[-1].stat().st_size // 1024} KB) -- will --continue it")
        info = {"harness": "claude", "model": self.cell["cc_model"], "session_no": self.sessions,
                "settings": self.cell["claude_settings"], "restored": self.restored,
                "transcript": saved[-1].name if saved else None}
        self.events.write("session_start", **info)
        return info

    async def prompt(self, text: str, fresh: bool = False) -> TurnEnd:
        c = self.cell["claude"]
        cmd = ["claude", "--settings", self.cell["claude_settings"], "--permission-mode", "acceptEdits",
               "--output-format", "stream-json", "--verbose"]
        if c.get("autocompact"):
            cmd += ["--autocompact", str(c["autocompact"])]
        if self.has_conversation and not fresh:
            cmd += ["--continue"]
        # The prompt goes on STDIN, not argv: an argv prompt is readable in the process table by every
        # other cell on the host, and on 2026-09-09 a sibling cell read "best official score so far"
        # out of it and searched toward that number (q38ac_r2/ac2_evo_cc). `claude -p` with no
        # argument reads the prompt from a piped stdin.
        cmd += ["-p"]
        t0 = time.time()
        self.events.write("prompt", chars=len(text), head=text[:200], continue_=self.has_conversation and not fresh)
        tools = 0; text_chars = 0; result_stop = None; api_errors = 0
        # Claude Code refuses some prompts locally, before any HTTP request: the stream then carries a
        # synthetic assistant message (model "<synthetic>", duration_api_ms 0) and a result with
        # is_error and terminal_reason "blocking_limit". The one seen in the wild is "Prompt is too
        # long" -- once the saved conversation is over the limit EVERY --continue fails that way, in
        # under half a second, forever. Two cells of m48_q38 died like that on 2026-09-21 (60 identical
        # retries over 2 h). It is not visible in stderr and never reaches the relay, so it has to be
        # read off the stream. Recovery is the conversation_lost path: drop --continue, start a fresh
        # session on the same workspace, keep every file and score.
        local_refusal = None
        with open(self.stderr_path, "ab") as err:
            # stream-json puts a whole assistant message or tool result on one line; asyncio's
            # default 64 KB StreamReader limit raised "Separator is found, but chunk is longer than
            # limit" and killed a cell after 15 min (2026-09-08). 512 MB limit + a fallback reader.
            self.proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(self.ws), env=self.env(), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=err, limit=512 * 1024 * 1024)
            assert self.proc.stdout is not None and self.proc.stdin is not None
            self.proc.stdin.write(text.encode("utf-8"))
            await self.proc.stdin.drain()
            self.proc.stdin.close()
            while True:
                try:
                    line = await self.proc.stdout.readline()
                except (asyncio.LimitOverrunError, ValueError) as exc:
                    self.events.write("cc_raw", error=f"oversized line: {exc}")
                    # drain the oversized chunk without dying
                    try:
                        await self.proc.stdout.read(64 * 1024 * 1024)
                    except Exception:
                        pass
                    continue
                if not line:
                    break
                s = line.decode("utf-8", "replace").rstrip("\n")
                if not s.strip():
                    continue
                try:
                    ev = json.loads(s)
                except ValueError:
                    self.events.write("cc_raw", line=s[:2000])
                    continue
                typ = ev.get("type")
                if typ == "assistant":
                    msg = ev.get("message") or {}
                    if msg.get("model") == "<synthetic>":
                        for blk in msg.get("content") or []:
                            if blk.get("type") == "text" and (blk.get("text") or "").strip():
                                local_refusal = (blk["text"] or "").strip()[:200]
                    for blk in (ev.get("message") or {}).get("content") or []:
                        if blk.get("type") == "tool_use":
                            tools += 1
                            self.last_tool_t = time.time()
                        elif blk.get("type") == "text":
                            t = blk.get("text") or ""
                            text_chars += len(t.strip())
                            if t.startswith("API Error") or "Request timed out" in t:
                                api_errors += 1
                elif typ == "result":
                    result_stop = ev.get("subtype") or "result"
                    if ev.get("is_error") and ev.get("terminal_reason") == "blocking_limit":
                        local_refusal = local_refusal or f"blocking_limit ({ev.get('subtype')})"
                self.events.write("cc", **{k: v for k, v in ev.items() if k in ("type", "subtype", "is_error", "num_turns", "duration_ms", "total_cost_usd")},
                                  payload=ev if typ in ("assistant", "user", "result", "system") else None)
            rc = await self.proc.wait()
        used_continue = "--continue" in cmd
        self.has_conversation = True
        ran = time.time() - t0
        err_tail = tail(self.stderr_path, 3000)
        if rc != 0 and used_continue and "No conversation found" in err_tail:
            # the saved transcript could not be continued: the driver re-sends the initial prompt
            self.has_conversation = False; self.conversation_lost = True; self.restored = False
            log(self.cell_dir, "claude refused --continue (no conversation found) -- next prompt starts cold")
        elif local_refusal and tools == 0 and used_continue:
            # refused locally before the model was reached (see local_refusal above): the saved
            # conversation is unusable from here on, so take the same route as a lost transcript.
            self.has_conversation = False; self.conversation_lost = True; self.restored = False
            log(self.cell_dir, f"claude refused the prompt locally ({local_refusal!r}) -- "
                               f"conversation retired, next prompt starts a fresh session")
        stop = result_stop or ("end_turn" if rc == 0 else "error")
        te = TurnEnd(stop_reason=stop, ran_s=ran, tools=tools, text_chars=text_chars, rc=rc,
                     error=None if rc == 0 else f"claude exited rc={rc}", stderr_tail=err_tail,
                     extra={"api_errors": api_errors, "local_refusal": local_refusal})
        self.events.write("turn_end", **{k: v for k, v in te.__dict__.items() if k != "stderr_tail"})
        self.proc = None
        return te

    def alive(self) -> bool:
        return True   # nothing persistent to die between turns; each prompt is its own process

    async def terminate(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=20)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
        self.proc = None

    async def restart(self) -> dict:
        """Fresh conversation: drop --continue on the next prompt."""
        await self.terminate()
        self.has_conversation = False
        self._fresh = True
        try:
            return await self.start()
        finally:
            self._fresh = False

#!/usr/bin/env python3
"""ACP handshake only: spawn `bnbcode acp` in a cell's workspace, initialize, session/new,
set model, then exit. No prompt is sent, so no LLM is needed.  usage: acp_handshake.py <cell_dir>"""
import asyncio, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "driver")); sys.path.insert(0, str(HERE.parent / "tracker"))
from common import EventLog, load_cell  # noqa: E402
from harness_bnbcode import BnbcodeHarness  # noqa: E402


async def main(cell_dir: Path) -> int:
    cell = load_cell(cell_dir)
    ev = EventLog(cell_dir / "handshake_events.jsonl")
    h = BnbcodeHarness(cell, ev)
    try:
        info = await asyncio.wait_for(h.start(), timeout=120)
        print(json.dumps(info, indent=2))
        await asyncio.sleep(1)
        print("process alive:", h.alive())
    finally:
        await h.terminate(); ev.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(Path(sys.argv[1]).resolve())))

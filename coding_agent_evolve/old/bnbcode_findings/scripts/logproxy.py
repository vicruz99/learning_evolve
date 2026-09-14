#!/usr/bin/env python3
"""Transparent HTTP proxy that records every request body on its way to the LLM.

    python3 logproxy.py --port 9101 --upstream 127.0.0.1:9001 --log /tmp/cap/bnb.jsonl

Sits between a coding-agent harness and the vLLM relay so the two harnesses' requests can
be diffed byte for byte. Streams the response through unbuffered, so SSE still works and
the agent behaves normally. Stdlib only.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailers", "transfer-encoding", "upgrade", "host"}

ARGS = None
LOCK = threading.Lock()
SEQ = [0]


def record(entry):
    with LOCK:
        SEQ[0] += 1
        entry["seq"] = SEQ[0]
        with open(ARGS.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                       # keep stderr quiet
        pass

    def _relay(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        try:
            parsed = json.loads(body) if body else None
        except Exception:
            parsed = None

        # Optionally pin sampling on the way through. bnbcode never sends `temperature`
        # (measured: 9 of 9 agent requests), so vLLM falls back to the checkpoint's
        # generation_config default of 1.0. This lets us A/B that without touching the
        # harness. Only chat requests are rewritten; the title-generation call is left alone.
        if (ARGS.force_temperature is not None and isinstance(parsed, dict)
                and "messages" in parsed and parsed.get("tools")):
            parsed["temperature"] = ARGS.force_temperature
            body = json.dumps(parsed).encode()
            parsed = dict(parsed, _forced_temperature=ARGS.force_temperature)
        record({
            "t": time.strftime("%H:%M:%S"),
            "method": method,
            "path": self.path,
            "n_bytes": len(body),
            "body": parsed if parsed is not None else body[:2000].decode("utf-8", "replace"),
        })

        host, _, port = ARGS.upstream.partition(":")
        conn = http.client.HTTPConnection(host, int(port or 80), timeout=1800)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Content-Length"] = str(len(body))
        headers["Host"] = ARGS.upstream
        try:
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(("logproxy upstream error: %s" % exc).encode())
            return

        self.send_response(resp.status)
        streaming = False
        for k, v in resp.getheaders():
            if k.lower() in HOP or k.lower() == "content-length":
                if k.lower() == "content-length":
                    self.send_header(k, v)
                continue
            if k.lower() == "content-type" and "event-stream" in v.lower():
                streaming = True
            self.send_header(k, v)
        if streaming:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # Stream through. Chunked framing by hand when the upstream was chunked, so the
        # client sees the same shape it would without us in the middle.
        TAILCAP = 60000
        collected = []
        head = []
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                if sum(len(c) for c in head) < 4000:
                    head.append(chunk)
                collected.append(chunk)
                while sum(len(c) for c in collected) > TAILCAP and len(collected) > 1:
                    collected.pop(0)
                if streaming:
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if streaming:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

        blob = b"".join(collected).decode("utf-8", "replace")
        headblob = b"".join(head).decode("utf-8", "replace")
        fin = [w for w in ("tool_calls", "\"stop\"", "length") if w in blob]
        # A stalled turn is one that finished with reason "stop" and no tool_calls: dump
        # a long tail for those so the raw generation can be inspected for a tool call the
        # parser may have dropped.
        stalled = ("tool_calls" not in blob) and ('"stop"' in blob)
        record({"t": time.strftime("%H:%M:%S"), "kind": "response",
                "status": resp.status, "streaming": streaming,
                "finish_hints": fin, "stalled": stalled,
                "head": headblob[:1500],
                "tail": blob[-40000:] if stalled else blob[-1500:]})

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", default="127.0.0.1:9001")
    ap.add_argument("--log", required=True)
    ap.add_argument("--force-temperature", type=float, default=None,
                    help="rewrite every tool-carrying chat request to use this temperature")
    ARGS = ap.parse_args()
    os.makedirs(os.path.dirname(ARGS.log), exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print("logproxy :%d -> %s   log=%s" % (ARGS.port, ARGS.upstream, ARGS.log), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""In-job HTTP relay that follows a moving llmtun tunnel. Stdlib only (python >= 3.8).

    python3.12 llm_relay.py --name gpu2 --port 18100

Clients in the job talk to http://127.0.0.1:18100/v1 for the whole run. Every request is forwarded
to the current address of the llmtun tunnel named --name, looked up in llmtun's manifest on GPFS
(~/.llmtun/endpoints: name<TAB>model<TAB>node<TAB>port<TAB>login_ip). The tunnel's PORT is sticky
per LSF job name; its IP is whichever Marvin login node hosts the tunnel and changes when a tunnel
is rebuilt elsewhere. The ICL driver (src/icl/loop.py::_refresh_endpoint) re-reads the manifest
itself; Shinka and LiteLLM cannot, so this relay does it for them. On a connection failure the
manifest is re-read and the request retried once against the new address.

Streaming (SSE) responses are passed through chunk by chunk, so Claude Code's streaming works.
The relay binds to loopback only and adds nothing to the traffic: no auth, no buffering of bodies.
"""
import argparse
import http.client
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MANIFEST = os.path.expanduser("~/.llmtun/endpoints")
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class Upstream:
    def __init__(self, name: str, ttl: float):
        self.name, self.ttl = name, ttl
        self._addr, self._read_at, self._lock = None, 0.0, threading.Lock()

    def _lookup(self):
        with open(MANIFEST) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 5 and f[0] == self.name:
                    return f[4], int(f[3])
        raise LookupError(f"{self.name!r} not in {MANIFEST}")

    def get(self, force: bool = False):
        with self._lock:
            if force or self._addr is None or time.time() - self._read_at > self.ttl:
                try:
                    new = self._lookup()
                    if new != self._addr:
                        log(f"upstream {self.name} -> http://{new[0]}:{new[1]}" +
                            (f"  (was {self._addr[0]}:{self._addr[1]})" if self._addr else ""))
                    self._addr = new
                except Exception as e:  # keep the last known address; the caller retries
                    log(f"manifest re-read failed ({e}); keeping {self._addr}")
                self._read_at = time.time()
            return self._addr


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), "[relay]", msg, file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream: Upstream = None       # set in main()
    timeout_s: float = 3600.0

    def log_message(self, *a):      # quiet; errors are logged explicitly
        pass

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        if body is not None:
            hdrs["Content-Length"] = str(len(body))
        for attempt in range(2):
            addr = self.upstream.get(force=(attempt == 1))
            if addr is None:
                self.send_error(503, "no upstream in llmtun manifest"); return
            conn = http.client.HTTPConnection(addr[0], addr[1], timeout=self.timeout_s)
            try:
                conn.request(self.command, self.path, body=body, headers=hdrs)
                resp = conn.getresponse()
            except (ConnectionError, OSError, http.client.HTTPException) as e:
                conn.close()
                log(f"{self.command} {self.path}: {addr[0]}:{addr[1]} unreachable ({e.__class__.__name__}); "
                    + ("re-reading manifest and retrying" if attempt == 0 else "giving up"))
                if attempt == 1:
                    try:
                        self.send_error(502, f"upstream unreachable: {e}")
                    except Exception:
                        pass
                    return
                continue
            try:
                self.send_response(resp.status, resp.reason)
                chunked = False
                for k, v in resp.getheaders():
                    kl = k.lower()
                    if kl == "transfer-encoding" and "chunked" in v.lower():
                        chunked = True
                    if kl in ("connection", "keep-alive", "transfer-encoding"):
                        continue
                    self.send_header(k, v)
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not chunk:
                        break
                    if chunked:
                        self.wfile.write(f"{len(chunk):x}\r\n".encode()); self.wfile.write(chunk); self.wfile.write(b"\r\n")
                    else:
                        self.wfile.write(chunk)
                    self.wfile.flush()
                if chunked:
                    self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                                    # client went away mid-stream
            finally:
                conn.close()
            return

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = _forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="llmtun model name, e.g. gpu2")
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ttl", type=float, default=60.0, help="seconds between manifest re-reads")
    ap.add_argument("--timeout", type=float, default=3600.0, help="upstream read timeout (long generations)")
    a = ap.parse_args()
    Handler.upstream = Upstream(a.name, a.ttl)
    Handler.timeout_s = a.timeout
    Handler.upstream.get(force=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    log(f"listening on http://{a.host}:{a.port} -> llmtun '{a.name}' (manifest {MANIFEST}, ttl {a.ttl:.0f}s)")
    srv.serve_forever()


if __name__ == "__main__":
    main()

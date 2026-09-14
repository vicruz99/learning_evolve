#!/usr/bin/env python3
"""A fixed address for a vLLM server that keeps moving.

The problem
-----------
bnbcode reads `provider.vllm.options.baseURL` from ~/.config/opencode/opencode.jsonc when its
backend starts. `bnbcode-go` rewrites that key to whichever GPU node is serving right now -- but
only at launch. When an LSF job dies or is requeued onto a different node mid-session, the agent
keeps talking to an address that no longer answers, and the only cure is to restart it.

The fix
-------
Point bnbcode at *this* process instead, on a loopback address that never changes:

    "baseURL": "http://127.0.0.1:9001/v1"

Every new TCP connection is resolved fresh: we ask LSF where the target jobs are running, probe
each candidate for a live OpenAI endpoint, and splice the client to whichever answers. A server
that moves costs one failed request, not a session.

Why per-connection and not a URL rewrite: this is a byte pipe, so it does not need to understand
chat completions, streaming, tool calls or any future endpoint. It only needs to know where to
point.

Targets are given as `jobname:port`, because the port is NOT a property of the cluster: two vLLM
jobs can land on the same GPU node, and the second one has to take a different port. As of
2026-08-29, gpu2 serves :8001 on its node while gpu6 serves :8002 -- and gpu4, which shares gpu6's
node and holds :8001 there, is NOT tool-capable. Selecting by hostname alone finds the wrong server.

Usage
-----
    python3 llm_relay.py &                                   # defaults: :9001 -> gpu2:8001,gpu6:8002
    python3 llm_relay.py --jobs gpu2:8001,gpu6:8002 --verbose

Run it on the SAME node as bnbcode (inside your interactive job). Stdlib only -- any python3.
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.request

# `bjobs` and DNS both live behind the corporate proxy env vars on this cluster; the probe below
# talks to a 10.124.x address that `no_proxy` (rng-dl01-*,localhost,127.0.0.1) does not cover, so
# we always build an opener that bypasses the proxy explicitly rather than trusting the environment.
_NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


OUTAGES: list[dict] = []
HEALTH_LOG = ""


def record_outages() -> None:
    """Persist endpoint outages so the analysis can tell an infrastructure failure from a
    model stall. Without this, a dead server looks exactly like a stalled agent."""
    if not HEALTH_LOG:
        return
    try:
        import json as _json
        with open(HEALTH_LOG, "w") as f:
            _json.dump(OUTAGES, f)
    except OSError:
        pass


_SEED = random.randrange(1 << 30)


def log(msg: str) -> None:
    print(f"[llm-relay {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def lsf_hosts(job_names: list[str]) -> list[tuple[str, str]]:
    """(job_name, host) for every RUNning LSF job whose name is in `job_names`, in that order.

    `bjobs -w` prints EXEC_HOST as "12*rng-dl01-w26n14" when several slots are on one host.
    """
    try:
        out = subprocess.run(["bjobs", "-w"], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:                      # LSF unreachable -> no candidates, not a crash
        log(f"bjobs failed: {exc}")
        return []
    found: dict[str, str] = {}
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) < 7 or f[2] != "RUN":
            continue
        name, host = f[6], f[5].split("*")[-1]
        if name not in found:
            found[name] = host
    # "*" means "any running job": the probes below still insist on the right model and on
    # tool-capability, so scanning everything is safe -- and it is what lets a REPLACEMENT
    # server with a brand new job name be picked up without touching any config.
    if "*" in job_names:
        # v2: shuffled with a per-process seed, so several relays (one per cell) spread over the
        # servers while each relay keeps a stable preference order (sticky -> prefix cache hits)
        items = sorted(found.items())
        random.Random(_SEED).shuffle(items)
        return items
    return [(n, found[n]) for n in job_names if n in found]


def resolve(host: str) -> str | None:
    try:
        return socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
    except OSError:
        return None


def alive(ip: str, port: int, timeout: float) -> bool:
    """Cheap liveness check: /v1/models answers with a model list.

    A bare TCP connect is not enough -- vLLM's listener comes up well before the engine does.
    This is not sufficient on its own either (see `agentic`), but it is cheap enough to re-run
    on every TTL expiry.
    """
    try:
        with _NOPROXY.open(f"http://{ip}:{port}/v1/models", timeout=timeout) as r:
            return bool(json.load(r).get("data"))
    except Exception:
        return False


def agentic(ip: str, port: int, model: str, timeout: float) -> bool:
    """Does this server actually do tool calls?

    A server launched without `--enable-auto-tool-choice --tool-call-parser` answers /v1/models
    perfectly and then 400s on the first tool call -- useless to a coding agent, and indistinguishable
    from a good one until you ask. Several of the gpuN jobs are launched that way, so probe rather
    than trust the job name. This also exercises the engine end to end, which is the only check that
    catches a wedged engine behind a healthy HTTP front end (the 2026-08-09 failure mode).
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "call tool t"}],
        "tools": [{"type": "function",
                   "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": "auto",
        "max_tokens": 16,
    }).encode()
    req = urllib.request.Request(f"http://{ip}:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with _NOPROXY.open(req, timeout=timeout) as r:
            return bool(json.load(r).get("choices"))
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", "replace")
        if "enable-auto-tool-choice" in detail:
            log(f"  {ip}:{port} is live but has no tool support -- skipping")
        return False
    except Exception:
        return False


class Upstream:
    """Current target, re-discovered on demand and memoised for `ttl` seconds."""

    def __init__(self, targets: list[tuple[str, int]], model: str, ttl: float,
                 probe_timeout: float, tool_timeout: float):
        self.targets = targets                    # [(job_name, port), ...] in preference order
        self.model, self.ttl = model, ttl
        self.probe_timeout, self.tool_timeout = probe_timeout, tool_timeout
        self._addr: tuple[str, int] | None = None
        self._name = ""
        self._at = 0.0
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._addr = None

    def get(self) -> tuple[tuple[str, int] | None, str]:
        with self._lock:
            if self._addr and (time.time() - self._at) < self.ttl:
                return self._addr, self._name
            # Keeping the current pick is the common case, and the expensive tool probe adds
            # nothing there -- it was already established when this address was chosen.
            if self._addr and alive(self._addr[0], self._addr[1], self.probe_timeout):
                self._at = time.time()
                return self._addr, self._name

            found = lsf_hosts([n for n, _ in self.targets])
            hosts = dict(found)
            for name, port in self.targets:
                if name == "*":
                    candidates = found
                elif name in hosts:
                    candidates = [(name, hosts[name])]
                elif "-" in name or "." in name:
                    # v2: a fixed host (rng-dl01-w26n16 or 10.124.x.x), for servers that are not
                    # LSF jobs of this user; still subject to the liveness and tool probes
                    candidates = [(f"host:{name}", name)]
                else:
                    candidates = []
                for cname, host in candidates:
                    ip = resolve(host)
                    if not ip or not alive(ip, port, self.probe_timeout):
                        continue
                    if not agentic(ip, port, self.model, self.tool_timeout):
                        continue
                    if (ip, port) != self._addr:
                        log(f"upstream -> {cname} {host} ({ip}:{port})")
                    self._addr, self._name, self._at = (ip, port), f"{cname}/{host}", time.time()
                    return self._addr, self._name
            if self._addr is not None:
                log("no live tool-capable upstream found")
            self._addr, self._name = None, ""
            return None, ""


def splice(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            chunk = a.recv(65536)
            if not chunk:
                break
            b.sendall(chunk)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def wait_for_upstream(up: "Upstream", wait_s: float) -> tuple[tuple[str, int] | None, str]:
    """Block until an upstream exists, up to wait_s.

    A vLLM server that dies, moves, or is replaced by a new LSF job must cost the agent a
    pause, not the run: an unaided cell that gets a 503 exits, and the transcript then shows
    an infrastructure failure that is indistinguishable from the model choosing to stop --
    which would silently corrupt the very thing this campaign measures.
    """
    addr, name = up.get()
    if addr is not None:
        return addr, name
    began = time.time()
    log(f"no upstream; holding the connection for up to {int(wait_s)}s")
    OUTAGES.append({"start": began})
    while time.time() - began < wait_s:
        time.sleep(5.0)
        up.invalidate()
        addr, name = up.get()
        if addr is not None:
            gone = time.time() - began
            log(f"upstream returned after {gone:.0f}s -> {name}")
            OUTAGES[-1]["end"] = time.time()
            record_outages()
            return addr, name
    log(f"upstream still absent after {wait_s:.0f}s; giving up on this request")
    OUTAGES[-1]["end"] = time.time()
    record_outages()
    return None, ""


def handle(client: socket.socket, up: Upstream, verbose: bool, wait_s: float = 900.0) -> None:
    with client:
        addr, name = wait_for_upstream(up, wait_s)
        if addr is None:
            # Answer rather than hang: a 503 surfaces in the agent's transcript as a readable
            # error, where a dropped connection shows up as an opaque client-side timeout.
            client.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                           b"Connection: close\r\nContent-Length: 63\r\n\r\n"
                           b'{"error":{"message":"llm-relay: no live vLLM upstream"}}\n')
            return
        try:
            server = socket.create_connection(addr, timeout=30)
        except OSError as exc:
            # The memoised address just went away. Drop it so the next request rediscovers.
            up.invalidate()
            log(f"connect to {name} {addr} failed ({exc}); cache invalidated")
            client.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                           b"Connection: close\r\nContent-Length: 66\r\n\r\n"
                           b'{"error":{"message":"llm-relay: upstream refused, retry"}}\n')
            return
        if verbose:
            log(f"connection -> {name}")
        with server:
            server.settimeout(None)
            client.settimeout(None)
            t = threading.Thread(target=splice, args=(server, client), daemon=True)
            t.start()
            splice(client, server)
            t.join()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", default="gpu2:8001,gpu6:8002",
                    # each token is NAME[:PORT]; NAME = LSF job name, "*" (any of your RUN jobs),
                    # or a fixed hostname/IP (anything containing "-" or ".")
                    help="comma-separated LSF job targets as name[:port], preferred first "
                         "(default: gpu2:8001,gpu6:8002). The port matters: two vLLM jobs can "
                         "share a GPU node and the second one is not on 8001.")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8", help="model id used by the tool probe")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=9001)
    ap.add_argument("--ttl", type=float, default=30.0,
                    help="seconds to reuse a discovered upstream before re-probing")
    ap.add_argument("--probe-timeout", type=float, default=5.0)
    ap.add_argument("--tool-timeout", type=float, default=30.0,
                    help="timeout for the tool-calling probe; it decodes, so it is not instant")
    ap.add_argument("--verbose", action="store_true", help="log every connection")
    ap.add_argument("--wait", type=float, default=900.0,
                    help="seconds to hold a request while no upstream is live, instead of "
                         "returning 503. A server that dies, moves, or is replaced by a new "
                         "LSF job then costs the agent a pause rather than the whole run.")
    ap.add_argument("--health-log", default="",
                    help="JSON file recording endpoint outages, so the analysis can tell an "
                         "infrastructure failure from the model deciding to stop")
    a = ap.parse_args()

    global HEALTH_LOG
    HEALTH_LOG = a.health_log

    targets = []
    for tok in a.jobs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, port = tok.partition(":")
        targets.append((name, int(port) if port else 8001))
    up = Upstream(targets, a.model, a.ttl, a.probe_timeout, a.tool_timeout)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.listen_host, a.listen_port))
    srv.listen(128)
    log("listening on http://{}:{}/v1  ->  {}".format(
        a.listen_host, a.listen_port, ", ".join(f"{n}:{p}" for n, p in targets)))
    addr, name = up.get()
    log(f"initial upstream: {name or 'NONE'}")

    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle, args=(conn, up, a.verbose, a.wait), daemon=True).start()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit tests for sft/online/orchestrate.py: schedule, round lifecycle, restarts. No GPU, no server.

The driver is tests/online_stub_icl.py (via ONLINE_RUN_ICL_CMD), the trainer is `fake_train` below
(moves queue jobs pending -> done and writes a dummy adapter), the server is FakeServer (fake://).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import pytest

from sft.online.orchestrate import FakeServer, Orchestrator, train_points

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB = os.path.join(SRC, "tests", "online_stub_icl.py")


# ---- schedule -------------------------------------------------------------------------------------
def test_train_points():
    assert train_points(1, 12) == list(range(11))                 # gen 11's round has no consumer
    assert train_points(3, 12) == [2, 5, 8]
    assert train_points(3, 12, final_round=True) == [2, 5, 8, 11]
    assert train_points(None, 12) == []


# ---- harness ----------------------------------------------------------------------------------------
def make_cfg(tmp_path, runs, N=4, every=None, arms=None):
    arms = arms or {"e1_top40": {"every": 1, "select": ["top_frac", 0.4]},
                    "e3_top30": {"every": 3, "select": ["top_frac", 0.3]},
                    "e1_p80": {"every": 1, "select": ["pct_threshold", 80]},
                    "frozen": {"every": None}}
    return {"runs_root": str(tmp_path / "runs"), "adapters_root": str(tmp_path / "adapters"),
            "queue_dir": str(tmp_path / "queue"), "servers": {"A": f"fake://{tmp_path}"},
            "base_model": "base", "num_generations": N, "final_round": False,
            "run_icl_args": ["--problem", "ac2", "--groups-per-batch", "2", "--group-size", "5"],
            "hparams": {"lora_r": 8, "lora_alpha": 16, "lr": 1e-4, "epochs": 1, "grad_accum": 1,
                        "max_len": 1024},
            "ray": {"mode": "skip"}, "adapter_settle_s": 0, "arms": arms, "runs": runs}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ONLINE_RUN_ICL_CMD", f"{sys.executable} {STUB}")
    monkeypatch.delenv("STUB_FAIL_AT_GEN", raising=False)
    monkeypatch.delenv("STUB_SLEEP", raising=False)


def wait_drivers(orch, timeout=30):
    """Block until no launched driver is alive (the stub finishes in well under a second)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(orch.driver_pid(r) for r in orch.cfg["runs"]):
            return
        time.sleep(0.05)
    raise AssertionError("stub driver did not exit")


def fake_train(queue: str, fail=False, examples=None):
    """Play the trainer daemon for every pending job."""
    pend = os.path.join(queue, "pending")
    n = 0
    for name in sorted(os.listdir(pend)):
        job = json.load(open(os.path.join(pend, name)))
        os.remove(os.path.join(pend, name))
        if fail:
            json.dump({"status": "failed", "job": job, "error": "boom"},
                      open(os.path.join(queue, "failed", name), "w"))
        else:
            os.makedirs(job["adapter_out"], exist_ok=True)
            open(os.path.join(job["adapter_out"], "adapter_config.json"), "w").write("{}")
            import hashlib
            blob = f"weights-{job['id']}".encode()
            open(os.path.join(job["adapter_out"], "adapter_model.safetensors"), "wb").write(blob)
            rows = sum(1 for _ in open(job["data"]))
            ex = rows if examples is None else examples
            json.dump({"status": "ok", "job": job, "examples": ex, "skipped": {"too_long": rows - ex},
                       "adapter_out": job["adapter_out"], "adapter_sha": hashlib.sha256(blob).hexdigest(),
                       "seconds": 1.0, "train_loss": 0.5, "steps": ex},
                      open(os.path.join(queue, "done", name), "w"))
        n += 1
    return n


def models_used(orch, run):
    p = os.path.join(orch.rundir(run), "online_stub_models.jsonl")
    return [json.loads(l)["model"] for l in open(p)]


# ---- the main lifecycle --------------------------------------------------------------------------------
def test_every_generation_arm_trains_between_generations(tmp_path, env):
    run = {"name": "e1_s1", "arm": "e1_top40", "seed": 1, "server": "A"}
    orch = Orchestrator(make_cfg(tmp_path, [run]), child_env=dict(os.environ))
    server = FakeServer.registry[f"fake://{tmp_path}"]

    assert orch.tick()["e1_s1"].startswith("generating")         # segment gens 0..0 on the base
    wait_drivers(orch)
    st = orch.tick()["e1_s1"]                                    # gen 0 done -> round 0 submitted
    assert st.startswith("training(submitted e1_s1_r00")
    assert orch.tick()["e1_s1"] == "training(pending e1_s1_r00)"
    job = json.load(open(tmp_path / "queue" / "pending" / "e1_s1_r00.json"))
    assert job["adapter_in"] is None and job["hparams"]["seed"] == 1000 and job["gen_lo"] == job["gen_hi"] == 0
    rows = [json.loads(l) for l in open(job["data"])]
    assert len(rows) == round(0.4 * 8)                            # 8 valid of 10 (child 0 of each parent fails)

    assert fake_train(str(tmp_path / "queue")) == 1
    assert orch.tick()["e1_s1"] == "loaded"
    assert server.loads == [("e1_s1_r00", str(tmp_path / "adapters" / "e1_s1" / "r00"), False)]
    rounds = orch.rounds(run)
    assert rounds[-1]["status"] == "trained" and rounds[-1]["model_name"] == "e1_s1_r00"
    assert rounds[-1]["adapter_sha"] == hashlib.sha256(b"weights-e1_s1_r00").hexdigest()

    assert orch.tick()["e1_s1"].startswith("generating")         # gen 1 under the adapter
    wait_drivers(orch)
    orch.tick(); fake_train(str(tmp_path / "queue")); orch.tick()   # round 1 continues r00
    job1 = json.load(open(tmp_path / "queue" / "done" / "e1_s1_r01.json"))["job"]
    assert job1["adapter_in"] == str(tmp_path / "adapters" / "e1_s1" / "r00")
    assert server.loads[-1][0] == "e1_s1_r01" and server.unloads == []   # new name per round, never unloaded

    # finish: gens 2 and 3; round after gen 2, none after gen 3 (no consumer)
    for _ in range(6):
        st = orch.tick()["e1_s1"]
        if st == "done":
            break
        wait_drivers(orch)
        fake_train(str(tmp_path / "queue"))
    assert orch.tick()["e1_s1"] == "done"
    assert models_used(orch, run) == ["base", "e1_s1_r00", "e1_s1_r01", "e1_s1_r02"]
    assert [r["round"] for r in orch.rounds(run)] == [0, 1, 2]
    shas = open(orch.online_dir(run) + "/trained_shas.txt").read().split()
    assert len(shas) == len(set(shas)) == 3 * round(0.4 * 8)


def test_every_three_arm_and_frozen_control(tmp_path, env):
    runs = [{"name": "e3_s1", "arm": "e3_top30", "seed": 1, "server": "A"},
            {"name": "fz_s1", "arm": "frozen", "seed": 1, "server": "A"}]
    orch = Orchestrator(make_cfg(tmp_path, runs, N=6), child_env=dict(os.environ))
    for _ in range(12):
        st = orch.tick()
        if all(v == "done" for v in st.values()):
            break
        wait_drivers(orch)
        fake_train(str(tmp_path / "queue"))
    assert orch.tick() == {"e3_s1": "done", "fz_s1": "done"}
    assert models_used(orch, runs[1]) == ["base"] * 6                     # frozen never trained
    assert not os.path.exists(orch.rounds_path(runs[1]))
    assert models_used(orch, runs[0]) == ["base"] * 3 + ["e3_s1_r00"] * 3     # one round after gen 2
    r = orch.rounds(runs[0])
    assert len(r) == 1 and (r[0]["gen_lo"], r[0]["gen_hi"]) == (0, 2)
    # top 30% of the 3-generation window pooled: 24 valid -> 7
    assert r[0]["examples"] == round(0.3 * 24)


def test_zero_selected_round_is_skipped_not_failed(tmp_path, env):
    run = {"name": "p80_s1", "arm": "e1_p80", "seed": 1, "server": "A"}
    orch = Orchestrator(make_cfg(tmp_path, [run], N=3), child_env=dict(os.environ))
    orch.tick(); wait_drivers(orch)
    orch.tick()                                                   # round 0 submitted
    # pretend every selected answer was already trained on -> the NEXT round selects nothing
    fake_train(str(tmp_path / "queue")); orch.tick()             # round 0 trained + loaded
    orch.tick(); wait_drivers(orch)                               # gen 1 under the adapter
    m0 = json.load(open(orch.online_dir(run) + "/round_00/manifest.json"))
    # poison trained_shas with everything gen 1 could select
    from sft.build_dataset import collect, answer_sha
    cands, _ = collect(orch.rundir(run))
    with open(orch.online_dir(run) + "/trained_shas.txt", "a") as f:
        f.writelines(answer_sha(c["answer"]) + "\n" for c in cands if c["generation"] == 1)
    assert orch.tick()["p80_s1"] == "skipped(no_examples)"
    rounds = orch.rounds(run)
    assert rounds[-1]["status"] == "skipped" and rounds[-1]["gen_lo"] == rounds[-1]["gen_hi"] == 1
    assert orch.tick()["p80_s1"].startswith("generating")         # continues on the ROUND-0 adapter
    wait_drivers(orch)
    assert models_used(orch, run) == ["base", "p80_s1_r00", "p80_s1_r00"]
    assert m0["counts"]["selected"] >= 1


def test_trainer_failure_retries_once_then_fails_run_only(tmp_path, env):
    runs = [{"name": "a", "arm": "e1_top40", "seed": 1, "server": "A"},
            {"name": "b", "arm": "frozen", "seed": 2, "server": "A"}]
    orch = Orchestrator(make_cfg(tmp_path, runs, N=2), child_env=dict(os.environ))
    orch.tick(); wait_drivers(orch)
    assert orch.tick()["a"].startswith("training(submitted a_r00")
    fake_train(str(tmp_path / "queue"), fail=True)
    assert orch.tick()["a"] == "training(submitted a_r00_retry1)"
    fake_train(str(tmp_path / "queue"), fail=True)
    st = orch.tick()
    assert st["a"] == "failed" and st["b"] == "done"
    assert os.path.exists(orch.online_dir(runs[0]) + "/FAILED")
    assert orch.rounds(runs[0])[-1]["status"] == "failed"
    assert orch.tick()["a"] == "failed"                           # stays failed, no relaunch


def test_all_too_long_is_a_skipped_round(tmp_path, env):
    run = {"name": "r", "arm": "e1_top40", "seed": 1, "server": "A"}
    orch = Orchestrator(make_cfg(tmp_path, [run], N=2), child_env=dict(os.environ))
    orch.tick(); wait_drivers(orch); orch.tick()
    fake_train(str(tmp_path / "queue"), examples=0)
    assert orch.tick()["r"] == "skipped(all_too_long)"
    assert orch.current_model(run) == "base"


def test_restart_adopts_live_driver_and_reloads_adapter_after_server_restart(tmp_path, env, monkeypatch):
    run = {"name": "x", "arm": "e1_top40", "seed": 3, "server": "A"}
    cfg = make_cfg(tmp_path, [run], N=3)
    monkeypatch.setenv("STUB_SLEEP", "3")
    orch = Orchestrator(cfg, child_env=dict(os.environ))
    st = orch.tick()["x"]
    pid = int(st.split("pid ")[1].rstrip(")"))
    # a brand-new orchestrator (restart) sees the same driver and does not launch another
    orch2 = Orchestrator(cfg, child_env=dict(os.environ))
    assert orch2.tick()["x"] == f"generating(pid {pid})"
    assert len(open(orch2.online_dir(run) + "/segments.jsonl").readlines()) == 1
    wait_drivers(orch2)
    monkeypatch.setenv("STUB_SLEEP", "0")
    orch2.tick(); fake_train(str(tmp_path / "queue")); assert orch2.tick()["x"] == "loaded"
    # server "restarts": forgets adapters; the next launch re-loads before starting the driver
    server = FakeServer.registry[f"fake://{tmp_path}"]
    server.adapters.clear()
    assert orch2.tick()["x"].startswith("generating")
    assert server.loads[-1] == ("x_r00", str(tmp_path / "adapters" / "x" / "r00"), False)
    wait_drivers(orch2)


def test_server_down_pauses_launch(tmp_path, env):
    run = {"name": "y", "arm": "frozen", "seed": 1, "server": "A"}
    orch = Orchestrator(make_cfg(tmp_path, [run], N=1), child_env=dict(os.environ))
    server = FakeServer.registry[f"fake://{tmp_path}"]
    server.up = False
    assert orch.tick()["y"] == "paused(server down)"
    server.up = True
    assert orch.tick()["y"].startswith("generating")
    wait_drivers(orch)
    assert orch.tick()["y"] == "done"


def test_segment_that_never_advances_three_times_quickly_fails_the_run(tmp_path, env, monkeypatch):
    run = {"name": "z", "arm": "frozen", "seed": 1, "server": "A"}
    monkeypatch.setenv("STUB_FAIL_AT_GEN", "0")            # before the env snapshot is taken
    orch = Orchestrator(make_cfg(tmp_path, [run], N=2), child_env=dict(os.environ))
    for _ in range(3):
        assert orch.tick()["z"].startswith("generating")   # three quick launches, none completes gen 0
        wait_drivers(orch)
    assert orch.tick()["z"] == "failed"


def test_dry_run_cli(tmp_path, env):
    cfg = make_cfg(tmp_path, [{"name": "q", "arm": "e3_top30", "seed": 1, "server": "A"}], N=12)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    out = subprocess.run([sys.executable, "-m", "sft.online.orchestrate", "--config", str(p), "--dry-run"],
                         cwd=SRC, capture_output=True, text=True)
    assert out.returncode == 0 and "train after gens [2, 5, 8]" in out.stdout


from sft.online import orchestrate  # noqa: E402  (module handle for the wedge-detector tests)


# ---- the live-wedge detector -------------------------------------------------------------------
# Two real wedges were measured on 2026-09-09 and neither is caught by the obvious test:
#   A: 0 running, 320 queued, generation counter frozen        -> "nothing running" would catch it
#   B: 6-9 running, 234 queued, +5 gen tokens in 41 s, prefill -> only a decode RATE catches it
# and a healthy server here sits IDLE for long stretches while grading catches up, so "no tokens"
# must never be enough on its own. These four cases pin all of that down.
class _ClockedServer(orchestrate.Server):
    """Serves a fixed queue_state, and owns the clock the orchestrator reads."""

    def __init__(self, state, step=25.0):
        self.url, self.timeout = "http://fake", 5.0
        self.state, self.step, self.now = state, step, 1_000_000.0

    def queue_state(self):
        return dict(self.state(self.now) if callable(self.state) else self.state)

    def tick(self):
        self.now += self.step


def _run_wedge(monkeypatch, state, ticks=8, step=25.0):
    srv = _ClockedServer(state, step)
    monkeypatch.setattr(orchestrate.time, "time", lambda: srv.now)
    o = orchestrate.Orchestrator.__new__(orchestrate.Orchestrator)
    o.cfg = {"wedged_ticks_before_restart": 4, "wedged_min_tok_per_s": 5.0}
    o.frozen, o.last_tokens, o.restarts = {}, {}, []
    o.restart_server = lambda k, why: (o.restarts.append(why), True)[1]
    for _ in range(ticks):
        o.check_wedged("A", srv)
        srv.tick()
    return o.restarts


def test_wedge_detector_leaves_an_idle_server_alone(monkeypatch):
    """The common case: nothing outstanding and no tokens for minutes. Never restart this."""
    assert _run_wedge(monkeypatch, {"running": 0.0, "waiting": 0.0, "tokens": 1000.0}) == []


def test_wedge_detector_leaves_a_saturated_server_alone(monkeypatch):
    """Server A healthy after its restart: 48 running, a long queue, ~2400 tok/s."""
    st = lambda now: {"running": 48.0, "waiting": 352.0, "tokens": 2400.0 * now}
    assert _run_wedge(monkeypatch, st) == []


def test_wedge_detector_catches_a_stopped_scheduler(monkeypatch):
    """Wedge A: requests queued, none admitted, counter frozen."""
    out = _run_wedge(monkeypatch, {"running": 0.0, "waiting": 320.0, "tokens": 331186.0})
    assert len(out) == 1 and "WEDGED" in out[0]


def test_wedge_detector_catches_decode_at_a_crawl(monkeypatch):
    """Wedge B: requests RUNNING and prefill progressing, but 0.12 tok/s of decode."""
    st = lambda now: {"running": 6.0, "waiting": 234.0, "tokens": 9_512_154.0 + 0.12 * now}
    out = _run_wedge(monkeypatch, st)
    assert len(out) == 1 and "WEDGED" in out[0]


def test_wedge_detector_needs_a_sustained_signal(monkeypatch):
    """One slow tick is not a wedge: three below the floor, then recovery, must not restart."""
    st = lambda now: {"running": 8.0, "waiting": 100.0,
                      "tokens": 0.0 if now < 1_000_075.0 else 5000.0 * (now - 1_000_000.0)}
    assert _run_wedge(monkeypatch, st) == []


def test_wedge_detector_survives_a_server_restart(monkeypatch):
    """A restart zeroes generation_tokens_total; that must not read as a wedge.

    Measured 2026-09-10: server A came back from its own restart and the very next tick logged
    "-10883.29 tok/s with 43 running and 37 queued -- tick 1 below 5 tok/s". The rate was negative
    because the new engine's counter started from zero. Four of those in a row would have restarted
    a healthy server, and each restart costs every run on it its in-flight generation.
    """
    t0 = 1_000_000.0

    def st(now):
        # 2400 tok/s, then a restart at t0+100 resets the counter and it climbs at the same rate
        tok = 2400.0 * (now - t0) if now < t0 + 100 else 2400.0 * (now - (t0 + 100))
        return {"running": 43.0, "waiting": 37.0, "tokens": tok}

    assert _run_wedge(monkeypatch, st, ticks=10) == []

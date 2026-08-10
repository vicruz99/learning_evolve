"""Unit tests for the TriMul kernel environment (pure; no GPU, no ray, no LLM).

``TrimulLocalReward`` shells out to ``evaluate.py``, so every test here fakes that subprocess and
asserts on what we do with its output. Two things are worth pinning down beyond the arithmetic:

* the **GPU lock**, because losing it is silent — evals still return, they just return timings taken
  on a contended card, which looks like a bad kernel rather than a broken harness; and
* the **infra-vs-genuine split**, because a missing grader must never be recorded as a candidate
  that failed to compile (that is the distinction the whole failure_type column exists to make).
"""
import json
import subprocess
import threading
import time

import pytest

from envs.kernel_trimul import (SCORE_SCALE, TrimulA100Env, TrimulH100Env,
                                TrimulLocalReward, _HW_RULE_A100, _HW_RULE_H100)
from puct import State


def _state() -> State:
    return State(timestep=-1, construction=None, code="", value=-1_000_000)


def _reward(problem="trimul_a100", **kw) -> TrimulLocalReward:
    return TrimulLocalReward(problem, "/tmp", **kw)


def _fake_run(payload, *, delay: float = 0.0, write: bool = True):
    """Stand in for ``subprocess.run``: write ``payload`` to the --json path evaluate.py was given."""
    def run(cmd, **kwargs):
        if delay:
            time.sleep(delay)
        if write:
            out = cmd[cmd.index("--json") + 1]
            with open(out, "w") as f:
                json.dump([payload], f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return run


OK = {"task": "trimul", "mode": "benchmark", "score_us": 2500.0,
      "phases": {"benchmark": {"passed": True, "stderr": ""}}}


# ---- the score path ------------------------------------------------------------------------------
def test_reward_is_score_scale_over_microseconds(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(OK))
    out = _reward().get_reward("code", _state())
    assert out["correctness"] == 1.0
    assert out["raw_score"] == 2500.0
    assert out["reward"] == pytest.approx(SCORE_SCALE / 2500.0)


def test_faster_kernel_earns_more_reward(monkeypatch):
    """Sanity on the direction: this is a MINIMISE problem behind a maximise-shaped reward."""
    fast = dict(OK, score_us=1000.0)
    monkeypatch.setattr(subprocess, "run", _fake_run(fast))
    quick = _reward().get_reward("code", _state())["reward"]
    monkeypatch.setattr(subprocess, "run", _fake_run(OK))
    slow = _reward().get_reward("code", _state())["reward"]
    assert quick > slow


def test_timings_are_recorded_for_progress_csv(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(OK, delay=0.05))
    r = _reward()
    r.get_reward("code", _state())
    assert r._last_timing["eval_seconds"] >= 0.05
    assert "queue_seconds" in r._last_timing


# ---- genuine candidate failures ------------------------------------------------------------------
def test_failed_phase_is_a_crash_not_a_score(monkeypatch):
    bad = {"task": "trimul", "mode": "benchmark", "score_us": None, "failed_phase": "benchmark",
           "phases": {"benchmark": {"passed": False, "stderr": "SyntaxError: invalid syntax"}}}
    monkeypatch.setattr(subprocess, "run", _fake_run(bad))
    out = _reward().get_reward("nonsense", _state())
    assert out["correctness"] == 0.0
    assert out["reward"] == 0.0
    assert out["failure_type"] == "process_crash"
    # the model has to see WHY it failed, so the stderr tail must survive into msg
    assert "SyntaxError: invalid syntax" in out["msg"]


def test_passed_false_beats_a_present_score(monkeypatch):
    """Defensive: never trust score_us if the phase did not pass."""
    weird = dict(OK, score_us=10.0, phases={"benchmark": {"passed": False, "stderr": "boom"}})
    monkeypatch.setattr(subprocess, "run", _fake_run(weird))
    assert _reward().get_reward("code", _state())["correctness"] == 0.0


def test_timeout_is_reported_as_eval_timeout(monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1200)
    monkeypatch.setattr(subprocess, "run", boom)
    out = _reward(eval_timeout=1200).get_reward("code", _state())
    assert out["failure_type"] == "eval_timeout"
    assert out["correctness"] == 0.0


# ---- our failures, which must not look like the candidate's --------------------------------------
def test_missing_grader_is_infra_not_a_bad_kernel(monkeypatch):
    monkeypatch.setenv("TRIMUL_EVALUATE_PY", "/nonexistent/evaluate.py")
    out = _reward().get_reward("code", _state())
    assert out["failure_type"] == "harness_error"
    assert out["correctness"] == 0.0


def test_grader_writing_no_results_is_infra(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(OK, write=False))
    out = _reward().get_reward("code", _state())
    assert out["failure_type"] == "harness_error"


# ---- the lock ------------------------------------------------------------------------------------
def test_concurrent_evals_are_serialised_on_the_gpu(monkeypatch):
    """Without _GPU_LOCK these two overlap, both time a contended card, and nothing complains."""
    live, peak, guard = [0], [0], threading.Lock()

    def run(cmd, **kwargs):
        with guard:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with guard:
            live[0] -= 1
        out = cmd[cmd.index("--json") + 1]
        with open(out, "w") as f:
            json.dump([OK], f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    threads = [threading.Thread(target=lambda: _reward().get_reward("code", _state()))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1, f"{peak[0]} evals shared the GPU at once"


# ---- env wiring ----------------------------------------------------------------------------------
def test_env_is_minimise_and_sandbox_free():
    env = TrimulA100Env.__new__(TrimulA100Env)          # no sampler/config needed for these two
    assert env.is_maximize() is False           # runtime: lower is better
    assert TrimulA100Env.uses_sandbox is False      # the loop must skip init_ray


def test_initial_state_carries_no_seed_program():
    """TTT-Discover starts trimul from nothing (env.py:178); a seed here would change the task."""
    st = TrimulA100Env.create_initial_state("trimul_a100")
    assert st.code == ""
    assert st.value == -1_000_000


def _prompt(cls=TrimulH100Env) -> str:
    from envs.base import EnvConfig
    env = cls(_state(), object(), EnvConfig(problem_type="trimul_h100", log_path="/tmp"))
    return env.get_question()


def test_prompt_keeps_the_upstream_rules_verbatim():
    q = _prompt()
    assert "You must use trition 3.3.1 and these kernels will be run on an H100." in q
    assert "you must implement at least part of the operations in a kernel." in q


def test_the_parent_solution_is_the_last_thing_in_the_prompt():
    """The prefix-caching contract (envs/base.py improvement_task): everything before the current
    kernel is constant across a generation's parents, so only the tail is re-prefilled. Upstream
    orders these the other way round; putting the rules after the parent would silently cost cache
    hits on every PUCT arm without failing anything.
    """
    q = _prompt()
    assert q.index("Rules:") < q.index("--- Current kernel to improve upon ---")
    assert q.index("Target: 1000") > q.index("Rules:")


def test_prompt_asks_for_a_strategy_block():
    """Without this, state.strategy is always empty and --include-strategy silently shows code."""
    q = _prompt()
    assert "<strategy>" in q and "</strategy>" in q


# ---- the two hardware variants -------------------------------------------------------------------
def test_a100_prompt_names_the_a100_and_its_shared_memory_ceiling():
    """The A100 variant exists because of one observed failure mode: an H100-legal block size asking
    for 393216 bytes of shared memory against the A100's 166912. Naming the number is the fix."""
    q = _prompt(TrimulA100Env)
    assert "A100 80GB (sm80)" in q
    assert "166912" in q
    assert "run on an H100" not in q


def test_h100_prompt_is_the_verbatim_upstream_line():
    q = _prompt(TrimulH100Env)
    assert _HW_RULE_H100 in q
    assert "166912" not in q


def test_the_variants_differ_only_in_the_hardware_rule():
    a, h = _prompt(TrimulA100Env), _prompt(TrimulH100Env)
    assert a.replace(_HW_RULE_A100, _HW_RULE_H100) == h


def test_each_problem_defaults_to_its_own_card(monkeypatch):
    from envs.kernel_trimul import _eval_settings
    monkeypatch.delenv("TRIMUL_EVAL_GPU", raising=False)
    assert _eval_settings("trimul_a100")["gpu"] == "1"   # guadiana: GPU 0 holds the vLLM server
    assert _eval_settings("trimul_h100")["gpu"] == "0"
    monkeypatch.setenv("TRIMUL_EVAL_GPU", "3")
    assert _eval_settings("trimul_a100")["gpu"] == "3"   # the env var always wins


def test_both_problems_grade_against_the_same_task_file(monkeypatch):
    """Different prompts, one task.yml -- otherwise the scores would not be comparable at all."""
    seen = []

    def run(cmd, **kwargs):
        seen.append(cmd[cmd.index("--task") + 1])
        out = cmd[cmd.index("--json") + 1]
        with open(out, "w") as f:
            json.dump([OK], f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    _reward("trimul_a100").get_reward("code", _state())
    _reward("trimul_h100").get_reward("code", _state())
    assert seen == ["trimul", "trimul"]


# ---- the cross-process half of the guard ---------------------------------------------------------
def test_the_gpu_guard_excludes_other_processes(tmp_path, monkeypatch):
    """A sweep runs each run as its own PROCESS, so the threading.Lock alone would let two runs time
    a contended card with nothing in the logs. This asserts the flock, not the thread lock: the
    'other process' here is a raw fd holding the same lock file.
    """
    import fcntl
    import os
    from envs.kernel_trimul import _gpu_guard

    monkeypatch.setenv("TRIMUL_LOCK_DIR", str(tmp_path))
    lock_file = tmp_path / "learning_evolve-trimul-gpu9.lock"

    with _gpu_guard("9"):
        assert lock_file.exists()
        fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            with pytest.raises(BlockingIOError):        # someone else holds it
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    # released on exit, so the next run can grade
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_guards_on_different_cards_do_not_block_each_other(tmp_path, monkeypatch):
    from envs.kernel_trimul import _gpu_guard
    monkeypatch.setenv("TRIMUL_LOCK_DIR", str(tmp_path))
    # Nested: if the flock were keyed globally rather than per GPU, this would deadlock the test.
    with _gpu_guard("0"):
        pass
    with _gpu_guard("1"):
        pass
    assert (tmp_path / "learning_evolve-trimul-gpu0.lock").exists()
    assert (tmp_path / "learning_evolve-trimul-gpu1.lock").exists()

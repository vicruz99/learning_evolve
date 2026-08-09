"""What a run does when the model server (or the grader) goes away mid-sweep.

The behaviour these pin down replaces the old one: ``icl.loop._run_group`` caught any exception,
recorded a parent group with no candidates, and let the loop walk on. The run then finished all its
generations, but every generation after the failure was conditioned on a buffer and a context pool
missing that group's children — the configured experiment on paper, a different one in fact — and
``results.resume`` refused to trust anything past it anyway, so a later ``--resume`` threw the whole
run away.

Now: wait the server out, and if it does not come back, stop at the last COMPLETE generation, leaving
a run dir that resumes exactly there.
"""
import asyncio
import dataclasses
import json
import os

import pytest


@pytest.fixture
def no_sleep(monkeypatch):
    """Collapse the retry backoff so the waiting tests run in milliseconds. Binds the real sleep
    first — patching ``asyncio.sleep`` with something that calls ``asyncio.sleep`` recurses."""
    real = asyncio.sleep

    async def instant(_seconds):
        await real(0)

    monkeypatch.setattr(asyncio, "sleep", instant)

from generation import GenResult
from icl.config import ICLConfig
from icl.loop import GenerationAborted, ICLRunner, LLMUnavailable
from results.resume import inspect_run


class _Boom(Exception):
    """A transport-level failure: no status_code, so it is transient and worth waiting on."""


class _Rejected(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeLLM:
    """Stands in for VLLMClient. ``script`` is consulted per generate() call: an exception instance
    is raised, anything else yields completions."""

    def __init__(self, script=None, healthy: bool = False):
        self.script = list(script or [])
        self.calls = 0
        self.healthy = healthy

    async def generate(self, prompt, n, temperature=1.0, max_tokens=100) -> GenResult:
        self.calls += 1
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, BaseException):
                raise step
        return GenResult(texts=[f"```text\nseeee {i}\n```" for i in range(n)],
                         reasonings=[""] * n, finish_reasons=["stop"] * n,
                         prompt_tokens=10, completion_tokens=10 * n)

    async def health(self) -> bool:
        return self.healthy

    async def cache_counters(self) -> dict:
        return {}

    async def count_tokens(self, texts) -> list:
        return []


def _cfg(tmp_path, **kw) -> ICLConfig:
    base = dict(problem="toy", log_path=str(tmp_path / "run"), groups_per_batch=2, group_size=2,
                num_generations=3, n_context=2, max_tokens=64, memory_stop_fraction=0,
                save_completions=False, save_reasoning=False, llm_max_wait=1.0)
    base.update(kw)
    return ICLConfig(**base)


def _runner(cfg, llm) -> ICLRunner:
    r = ICLRunner(cfg)
    r.llm = llm
    return r


# ---- waiting the server out ----------------------------------------------------------------------
def test_a_transient_failure_is_waited_out_rather_than_lost(tmp_path, no_sleep):
    """The common case on a cluster: the server's own job is requeued and comes back a while later.
    The generation must not advance in the meantime, and nothing may be recorded as damaged."""
    llm = FakeLLM(script=[_Boom("connection refused"), _Boom("connection refused")])
    runner = _runner(_cfg(tmp_path, llm_max_wait=0), llm)                # 0 = wait indefinitely

    asyncio.run(runner.run())

    assert llm.calls == 8            # 2 failures + 6 real requests (2 parents x 3 generations)
    prog = inspect_run(str(tmp_path / "run"), 3)
    assert prog.good_generations == 3 and prog.complete
    assert prog.damage == []


def test_a_rejected_request_is_not_waited_on(tmp_path):
    """A 404 for a model the server does not serve will not fix itself; retrying it for an hour only
    delays finding out."""
    llm = FakeLLM(script=[_Rejected(404)])
    # One parent, so the assertion below is about retries and not about the sibling group that would
    # otherwise have been generating concurrently.
    runner = _runner(_cfg(tmp_path, llm_max_wait=0, groups_per_batch=1), llm)

    with pytest.raises(LLMUnavailable, match="retrying will not change that"):
        asyncio.run(runner.run())
    assert llm.calls == 1


def test_giving_up_stops_the_run_instead_of_recording_an_empty_group(tmp_path, no_sleep):
    llm = FakeLLM(script=[_Boom("gone")] * 50)
    runner = _runner(_cfg(tmp_path, llm_max_wait=1.0), llm)

    with pytest.raises(LLMUnavailable, match="Stopping at the last complete generation"):
        asyncio.run(runner.run())


# ---- what the abandoned generation leaves on disk ------------------------------------------------
def test_a_lost_generation_is_abandoned_and_the_run_resumes_at_the_previous_one(tmp_path, no_sleep):
    """The point of the whole change: the run stops, and it stops somewhere resumable.

    Generations 0 and 1 complete; generation 2 loses its first parent group. The old code recorded
    that group empty and finished the run; results.resume then capped the run at generation 2 and a
    --resume discarded all three.
    """
    run_dir = str(tmp_path / "run")
    # 2 parents x 3 generations = 6 requests; kill the 5th (generation 2, first parent).
    script = [None] * 4 + [_Boom("server died")] * 50
    llm = FakeLLM(script=script)
    runner = _runner(_cfg(tmp_path, llm_max_wait=1.0), llm)

    with pytest.raises(GenerationAborted):
        asyncio.run(runner.run())

    # No meta.json for the abandoned generation -> resume reads it as "never finished".
    assert not os.path.exists(os.path.join(run_dir, "generations", "gen_0002", "meta.json"))
    prog = inspect_run(run_dir, 3)
    assert prog.good_generations == 2
    assert prog.resume_step == 2                     # the buffer snapshot for step 2 exists
    assert not prog.complete
    # And the damage report is about an unfinished generation, NOT about groups with no candidates:
    # nothing was recorded as a complete-but-hollow generation.
    assert any("never finished" in d for d in prog.damage)
    assert not any("recorded no candidates" in d for d in prog.damage)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["status"] == "aborted"
    assert len(summary["per_generation"]) == 2


def test_a_grading_failure_is_reported_as_grading_not_as_an_unreachable_llm(tmp_path):
    """A dead Ray worker used to surface as 'the LLM was unreachable for them'. It is a different
    fault with a different fix, and re-generating on it would pay for the decode twice."""
    runner = _runner(_cfg(tmp_path, llm_max_wait=1.0), FakeLLM())

    class BrokenGrading(runner.spec.env_type):
        async def rollout_step(self, completion_text, step_idx):
            raise RuntimeError("Ray worker died")

    # dataclasses.replace, not a write through the frozen instance: runner.spec IS the registry's
    # ProblemSpec object, and mutating it would break every later test in the session.
    runner.spec = dataclasses.replace(runner.spec, env_type=BrokenGrading)

    with pytest.raises(GenerationAborted, match="grading failed after the model had answered"):
        asyncio.run(runner.run())
    assert runner.llm.calls == 2                     # generated once per parent, never retried


# ---- an evaluator that grades nothing valid ------------------------------------------------------
class EmptyAnswerLLM(FakeLLM):
    """Answers every request, but nothing it returns ever grades valid — the shape a broken sandbox
    produces: candidates come back, and every one of them fails."""

    async def generate(self, prompt, n, temperature=1.0, max_tokens=100) -> GenResult:
        self.calls += 1
        return GenResult(texts=["```text\n\n```"] * n, reasonings=[""] * n,
                         finish_reasons=["stop"] * n, prompt_tokens=10, completion_tokens=10 * n)


def test_a_run_that_grades_nothing_valid_stops_instead_of_finishing(tmp_path):
    """The demonstrated hole: such generations are structurally perfect (full groups, full complement
    of children, all invalid), so a run of them used to verify as COMPLETE and --resume skipped it —
    a sweep handing back green runs that hold nothing."""
    run_dir = str(tmp_path / "run")
    runner = _runner(_cfg(tmp_path, num_generations=10, max_empty_generations=3), EmptyAnswerLLM())

    asyncio.run(runner.run())                       # a clean stop, not an exception

    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["status"] == "stopped_no_yield"
    assert len(summary["per_generation"]) == 3      # stopped at the threshold, not at generation 10
    assert summary["totals"]["succeeded"] == 0

    prog = inspect_run(run_dir, 10)
    assert not prog.complete                        # ...so --resume will not skip it
    assert any("no result whatever its summary says" in d for d in prog.damage)


def test_the_no_yield_stop_can_be_turned_off(tmp_path):
    runner = _runner(_cfg(tmp_path, num_generations=2, max_empty_generations=0), EmptyAnswerLLM())
    asyncio.run(runner.run())
    summary = json.load(open(os.path.join(tmp_path / "run", "summary.json")))
    assert summary["status"] == "complete"          # the run's own claim...
    # ...which results.resume still refuses to believe, because the run holds nothing.
    assert not inspect_run(str(tmp_path / "run"), 2).complete


def test_a_barren_streak_is_forgiven_if_the_run_recovers(tmp_path):
    """An early generation can legitimately come back empty before the buffer has anything good in
    it. Only a SUSTAINED streak is an evaluator fault."""
    class SometimesLLM(EmptyAnswerLLM):
        async def generate(self, prompt, n, temperature=1.0, max_tokens=100):
            self.calls += 1
            if self.calls <= 4:                     # generations 0 and 1 (2 parents each) yield nothing
                return await EmptyAnswerLLM.generate(self, prompt, n, temperature, max_tokens)
            return await FakeLLM.generate(self, prompt, n, temperature, max_tokens)

    runner = _runner(_cfg(tmp_path, num_generations=4, max_empty_generations=3), SometimesLLM())
    asyncio.run(runner.run())

    summary = json.load(open(os.path.join(tmp_path / "run", "summary.json")))
    assert summary["status"] == "complete"
    assert len(summary["per_generation"]) == 4
    assert inspect_run(str(tmp_path / "run"), 4).complete

"""End-to-end ICL loop smoke test with a STUBBED LLM (no vLLM server needed).

Exercises the full plumbing: PUCT sample -> prompt + n-best context block -> grade in the ray
sandbox -> buffer update/flush -> best-so-far logging. Generation is replaced by a canned valid
solution so this runs offline. Run directly:

    python tests/smoke_icl.py
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import tempfile

from icl.config import ICLConfig
from icl.loop import ICLRunner
from context import select_best_n, build_context_block
from generation.vllm_client import GenResult
from results.resume import inspect_run


class _StubLLM:
    """Returns canned valid ac2 solutions, each a DISTINCT construction (so the buffer's dedup
    doesn't reject them): the seed sequence plus a unique number of small positive elements.

    Stands in for ``VLLMClient``, so it has to answer the same two calls the loop makes per
    generation: ``generate`` (a GenResult, not a list of strings) and ``cache_counters``.
    """
    def __init__(self, counter: int = 0):
        self.prompts_seen: list[str] = []
        self._counter = counter

    async def cache_counters(self):
        return {}                            # no server to scrape; the loop treats {} as "unknown"

    async def count_tokens(self, texts):
        return []                            # no /tokenize; the loop falls back to its chars estimate

    async def generate(self, prompt, n, temperature, max_tokens):
        self.prompts_seen.append(prompt)
        outs = []
        for _ in range(n):
            self._counter += 1
            k = self._counter
            outs.append(
                "```python\n"
                "def construct_function():\n"
                f"    return list(height_sequence_1) + [0.001] * {k}\n"
                "```"
            )
        return GenResult(texts=outs, reasonings=[""] * n, finish_reasons=["stop"] * n,
                         prompt_tokens=100, completion_tokens=20 * n, latency=0.01)


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        cfg = ICLConfig(
            problem="ac2",
            log_path=td,
            groups_per_batch=1,
            group_size=2,
            num_generations=2,
            n_context=3,
        )
        runner = ICLRunner(cfg)
        stub = _StubLLM()
        runner.llm = stub
        await runner.run()

        # --- buffer snapshot now lives under buffer/ ---
        final = os.path.join(td, "buffer", "puct_sampler_step_000002.json")
        assert os.path.exists(final), f"missing {final}"
        n_children = sum(1 for s in runner.sampler._states if s.timestep >= 0)
        print(f"buffer size={len(runner.sampler._states)} children={n_children} T={runner.sampler._T}")
        assert runner.sampler._T >= 1 and n_children >= 1, "no valid children entered the buffer"

        # --- results-tracking artifacts ---
        for name in ("config.json", "summary.json", "progress.csv", "events.jsonl"):
            assert os.path.exists(os.path.join(td, name)), f"missing {name}"
        assert glob.glob(os.path.join(td, "solutions", "sol_*.py")), "no solution .py files written"
        assert os.path.exists(os.path.join(td, "solutions", "manifest.jsonl"))
        assert os.path.exists(os.path.join(td, "generations", "gen_0000", "parent_00", "prompt.txt"))
        assert glob.glob(os.path.join(td, "generations", "gen_0000", "parent_00", "child_*.txt")), \
            "completions not saved"
        assert os.path.exists(os.path.join(td, "generations", "gen_0000", "meta.json"))

        summ = json.load(open(os.path.join(td, "summary.json")))
        assert summ["status"] == "complete"
        assert summ["totals"]["succeeded"] >= 1
        assert summ["best"] is not None
        n_sol = len(glob.glob(os.path.join(td, "solutions", "sol_*.py")))
        print(f"solutions={n_sol} best_score={summ['best']['score']:.6f} "
              f"succeeded={summ['totals']['succeeded']} failed={summ['totals']['failed']}")

        # From generation 2, the prompt should carry the context block (gen 1 produced children).
        assert any("Best solutions found so far" in p for p in stub.prompts_seen), \
            "context block never injected"

        # The standalone context builder renders selected states.
        block = build_context_block(select_best_n(runner.sampler._states, 3),
                                    metric_name="lower bound", maximize=True)
        assert "lower bound" in block
        print("\nICL loop smoke test passed.")

        await _resume_leg(td, summ)


async def _resume_leg(run_dir: str, before: dict) -> None:
    """Continue the finished run for two more generations via --resume-step, the way run_sweep does.

    This is the leg that used to lose data: the tracker built empty books, so summary.json came back
    describing only the generations done after the resume (and still said `complete`), which then made
    --resume rewind to that wrong point.
    """
    prog = inspect_run(run_dir)
    assert prog.complete and prog.good_generations == 2 and prog.resume_step == 2, prog
    assert prog.damage == [], prog.damage

    cfg = ICLConfig(problem="ac2", log_path=run_dir, groups_per_batch=1, group_size=2,
                    num_generations=4, n_context=3, resume_step=2)
    runner = ICLRunner(cfg)
    runner.llm = _StubLLM(counter=1000)          # distinct constructions from the first leg
    await runner.run()

    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    assert [g["generation"] for g in summ["per_generation"]] == [0, 1, 2, 3], summ["per_generation"]
    assert summ["totals"]["candidates"] == before["totals"]["candidates"] + 4
    assert summ["totals"]["succeeded"] >= before["totals"]["succeeded"]
    assert summ["started_at"] == before["started_at"], "a resume must not restart the run's clock"
    assert summ["resumes"] and summ["resumes"][-1]["from_generation"] == 2
    # every solution file is still there: numbering continued instead of overwriting sol_000001
    n_sol = len(glob.glob(os.path.join(run_dir, "solutions", "sol_*.py")))
    assert n_sol == summ["totals"]["unique_solutions"] == sum(
        1 for _ in open(os.path.join(run_dir, "solutions", "manifest.jsonl"))), n_sol
    assert len({json.loads(l)["sol"] for l in
                open(os.path.join(run_dir, "solutions", "manifest.jsonl"))}) == n_sol
    rows = open(os.path.join(run_dir, "progress.csv")).read().strip().splitlines()
    assert [r.split(",")[0] for r in rows[1:]] == ["0", "1", "2", "3"], rows

    prog = inspect_run(run_dir)
    assert prog.complete and prog.good_generations == 4, prog
    assert prog.damage == [], prog.damage
    print(f"resume leg passed: {prog.good_generations} generations, {n_sol} solutions, "
          f"one cumulative summary.")


if __name__ == "__main__":
    asyncio.run(main())

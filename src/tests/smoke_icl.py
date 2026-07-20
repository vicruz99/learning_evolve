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


class _StubLLM:
    """Returns canned valid ac2 solutions, each a DISTINCT construction (so the buffer's dedup
    doesn't reject them): the seed sequence plus a unique number of small positive elements."""
    def __init__(self):
        self.prompts_seen: list[str] = []
        self._counter = 0

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
        return outs


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


if __name__ == "__main__":
    asyncio.run(main())

"""Eval smoke test (needs ray). Grades one trivial valid solution per problem through the real
Environment -> SandboxRewardEvaluator path. Run directly:

    python tests/smoke_eval.py
"""
from __future__ import annotations

import asyncio
import tempfile

from envs import EnvConfig, get_problem
from sandbox import init_ray


class _DummySampler:
    """Environment.__init__ requires a non-None sampler; grading never touches it."""
    def update_states(self, *a, **k):
        pass

    def record_failed_rollout(self, *a, **k):
        pass


# Trivial-but-valid solutions per problem (entrypoint name matches each evaluator).
TRIVIAL = {
    "erdos": '''```python
def run(seed=42, budget_s=1000, **kwargs):
    import numpy as np
    n = 50
    h = np.ones(n) * 0.5
    dx = 2.0 / n
    c5 = float(np.max(np.correlate(h, 1 - h, mode="full") * dx))
    return h, c5, n
```''',
    "circle_packing_26": '''```python
def run_packing():
    import numpy as np
    centers, radii = [], []
    r = 0.05
    for k in range(26):
        i, j = k % 6, k // 6
        centers.append([(i + 0.5) / 6.0, (j + 0.5) / 6.0])
        radii.append(r)
    return np.array(centers), np.array(radii), float(np.sum(radii))
```''',
    "ac1": '''```python
def propose_candidate(seed=42, budget_s=1000, **kwargs):
    return list(height_sequence_1)
```''',
    "ac2": '''```python
def construct_function():
    return list(height_sequence_1)
```''',
}


async def _grade_one(name: str, code: str) -> None:
    spec = get_problem(name)
    initial = spec.env_type.create_initial_state(spec.problem_type)
    with tempfile.TemporaryDirectory() as td:
        cfg = EnvConfig(
            problem_type=spec.problem_type,
            log_path=td,
            num_cpus_per_task=spec.num_cpus_per_task,
            eval_timeout=120,
            timeout=180.0,
        )
        env = spec.env_type(initial_state=initial, sampler=_DummySampler(), config=cfg)
        outs = await env.check_answer(code, step=0)
        status = "OK" if outs.correctness > 0 else "FAIL"
        print(f"[{name}] {status}  correctness={outs.correctness}  raw_score={outs.raw_score}  msg={outs.msg[:120]}")
        assert outs.correctness > 0, f"{name} trivial solution failed: {outs.msg}"


async def main() -> None:
    init_ray(2)
    for name, code in TRIVIAL.items():
        await _grade_one(name, code)
    print("\nAll eval smoke checks passed.")


if __name__ == "__main__":
    asyncio.run(main())

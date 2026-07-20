"""Unit tests for ExperimentTracker (pure; no ray/server)."""
import json
import os
import glob

from puct import State
from envs.base import RolloutResult
from results import ExperimentTracker


class _Spec:
    metric_name = "score"
    maximize = True
    entrypoint = "run"


class _SpecMin:
    metric_name = "bound"
    maximize = False
    entrypoint = "run"


class _Sampler:
    def __init__(self):
        self._t = 0

    def get_sample_stats(self):
        self._t += 1
        return {"puct/buffer_size": self._t + 1, "puct/T": self._t}


def _valid(raw, gen, maximize=True):
    value = raw if maximize else -raw
    st = State(timestep=gen, construction=[raw], code=f"```python\ndef run():\n    return {raw}\n```", value=value)
    return RolloutResult(reward=raw, correctness=1.0, raw_score=raw, msg="ok",
                         parsed_code=st.code, correct_format=True, next_state=st)


def _fail():
    return RolloutResult(reward=0.0, correctness=0.0, raw_score=0.0, msg="bad code",
                         parsed_code="", correct_format=False, next_state=None)


def _cfg():
    return {"problem": "toy", "context_strategy": "best", "n_context": 4,
            "group_size": 2, "groups_per_batch": 1, "num_generations": 2}


def test_tracker_tree_and_summary(tmp_path):
    run_dir = tmp_path / "run"
    tr = ExperimentTracker(str(run_dir), _cfg(), _Spec(), save_completions=True)
    sampler = _Sampler()
    seed = State(timestep=-1, construction=None, code="", value=None)

    tr.start_generation(0, [seed])
    tr.record_group(0, 0, seed, "PROMPT0", ["c0", "c1"], [_valid(1.0, 0), _fail()])
    tr.end_generation(0, sampler)

    tr.start_generation(1, [seed])
    tr.record_group(1, 0, seed, "PROMPT1", ["c0", "c1"], [_valid(2.0, 1), _valid(1.5, 1)])
    tr.end_generation(1, sampler)
    tr.close()

    # top-level files
    for name in ("config.json", "summary.json", "progress.csv", "events.jsonl"):
        assert (run_dir / name).exists(), name

    # one .py per valid candidate (1 + 2), manifest & events line counts
    assert len(glob.glob(str(run_dir / "solutions" / "sol_*.py"))) == 3
    assert sum(1 for _ in open(run_dir / "solutions" / "manifest.jsonl")) == 3
    assert sum(1 for _ in open(run_dir / "events.jsonl")) == 4  # 2 candidates x 2 gens

    # nested per-generation artifacts
    assert (run_dir / "generations" / "gen_0000" / "parent_00" / "prompt.txt").exists()
    assert (run_dir / "generations" / "gen_0000" / "parent_00" / "child_00.txt").exists()
    assert (run_dir / "generations" / "gen_0000" / "meta.json").exists()

    # summary correctness
    summ = json.load(open(run_dir / "summary.json"))
    assert summ["status"] == "complete"
    assert summ["totals"] == {**summ["totals"], "candidates": 4, "succeeded": 3, "failed": 1}
    assert summ["best"]["score"] == 2.0
    assert summ["worst_valid"]["score"] == 1.0
    assert len(summ["per_generation"]) == 2

    # progress.csv: header + 2 rows, with the self-descriptive column names
    lines = open(run_dir / "progress.csv").read().strip().splitlines()
    assert len(lines) == 3
    header = lines[0].split(",")
    assert {"puct_expansions", "gen_best_score", "best_so_far_score",
            "valid_candidates", "failed_candidates"} <= set(header)

    # cross-experiment index at the parent dir
    assert (tmp_path / "index.csv").exists()


def test_tracker_minimize_direction(tmp_path):
    """For a minimize problem, 'best' must be the LOWEST native score."""
    run_dir = tmp_path / "run_min"
    tr = ExperimentTracker(str(run_dir), _cfg(), _SpecMin(), save_completions=False)
    sampler = _Sampler()
    seed = State(timestep=-1, construction=None, code="", value=None)

    tr.start_generation(0, [seed])
    tr.record_group(0, 0, seed, "P", ["a", "b"], [_valid(0.5, 0, maximize=False), _valid(0.3, 0, maximize=False)])
    tr.end_generation(0, sampler)
    tr.close()

    summ = json.load(open(run_dir / "summary.json"))
    assert summ["best"]["score"] == 0.3        # lower bound is better
    assert summ["worst_valid"]["score"] == 0.5
    # save_completions=False => no child_*.txt
    assert not glob.glob(str(run_dir / "generations" / "gen_0000" / "parent_00" / "child_*.txt"))

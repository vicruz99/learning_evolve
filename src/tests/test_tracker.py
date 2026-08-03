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


def test_a_resumed_tracker_continues_the_run_instead_of_restarting_its_books(tmp_path):
    """A resumed run used to open a NEW set of books: per_generation started empty, so summary.json was
    rewritten with only the post-resume generations (a 3-generation summary for a 15-generation run,
    then 'status: complete'), totals/best reset, and solution ids restarted at sol_000001 — silently
    overwriting the .py files of the generations that were kept."""
    run_dir = tmp_path / "run"
    cfg = {**_cfg(), "num_generations": 3}
    sampler = _Sampler()
    seed = State(timestep=-1, construction=None, code="", value=None)

    tr = ExperimentTracker(str(run_dir), cfg, _Spec())
    for gen in (0, 1):
        tr.start_generation(gen, [seed])
        tr.record_group(gen, 0, seed, f"P{gen}", ["c0", "c1"], [_valid(1.0 + gen, gen), _fail()])
        tr.end_generation(gen, sampler)
    tr.close(status="failed")                       # killed after two generations
    first_started_at = json.load(open(run_dir / "summary.json"))["started_at"]

    tr = ExperimentTracker(str(run_dir), cfg, _Spec(), resume_step=2)
    tr.start_generation(2, [seed])
    tr.record_group(2, 0, seed, "P2", ["c0", "c1"], [_valid(5.0, 2), _valid(0.5, 2)])
    tr.end_generation(2, sampler)
    tr.close()

    summ = json.load(open(run_dir / "summary.json"))
    assert [g["generation"] for g in summ["per_generation"]] == [0, 1, 2]
    assert summ["totals"]["candidates"] == 6 and summ["totals"]["succeeded"] == 4
    assert summ["best"]["score"] == 5.0             # from the resumed generation
    assert summ["worst_valid"]["score"] == 0.5
    assert summ["started_at"] == first_started_at   # one run, not a new one
    assert summ["resumes"] == [{**summ["resumes"][0], "from_generation": 2}]

    # the resumed generation's solutions keep numbering where the run left off
    sols = sorted(os.path.basename(p) for p in glob.glob(str(run_dir / "solutions" / "sol_*.py")))
    assert sols == ["sol_000001.py", "sol_000002.py", "sol_000003.py", "sol_000004.py"]
    manifest = [json.loads(l) for l in open(run_dir / "solutions" / "manifest.jsonl")]
    assert len({m["sol"] for m in manifest}) == len(manifest) == 4
    assert sum(1 for _ in open(run_dir / "events.jsonl")) == 6

    # the per-generation meta.json files (what a later resume rebuilds from) agree with that summary
    from results.resume import prior_state
    prior = prior_state(str(run_dir), 3)
    assert [g["generation"] for g in prior.per_generation] == [0, 1, 2]
    assert prior.candidates == 6 and prior.succeeded == 4 and prior.sol_seq == 4

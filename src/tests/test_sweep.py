"""Unit tests for the sweep launcher's parsing/validation (pure; launches nothing)."""
import glob
import json
import os
import shutil

import pytest
import yaml

import run_sweep
from run_sweep import SweepError, _expand, _flag_tables, _state, _to_argv, build_specs
from tests.test_resume import _write_run


def _write(tmp_path, doc: dict, name: str = "s.yaml") -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return str(path)


# ---- grid expansion ------------------------------------------------------------------------------
def test_grid_cross_product_and_names():
    runs = _expand({
        "common": {"problem": "toy", "n-context": 4},
        "grid": {"context-strategy": ["best", "random"], "n-context": [10, 20]},
    })
    assert len(runs) == 4
    assert [r["name"] for r in runs] == [
        "toy_cs-best_nc-10", "toy_cs-best_nc-20", "toy_cs-random_nc-10", "toy_cs-random_nc-20",
    ]
    # grid overrides common
    assert runs[0]["flags"]["n-context"] == 10


def test_single_value_axis_is_omitted_from_the_name():
    runs = _expand({"common": {"problem": "toy"},
                    "grid": {"context-strategy": ["best", "random"], "n-context": [20]}})
    assert [r["name"] for r in runs] == ["toy_cs-best", "toy_cs-random"]
    assert all(r["flags"]["n-context"] == 20 for r in runs)


def test_explicit_runs_override_common_and_keep_their_name():
    runs = _expand({"common": {"problem": "toy", "n-context": 4},
                    "runs": [{"name": "mine", "n-context": 30}]})
    assert runs == [{"name": "mine", "flags": {"problem": "toy", "n-context": 30}}]


def test_grid_and_runs_combine():
    runs = _expand({"common": {"problem": "toy"},
                    "grid": {"context-strategy": ["best", "random"]},
                    "runs": [{"name": "extra", "context-strategy": "recent"}]})
    assert [r["name"] for r in runs] == ["toy_cs-best", "toy_cs-random", "extra"]


def test_empty_sweep_and_duplicate_names_are_errors():
    with pytest.raises(SweepError, match="neither"):
        _expand({"common": {"problem": "toy"}})
    with pytest.raises(SweepError, match="duplicate run names"):
        _expand({"runs": [{"name": "x"}, {"name": "x"}]})
    with pytest.raises(SweepError, match="non-empty list"):
        _expand({"grid": {"context-strategy": "best"}})


# ---- flag validation -----------------------------------------------------------------------------
def test_unknown_flag_is_rejected_with_a_suggestion():
    by_flag, bool_pair = _flag_tables()
    with pytest.raises(SweepError, match=r"did you mean: n-context"):
        _to_argv({"n-contexts": 4}, by_flag, bool_pair)


def test_launcher_owned_flags_are_rejected():
    by_flag, bool_pair = _flag_tables()
    for reserved in ("log-path", "resume-step"):
        with pytest.raises(SweepError, match="owned by the launcher"):
            _to_argv({reserved: "x"}, by_flag, bool_pair)


def test_bools_map_to_the_right_polarity_flag():
    by_flag, bool_pair = _flag_tables()
    assert _to_argv({"include-code": False}, by_flag, bool_pair) == ["--no-include-code"]
    assert _to_argv({"save-reasoning": True}, by_flag, bool_pair) == ["--save-reasoning"]
    # a store_true with no paired negative cannot express False, and says so
    with pytest.raises(SweepError, match="no negative flag"):
        _to_argv({"include-strategy": False}, by_flag, bool_pair)
    # non-bool value for a bool flag
    with pytest.raises(SweepError, match="boolean flag"):
        _to_argv({"include-code": "yes"}, by_flag, bool_pair)


def test_value_flags_pass_through_as_strings():
    by_flag, bool_pair = _flag_tables()
    assert _to_argv({"n-context": 20}, by_flag, bool_pair) == ["--n-context", "20"]


# ---- sweep file -> manifest ----------------------------------------------------------------------
def test_build_specs_assigns_log_paths_and_rejects_unknown_sections(tmp_path):
    path = _write(tmp_path, {"sweep": {"name": "sw"}, "common": {"problem": "toy"},
                             "grid": {"context-strategy": ["best"]}})
    sweep_dir, settings, manifest = build_specs(path, str(tmp_path / "out"))
    entry = manifest["entries"][0]
    assert entry["log_path"] == os.path.join(str(tmp_path / "out"), "toy_cs-best")
    assert entry["cmd"][-2:] == ["--log-path", entry["log_path"]]
    assert "run_icl.py" in entry["cmd"][1]

    bad = _write(tmp_path, {"commons": {"problem": "toy"}}, "bad.yaml")
    with pytest.raises(SweepError, match="unknown top-level key"):
        build_specs(bad, None)
    bad2 = _write(tmp_path, {"sweep": {"nmae": "typo"}, "grid": {"n-context": [1]}}, "bad2.yaml")
    with pytest.raises(SweepError, match=r"unknown key\(s\) in `sweep`"):
        build_specs(bad2, None)


def test_sweep_dir_defaults_to_the_file_stem(tmp_path):
    path = _write(tmp_path, {"common": {"problem": "toy"}, "grid": {"n-context": [1]}}, "mysweep.yaml")
    sweep_dir, _settings, manifest = build_specs(path, None)
    assert sweep_dir == os.path.join("runs", "mysweep")
    assert manifest["name"] == "mysweep"


# ---- state reconciliation ------------------------------------------------------------------------
def test_state_distinguishes_died_from_complete_and_pending():
    complete = {"gens": 3, "best": 1.0, "status": "complete", "complete": True, "wall": 1,
                "tok_s": 1, "updated": None}
    partial = {**complete, "status": "running", "complete": False, "gens": 1}
    # a pid that is gone while the run never reported completion is the case that was invisible before
    assert _state({"pid": 999_999_999}, partial) == "DIED"
    assert _state({"pid": 999_999_999}, complete) == "complete"
    assert _state({"pid": None}, partial) == "pending"
    assert _state({"pid": 999_999_999, "returncode": 1}, partial) == "exit 1"
    assert _state({"pid": os.getpid()}, partial) == "running"


def test_state_flags_a_complete_summary_its_files_do_not_back():
    """summary.json says the run finished, its generations are gone: the state that used to read
    'complete' (and made --resume skip the run) has to be visible."""
    hollow = {"gens": 3, "status": "complete", "complete": False, "best": None, "wall": None,
              "tok_s": None, "updated": None}
    assert _state({"pid": None}, hollow) == "DAMAGED"
    assert _state({"pid": 999_999_999}, hollow) == "DAMAGED"


def test_progress_reads_wall_and_tokens_from_summary(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "status": "running",
        "best": {"score": 2.5},
        "updated_at": "2026-07-26T00:00:00",
        "per_generation": [{"wall_seconds": 100.0, "usage": {"completion_tokens": 50_000}}],
    }))
    prog = run_sweep._run_progress(str(run_dir))
    assert prog["gens"] == 1 and prog["best"] == 2.5
    assert prog["wall"] == 100.0 and prog["tok_s"] == 500


def test_progress_tolerates_a_missing_or_half_written_summary(tmp_path):
    assert run_sweep._run_progress(str(tmp_path / "nope"))["gens"] == 0
    run_dir = tmp_path / "half"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text('{"status": "runn')
    assert run_sweep._run_progress(str(run_dir))["status"] is None


def test_progress_separates_what_the_summary_claims_from_what_is_verifiable(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, snapshot_steps=[2, 3], drop_group_in=2)
    prog = run_sweep._run_progress(run_dir, 5)
    assert prog["gens"] == 3                     # summary.json's own account
    assert prog["good"] == 2 and prog["resume_step"] == 2 and prog["complete"] is False


# ---- resume planning -----------------------------------------------------------------------------
def _entry(name: str, run_dir: str, want: int = 5, cmd_extra: list | None = None) -> dict:
    return {"name": name, "log_path": run_dir, "num_generations": want, "pid": 4242, "returncode": 0,
            "cmd": ["python", "run_icl.py", "--log-path", run_dir] + (cmd_extra or [])}


def test_resume_restarts_an_incomplete_run_from_its_first_generation(tmp_path, capsys):
    """A sweep resume is whole-run granular: partial runs start over rather than being continued, so
    no run's generations are a mixture of two processes with the interruption buried inside."""
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed", snapshot_steps=[2, 3])
    manifest = {"entries": [_entry("r", run_dir)]}
    run_sweep.plan_resume(manifest)
    assert "--resume-step" not in manifest["entries"][0]["cmd"]
    assert manifest["entries"][0]["pid"] is None
    # everything the interrupted attempt produced is set aside, not appended to
    assert os.listdir(os.path.join(run_dir, "generations")) == []
    assert sum(1 for _ in open(os.path.join(run_dir, "events.jsonl"))) == 0
    assert sum(1 for _ in open(os.path.join(run_dir, "buffer", "context_pool.jsonl"))) == 0
    assert not glob.glob(os.path.join(run_dir, "solutions", "sol_*.py"))
    stale = [d for d in os.listdir(run_dir) if d.startswith("stale_")]
    assert len(stale) == 1
    assert sorted(os.listdir(os.path.join(run_dir, stale[0], "generations"))) == [
        "gen_0000", "gen_0001", "gen_0002"]
    out = capsys.readouterr().out
    assert "restarting from generation 0 (discarding 3 verified generation(s))" in out


def test_resume_redoes_a_run_whose_data_was_deleted_under_a_complete_summary(tmp_path, capsys):
    """The reported bug: the folders were deleted, --resume called the runs complete and skipped them."""
    run_dir = _write_run(tmp_path / "r", 5, want=5)
    shutil.rmtree(os.path.join(run_dir, "generations"))
    shutil.rmtree(os.path.join(run_dir, "buffer"))
    manifest = {"entries": [_entry("r", run_dir)]}
    run_sweep.plan_resume(manifest)
    assert "--resume-step" not in manifest["entries"][0]["cmd"]
    assert "restarting from generation 0" in capsys.readouterr().out
    assert run_sweep._run_progress(run_dir, 5)["complete"] is False


def test_resume_leaves_a_verifiably_complete_run_alone(tmp_path, capsys):
    run_dir = _write_run(tmp_path / "r", 5, want=5)
    manifest = {"entries": [_entry("r", run_dir)]}
    run_sweep.plan_resume(manifest)
    assert "--resume-step" not in manifest["entries"][0]["cmd"]
    assert "complete (5/5) — skipping" in capsys.readouterr().out
    assert not any(d.startswith("stale_") for d in os.listdir(run_dir))


def test_resume_drops_a_resume_step_left_by_an_earlier_relaunch(tmp_path):
    run_dir = _write_run(tmp_path / "r", 2, want=5, status="failed")
    manifest = {"entries": [_entry("r", run_dir, cmd_extra=["--resume-step", "9", "--seed", "1"])]}
    run_sweep.plan_resume(manifest)
    cmd = manifest["entries"][0]["cmd"]
    assert "--resume-step" not in cmd and cmd[-2:] == ["--seed", "1"]


def test_resume_with_print_cmds_touches_nothing(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed", snapshot_steps=[2, 3])
    before = sorted(os.listdir(os.path.join(run_dir, "generations")))
    manifest = {"entries": [_entry("r", run_dir)]}
    run_sweep.plan_resume(manifest, dry_run=True)
    assert "--resume-step" not in manifest["entries"][0]["cmd"]
    assert sorted(os.listdir(os.path.join(run_dir, "generations"))) == before
    assert not any(d.startswith("stale_") for d in os.listdir(run_dir))


def test_launching_at_generation_zero_sets_the_old_attempt_aside(tmp_path):
    """A sweep relaunch without --resume-step starts at generation 0; the previous attempt's
    events/progress/solutions must not stay behind for it to append to (one run dir on disk ended up
    with 21 progress rows for 15 generations that way)."""
    run_dir = _write_run(tmp_path / "r", 2, want=5, status="failed")
    run_sweep._clear_for_launch(_entry("r", run_dir))
    assert os.listdir(os.path.join(run_dir, "generations")) == []
    assert sum(1 for _ in open(os.path.join(run_dir, "events.jsonl"))) == 0


def test_grid_over_problems_does_not_repeat_the_problem_in_the_name():
    """`problem` is already the name prefix, so a grid over problems must not duplicate it."""
    runs = _expand({
        "common": {"n-context": 0},
        "grid": {"problem": ["erdos", "ac1"], "parent-source": ["initial", "puct"], "seed": [1, 2]},
    })
    assert [r["name"] for r in runs] == [
        "erdos_ps-initial_s-1", "erdos_ps-initial_s-2", "erdos_ps-puct_s-1", "erdos_ps-puct_s-2",
        "ac1_ps-initial_s-1", "ac1_ps-initial_s-2", "ac1_ps-puct_s-1", "ac1_ps-puct_s-2",
    ]
    assert runs[0]["flags"] == {"n-context": 0, "problem": "erdos",
                                "parent-source": "initial", "seed": 1}


def test_grid_over_problem_only_names_runs_after_the_problem():
    runs = _expand({"common": {"n-context": 0}, "grid": {"problem": ["erdos", "ac1"]}})
    assert [r["name"] for r in runs] == ["erdos", "ac1"]

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


# ---- continuing one run mid-run -------------------------------------------------------------------
def _sweep(tmp_path, *entries) -> str:
    sweep_dir = str(tmp_path)
    json.dump({"name": "s", "entries": list(entries), "settings": {"max_parallel": 2}},
              open(os.path.join(sweep_dir, "sweep.json"), "w"))
    return sweep_dir


def test_continue_run_picks_up_at_the_last_verifiable_generation(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed", snapshot_steps=[2, 3])
    _sweep(tmp_path, _entry("r", run_dir), _entry("other", str(tmp_path / "other")))
    _sd, _settings, manifest, name = run_sweep.plan_continue_run(run_dir)
    assert name == "r"
    entry = next(e for e in manifest["entries"] if e["name"] == "r")
    assert entry["cmd"][-2:] == ["--resume-step", "3"]
    assert entry["pid"] is None
    # generations 0..2 stay; only what the resume would append onto is set aside
    assert sorted(os.listdir(os.path.join(run_dir, "generations"))) == [
        "gen_0000", "gen_0001", "gen_0002"]
    # the rest of the sweep is untouched and still in the manifest, so --status keeps working
    assert [e["name"] for e in manifest["entries"]] == ["r", "other"]


def test_continue_run_takes_an_explicit_generation(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed", snapshot_steps=[1, 2, 3])
    _sweep(tmp_path, _entry("r", run_dir))
    _sd, _s, manifest, _n = run_sweep.plan_continue_run(run_dir, "1")
    assert manifest["entries"][0]["cmd"][-2:] == ["--resume-step", "1"]
    assert sorted(os.listdir(os.path.join(run_dir, "generations"))) == ["gen_0000"]
    stale = [d for d in os.listdir(run_dir) if d.startswith("stale_")][0]
    assert sorted(os.listdir(os.path.join(run_dir, stale, "generations"))) == ["gen_0001", "gen_0002"]
    assert sum(1 for _ in open(os.path.join(run_dir, "buffer", "context_pool.jsonl"))) == 2


def test_continue_run_from_generation_zero_restarts_that_one_run(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed")
    _sweep(tmp_path, _entry("r", run_dir))
    _sd, _s, manifest, _n = run_sweep.plan_continue_run(run_dir, "0")
    assert "--resume-step" not in manifest["entries"][0]["cmd"]
    assert os.listdir(os.path.join(run_dir, "generations")) == []


def test_continue_run_refuses_a_generation_with_no_surviving_buffer(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed", snapshot_steps=[3])
    _sweep(tmp_path, _entry("r", run_dir))
    with pytest.raises(SweepError, match="no loadable PUCT snapshot"):
        run_sweep.plan_continue_run(run_dir, "2")
    # ...and does not touch the run dir on the way out
    assert sorted(os.listdir(os.path.join(run_dir, "generations"))) == [
        "gen_0000", "gen_0001", "gen_0002"]


def test_continue_run_refuses_a_finished_run_unless_a_generation_is_named(tmp_path):
    run_dir = _write_run(tmp_path / "r", 5, want=5, snapshot_steps=[3, 5])
    _sweep(tmp_path, _entry("r", run_dir))
    with pytest.raises(SweepError, match="already complete"):
        run_sweep.plan_continue_run(run_dir)
    _sd, _s, manifest, _n = run_sweep.plan_continue_run(run_dir, "3")   # redoing the tail is allowed
    assert manifest["entries"][0]["cmd"][-2:] == ["--resume-step", "3"]


def test_continue_run_rejects_unknown_runs_and_bad_generations(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed")
    _sweep(tmp_path, _entry("r", run_dir))
    with pytest.raises(SweepError, match="not run.s. of this sweep"):
        run_sweep.plan_continue_run(str(tmp_path / "nope"))
    with pytest.raises(SweepError, match="expected a generation number"):
        run_sweep.plan_continue_run(run_dir, "seven")
    with pytest.raises(SweepError, match="leave nothing to run"):
        run_sweep.plan_continue_run(run_dir, "5")


def test_continue_run_with_print_cmds_touches_nothing(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, status="failed", snapshot_steps=[1, 2, 3])
    _sweep(tmp_path, _entry("r", run_dir))
    _sd, _s, manifest, _n = run_sweep.plan_continue_run(run_dir, "1", dry_run=True)
    assert manifest["entries"][0]["cmd"][-2:] == ["--resume-step", "1"]
    assert sorted(os.listdir(os.path.join(run_dir, "generations"))) == [
        "gen_0000", "gen_0001", "gen_0002"]
    assert not any(d.startswith("stale_") for d in os.listdir(run_dir))


# ---- resuming a sweep while continuing named runs mid-run ------------------------------------------
def test_resume_can_continue_one_run_while_restarting_the_others(tmp_path):
    """The mode that matters when one long run is deep in and the rest of the sweep still has to go:
    that run picks up where it stopped, everything else starts over, one queue supervises the lot."""
    deep = _write_run(tmp_path / "deep", 12, want=15, status="failed", snapshot_steps=[12])
    shallow = _write_run(tmp_path / "shallow", 2, want=15, status="failed", snapshot_steps=[2])
    done = _write_run(tmp_path / "done", 15, want=15)
    manifest = {"name": "s", "entries": [_entry("deep", deep, want=15),
                                         _entry("shallow", shallow, want=15),
                                         _entry("done", done, want=15)]}
    queued = run_sweep.plan_runs(manifest, continue_at={"deep": "auto"}, restart_others=True)

    assert queued == {"deep", "shallow"}                       # 'done' verifies complete, not queued
    by_name = {e["name"]: e for e in manifest["entries"]}
    assert by_name["deep"]["cmd"][-2:] == ["--resume-step", "12"]
    assert len(os.listdir(os.path.join(deep, "generations"))) == 12      # kept
    assert "--resume-step" not in by_name["shallow"]["cmd"]
    assert os.listdir(os.path.join(shallow, "generations")) == []        # restarted
    assert len(os.listdir(os.path.join(done, "generations"))) == 15      # untouched
    assert not any(d.startswith("stale_") for d in os.listdir(done))


def test_continuing_several_runs_at_different_generations(tmp_path):
    a = _write_run(tmp_path / "a", 12, want=15, status="failed", snapshot_steps=[8, 12])
    b = _write_run(tmp_path / "b", 5, want=15, status="failed", snapshot_steps=[5])
    manifest = {"name": "s", "entries": [_entry("a", a, want=15), _entry("b", b, want=15)]}
    queued = run_sweep.plan_runs(manifest, continue_at={"a": "8", "b": "auto"}, restart_others=True)
    assert queued == {"a", "b"}
    by_name = {e["name"]: e for e in manifest["entries"]}
    assert by_name["a"]["cmd"][-2:] == ["--resume-step", "8"]
    assert by_name["b"]["cmd"][-2:] == ["--resume-step", "5"]
    assert len(os.listdir(os.path.join(a, "generations"))) == 8


def test_continue_only_mode_leaves_the_sweeps_other_unfinished_runs_alone(tmp_path):
    deep = _write_run(tmp_path / "deep", 12, want=15, status="failed", snapshot_steps=[12])
    other = _write_run(tmp_path / "other", 2, want=15, status="failed", snapshot_steps=[2])
    manifest = {"name": "s", "entries": [_entry("deep", deep, want=15),
                                         _entry("other", other, want=15)]}
    queued = run_sweep.plan_runs(manifest, continue_at={"deep": "auto"}, restart_others=False)
    assert queued == {"deep"}
    assert len(os.listdir(os.path.join(other, "generations"))) == 2      # untouched, not queued
    assert "--resume-step" not in next(e for e in manifest["entries"] if e["name"] == "other")["cmd"]


def test_a_named_run_that_is_complete_can_still_be_redone_from_a_generation(tmp_path):
    done = _write_run(tmp_path / "done", 15, want=15, snapshot_steps=[10, 15])
    manifest = {"name": "s", "entries": [_entry("done", done, want=15)]}
    with pytest.raises(SweepError, match="already complete"):
        run_sweep.plan_runs(manifest, continue_at={"done": "auto"})
    queued = run_sweep.plan_runs(manifest, continue_at={"done": "10"})
    assert queued == {"done"} and manifest["entries"][0]["cmd"][-2:] == ["--resume-step", "10"]


# ---- --continue-run spec parsing -------------------------------------------------------------------
def test_continue_specs_accept_names_paths_and_generations():
    assert run_sweep.parse_continue_specs(["r1"], None, "runs/sw") == ("runs/sw", {"r1": "auto"})
    assert run_sweep.parse_continue_specs(["r1:7"], None, "runs/sw") == ("runs/sw", {"r1": "7"})
    assert run_sweep.parse_continue_specs(["runs/sw/r1"], None, None) == ("runs/sw", {"r1": "auto"})
    assert run_sweep.parse_continue_specs(["runs/sw/r1:7"], None, None) == ("runs/sw", {"r1": "7"})
    # --from-generation is sugar for a single run's :N
    assert run_sweep.parse_continue_specs(["r1"], "3", "runs/sw") == ("runs/sw", {"r1": "3"})
    assert run_sweep.parse_continue_specs(["a:1", "b"], None, "runs/sw")[1] == {"a": "1", "b": "auto"}


def test_continue_specs_reject_ambiguity():
    with pytest.raises(SweepError, match="give the run's directory"):
        run_sweep.parse_continue_specs(["r1"], None, None)          # which sweep?
    with pytest.raises(SweepError, match="but the sweep being resumed is"):
        run_sweep.parse_continue_specs(["runs/other/r1"], None, "runs/sw")
    with pytest.raises(SweepError, match="named twice"):
        run_sweep.parse_continue_specs(["r1", "r1:4"], None, "runs/sw")
    with pytest.raises(SweepError, match="applies to a single --continue-run"):
        run_sweep.parse_continue_specs(["a", "b"], "3", "runs/sw")
    with pytest.raises(SweepError, match="both say where to continue"):
        run_sweep.parse_continue_specs(["a:2"], "3", "runs/sw")

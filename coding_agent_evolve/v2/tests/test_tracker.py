"""Tracker tests. Run with an interpreter that has numpy (the agent-eval venv):

    ~/venvs/agent-eval/bin/python -m unittest v2/tests/test_tracker.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "tracker"))
from tracker import install_tracker, read_iterations, stop_reason_file, best_record, improvement_banked, load_meta  # noqa: E402
from score import score_cell  # noqa: E402

PY = sys.executable


def make_cell(tmp: Path, problem: str, max_evals: int = 0) -> Path:
    cell = tmp / f"cell_{problem}"
    (cell / "workspace").mkdir(parents=True)
    meta = load_meta(V2 / "problems" / problem / "meta.yaml")
    install_tracker(cell, V2 / "problems" / problem / "eval.py", meta, max_evals=max_evals, t0=0.0)
    return cell


def official(cell: Path, npy: Path):
    return subprocess.run([PY, "eval.py", str(npy)], cwd=cell / "workspace", capture_output=True, text=True)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2trk_"))

    def test_ac2_records_valid_invalid_and_best(self):
        cell = make_cell(self.tmp, "AC2")
        ws = cell / "workspace"
        a = ws / "a.npy"; np.save(a, np.full(50, 0.3))                  # valid, 0.6667-ish
        b = ws / "b.npy"; np.save(b, np.array([1.0, 0.5, 0.25, 0.1, 0.0, 0.0, 0.1, 0.25, 0.5, 1.0]))
        bad = ws / "bad.npy"; np.save(bad, np.zeros(10))                 # sum ~ 0 -> ValueError
        r1 = official(cell, a); self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertIn("SCORE", r1.stdout)                                # same stdout as grader
        r2 = official(cell, bad); self.assertNotEqual(r2.returncode, 0)
        self.assertIn("Sum of sequence", r2.stderr)
        r3 = official(cell, b); self.assertEqual(r3.returncode, 0)
        rows = read_iterations(cell)
        self.assertEqual([r["i"] for r in rows], [1, 2, 3])
        self.assertIsNone(rows[1]["score"]); self.assertIn("error", rows[1])
        self.assertTrue(all((cell / "submissions" / r["snap"]).is_file() for r in rows))
        best = best_record(cell)
        scores = [r["score"] for r in rows if r["score"] is not None]
        self.assertEqual(best["score"], max(scores))                     # maximise
        self.assertTrue((cell / "best.npy").is_file())
        self.assertTrue(improvement_banked(cell, 0.6667, True) or max(scores) <= 0.6667)
        # unrecorded import path
        r = subprocess.run([PY, "-c", "from eval import evaluate_sequence; print(evaluate_sequence([1.0,2.0]))"],
                           cwd=ws, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(read_iterations(cell)), 3)
        # host mode scores the best snapshot and agrees with the recorded value
        s = score_cell(cell, PY)
        self.assertTrue(s["ok"]); self.assertAlmostEqual(s["score"], best["score"], places=12)
        self.assertEqual(len(read_iterations(cell)), 3)                  # host scoring not recorded

    def test_ac1_minimises_and_treats_inf_as_invalid(self):
        cell = make_cell(self.tmp, "AC1")
        ws = cell / "workspace"
        a = ws / "a.npy"; np.save(a, np.full(64, 0.2))                   # scores 2.0
        b = ws / "b.npy"; np.save(b, np.linspace(0.1, 1.0, 64))          # different score
        z = ws / "z.npy"; np.save(z, np.zeros(8))                        # inf -> invalid, rc 1
        for f in (a, b, z):
            official(cell, f)
        rows = read_iterations(cell)
        self.assertIsNone(rows[2]["score"])
        best = best_record(cell)
        self.assertEqual(best["score"], min(r["score"] for r in rows if r["score"] is not None))

    def test_erdos_and_cap_writes_stop(self):
        cell = make_cell(self.tmp, "Erdos", max_evals=2)
        ws = cell / "workspace"
        h = np.load(V2 / "problems/Erdos/initial_h_values.npy")
        a = ws / "a.npy"; np.save(a, h)
        r = official(cell, a); self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("C5", r.stdout)
        self.assertIsNone(stop_reason_file(cell))
        official(cell, a)
        self.assertEqual(stop_reason_file(cell), "iteration_limit")
        self.assertAlmostEqual(best_record(cell)["score"], 0.49399, places=4)

    def test_concurrent_calls_get_distinct_indices(self):
        cell = make_cell(self.tmp, "AC2")
        ws = cell / "workspace"
        files = []
        for k in range(8):
            f = ws / f"c{k}.npy"; np.save(f, np.full(20 + k, 0.5)); files.append(f)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda f: official(cell, f), files))
        rows = read_iterations(cell)
        self.assertEqual(sorted(r["i"] for r in rows), list(range(1, 9)))
        self.assertEqual(len({r["snap"] for r in rows}), 8)

    def test_protected_files_are_read_only(self):
        cell = make_cell(self.tmp, "AC2")
        for n in ("eval.py", "_official_evaluator.py", "_adrs_track.json"):
            self.assertFalse(os.access(cell / "workspace" / n, os.W_OK), n)


if __name__ == "__main__":
    unittest.main()

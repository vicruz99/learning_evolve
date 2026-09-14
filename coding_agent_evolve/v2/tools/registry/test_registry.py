#!/usr/bin/env python3
"""Tests for registry.py using a toy evaluator. Run:  python3 test_registry.py [-v]

Every test drives the CLI as a subprocess (the way agents use it), in a temp dir with its own
REGISTRY_HOME. Needs only the stdlib.
"""
import concurrent.futures
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "registry.py")
PY = sys.executable

TOY_EVAL = r'''
import json, sys, time
d = json.load(open(sys.argv[1]))
if d.get("sleep"): time.sleep(d["sleep"])
if d.get("invalid"):
    print("INVALID: bad construction", file=sys.stderr); sys.exit(1)
if d.get("style") == "c5":
    print(f"C5          = {d['value']!r}   (lower is better)")
elif d.get("style") == "gpumode":
    print(f"SCORE (geom of 4 benchmarks): {d['value']}")
elif d.get("style") == "inf":
    print("SCORE = inf")
elif d.get("style") == "silent":
    print("done")
else:
    print(f"n           = {d.get('n', 0)}")
    print(f"SCORE       = {d['value']!r}   (higher is better)")
'''


class Base(unittest.TestCase):
    direction = "max"
    policy = "on-claimed-best"
    timeout = 3

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="regtest_")
        self.home = os.path.join(self.tmp, "home")
        self.env = dict(os.environ, REGISTRY_HOME=self.home)
        pdir = os.path.join(self.tmp, "problem")
        os.makedirs(pdir)
        with open(os.path.join(pdir, "eval.py"), "w") as fh:
            fh.write(TOY_EVAL)
        self.write_json(os.path.join(pdir, "seed.json"), {"value": 0.5, "n": 10})
        self.run_dir = os.path.join(self.tmp, "run1")
        os.makedirs(self.run_dir)
        self.ok(["problem", "add", "toy", "--direction", self.direction, "--eval", f"{PY} eval.py {{artifact}}",
                 "--files", os.path.join(pdir, "eval.py"), "--baseline", os.path.join(pdir, "seed.json"),
                 "--timeout", str(self.timeout)])
        self.init_out = self.ok(["init", "--problem", "toy", "--eval-policy", self.policy, "--baseline-expect", "0.5"], cwd=self.run_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # helpers
    def write_json(self, path, obj):
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    def cand(self, name, **fields):
        return self.write_json(os.path.join(self.run_dir, f"{name}.json"), fields)

    def reg(self, args, cwd=None, check=True):
        r = subprocess.run([PY, REGISTRY] + args, cwd=cwd or self.run_dir, env=self.env, capture_output=True, text=True)
        if check and r.returncode != 0:
            self.fail(f"registry {' '.join(args)} failed rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
        return r

    def ok(self, args, cwd=None):
        return self.reg(args, cwd).stdout

    def submit(self, desc, **kw):
        args = ["submit", "--desc", desc, "--json"]
        for k, v in kw.items():
            if v is True:
                args.append(f"--{k.replace('_', '-')}")
            elif v is not None:
                args += [f"--{k.replace('_', '-')}", str(v)]
        return json.loads(self.ok(args))

    def db(self):
        return sqlite3.connect(os.path.join(self.run_dir, "run", "registry", "registry.sqlite"))

    def row(self, cid):
        c = self.db(); c.row_factory = sqlite3.Row
        return c.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()

    def best_json(self):
        with open(os.path.join(self.run_dir, "run", "registry", "best.json")) as fh:
            return json.load(fh)

    def sha(self, path):
        import hashlib
        return hashlib.sha256(open(path, "rb").read()).hexdigest()


class TestInitAndBaseline(Base):
    def test_baseline_row(self):
        self.assertIn("baseline id=0 score=0.5", self.init_out)
        r = self.row(0)
        self.assertEqual(r["status"], "done"); self.assertEqual(r["tag"], "baseline"); self.assertEqual(r["score"], 0.5)
        self.assertTrue(os.path.exists(os.path.join(self.run_dir, "run", "best.json")))
        self.assertEqual(self.best_json()["id"], 0)
        self.assertTrue(os.path.exists(os.path.join(self.run_dir, "eval.py")), "problem files copied into run folder")

    def test_baseline_expect_mismatch_fails(self):
        d = os.path.join(self.tmp, "run2"); os.makedirs(d)
        r = self.reg(["init", "--problem", "toy", "--baseline-expect", "2.0"], cwd=d, check=False)
        self.assertEqual(r.returncode, 3); self.assertIn("baseline scores 0.5", r.stderr)

    def test_double_init_refused(self):
        r = self.reg(["init", "--problem", "toy"], check=False)
        self.assertNotEqual(r.returncode, 0); self.assertIn("already initialised", r.stderr)

    def test_unknown_problem(self):
        d = os.path.join(self.tmp, "run3"); os.makedirs(d)
        r = self.reg(["init", "--problem", "nope"], cwd=d, check=False)
        self.assertNotEqual(r.returncode, 0); self.assertIn("unknown problem", r.stderr)

    def test_no_registry_here(self):
        r = self.reg(["list"], cwd=self.tmp, check=False)
        self.assertNotEqual(r.returncode, 0); self.assertIn("no registry found", r.stderr)

    def test_root_found_from_subdir(self):
        sub = os.path.join(self.run_dir, "run", "attempts"); os.makedirs(sub, exist_ok=True)
        out = self.ok(["best"], cwd=sub)
        self.assertIn("id=0", out)


class TestSubmitPolicyOnClaimedBest(Base):
    def test_claimed_beats_best_is_evaluated_and_promoted(self):
        out = self.submit("first", artifact=self.cand("c1", value=0.7), claimed=0.7, program=self.write_json(os.path.join(self.run_dir, "c1.py"), {}))
        self.assertEqual(out["id"], 1); self.assertEqual(out["score"], "0.7"); self.assertEqual(out["new_best"], "yes")
        self.assertEqual(out["promoted"], "run/best.json")
        self.assertEqual(self.best_json()["id"], 1)
        self.assertEqual(self.sha(os.path.join(self.run_dir, "run", "best.json")), self.row(1)["artifact_sha256"])
        self.assertTrue(os.path.exists(os.path.join(self.run_dir, "run", "best.py")))
        self.assertTrue(os.path.exists(os.path.join(self.run_dir, "run", "registry", "programs", "000001.py")))

    def test_claimed_below_best_is_skipped(self):
        self.submit("first", artifact=self.cand("c1", value=0.7), claimed=0.7)
        out = self.submit("worse", artifact=self.cand("c2", value=0.65), claimed=0.65)
        self.assertEqual(out["score"], "null"); self.assertIn("skipped", out["eval"]); self.assertEqual(self.row(2)["eval_status"], "skipped")

    def test_no_claimed_is_unverified(self):
        out = self.submit("blind", artifact=self.cand("c1", value=0.9))
        self.assertIn("no --claimed", out["eval"]); self.assertIsNone(self.row(1)["score"])
        self.assertIn("unverified=1", self.ok(["list"]))

    def test_overclaim_warns(self):
        out = self.submit("lie", artifact=self.cand("c1", value=0.75), claimed=0.95)
        self.assertEqual(out["score"], "0.75"); self.assertIn("claimed differs", out["warning"])
        self.assertIn("0.950000!", self.ok(["list"]))

    def test_invalid(self):
        out = self.submit("cheat", artifact=self.cand("c1", value=0.99, invalid=1), claimed=0.99)
        self.assertEqual(out["status"], "invalid"); self.assertIn("INVALID: bad construction", out["eval"])
        self.assertIsNone(self.row(1)["score"]); self.assertEqual(self.best_json()["id"], 0)

    def test_inf_score_is_invalid(self):
        out = self.submit("inf", artifact=self.cand("c1", value=1, style="inf"), claimed=9)
        self.assertEqual(out["status"], "invalid")

    def test_no_score_line_is_error(self):
        out = self.submit("silent", artifact=self.cand("c1", value=1, style="silent"), claimed=9)
        self.assertIn("no SCORE line", out["eval"]); self.assertEqual(self.row(1)["eval_status"], "error")

    def test_timeout_then_pending(self):
        out = self.submit("slow", artifact=self.cand("c1", value=0.9, sleep=10), claimed=0.99)
        self.assertIn("timeout", out["eval"]); self.assertEqual(self.row(1)["status"], "done")
        # make it fast and sweep
        self.write_json(os.path.join(self.run_dir, "run", "registry", "artifacts", "000001.json"), {"value": 0.9})
        out2 = self.ok(["eval", "--pending"])
        self.assertIn("id=1 eval=ok score=0.9", out2); self.assertIn("new_best=yes", out2)
        self.assertEqual(self.best_json()["id"], 1)

    def test_pending_order_and_nothing_pending(self):
        self.submit("a", artifact=self.cand("a", value=0.6), claimed=0.3)
        self.submit("b", artifact=self.cand("b", value=0.8), claimed=0.4)
        out = self.ok(["eval", "--pending"]).splitlines()
        self.assertTrue(out[0].startswith("id=2"), out)   # higher claimed first
        self.assertEqual(self.ok(["eval", "--pending"]).strip(), "nothing pending")

    def test_no_eval_flag(self):
        out = self.submit("x", artifact=self.cand("c1", value=0.9), claimed=0.9, no_eval=True)
        self.assertIn("--no-eval", out["eval"])

    def test_note_rejects_scores(self):
        r = self.reg(["submit", "--kind", "note", "--desc", "bound", "--claimed", "0.99"], check=False)
        self.assertEqual(r.returncode, 2); self.assertIn("--kind note", r.stderr)
        out = self.submit("bound C<=0.95", kind="note", tag="bounds")
        self.assertEqual(out["kind"], "note"); self.assertNotIn("score", out)
        self.assertIn("note", self.ok(["list"]))

    def test_done_needs_artifact(self):
        r = self.reg(["submit", "--desc", "x"], check=False)
        self.assertEqual(r.returncode, 2); self.assertIn("--artifact", r.stderr)

    def test_failed_without_artifact(self):
        out = self.submit("crash", status="failed", error="TypeError")
        self.assertEqual(out["status"], "failed"); self.assertEqual(self.row(1)["error"], "TypeError")

    def test_parents_validated(self):
        r = self.reg(["submit", "--desc", "x", "--artifact", self.cand("c", value=1), "--parents", "42"], check=False)
        self.assertEqual(r.returncode, 2); self.assertIn("no candidate with id 42", r.stderr)
        self.submit("p", artifact=self.cand("p", value=0.6), claimed=0.6)
        self.submit("child", artifact=self.cand("k", value=0.61), claimed=0.61, parents="0, 1")
        self.assertEqual(json.loads(self.row(2)["parents"]), [0, 1])
        lin = self.ok(["lineage", "2"])
        self.assertIn("#2", lin); self.assertIn("  #0", lin); self.assertIn("  #1", lin)
        self.assertIn("children of #1: #2", self.ok(["lineage", "1"]))

    def test_config_json(self):
        r = self.reg(["submit", "--desc", "x", "--artifact", self.cand("c", value=1), "--config", "notjson"], check=False)
        self.assertEqual(r.returncode, 2); self.assertIn("not valid JSON", r.stderr)
        cfgfile = self.write_json(os.path.join(self.run_dir, "cfg.json"), {"n": 4096, "seed": 3})
        self.submit("x", artifact=self.cand("c", value=0.6), config=f"@{cfgfile}")
        self.assertEqual(json.loads(self.row(1)["config"]), {"n": 4096, "seed": 3})
        self.assertIn("   1 done", self.ok(["list", "--grep", "4096"]))
        self.assertNotIn("   1 done", self.ok(["list", "--grep", "nomatch"]))

    def test_duplicate_detection(self):
        a = self.cand("c1", value=0.7)
        self.submit("first", artifact=a, claimed=0.7)
        out = self.submit("again", artifact=a, claimed=0.7)
        self.assertEqual(out["duplicate_of"], 1)

    def test_artifact_copy_is_independent(self):
        a = self.cand("c1", value=0.7)
        self.submit("first", artifact=a, claimed=0.7)
        self.write_json(a, {"value": 0.99})  # agent overwrites its file afterwards
        self.assertIn("artifact_drift=no", self.ok(["show", "1"]))
        self.assertEqual(json.load(open(os.path.join(self.run_dir, "run", "registry", "artifacts", "000001.json")))["value"], 0.7)

    def test_export(self):
        self.submit("first", artifact=self.cand("c1", value=0.7), claimed=0.7)
        lines = self.ok(["export"]).strip().splitlines()
        self.assertEqual(len(lines), 2); self.assertEqual(json.loads(lines[1])["score"], 0.7)
        p = os.path.join(self.tmp, "o.csv"); self.ok(["export", "--csv", p])
        self.assertEqual(open(p).readline().split(",")[0], "id")

    def test_eval_rerun_records_previous(self):
        self.submit("first", artifact=self.cand("c1", value=0.7), claimed=0.7)
        self.write_json(os.path.join(self.run_dir, "run", "registry", "artifacts", "000001.json"), {"value": 0.72})
        out = self.ok(["eval", "1"])
        self.assertIn("score=0.72", out); self.assertIn("previous=0.7", out)


class TestPolicyAlways(Base):
    policy = "always"

    def test_evaluates_without_claim(self):
        out = self.submit("x", artifact=self.cand("c1", value=0.6))
        self.assertEqual(out["score"], "0.6"); self.assertEqual(out["new_best"], "yes")
        out = self.submit("y", artifact=self.cand("c2", value=0.55), claimed=0.99)
        self.assertEqual(out["score"], "0.55"); self.assertEqual(out["new_best"], "no"); self.assertEqual(out["delta"], "-0.05")


class TestPolicyEveryN(Base):
    policy = "every-3"

    def test_every_third_plus_claimed_best(self):
        evaluated = []
        for i in range(1, 7):
            out = self.submit(f"c{i}", artifact=self.cand(f"c{i}", value=0.4), claimed=0.4)  # never beats 0.5
            evaluated.append(out["score"] != "null")
        self.assertEqual(evaluated, [False, False, True, False, False, True])
        out = self.submit("best", artifact=self.cand("b", value=0.9), claimed=0.9)
        self.assertEqual(out["new_best"], "yes")


class TestPolicyNever(Base):
    policy = "never"

    def test_never(self):
        out = self.submit("x", artifact=self.cand("c1", value=0.9), claimed=0.9)
        self.assertIn("policy=never", out["eval"]); self.assertEqual(self.best_json()["id"], 0)
        self.assertIn("new_best=yes", self.ok(["eval", "1"]))


class TestMinDirection(Base):
    direction = "min"

    def test_lower_is_better(self):
        out = self.submit("worse", artifact=self.cand("c1", value=0.6), claimed=0.6)
        self.assertIn("does not beat", out["eval"])
        out = self.submit("better", artifact=self.cand("c2", value=0.4), claimed=0.4)
        self.assertEqual(out["new_best"], "yes"); self.assertEqual(out["delta"], "-0.1")
        self.assertTrue(self.ok(["list", "--sort", "score", "--last", "1"]).splitlines()[2].strip().startswith("2 "))


class TestScoreRegexes(Base):
    policy = "always"

    def test_c5_and_gpumode_lines(self):
        out = self.submit("c5", artifact=self.cand("c1", value=0.62, style="c5"))
        self.assertEqual(out["score"], "0.62")
        out = self.submit("gpu", artifact=self.cand("c2", value=1234.5, style="gpumode"))
        self.assertEqual(out["score"], "1234.5")


class TestConcurrency(Base):
    policy = "always"
    timeout = 120  # 24 python interpreters starting at once on a throttled login node can take >3 s each

    def test_24_parallel_submits(self):
        files = [self.cand(f"p{i}", value=0.5 + i / 100) for i in range(1, 25)]

        def one(i):
            return self.submit(f"p{i}", artifact=files[i - 1], claimed=0.5 + i / 100)
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
            outs = list(ex.map(one, range(1, 25)))
        ids = sorted(o["id"] for o in outs)
        self.assertEqual(ids, list(range(1, 25)))
        c = self.db()
        self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        dist = c.execute("SELECT eval_status, eval_error, count(*) FROM candidates GROUP BY 1, 2").fetchall()
        self.assertEqual(c.execute("SELECT count(*) FROM candidates WHERE score IS NOT NULL").fetchone()[0], 25,
                         f"eval status distribution: {dist}")
        # ids are assigned in arrival order, so check invariants rather than id==file number
        best = c.execute("SELECT id, score FROM candidates ORDER BY score DESC LIMIT 1").fetchone()
        self.assertAlmostEqual(best[1], 0.74)
        for cid, score, art in c.execute("SELECT id, score, artifact FROM candidates WHERE id > 0"):
            stored = json.load(open(os.path.join(self.run_dir, art)))["value"]
            self.assertAlmostEqual(score, stored, msg=f"row {cid} score does not match its stored artifact")
        bj = self.best_json()
        self.assertEqual(bj["id"], best[0])
        self.assertEqual(self.sha(os.path.join(self.run_dir, "run", "best.json")), bj["sha256"])
        self.assertEqual(sum(1 for o in outs if o["new_best"] == "yes"), len({o["id"] for o in outs if o["new_best"] == "yes"}))


if __name__ == "__main__":
    unittest.main()

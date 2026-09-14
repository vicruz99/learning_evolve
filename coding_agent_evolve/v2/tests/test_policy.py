import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driver"))
from policy import Rules, State, decide  # noqa: E402

R = Rules(min_evals=5, min_hours=1.0, keep_going=True, nudge_min_gap_s=300)


def S(**kw):
    base = dict(elapsed_s=100, hard_budget_s=7200, evals=0, banked=False, continues=1, since_last_nudge_s=0, stop_reason=None)
    base.update(kw)
    return State(**base)


class PolicyTests(unittest.TestCase):
    def test_first_stop_always_continues(self):
        self.assertTrue(decide(S(continues=0, evals=99, banked=True, elapsed_s=7000), R)[0])

    def test_stop_file_and_budget_end(self):
        self.assertFalse(decide(S(stop_reason="iteration_limit"), R)[0])
        self.assertFalse(decide(S(elapsed_s=7200), R)[0])

    def test_minima_force_continue(self):
        self.assertTrue(decide(S(evals=2, elapsed_s=5000, banked=True), R)[0])       # evals below minimum
        self.assertTrue(decide(S(evals=9, elapsed_s=1000, banked=True), R)[0])       # time below minimum
        self.assertTrue(decide(S(evals=9, elapsed_s=5000, banked=False), R)[0])      # nothing banked

    def test_after_minima_throttled_or_ended(self):
        met = dict(evals=9, elapsed_s=5000, banked=True)
        go, why = decide(S(since_last_nudge_s=10, **met), R)
        self.assertTrue(go); self.assertTrue(why.startswith("WAIT 290s"), why)
        self.assertTrue(decide(S(since_last_nudge_s=400, **met), R)[0])
        r2 = Rules(min_evals=5, min_hours=1.0, keep_going=False, nudge_min_gap_s=300)
        self.assertFalse(decide(S(since_last_nudge_s=400, **met), r2)[0])


if __name__ == "__main__":
    unittest.main()

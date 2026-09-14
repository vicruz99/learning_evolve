# -*- coding: utf-8 -*-
"""Score every run's best.npy with its own eval.py. AC2: higher is better."""
import glob, importlib.util, os, sys
import numpy as np

runs = ["ac2_evo_cc_s1", "ac2_evo_s3", "ac2_plain_s3",
        "ac2_evo_s1", "ac2_plain_s1", "ac2_evo_s2", "ac2_plain_s2"]
base = os.path.expanduser("~/agent_runs")
print("%-18s %-10s %-12s %-9s %s" % ("run", "attempts", "best score", "n", "note"))
for r in runs:
    d = os.path.join(base, r)
    if not os.path.isdir(d):
        continue
    nat = len(glob.glob(os.path.join(d, "run", "attempts", "*.py")))
    bp = os.path.join(d, "run", "best.npy")
    if not os.path.exists(bp):
        print("%-18s %-10d %-12s %-9s %s" % (r, nat, "-", "-", "no run/best.npy"))
        continue
    spec = importlib.util.spec_from_file_location("ev", os.path.join(d, "eval.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    try:
        a = np.load(bp)
        s = ev.evaluate_sequence([float(x) for x in np.asarray(a).ravel()])
        print("%-18s %-10d %-12.7f %-9d %s"
              % (r, nat, s, a.size, "ledger" if os.path.exists(os.path.join(d, "run", "LEDGER.md")) else "no ledger"))
    except Exception as exc:
        print("%-18s %-10d %-12s %-9s %s" % (r, nat, "ERR", "-", str(exc)[:50]))
print("\nbaseline (uniform) = 0.6667   target = 0.97")

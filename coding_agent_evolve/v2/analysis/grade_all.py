"""Re-grade EVERY candidate .npy in every AC2 agent run with the pristine eval.py.

Exact evaluate_sequence for n <= EXACT_CAP; above that an FFT-based numerically
equivalent evaluation (method="fft") because np.convolve is O(n^2). Output: JSONL,
one line per file, written incrementally so a partial run is still usable.
"""
import glob, importlib.util, json, os, sys, time
import numpy as np

REF = os.path.expanduser("~/work/learning_evolve/coding_agent_evolve/portable/tasks/AC2/eval.py")
sp = importlib.util.spec_from_file_location("ac2eval", REF); ev = importlib.util.module_from_spec(sp)
sp.loader.exec_module(ev)
EXACT_CAP = int(os.environ.get("EXACT_CAP", "2500000"))
OUT = sys.argv[1]

def fft_score(seq):
    seq = np.asarray(seq, dtype=np.float64)
    seq = np.maximum(seq, 0.0)
    if seq.sum() < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    seq = np.minimum(1000.0, seq)
    n = seq.size
    L = 1
    while L < 2 * n - 1:
        L *= 2
    F = np.fft.rfft(seq, L)
    conv = np.fft.irfft(F * F, L)[: 2 * n - 1]
    conv = np.maximum(conv, 0.0)  # exact conv of non-negatives is non-negative; kill fft noise
    m = conv.size
    h = 1.0 / (m + 1)              # np.diff(np.linspace(-0.5,0.5,m+2)) is constant
    y = np.concatenate(([0.0], conv, [0.0]))
    l2 = (h / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(np.abs(conv)) / (m + 1)
    ninf = np.max(np.abs(conv))
    return float(l2 / (n1 * ninf))

roots = [os.path.expanduser("~/agent_runs/campaign"), os.path.expanduser("~/agent_runs")]
files = set()
for r in roots:
    for f in glob.glob(os.path.join(r, "ac2_*", "**", "*.npy"), recursive=True):
        if os.path.basename(f) == "height_sequence_1.npy":
            continue
        files.add(f)
files = sorted(files, key=lambda f: os.path.getsize(f))
print("files:", len(files), flush=True)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: done.add(json.loads(line)["path"])
        except Exception: pass

with open(OUT, "a") as out:
    for f in files:
        if f in done:
            continue
        rec = {"path": f, "mtime": int(os.path.getmtime(f)), "size": os.path.getsize(f)}
        rel = f.replace(os.path.expanduser("~/agent_runs/"), "")
        parts = rel.split("/")
        rec["cell"] = parts[1] if parts[0] == "campaign" else parts[0]
        rec["set"] = "campaign" if parts[0] == "campaign" else "pilot"
        t = time.time()
        try:
            a = np.load(f, allow_pickle=False)
            if hasattr(a, "files"):
                a = a[a.files[0]]
            a = np.asarray(a)
            rec["dtype"] = str(a.dtype); rec["shape"] = list(a.shape)
            a = np.asarray(a, dtype=np.float64).ravel()
            rec["n"] = int(a.size)
            if a.size == 0:
                raise ValueError("empty")
            if a.size <= EXACT_CAP:
                rec["score"] = float(ev.evaluate_sequence(a.tolist())); rec["method"] = "exact"
            else:
                rec["score"] = fft_score(a); rec["method"] = "fft"
        except Exception as ex:
            rec["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:150])
        rec["sec"] = round(time.time() - t, 2)
        out.write(json.dumps(rec) + "\n"); out.flush(); os.fsync(out.fileno())
print("done", flush=True)

"""Does the power cap reorder kernel variants, or just scale them uniformly?

Times each variant under two power regimes on the SAME GPU:

  throttled : reps back-to-back, exactly like eval.py's benchmark loop. The
              card pins at its 300W limit and SM clock sags 1410 -> ~1020 MHz.
  cold      : one rep, then sleep, so every rep starts near base power and the
              card never enters the cap.

If the cold/throttled ratio is the same for every variant, the cap is a uniform
scale factor and local optimisation transfers. If the ratio varies, the cap
changes which variant looks best, and it does not.

RESULT: THIS EXPERIMENT DOES NOT WORK -- kept only to document the negative.
The sleep drops the card to idle (210 MHz), so every "cold" rep pays a DVFS
ramp-up and reads *slower* than the throttled one (2911 vs 2401us for the stock
kernel). It measures clock ramp latency, not power headroom, and the single
reordering it produced was between two variants tied to 0.1%. Isolating the cap
properly needs locked clocks (`nvidia-smi -lgc`, root-only). The transferability
question was instead answered by ranking the same variants on local vs Modal:
6/6 pairwise orderings agreed.
"""

import importlib.util
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "test" / "trimul"))
import torch  # noqa: E402
from reference import generate_input  # noqa: E402

# the task.yml benchmark shapes
SHAPES = [
    dict(seqlen=256, bs=2, dim=128, hiddendim=128, seed=9371, nomask="True", distribution="normal"),
    dict(seqlen=768, bs=1, dim=128, hiddendim=128, seed=381, nomask="True", distribution="cauchy"),
    dict(seqlen=256, bs=2, dim=384, hiddendim=128, seed=2301, nomask="False", distribution="normal"),
    dict(seqlen=512, bs=1, dim=128, hiddendim=128, seed=12819, nomask="True", distribution="normal"),
    dict(seqlen=1024, bs=1, dim=128, hiddendim=128, seed=381, nomask="True", distribution="cauchy"),
    dict(seqlen=768, bs=1, dim=384, hiddendim=128, seed=481, nomask="False", distribution="normal"),
    dict(seqlen=1024, bs=1, dim=384, hiddendim=128, seed=23291, nomask="True", distribution="normal"),
]


def load(path):
    spec = importlib.util.spec_from_file_location(f"v_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def clocks():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True,
    ).stdout.strip()
    sm, pw = out.split(",")
    return float(sm), float(pw)


def time_one(fn, data, reps, cooldown):
    """Median rep time in us, plus the SM clock seen mid-run."""
    for _ in range(3):  # warmup / triton autotune
        fn(data)
    torch.cuda.synchronize()

    times, mid_clock = [], None
    for i in range(reps):
        if cooldown:
            time.sleep(cooldown)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(data)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)  # ms -> us
        del out
        if i == reps // 2:
            mid_clock = clocks()
    return statistics.median(times), mid_clock


def main():
    variants = sorted(Path("variants").glob("v*.py"))
    # pytorch baseline is 10x slower; skip it here so a single sweep stays short
    variants = [v for v in variants if "pytorch" not in v.name]

    results = {}
    for regime, reps, cooldown in [("throttled", 40, 0.0), ("cold", 12, 0.45)]:
        for vpath in variants:
            fn = load(str(vpath))
            per_shape, clock_samples = [], []
            for shape in SHAPES:
                data = generate_input(**shape)
                t, ck = time_one(fn, data, reps, cooldown)
                per_shape.append(t)
                if ck:
                    clock_samples.append(ck)
                del data
                torch.cuda.empty_cache()
            geo = math.exp(statistics.fmean(math.log(t) for t in per_shape))
            results[(regime, vpath.stem)] = geo
            mhz = statistics.fmean(c[0] for c in clock_samples)
            watt = statistics.fmean(c[1] for c in clock_samples)
            print(f"{regime:>9} {vpath.stem:<14} geomean={geo:8.1f}us   sm={mhz:6.0f}MHz  {watt:5.0f}W", flush=True)

    print("\n=== does the cap reorder variants? ===")
    names = [v.stem for v in variants]
    print(f"{'variant':<14} {'cold(us)':>10} {'throttled(us)':>14} {'throttled/cold':>15}")
    ratios = {}
    for n in names:
        c, t = results[("cold", n)], results[("throttled", n)]
        ratios[n] = t / c
        print(f"{n:<14} {c:10.1f} {t:14.1f} {t / c:15.3f}")

    rank_cold = sorted(names, key=lambda n: results[("cold", n)])
    rank_thr = sorted(names, key=lambda n: results[("throttled", n)])
    print(f"\nranking cold     : {' < '.join(rank_cold)}")
    print(f"ranking throttled: {' < '.join(rank_thr)}")
    print(f"ranking preserved: {rank_cold == rank_thr}")
    spread = (max(ratios.values()) - min(ratios.values())) / statistics.fmean(list(ratios.values())) * 100
    print(f"throttle penalty spread across variants: {spread:.1f}% "
          f"(min {min(ratios.values()):.3f}x, max {max(ratios.values()):.3f}x)")


if __name__ == "__main__":
    main()

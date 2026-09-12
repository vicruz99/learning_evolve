import json, glob, os, re, collections
ONLINE = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/vicruz/online"
cfg = json.load(open(os.path.expanduser("~/work/learning_evolve/src/sft/online/ac2_online.json")))
seed = {r["name"]: r["seed"] for r in cfg["runs"]}
rows = []
hosts = collections.Counter()
for d in sorted(glob.glob(ONLINE + "/runs/ac2_*/")):
    n = os.path.basename(d.rstrip("/"))
    per = collections.defaultdict(lambda: collections.Counter())
    for line in open(d + "events.jsonl", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        g = e.get("generation")
        if g is None or "raw_score" not in e:
            continue
        per[g]["ev"] += 1
        ft = e.get("failure_type") or ""
        if ft == "cpu_starvation":
            per[g]["starv"] += 1
            m = re.search(r"ip=(\d+\.\d+\.\d+\.\d+)", str(e.get("msg") or ""))
            if m:
                hosts[m.group(1)] += 1
                per[g][m.group(1)] += 1
    for g, c in sorted(per.items()):
        if c["starv"]:
            ips = " ".join("%s=%d" % (k.split(".")[-1], v) for k, v in c.items()
                           if k.count(".") == 3)
            rows.append((n, seed.get(n), g, c["ev"], c["starv"], ips))
print("%-22s %4s %3s %4s %7s  %s" % ("run", "seed", "gen", "ev", "starv", "ip"))
for r in rows:
    print("%-22s %4s %3d %4d %7d  %s" % (r[0], r[1], r[2], r[3], r[4], r[5]))
print()
print("cpu_starvation by host ip:", dict(hosts))
print("total cpu_starvation:", sum(hosts.values()))

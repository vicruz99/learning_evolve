# Bosch cluster (`rng-dl01`) — LSF cheat sheet

Snapshot taken **2026-07-28**. Slot occupancy changes by the minute; the *structure* (queues, host
inventory, imposed defaults) changes rarely. Re-check with the commands in the last section.

Site: IBM Spectrum LSF, 93 hosts, 8 login nodes. Project `BH-000557-01`, user group
`rb_bd_dlp_rng-dl01_cr_AIQ_employees`.

---

## 1. Never compute on a login node

`rng-dl01-login1..8` appear in `lshosts` with `ncpus = -` and `server = No` — LSF does not consider
them compute resources at all. A cgroup caps the whole user slice at **5.0 cores of CPU time while
exposing 64**, and that cap is invisible to `nproc`, `sched_getaffinity`, Ray and `queue_seconds`.

Measured cost (2026-07-28, `runs/ctx_qwen/cp26_n10_s1_random`): evals ran **11.5× slower** than on the
INESC box and **56% of 1200 candidates died on the 530 s `eval_timeout`**, including several scoring
within 2% of that run's best. Full write-up in `IMPLEMENTATION_LOG.md` (2026-07-28 entry).

---

## 2. `batch_cpu` is a trap — do not queue the ICL driver there

It looks like the obvious home for CPU-only work. It is not:

| | |
|---|---|
| Hosts | **4** — `w24c01`–`w24c04` only |
| Slots per host | `MAX 256` in `bhosts`, but all four sit `closed_Full` at **NJOBS 128** (the `define_ncpus_threads` resource means `ncpus` counts threads; usable slots are the 128 physical cores) |
| Total capacity | ~512 slots for the entire site |
| Backlog | **5,961 jobs, 5,449 pending, 434 running** |
| `RUNLIMIT` | **360 min (6 h)** |
| `PROCLIMIT` | not set — the per-host slot cap is what binds |

A 6-hour wall clock also cannot hold a full sweep.

**Instead: take CPU cores on a GPU-queue node.** The GPU nodes are large and half-empty — `batch_b200`
hosts are 128-core with ~36–70 slots used. Since the vLLM server has to live on a GPU node anyway
(it already runs on `w26n05`, a b200), the cleanest design is **one job** that requests a GPU *and*
32+ cores, runs vLLM in the background and the driver in the foreground against `localhost:8001`.
That also removes the cross-node hostname discovery and the port-forwarding entirely.

---

## 3. Queues

`bqueues` as of the snapshot. `PROCLIMIT`/`JL/U`/`JL/P` are unset everywhere — per-host slots bind.

| Queue | Prio | NJOBS | PEND | RUN | Notes |
|---|---|---|---|---|---|
| `admin` | 90 | 64 | 64 | 0 | not for us |
| `dev` | 80 | 0 | 0 | 0 | **empty** — worth probing |
| `inter_a100` | 70 | 128 | **0** | 128 | **interactive, zero backlog** — best bet for a shell |
| `inter_a100_full_gpu` | 70 | 20 | 8 | 12 | |
| `short` | 70 | 0 | 0 | 0 | **empty** — check its RUNLIMIT |
| `batch_b200` | 70 | 2728 | 1791 | 937 | where the vLLM server currently lands |
| `batch_mi355x` | 70 | 48 | 32 | 16 | `w26a01/a02` were fully idle |
| `batch_h200` | 60 | 1733 | 1018 | 609 | |
| `batch_h100` | 60 | 701 | 437 | 209 | |
| `batch_a100` | 60 | 1235 | 677 | 524 | |
| `batch_cpu` | 60 | 5961 | **5449** | 434 | see §2 |
| `tryout` | 20 | 0 | 0 | 0 | `MAX 16` |

Also present: `batch_v100`, `batch_rtx6000_tfx`, `batch_*_mig`, `batch_h200_ega`, `batch_h200_fm`,
`batch_gaudi`.

**`inter_v100` no longer exists.** The `interj()` helper we were given references it, so its CPU
branch fails. Use `inter_a100` instead — which currently has **zero pending jobs**, so an interactive
request there starts immediately.

---

## 4. Host inventory

| Family | Hosts | Cores/host | RAM |
|---|---|---|---|
| a100 | `w001`–`w015`, `w079`–`w081` | 64 | 1.0 T |
| a100 (big) | `w082`–`w089` | 96 | 1.0 T |
| h100 | `w090`–`w093` | 96 | 1.4 T |
| h100 | `w24n27` | 192 | 1.4 T |
| h200 | `w24n01`–`w24n16` | 96 | 1.9 T |
| h200 | `w25n01`–`w25n17` | 128 | 2.2 T |
| b200 | `w26n01`–`w26n20` | 128 | 1.9 T |
| mi355x | `w26a01`–`w26a03` | 128 | 2.2 T |
| rtx | `w25r01` | 128 | 1.9 T |
| gaudi2 | `w094` | 96 | 1.0 T |
| volta | `w057` | 40 | 755 G |
| **CPU-only** | `w24c01`–`w24c04` | 256 (128 usable) | 1.4 T |

Largest single-host allocation possible is therefore **128 cores** on a b200/h200/mi355x node
(192 on `w24n27`), assuming they are free. Nothing gets you more than that with `span[hosts=1]`, and
we need one host because Ray's head and the `cpu_scheduler` pin cores on a single node.

---

## 5. Defaults the site's esub imposes — these will bite

`bjobs -l` on a submitted job shows what LSF actually applied:

```
Requested Resources < affinity[core] rusage[mem=1024] >
Combined: select[(ad_access==0 && type == any)] order[r15s:pg] rusage[mem=1024.00] affinity[core(1)*1]
CORELIMIT MEMLIMIT
     0 M       1 G
```

* **`MEMLIMIT` defaults to 1 GB.** The ICL driver holds 1200 candidates plus ~28k-token prompt blocks
  and will be killed. **Always pass memory explicitly**: `-M 32768MB -R "rusage[mem=32768]"`.
* **`CORELIMIT 0 M`** — no core dumps.
* **`affinity[core(1)*1]`** — LSF binds each task to a physical core. Good news: unlike the login
  node's quota, this *is* visible to `sched_getaffinity`, so `python -m sandbox.ray_doctor` inside a
  job reports the truth. Still pass `ray start --head --num-cpus=$LSB_DJOB_NUMPROC` explicitly —
  `run_sweep.py`'s `ensure_ray_head()` runs a bare `ray start --head`, which sizes itself to the node
  rather than to your reservation.
* `select[ad_access==0]` is added automatically; it is not something we set.

---

## 6. Reading a pending job

```
$ bjobs -p
 Job slot limit reached: 4 hosts;
 Not specified in job submission: 82 hosts;
 Closed by LSF administrator: 7 hosts;
```

Those three numbers sum to 93 = every host in the cluster, and that is how to read them:

* **4 hosts** — the queue's actual members (`batch_cpu` = `w24c01`–`w24c04`), all full. *This line
  tells you how many hosts your queue owns.*
* **82 hosts** — belong to other queues; your submission didn't ask for them.
* **7 hosts** — `closed_Adm`: `infra1`, `mgmt1`, `mgmt2`, `w003`, `w24n12`, `w24n14`, `w24n16`.

The same `bjobs -l` also showed the job requesting **256 Task(s)** — more than any single `batch_cpu`
host can supply (128 usable), so it could never start regardless of the backlog. Check the
`N Task(s)` line first when a job pends forever.

---

## 7. Submitting

Interactive shell with cores, on the queue that actually has room:

```bash
bsub -Is -q inter_a100 -J icl -P BH-000557-01 \
     -n 32 -R "span[hosts=1] rusage[mem=32768]" -M 32768MB -W 6:00 /bin/bash
```

Then, on the compute node:

```bash
cd ~/projects/phd/learning_evolve/src && source .venv/bin/activate
ray start --head --num-cpus=$LSB_DJOB_NUMPROC       # never a bare `ray start --head`
python -m sandbox.ray_doctor                        # expect no FAIL
```

Batch jobs live in `src/jobs/` (`icl_sweep.bsub`, `vllm_server.bsub`, and `src/jobs/README.md`).
`-J` is only a display name; `-G rb_bd_dlp_rng-dl01_cr_AIQ_employees` may be required by the esub.

Note `bslots -R "span[hosts=1]"` does **not** work here — this LSF build only accepts a `select`
string in `bslots -R`. Use `bhosts -w` and subtract `NJOBS` from `MAX`.

---

## 8. Refreshing this file

```bash
bqueues -w                       # queues, backlog, priorities
bqueues -l <queue>               # RUNLIMIT, PROCLIMIT, HOSTS, interactive allowed?
bhosts -w                        # per-host MAX vs NJOBS -> free slots
lshosts -w                       # cores / RAM / GPU family per host
bjobs -p ; bjobs -l <jobid>      # pending reasons, and what the job really requested
busers $USER ; blimits -c        # your slot ceiling and any project-wide caps
bmgroup                          # host groups
```

# Running on Bosch (LSF) without the login node

The login node caps the whole user slice at **5 cores of CPU time** while showing 64. Nothing in the
stack sees that cap — `nproc`, `sched_getaffinity`, Ray and `queue_seconds` all report a healthy
64-core box — so the only symptom is that every eval runs ~12x slow and dies on `eval_timeout`.
Measured 2026-07-28: 11.5x slower than the INESC box, 56% of 1200 candidates timed out, and several
of the discarded ones scored within 2% of that run's best. See `docs/IMPLEMENTATION_LOG.md`.

**Queue and host facts (measured 2026-07-28) live in `docs/BOSCH_CLUSTER.md`.** Two of them override
what the scripts below assume by default: `batch_cpu` is only 4 hosts with a 5,400-job backlog and a
6 h limit, and LSF imposes a **1 GB memory limit** unless you ask for more.

## "bqueues only shows GPU machines"

Queue names here are GPU-flavoured (`inter_v100`, `inter_a100`, `batch_h200`), but **a queue does not
force a GPU onto you**. You get a GPU only if the job asks with `-gpu "num=N"`. Your supervisor's own
helper makes this explicit — its CPU branch submits to `inter_v100` with no `-gpu` flag:

```bash
bsub -Is -q inter_v100 -J vsc -P BH-000557-01 -n 2 -W 6:00 /bin/bash   # "CPU-JOB": 2 cores, no GPU
```

So `jobs/icl_sweep.bsub` requesting `-n 32` with no `-gpu` line gets 32 CPU cores on a compute node.

Worth confirming whether the site also has a plain CPU queue (cheaper, no GPU node wasted):

```bash
bqueues -w                       # every queue, full names
bqueues -l batch_h200            # wall-clock ceiling, max slots per job, who may submit
bhosts -w                        # hosts and their slot counts
lshosts -w                       # cores / memory / model per host
bmgroup                          # host groups (CPU-only pools usually show up here)
```

Pick whichever queue has the wall-clock limit your sweeps need and pass it at submit time:
`bsub -q <queue> < jobs/icl_sweep.bsub` (a command-line flag overrides the `#BSUB` line in the file).

## Two jobs, one server

```bash
cd ~/projects/phd/learning_evolve/src
mkdir -p jobs/logs

bsub < jobs/vllm_server.bsub                              # GPU job; writes jobs/vllm_host.txt
bjobs -w                                                  # wait until RUN
SWEEP=sweeps/ctx_qwen.yaml bsub < jobs/icl_sweep.bsub     # CPU job; finds the server by hostname
```

The driver job waits for the server's `/health`, rewrites `vllm-base-url` in a scratch copy of the
sweep yaml (the server's node changes on every resubmit), starts a Ray head **sized to
`$LSB_DJOB_NUMPROC`**, prints `ray_doctor`, and runs `run_sweep.py --ray-head require`.

That sizing is the part that matters. `run_sweep.py`'s `ensure_ray_head()` runs a bare
`ray start --head`, which reads the **node's** core count rather than your reservation — on a shared
node that reproduces the login-node failure exactly. Starting the head ourselves with an explicit
`--num-cpus` is what prevents it, and `--ray-head require` makes the job fail loudly instead of
silently falling back.

Check the first minute of `jobs/logs/icl_sweep.<jobid>.out`: `ray_doctor` should report no FAIL, and
`slots (LSF)` / `affinity` / `cgroup cpu.max` should agree. If `cpu.max` shows a quota well below
your slot count, LSF is enforcing by quota rather than cpuset — lower `-n` to match and resubmit.

## Trimul (kernel) sweeps: `jobs/trimul_sweep.bsub`

The kernel task inverts both of `icl_sweep.bsub`'s defining choices, so it has its own script:
the **driver's job needs a GPU** (`-gpu "num=1"` — grading is a local subprocess on this node's
card, not a Ray task), and it needs **no Ray at all** (`--ray-head skip`; there is no `ray start`,
no `ray_doctor`). Only `-n 4`: the flock serialises grading to one eval at a time.

```bash
SWEEP=sweeps/trimul_bon_qwen_bosch.yaml bsub < jobs/trimul_sweep.bsub    # default queue batch_h100
```

The tracked `sweeps/trimul_*_qwen.yaml` are written for guadiana — make the Bosch copy first
("ON ANOTHER MACHINE" in `sweeps/trimul_bon_qwen.yaml`). The job refuses a `problem:` that doesn't
match the queue's card, and refuses to start unless `$KPY` (default `~/venvs/kernel-eval/bin/python`;
creation instructions in `gpumode_local/reference/README.md`) carries torch + triton 3.3.1. It
rewrites `vllm-base-url`, `trimul-eval-python` and `trimul-eval-gpu` into the scratch copy of the
yaml — inside the job LSF's `CUDA_VISIBLE_DEVICES` makes the allocated card "GPU 0".

## Interactive instead

Same allocation, a shell you can watch. Wall-clock limits on interactive queues are short (6h), so
this is for debugging, not for a full sweep:

```bash
bsub -Is -q inter_v100 -J icl -P BH-000557-01 -n 32 -R "span[hosts=1]" -W 6:00 /bin/bash
# then, on the compute node:
cd ~/projects/phd/learning_evolve/src && source .venv/bin/activate
ray start --head --num-cpus=$LSB_DJOB_NUMPROC
python -m sandbox.ray_doctor
tmux new -s sweep    # so the run survives a dropped ssh (the job still dies at -W)
python run_sweep.py sweeps/ctx_qwen.yaml --ray-head require
```

## Using fewer cores than LSF gave you

Sizing is normally detected (affinity mask, cgroup quota, `LSB_DJOB_NUMPROC`, minus a core per
driver). To choose it yourself — sharing the node with another job of yours, leaving room for a
server on the same host, or reproducing a run made on a smaller box:

```bash
python run_sweep.py sweeps/ctx_qwen.yaml --ray-num-cpus 16      # or sweep.ray.num_cpus: 16 in the yaml
python run_sweep.py sweeps/ctx_qwen.yaml --ray-num-cpus 16 --print-cmds   # see the sizing, start nothing
```

That number **is** the head's `--num-cpus`: `reserve_cpus` is not subtracted on top, and the rest of
the allocation is left idle. It only applies on `--ray-head auto` — a head that is already running
cannot be resized, so with `--ray-head require` (what `icl_sweep.bsub` uses) put the number on the
`ray start --head --num-cpus=N` line instead. Asking for more than the affinity mask holds is warned
about, not refused: `cpu_scheduler` builds its groups from that mask, so the surplus evals would spin
until they failed as `cpu_starvation`.

## Port-forwarding to your laptop

This is the *access* half, not a substitute for the reservation — it lets you reach a service running
on a compute node. Compute nodes are not directly reachable, so hop through the login node:

```bash
# on your laptop; rng-dl01-w26n05 is the compute node (bjobs -w tells you which)
ssh -N -L 8001:rng-dl01-w26n05:8001 <user>@rng-dl01-login1.de.bosch.com
```

Then `http://localhost:8001/v1` on the laptop is the vLLM server. Same pattern for a Ray dashboard
(`-L 8265:<node>:8265`) or a notebook. Add `-o ServerAliveInterval=60` so the tunnel survives idling.

Put it in `~/.ssh/config` and it becomes one command:

```
Host bosch-login
    HostName rng-dl01-login1.de.bosch.com
    User <user>
    ServerAliveInterval 60
Host bosch-worker
    HostName rng-dl01-w26n05
    User <user>
    ProxyJump bosch-login
```

`ssh -N -L 8001:localhost:8001 bosch-worker` then tunnels straight to the worker.

## Do not

- Run `run_icl.py`, `run_sweep.py` or `ray start` on the login node.
- Run `ray start --head` without `--num-cpus` inside a job.
- Trust `nproc` for how many evals you may run — it reports the node, not your share.

# TriMul reference kernel — use this to sanity-check a new GPU

`trimul_best.py` is **the best TriMul kernel TTT-Discover discovered**, byte-identical to
`discover/results/kernel-engineering/trimul.py` (and to `coding_agent_evolve/gpumode/test/candidate.py`).
16420 bytes, `sha256 3ff4aed0d3385cc07b74035c75eb679874366c631b396005e12b9a2093657b79`.

Re-verify the copy at any time:

```bash
diff src/gpumode_local/reference/trimul_best.py discover/results/kernel-engineering/trimul.py && echo identical
```

## What it should score

| card | score | source |
|---|---:|---|
| A100 | **2198 µs** | TTT-Discover's reported best |
| H100 | **1161 µs** | TTT-Discover's reported best |
| A100 80GB **PCIe** (guadiana, GPU 1) | **2467 µs** | measured here 2026-08-10, mean of 3, spread 1.0 % |

⚠️ **Read the third row before using the first as a pass/fail gate.** On guadiana we measure
2467 µs, i.e. **12 % above** the 2198 µs reference, reproducibly (2457.4 / 2482.5 / 2461.2, and
2491.0 / 2487.3 in earlier standalone runs — every measurement lands in 2457–2491). Correctness
passes 18/18 shapes every time, so this is not a broken kernel or a broken harness. The most likely
cause is that guadiana's card is an **A100 80GB PCIe** while the reference is presumably an
**A100 SXM** (higher memory bandwidth, higher sustained clocks); torch/triton build differences can
add a little more. Treat "A100" as a family, not a number: **establish your own baseline per machine**
with the command below and compare against that.

The ~1 % run-to-run spread only holds on an **idle** card. On a shared GPU `eval.py`'s convergence
rule (relative error < 0.1 %) can never be met, so every benchmark burns its full rep budget and the
numbers are junk. Check with `nvidia-smi` first.

## Debugging a new GPU — the commands

Everything below runs the *grader* directly; no LLM, no ICL loop, no Ray.

**Step 0 is not optional, and skipping it produces a misleading error.** `$KPY` must name an
interpreter that has torch 2.7.1 / triton 3.3.1 (see `coding_agent_evolve/gpumode/requirements.txt`).
If it is unset, `$KPY evaluate.py ...` collapses to `evaluate.py ...`, bash tries to execute the script
itself, and you get **`Permission denied`** — see Troubleshooting below. Run this first, in every new
shell:

```bash
export KPY=/scratch/vicstorage/learning_evolve/.venv/bin/python   # guadiana
# export KPY=~/venvs/kernel-eval/bin/python                       # Bosch, once created (see below)
# triton MUST be 3.3.1; the cu suffix may be cu126 or cu128 (measured identical on this task).
# guadiana's venv reports 2.7.1+cu126; a fresh cu128 install reports 2.7.1+cu128.
"$KPY" -c "import torch, triton; print(torch.__version__, triton.__version__)"
```

Then, from the repo root:

```bash

# 1. correctness only — fastest signal that the card and the stack work at all (~28 s)
$KPY coding_agent_evolve/gpumode/evaluate.py \
    src/gpumode_local/reference/trimul_best.py --gpu 0 --mode test

# 2. the score, and the number to compare against the table above (~11-23 s)
$KPY coding_agent_evolve/gpumode/evaluate.py \
    src/gpumode_local/reference/trimul_best.py --gpu 0 --mode benchmark

# 3. establish this machine's baseline + its noise floor before trusting any small win
$KPY coding_agent_evolve/gpumode/evaluate.py \
    src/gpumode_local/reference/trimul_best.py --gpu 0 --mode benchmark \
    --repeats 5 --json /tmp/trimul_baseline.json

# 4. the official ranked path (correctness re-checked every rep). ~100x slower — only for a final
#    number you intend to quote, never in a loop.
$KPY coding_agent_evolve/gpumode/evaluate.py \
    src/gpumode_local/reference/trimul_best.py --gpu 0 --mode leaderboard
```

Output ends with `SCORE (geom of 7 benchmarks): <us>`; exit code is 0 only if everything passed.
Add `-v` for stderr tails when something fails.

On a **shared** card, `--max-input-gib 4` drops the largest shapes so the run does not OOM — but the
score then covers fewer benchmarks and is no longer comparable to the table.

### Troubleshooting: `Permission denied`

```
bash: coding_agent_evolve/gpumode/evaluate.py: Permission denied
```

This is never a filesystem-permissions problem. It means the script was **executed directly** instead
of being handed to an interpreter — either because you typed `./evaluate.py`, or because `$KPY` was
unset so `$KPY evaluate.py …` collapsed to `evaluate.py …`. Fix: `echo "$KPY"`, re-run the `export`
above, and always invoke it as `"$KPY" coding_agent_evolve/gpumode/evaluate.py …`.

`evaluate.py` is mode `644` **on purpose**, even though it carries a `#!/usr/bin/env python3` shebang.
Making it executable would let that shebang pick up whatever `python3` comes first on `PATH` — on Bosch
the system python, which has no torch — turning a loud, obvious `Permission denied` into either a
confusing `ModuleNotFoundError` or, far worse, a *successful* run on an unknown torch/triton pair. This
harness only produces meaningful timings against a known stack, so it should refuse to run rather than
guess the interpreter. Do not `chmod +x` it.

### Same check, through the ICL harness's own evaluator

Use this to confirm the *reward path* (not just the grader) works on a new box — it is what a real run
calls, `flock` and all:

```bash
cd src && TRIMUL_EVAL_GPU=0 .venv/bin/python -c "
from envs.kernel_trimul import TrimulLocalReward
from puct import State
code = open('gpumode_local/reference/trimul_best.py').read()
r = TrimulLocalReward('trimul_h100', '/tmp', eval_timeout=1200)   # or trimul_a100
o = r.get_reward(code, State(timestep=-1, construction=None, code='', value=-1_000_000))
print('correctness', o['correctness'], 'score_us', o['raw_score'], 'reward', o['reward'])
print('timing', r._last_timing)
"
```

On a machine that is not guadiana, also set:

- `TRIMUL_EVAL_PYTHON=/path/to/torch2.7.1-triton3.3.1/bin/python`
- `TRIMUL_EVALUATE_PY=/path/to/coding_agent_evolve/gpumode/evaluate.py`

For an actual **run**, prefer the sweep keys over these variables — `trimul-eval-python`,
`trimul-eval-gpu`, `trimul-evaluate-py`, `trimul-eval-mode`. They are validated at parse time, printed
by `--print-cmds`, and recorded in the run's `config.json`, so a finished run says which interpreter
and which card produced its timings. The environment variables remain the fallback (a flag always
wins) and are the right thing for the one-off commands above.

## Setting this up on the Bosch cluster (`rng-dl01`)

Read `docs/BOSCH_CLUSTER.md` alongside this. Two site facts drive everything below: **never compute on
a login node** (a cgroup caps the user slice at 5.0 cores while exposing 64 — measured 11.5× slower
evals), and outbound network needs the **proxy module** or every download hangs.

### Where the venv goes

Both existing envs live in home — `src/.venv` (ICL) and `~/envs` (vLLM, activated as
`~/envs/bin/activate` in `jobs/vllm_server.bsub`, so **`~/envs` is itself a venv root, not a directory
of venvs** — do not nest inside it). Use a sibling: **`~/venvs/kernel-eval`**. There is no scratch
filesystem in use here, and `$TMPDIR` is node-local and reclaimed when the job ends, so it cannot hold
a venv.

Check space first: torch cu128 plus its bundled `nvidia-*` wheels is **~5–7 GB**, on top of the two
venvs already in home.

```bash
df -h ~ ; du -sh ~/envs ~/projects/phd/learning_evolve/src/.venv
```

### Create it

Get an interactive shell **with a GPU**, so the install can be verified where it will run.
`inter_a100` had zero backlog at the last snapshot, so it starts immediately; fall back to
`inter_a100_full_gpu` or a `batch_*` queue if the GPU request is refused.

```bash
bsub -Is -q inter_a100 -gpu "num=1" -J kvenv -P BH-000557-01 \
     -G rb_bd_dlp_rng-dl01_cr_AIQ_employees \
     -n 8 -R "span[hosts=1] rusage[mem=16384]" -M 16384MB -W 2:00 /bin/bash
```

Then on the compute node. **The proxy block is not optional** — it is what gives the job outbound
network, and it is why `jobs/vllm_server.bsub` and `jobs/icl_sweep.bsub` both start with it:

```bash
source /fs/applications/modules/current/init/bash
module load proxy4server-access/2.0
sleep 1
source /fs/applications/p4s-access/2.0/ActivateP4S.sh -a

python3 -m venv ~/venvs/kernel-eval          # or `uv venv ~/venvs/kernel-eval` if uv is available
source ~/venvs/kernel-eval/bin/activate
pip install --upgrade pip
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install pyyaml numpy
```

- **`cu126` works identically** — `coding_agent_evolve/gpumode/requirements.txt` records that both
  builds of torch 2.7.1 were *measured* to give the same timings on this task. cu128 matches the
  official GPU-mode harness image, so prefer it. torch bundles its own CUDA runtime, so it does not
  have to match `module load cuda/12.6.0`.
- **Do not install triton separately.** It ships with torch, and 2.7.1 provides exactly the 3.3.1 the
  harness pins. Installing it by hand is how you end up with a mismatched pair.

### Verify, still on the GPU node

```bash
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
```

Expect `2.7.1+cu128 3.3.1 NVIDIA A100-SXM4-...`. If `torch.cuda.is_available()` is False the driver is
older than that build needs — reinstall from the `cu126` index.

Then run step 3 from "Debugging a new GPU" above with
`export KPY=~/venvs/kernel-eval/bin/python` — always naming the interpreter explicitly, never
`./evaluate.py` — and **record the number as this machine's baseline.** Do not compare it to guadiana's 2467 µs: Bosch's A100s are
likely SXM, so they may land *closer* to the 2198 µs reference than guadiana does. `--repeats 5` gives
the noise floor to judge later wins against.

### The driver job needs a GPU — `jobs/icl_sweep.bsub` does not request one

Every math problem grades on CPU through Ray, so that script has **no `-gpu` line by design**, starts a
Ray head, and passes `--ray-head require`. trimul is the opposite on both counts: the card must be in
the **driver's** job (grading is a local subprocess, not a Ray task) and Ray is never initialised at all
(`uses_sandbox = False`). To adapt it:

1. add `#BSUB -gpu "num=1"`;
2. match the queue to the problem — `batch_a100` → `trimul_a100`, `batch_h100` / `batch_h200` →
   `trimul_h100`. **Avoid `batch_b200`**: sm100 Blackwell predates torch 2.7.1 / triton 3.3.1, so it
   may not compile, and upgrading torch to fix that breaks comparability with every other number here;
3. drop the `ray start` / `ray_doctor` / cleanup block;
4. pass **`--ray-head skip`**, not `require`. `run_sweep.py:308` raises
   `SweepError("no Ray head reachable")` under `require`, and nothing in `run_sweep` consults
   `uses_sandbox` — so the sweep would refuse to launch over a head trimul never wanted. `auto` would
   instead start a head nobody uses.

The vLLM server can stay in its own job on another node, exactly as it does for the math sweeps; only
grading has to be local to the card.

Then the sweep file needs three changes: `problem: trimul_a100` (or `_h100`) per run,
`trimul-eval-python: ~/venvs/kernel-eval/bin/python`, and `trimul-eval-gpu: "0"` — with `-gpu "num=1"`
LSF gives you a single card, visible as index 0.

Leave `TRIMUL_LOCK_DIR` alone. The `/tmp` default is correct **because** it is node-local: a GPU
belongs to one host, so every process that can contend for it runs on that host, and `flock` over NFS
is not dependable. Pointing it at home would buy nothing and could silently fail to exclude.

## Which problem to run on which card

`trimul_a100` and `trimul_h100` differ **only** in the rules line naming the target GPU, but that line
steers the model's block sizes: an H100-legal `BLOCK_H=128, BLOCK_K=64` wants 180 KB of shared memory
and dies on the A100's 166912-byte limit. The first smoke run lost a candidate to exactly that. So
match the problem to the card, and **never compare scores across architectures** — 2198 vs 1161 µs for
the same kernel is the size of the effect.

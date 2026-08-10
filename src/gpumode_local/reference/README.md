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

```bash
# 0. one-time: an env with torch 2.7.1 + triton 3.3.1 (see coding_agent_evolve/gpumode/requirements.txt)
#    On guadiana that already exists:
KPY=/scratch/vicstorage/learning_evolve/.venv/bin/python

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

## Which problem to run on which card

`trimul_a100` and `trimul_h100` differ **only** in the rules line naming the target GPU, but that line
steers the model's block sizes: an H100-legal `BLOCK_H=128, BLOCK_K=64` wants 180 KB of shared memory
and dies on the A100's 166912-byte limit. The first smoke run lost a candidate to exactly that. So
match the problem to the card, and **never compare scores across architectures** — 2198 vs 1161 µs for
the same kernel is the size of the effect.

# TTT-Discover kernel task (GPU-mode TriMul) for ShinkaEvolve

The kernel-engineering counterpart of `../ttt_discover_math/`: the same TriMul task the
`trimul_h100` ICL sweeps and the coding-agent arm solve, handed to ShinkaEvolve. Symlinked into
the pinned checkout as `ShinkaEvolve/examples/ttt_discover_kernel` by `../setup_shinka.sh`.

| file | what |
|---|---|
| `trimul/run_evo.py` | Shinka driver. `task_sys_msg` = `TRIMUL_PROMPT` (from `src/envs/kernel_prompt.py`, loaded by path) + the ICL Rules block with `_HW_RULE_H100` verbatim. |
| `trimul/evaluate.py` | **External grader.** Hands the candidate file to the frozen GPU-mode harness (`coding_agent_evolve/gpumode/evaluate.py`) under `~/venvs/kernel-eval/bin/python`, then writes Shinka's `metrics.json` / `correct.json`. `combined_score = 1500 / runtime_us` (TTT-Discover's reward), 0 on failure. |
| `trimul/initial.py` | NotImplemented stub inside the EVOLVE-BLOCK — the ICL arm starts from *no* kernel, and this is the closest Shinka equivalent. |
| `trimul/shinka_qwen.yaml` | Budget-matched config: 2400 generations, `max_evaluation_jobs: 1` (one card), Qwen kwargs as the trimul ICL sweeps. |

## What differs from the math ports, and why

- **Grading is out-of-process.** The math ports call `shinka.core.run_shinka_eval` and import the
  candidate; a kernel must be graded by the pinned torch 2.7.1 / triton 3.3.1 interpreter on an
  idle GPU, so `evaluate.py` shells out exactly as `src/envs/kernel_trimul.py` does and applies the
  same two gates (`@triton.jit` required, `identity` banned).
- **One evaluation at a time.** Leaderboard timings are meaningless on a shared card. Never raise
  `max_evaluation_jobs` above 1 unless the job holds more than one GPU and the grader is taught to
  spread candidates across them.
- **Embeddings on CPU.** On Marvin the job's single GPU is exclusive-process; the duplicate gate's
  embedder (`Qwen/Qwen3-Embedding-0.6B`) runs on the job's cores via `../embed_server.py`.

## Running on Marvin (rb-hpc)

```bash
cd ~/work/learning_evolve/src && mkdir -p jobs/logs
REP=r1 SHINKA_SMOKE=1 bsub -W 00:30 < jobs/marvin_shinka_run.bsub   # grader check + 3 generations
REP=r1 bsub < jobs/marvin_shinka_run.bsub                             # the run; resubmit to resume
```
Results: `ShinkaEvolve/examples/ttt_discover_kernel/trimul/results/trimul_qwen_<REP>/` (gitignored
through the `ShinkaEvolve/` clone; copy what you need).

Yardstick on these H100s: the published TTT-Discover kernel scores **1182 µs** (5 repeats, 0.6 %
spread) → `combined_score ≈ 1.27`.

## The seed program (`trimul/initial.py`) — read before changing it

The whole file, **imports included**, is inside the single EVOLVE-BLOCK, and it carries no
docstring or comments. Both are deliberate:

- Shinka only ever rewrites the text between the markers; anything outside is copied verbatim
  into every child. The first version kept `import torch` outside the block, so every
  `import triton` the model wrote at the top of its rewrite was discarded and 82 of the first
  111 candidates died with `NameError: name 'triton' is not defined` (runs r1/r2 of
  2026-08-27, discarded).
- The program text is shown to the model as the parent. A docstring explaining the experiment
  ("do not put a working kernel here") is an instruction the model reads.

Why a stub at all: TTT-Discover and the ICL arm start from an empty program (value -1e6); Shinka
needs a parseable initial program, so this is the prompt's entrypoint skeleton raising
`NotImplementedError`. It fails the `@triton.jit` gate without touching the GPU, scores 0, and
the first real generation is a full rewrite -- the same starting point the other arms get. Do
not seed a working kernel (see `coding_agent_evolve/gpumode/b200_task/README.md`, "No seed kernel").

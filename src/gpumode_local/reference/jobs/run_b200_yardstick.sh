#!/usr/bin/env bash
# B200 yardstick for the trimul task: grades the B200-patched copy of the
# published TTT-Discover kernel (see trimul_best_b200.py header for why the
# pristine trimul_best.py cannot pass on this card).
#
# Run this INSIDE a GPU job shell (bsub -Is ... -gpu num=1) so LSF has set
# CUDA_VISIBLE_DEVICES. The task copy raises only ranked_timeout: the stock
# 1200 s is too small for the leaderboard phase even on a healthy run
# (7 shapes x up to 120 s each, plus a reference recheck every iteration).
#
# Known failure mode: the B200 stack (driver 610.43.02 / triton 3.3.1)
# intermittently wedges the GPU mid-kernel -- the run then sits at 100% GPU
# forever and the phase dies on its timeout with no output. That is a wedge,
# not a slow eval: kill it and rerun.
set -euo pipefail
cd "$(dirname "$0")"
KPY=${KPY:-$HOME/venvs/kernel-eval/bin/python}
TASK_COPY=$(mktemp -d)/trimul
cp -r coding_agent_evolve/gpumode/trimul "$TASK_COPY"
sed -i "s/ranked_timeout: 1200/ranked_timeout: 3600/" "$TASK_COPY/task.yml"
"$KPY" coding_agent_evolve/gpumode/evaluate.py \
    src/gpumode_local/reference/trimul_best_b200.py \
    --task "$TASK_COPY" --mode leaderboard --repeats "${1:-5}"

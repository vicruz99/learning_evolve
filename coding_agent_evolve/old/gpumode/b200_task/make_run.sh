#!/usr/bin/env bash
# Materialise a self-contained TriMul/B200 run folder for a coding-agent run.
#
#   ./make_run.sh ~/agent_runs/trimul_b200_qwen1
#
# The run folder must live OUTSIDE this repo. An agent working inside it would inherit the
# repo's CLAUDE.md (which orders it to read docs/PROJECT_CONTEXT.md and docs/EXPERIMENT_PLAN.md,
# both of which discuss this exact task), reuse the project's auto-memory, and find the
# published TTT-Discover kernel in a sibling directory. See gpumode/run_description.md.
#
# What lands in the folder: the prompt, the frozen harness, and nothing else. In particular
# NOT test/candidate.py or gpumode_local/reference/trimul_best.py -- those ARE the published
# solution, and the ICL arm starts from no seed at all, so this one does too.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUMODE="$(dirname "$HERE")"
DEST="${1:?usage: make_run.sh <run_dir>}"

[ -e "$DEST" ] && { echo "refusing to overwrite existing $DEST" >&2; exit 1; }
case "$(readlink -f "$DEST")" in
  "$(readlink -f "$GPUMODE/../..")"/*)
     echo "refusing: $DEST is inside the repo -- see the comment at the top of this script" >&2
     exit 1;;
esac

mkdir -p "$DEST/attempts"
cp "$HERE/INITIAL_PROMPT.md" "$DEST/"
cp "$GPUMODE/evaluate.py"    "$DEST/"
cp -r "$GPUMODE/trimul"      "$DEST/"
chmod 644 "$DEST/evaluate.py"      # never executable: see gpumode_local/reference/README.md

cat > "$DEST/LEDGER.md" <<'LEDGER'
# Candidate ledger

Score = `SCORE (geom of 7 benchmarks)` from `--mode leaderboard`, in microseconds, lower better.

| # | family | description | correct | score (us) | vs champion | verdict |
|---|---|---|---:|---:|---:|---|
LEDGER

echo "run folder ready: $DEST"
echo
echo "Before starting the agent, on the node that holds the card:"
echo "  export KPY=~/venvs/kernel-eval/bin/python"
echo "  \"\$KPY\" -c \"import torch,triton;print(torch.__version__,triton.__version__,torch.cuda.get_arch_list())\""
echo "  # must be 2.7.1+cu128 / 3.3.1 and the arch list MUST contain sm_100, or nothing here is valid"

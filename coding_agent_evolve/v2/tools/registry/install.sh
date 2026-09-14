#!/usr/bin/env bash
# Install the registry CLI for the coding agents and register the AC1 / AC2 / Erdos problems.
#
#   bash tools/registry/install.sh            # install + register problems (idempotent)
#   bash tools/registry/install.sh --no-problems
#
# The CLI is installed as a COPY at ~/.local/bin/registry (not a symlink), so editing the repo
# while cells are running cannot change the tool they use; each run records the installed copy's
# sha256 in its config table. Re-run this script to roll a new version out to future cells.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"                 # coding_agent_evolve/
EXP="$REPO/experiments"
BIN="${REGISTRY_BIN:-$HOME/.local/bin}"
PY="${REGISTRY_PYTHON:-/home/crv1pi/venvs/agent-eval/bin/python}"

mkdir -p "$BIN"
tmp="$BIN/.registry.tmp.$$"
cp "$HERE/registry.py" "$tmp"
chmod 755 "$tmp"
mv -f "$tmp" "$BIN/registry"                      # atomic: a running `registry` keeps its old inode
echo "installed $BIN/registry  sha256=$(sha256sum "$BIN/registry" | cut -c1-16)..."
if [ ! -x "$PY" ]; then
  echo "WARNING: $PY not found; the shebang in registry.py points there" >&2
fi

[ "${1:-}" = "--no-problems" ] && exit 0

reg() {  # name direction seed-file
  local name=$1 dir=$2 seed=$3 d="$EXP/$1"
  if [ ! -f "$d/eval.py" ]; then echo "skip $name: $d/eval.py missing" >&2; return; fi
  "$BIN/registry" problem add "$name" --force --direction "$dir" \
      --eval "$PY eval.py {artifact}" --files "$d/eval.py" --baseline "$d/$seed" --timeout 1100 \
      --description "$4"
}
reg AC1   min height_sequence_1.npy "Autocorrelation inequality, upper bound (minimise). Target 1.5030. Entrypoint propose_candidate."
reg AC2   max height_sequence_1.npy "Autocorrelation inequality, lower bound (maximise). Target 0.97. Entrypoint construct_function."
reg Erdos min initial_h_values.npy  "Erdos minimum overlap C5 (minimise). Record 0.38092, milestone 0.38080. Entrypoint run."
echo
"$BIN/registry" problem list
cat <<EOF

Next, in a run folder:   registry init --problem AC2 --baseline-expect 0.6667
(TriMul is not registered here: its evaluator needs the kernel-eval venv and a GPU. When wanted:
  registry problem add trimul_b200 --direction min --timeout 1500 \\
      --eval "\$HOME/venvs/kernel-eval/bin/python evaluate.py {program} --mode leaderboard" \\
      --files $REPO/gpumode/evaluate.py <trimul harness files> --artifact-ext .py )
EOF

#!/bin/bash
# =================================================================================================
# Recreates the patched ShinkaEvolve checkout used for the baseline campaign, on a machine that
# doesn't have one yet (Bosch). After this, src/jobs/{vllm_server,vllm_embed,shinka_run}.bsub
# work unchanged — they expect $ROOT/ShinkaEvolve/examples/ttt_discover_math/<problem> and
# $ROOT/ShinkaEvolve/.venv.
#
# What it does:
#   1. clone SakanaAI/ShinkaEvolve into <project root>/ShinkaEvolve, pinned to the exact
#      upstream commit the guadiana runs used
#   2. apply shinka_local.patch (the 5 local fixes: eval kill-clock start time, 5k-char embed
#      prefix, extra_body passthrough for vLLM's thinking_token_budget, reasoning/
#      reasoning_content compat)
#   3. symlink examples/ttt_discover_math -> src/shinka_baseline/ttt_discover_math, so the
#      problem dirs stay tracked in THIS repo (results/ and logs are gitignored)
#
# It does NOT build the venv (Bosch needs the proxy modules for pip) — it prints the commands.
# =================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"      # the learning_evolve checkout
HERE="$ROOT/src/shinka_baseline"
UPSTREAM="https://github.com/SakanaAI/ShinkaEvolve.git"
PIN="b67a07328ab7e21e999d9e20a44f4f0054a4b83c"                  # upstream main, 2026-08 campaign

if [ -e "$ROOT/ShinkaEvolve" ]; then
    echo "ERROR: $ROOT/ShinkaEvolve already exists — refusing to touch it." >&2
    echo "(On guadiana the live checkout IS the original; this script is for fresh machines.)" >&2
    exit 1
fi

git clone "$UPSTREAM" "$ROOT/ShinkaEvolve"
git -C "$ROOT/ShinkaEvolve" checkout "$PIN"
git -C "$ROOT/ShinkaEvolve" apply --stat --check "$HERE/shinka_local.patch"
git -C "$ROOT/ShinkaEvolve" apply "$HERE/shinka_local.patch"
ln -s "$HERE/ttt_discover_math" "$ROOT/ShinkaEvolve/examples/ttt_discover_math"

echo
echo "ShinkaEvolve checkout ready at $ROOT/ShinkaEvolve (upstream $PIN + local patches)."
echo "Next (on Bosch, activate the proxy first — see docs/BOSCH_CLUSTER.md):"
echo "  source /fs/applications/modules/current/init/bash"
echo "  module load proxy4server-access/2.0 && source /fs/applications/p4s-access/2.0/ActivateP4S.sh -a"
echo "  cd $ROOT/ShinkaEvolve && python3 -m venv .venv && .venv/bin/pip install -e ."
echo "Then follow src/shinka_baseline/README.md to launch runs."

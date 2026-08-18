#!/usr/bin/env bash
# Fetch froggeric's fixed Qwen chat template and check it against this checkpoint.
#
#   ./fetch_chat_template.sh [dest_dir]     # default: alongside this script
#
# On Bosch this needs outbound network, so run it on a COMPUTE NODE, not the login node:
#   source /fs/applications/modules/current/init/bash
#   module load proxy4server-access/2.0 && sleep 1
#   source /fs/applications/p4s-access/2.0/ActivateP4S.sh -a
#
# Pin the VERSION. The repo's root chat_template.jinja is the Qwen3.8 template (v22.1) --
# wrong model. The 3.6 line lives under archive/qwen3.6/, and v19 is its newest. A template
# change alters every rendered prompt, so runs across two versions are not comparable:
# treat this like a model version, not a bug fix.
set -euo pipefail

VERSION=${VERSION:-v19}
DEST=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
URL="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/archive/qwen3.6/chat_template-${VERSION}.jinja"
OUT="$DEST/qwen3.6_chat_template-${VERSION}.jinja"

echo "[fetch] $URL"
curl -fsSL -m 120 -o "$OUT" "$URL"
echo "[fetch] wrote $OUT ($(wc -c < "$OUT") bytes)"
echo "[fetch] sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"
echo
echo "Serve with it:  CHAT_TEMPLATE=$OUT ./serve_qwen.sh"
echo "Check it:       python compare_templates.py /scratch/vicstorage/qwen/chat_template.jinja $OUT"

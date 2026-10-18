#!/usr/bin/env bash
# Writes demo/smol.bend's data into data/smol135/ (or data/smol360/; reference/smol_ref.py --prep): init.bin, the
# weights as F32 in reference/smol_layout.txt's order (smol360_layout.txt), and batches.bin, 64 steps of 2 x 129 token
# ids from Tiny Shakespeare (seed 0). The checkpoints and the Python that reads them (safetensors, tokenizers) come from
# llama.bend, which downloads the pinned revisions (reference/models.py there); their files are checked by SHA256 here.
#
#   scripts/smol_data.sh [135m | 360m]   # LLAMA_BEND=<path to llama.bend> if it is not ../llama.bend
set -euo pipefail
cd "$(dirname "$0")/.."
SMOL=${1:-135m}
case "$SMOL" in
  135m) D=data/smol135; W=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1; N=538060032 ;;
  360m) D=data/smol360; W=7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f; N=1447284480 ;;
  *) echo "usage: scripts/smol_data.sh [135m | 360m]"; exit 2 ;;
esac
export SMOL
LB=$(realpath "${LLAMA_BEND:-../llama.bend}")
HF=$LB/data/smollm2-$SMOL/hf
PY=$LB/reference/.venv/bin/python
[ -f "$HF/model.safetensors" ] && [ -x "$PY" ] || { echo "no SmolLM2-$SMOL checkpoint or venv in $LB (set LLAMA_BEND)"; exit 1; }
[ -f data/tiny/input.txt ] || scripts/tiny.sh
sha256sum -c - >/dev/null <<EOF || { echo "SHA256 mismatch in $HF"; exit 1; }
$W  $HF/model.safetensors
9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c  $HF/tokenizer.json
EOF
[ -f "$D/init.bin" ] && [ -f "$D/batches.bin" ] || "$PY" reference/smol_ref.py --prep "$HF"
[ "$(stat -c %s "$D/init.bin")" = $N ] || { echo "$D/init.bin: wrong size"; exit 1; }
[ "$(stat -c %s "$D/batches.bin")" = $((64 * 258 * 4)) ] || { echo "$D/batches.bin: wrong size"; exit 1; }
echo "SmolLM2-$SMOL data ready in $D"

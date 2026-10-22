#!/usr/bin/env bash
# Builds demo/smol_block.bend at one of SmolLM2's sizes into .scratch/smol_block_<size>: a copy of the program with its
# five size defs rewritten (width, key/value width, feed-forward, key/value heads, query heads per key/value head).
#
#   scripts/block.sh 135m | 360m | 1.7b
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  135m) d=576;  kvw=192;  ff=1536; kv=3;  g=3 ;;
  360m) d=960;  kvw=320;  ff=2560; kv=5;  g=3 ;;
  1.7b) d=2048; kvw=2048; ff=8192; kv=32; g=1 ;;
  *) echo "usage: scripts/block.sh 135m | 360m | 1.7b"; exit 2 ;;
esac
mkdir -p .scratch
src=.scratch/smol_block_$1.bend
perl -0pe "s/def D\(\) -> Nat:\n  \d+n/def D() -> Nat:\n  ${d}n/; s/def KVW\(\) -> Nat:\n  \d+n/def KVW() -> Nat:\n  ${kvw}n/; s/def FF\(\) -> Nat:\n  \d+n/def FF() -> Nat:\n  ${ff}n/; s/def HKV\(\) -> Nat:\n  \d+n/def HKV() -> Nat:\n  ${kv}n/; s/def GRP\(\) -> Nat:\n  \d+n/def GRP() -> Nat:\n  ${g}n/" demo/smol_block.bend > "$src"
bend "$src" -o .scratch/smol_block_$1 > /dev/null
echo ".scratch/smol_block_$1"

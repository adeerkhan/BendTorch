#!/usr/bin/env bash
# The exact outputs, as SHA256 digests: a refactor must not change one bit. Each program's output is deterministic
# (seeded data, fixed orders of summation, timing lines removed) and independent of the thread count.
#
#   scripts/bits.sh check     # compare with bench/bits.txt (make bits)
#   scripts/bits.sh save      # write bench/bits.txt (only when the bits move on purpose; say why in NOTES.md)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .scratch/tiny_llama ] && [ -x .scratch/mnist ] && [ -x .scratch/mlp_test ] || { echo "run make build first"; exit 2; }
digest() { sed -E 's/ train_seconds=[0-9.]+//' | sha256sum | cut -c1-16; }
P=$'ROMEO:\nBut soft, wh'
{
  for th in 1 16; do
    echo "tiny_llama train 50, $th threads: $(.scratch/tiny_llama --threads $th train 50 | digest)"
    echo "tiny_llama sample 100 64, $th threads: $(.scratch/tiny_llama --threads $th sample 100 64 "$P" | digest)"
    echo "mnist 1 epoch 50 batches adam, $th threads: $(.scratch/mnist --threads $th 1 50 adam 0.001 | digest)"
  done
} > .scratch/bits.txt
case "${1:-check}" in
  save) cp .scratch/bits.txt bench/bits.txt; cat bench/bits.txt ;;
  check) diff bench/bits.txt .scratch/bits.txt && echo "bits: every output identical" ;;
esac

#!/usr/bin/env bash
# The exact outputs, as SHA256 digests: a refactor must not change one bit. Each program's output is deterministic
# (seeded data, fixed orders of summation, timing lines removed) and independent of the thread count.
#
#   scripts/bits.sh check     # compare with bench/bits.txt (make bits)
#   scripts/bits.sh save      # write bench/bits.txt (only when the bits move on purpose; say why in NOTES.md)
#   scripts/bits.sh gpu       # make gpu's GPU builds against bench/bits.txt's 16-thread lines
#   scripts/bits.sh rt        # demo/tiny_rt.bend at B = 2, T = 16 against the tiny Llama's lines (make bits)
#   scripts/bits.sh smol [check|save]   # SmolLM2-135M's loss and 3 AdamW steps, 1 and 16 threads, against the "smol"
#                                       # lines (needs .scratch/smol and data/smol135, so not in CI; 5-13 GB a run)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = smol ]; then
  [ -x .scratch/smol ] && [ -f data/smol135/init.bin ] || { echo "build .scratch/smol (make smol) and run scripts/smol_data.sh 135m first"; exit 2; }
  digest() { sed -E '/^train_ms /d' | sha256sum | cut -c1-16; }
  {
    for th in 1 16; do
      # "threads N", not "N threads": check_js.sh and the gpu mode select the other lines by their "N threads"
      echo "smol loss, threads $th: $(.scratch/smol +RTS --threads $th -RTS loss | digest)"
      echo "smol train 3, threads $th: $(.scratch/smol +RTS --threads $th -RTS train 3 | digest)"
    done
  } > .scratch/bits_smol.txt
  case "${2:-check}" in
    save) { grep -v '^smol ' bench/bits.txt; cat .scratch/bits_smol.txt; } > .scratch/bits_all.txt && cp .scratch/bits_all.txt bench/bits.txt; cat bench/bits.txt ;;
    check) diff <(grep '^smol ' bench/bits.txt) .scratch/bits_smol.txt && echo "bits: SmolLM2's loss and training steps identical" ;;
  esac
  exit
fi
if [ "${1:-}" = rt ]; then
  # demo/tiny_rt.bend with B = 2 and T = 16 read from the command line: its outputs must be the tiny Llama's lines
  [ -x .scratch/tiny_rt ] || { echo "run make build first"; exit 2; }
  digest() { sha256sum | cut -c1-16; }
  P=$'ROMEO:\nBut soft, wh'
  {
    for th in 1 16; do
      echo "tiny_llama train 50, $th threads: $(.scratch/tiny_rt +RTS --threads $th -RTS 2 16 data/tiny/batches.bin train 50 2>/dev/null | digest)"
      echo "tiny_llama sample 100 64, $th threads: $(.scratch/tiny_rt +RTS --threads $th -RTS 2 16 data/tiny/batches.bin sample 100 64 "$P" 2>/dev/null | digest)"
    done
  } > .scratch/bits_rt.txt
  diff <(grep '^tiny_llama ' bench/bits.txt) .scratch/bits_rt.txt && echo "bits: tiny_rt with B = 2, T = 16 read at run time gives the tiny Llama's outputs"
  exit
fi
if [ "${1:-}" = gpu ]; then
  # the GPU builds of make gpu: their outputs must be bench/bits.txt's (the same bits as the CPU's)
  [ -x .scratch/tiny_llama_gpu ] && [ -x .scratch/mnist_gpu ] || { echo "run make gpu first"; exit 2; }
  digest() { sed -E 's/ train_seconds=[0-9.]+//' | sha256sum | cut -c1-16; }
  P=$'ROMEO:\nBut soft, wh'
  {
    echo "tiny_llama train 50, 16 threads: $(.scratch/tiny_llama_gpu train 50 | digest)"
    echo "tiny_llama sample 100 64, 16 threads: $(.scratch/tiny_llama_gpu sample 100 64 "$P" | digest)"
    echo "mnist 1 epoch 50 batches adam, 16 threads: $(.scratch/mnist_gpu 1 50 adam 0.001 | digest)"
  } > .scratch/bits_gpu.txt
  diff <(grep '16 threads' bench/bits.txt) .scratch/bits_gpu.txt && echo "bits: the GPU builds' outputs are the CPU's"
  exit
fi
[ -x .scratch/tiny_llama ] && [ -x .scratch/mnist ] && [ -x .scratch/mlp_test ] || { echo "run make build first"; exit 2; }
digest() { sed -E 's/ train_seconds=[0-9.]+//' | sha256sum | cut -c1-16; }
P=$'ROMEO:\nBut soft, wh'
{
  for th in 1 16; do
    echo "tiny_llama train 50, $th threads: $(.scratch/tiny_llama +RTS --threads $th -RTS train 50 | digest)"
    echo "tiny_llama sample 100 64, $th threads: $(.scratch/tiny_llama +RTS --threads $th -RTS sample 100 64 "$P" | digest)"
    echo "mnist 1 epoch 50 batches adam, $th threads: $(.scratch/mnist +RTS --threads $th -RTS 1 50 adam 0.001 | digest)"
  done
} > .scratch/bits.txt
case "${1:-check}" in
  save) { cat .scratch/bits.txt; grep '^smol ' bench/bits.txt || true; } > .scratch/bits_all.txt && cp .scratch/bits_all.txt bench/bits.txt; cat bench/bits.txt ;;
  check) diff <(grep -v '^smol ' bench/bits.txt) .scratch/bits.txt && echo "bits: every output identical" ;;
esac

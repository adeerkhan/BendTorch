#!/usr/bin/env bash
# Training speed, BendTorch against PyTorch on the same work, alternated in rounds:
#   mnist: one epoch of the 784-128-10 MLP with SGD (600 batches of 100), the program's own train_seconds
#   tiny:  200 AdamW steps of the two-layer Llama (32 rows x 32 wide): the wall time of `train 200` minus that of
#          `train 0` (start-up and loading: ~1.4 s for Python and torch, < 10 ms for Bend), per step
#   block: 3 AdamW steps of one decoder block at SmolLM2-135M's sizes (512 rows x 576 wide, MSE), the programs' own
#          train_ms after a warm-up step
# "1 thread" is pinned to CPU 3 (a performance core of the reference machine); "16 threads" is unpinned.
#
#   scripts/bench.sh            # 5 rounds; ROUNDS=n for another count
# Prints one line per run, then the medians. Needs `make data` and `make build`.
set -euo pipefail
cd "$(dirname "$0")/.."
ROUNDS=${ROUNDS:-5}
PY=reference/.venv/bin/python
[ -x .scratch/mnist ] && [ -x .scratch/tiny_llama ] && [ -x .scratch/smol_block ] || { echo "run make build first"; exit 2; }
out=.scratch/bench-$(date +%Y%m%d-%H%M%S).tsv
echo -e "round\tsystem\twork\tthreads\tseconds" > "$out"
pin() { if [ "$1" = 1 ]; then echo "taskset -c 3"; fi; }
for r in $(seq 1 "$ROUNDS"); do
  for th in 1 16; do
    s=$($(pin $th) .scratch/mnist +RTS --threads $th -RTS 1 0 sgd 0.1 | sed -n 's/.*train_seconds=\([0-9.]*\).*/\1/p')
    echo -e "$r\tbend\tmnist\t$th\t$s" | tee -a "$out"
    s=$(THREADS=$th $(pin $th) $PY reference/mnist_ref.py 1 0 sgd 0.1 | sed -n 's/.*train_seconds=\([0-9.]*\).*/\1/p')
    echo -e "$r\ttorch\tmnist\t$th\t$s" | tee -a "$out"
    t0=$(date +%s.%N); $(pin $th) .scratch/tiny_llama +RTS --threads $th -RTS train 0 > /dev/null; t1=$(date +%s.%N)
    $(pin $th) .scratch/tiny_llama +RTS --threads $th -RTS train 200 > /dev/null; t2=$(date +%s.%N)
    echo -e "$r\tbend\ttiny\t$th\t$(awk "BEGIN{printf \"%.5f\", (($t2 - $t1) - ($t1 - $t0)) / 200}")" | tee -a "$out"
    t0=$(date +%s.%N); THREADS=$th $(pin $th) $PY reference/tiny_ref.py train 0 > /dev/null; t1=$(date +%s.%N)
    THREADS=$th $(pin $th) $PY reference/tiny_ref.py train 200 > /dev/null; t2=$(date +%s.%N)
    echo -e "$r\ttorch\ttiny\t$th\t$(awk "BEGIN{printf \"%.5f\", (($t2 - $t1) - ($t1 - $t0)) / 200}")" | tee -a "$out"
    ms=$($(pin $th) .scratch/smol_block +RTS --threads $th -RTS train 3 | sed -n 's/train_ms //p')
    echo -e "$r\tbend\tblock\t$th\t$(awk "BEGIN{printf \"%.5f\", $ms / 3000}")" | tee -a "$out"
    ms=$(THREADS=$th $(pin $th) $PY reference/block_ref.py 135m train 3 | sed -n 's/train_ms //p')
    echo -e "$r\ttorch\tblock\t$th\t$(awk "BEGIN{printf \"%.5f\", $ms / 3000}")" | tee -a "$out"
  done
done
echo "== medians (seconds; mnist per epoch, tiny and block per step) from $out"
$PY - "$out" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
for work in ["mnist", "tiny", "block"]:
    for th in ["1", "16"]:
        cells = []
        for system in ["bend", "torch"]:
            v = sorted(float(r["seconds"]) for r in rows if r["system"] == system and r["work"] == work and r["threads"] == th)
            cells.append(f"{system} {statistics.median(v):.4f} ({v[0]:.4f}-{v[-1]:.4f})")
        print(f"{work:5} {th:>2} threads: " + ", ".join(cells))
PY

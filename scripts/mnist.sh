#!/usr/bin/env bash
# Downloads MNIST into data/mnist/ (once) and writes the initial weights of the PyTorch twin (seed 0).
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/mnist
mkdir -p "$D"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "$D/$f" ] || curl -sSfL "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" | gunzip > "$D/$f"
done
[ -f "$D/init/w1.bin" ] || reference/.venv/bin/python reference/mnist_ref.py --init
echo "MNIST ready in $D"

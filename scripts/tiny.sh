#!/usr/bin/env bash
# Downloads Tiny Shakespeare into data/tiny/ (once, checked by SHA256) and writes the initial weights and the batches
# of the tiny Llama (reference/tiny_ref.py --prep, seed 0).
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/tiny
SHA=86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
mkdir -p "$D"
[ -f "$D/input.txt" ] || curl -sSfL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o "$D/input.txt"
echo "$SHA  $D/input.txt" | sha256sum -c - >/dev/null || { echo "SHA256 mismatch for $D/input.txt"; exit 1; }
[ -f "$D/batches.bin" ] || reference/.venv/bin/python reference/tiny_ref.py --prep
echo "Tiny Shakespeare ready in $D"

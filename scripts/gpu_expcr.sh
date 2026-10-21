#!/usr/bin/env bash
# make gpu's exactness check: demo/gpu_layers_test.bend built with CUDA against a copy of torch/ in .scratch/gpu_expcr/
# where every F32.exp of main.bend and gpu.bend is demo/exp_cr.bend's exp_cr (the host's F32.exp bits, in plain F32 and
# U32 arithmetic), each case run alone so that its bangs run on the device. Prints the 13 cases' ulps; with the
# exponential the same on the host and the device, every case must be 0, which shows that the device code differs from
# main.bend's only through CUDA's float exp. torch/ itself is not touched.
#
#   scripts/gpu_expcr.sh          # needs CUDA_HOME (default ~/.local/cuda12); run it under the heavy-run lock
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME=${CUDA_HOME:-$HOME/.local/cuda12}
D=.scratch/gpu_expcr
rm -rf "$D" && mkdir -p "$D/demo"
cp -r torch "$D/torch"
cp demo/exp_cr.bend "$D/torch/exp_cr.bend"
for f in main gpu; do
  sed -i 's/F32\.exp(/XE.exp_cr(/g; 0,/^import Base$/s//import Base\nimport .\/exp_cr.bend as XE/' "$D/torch/$f.bend"
  grep -q 'import ./exp_cr.bend as XE' "$D/torch/$f.bend" && ! grep -q 'F32\.exp(' "$D/torch/$f.bend" || { echo "gpu_expcr: the swap failed in $f.bend"; exit 1; }
done
cp demo/gpu_layers_test.bend "$D/demo/gpu_layers_test.bend"
CUDA_HOME=$CUDA_HOME LD_LIBRARY_PATH=$CUDA_HOME/lib bend "$D/demo/gpu_layers_test.bend" -o "$D/gpu_layers_test_cuda"
for k in 0 1 2 3 4 5 6 7 8 9 10 11 12; do LD_LIBRARY_PATH=$CUDA_HOME/lib "$D/gpu_layers_test_cuda" $k; done | tr '\n' ' '
echo

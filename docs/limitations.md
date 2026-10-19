# Limitations

Everything I know BendTorch does not do, does worse than it could, or does differently from PyTorch, with the
measurement or the reason next to each. It is kept up to date as things are found; `NOTES.md` has the history.
Numbers marked "busy machine" were taken with other jobs running and are a first look, not a benchmark.

## What is proved, and what is not

- The laws (`LAWS.bend`, `PROOF.bend`, `torch/dtype.bend`) cover structure: shapes, composition, the backward formulas of
  the linear layers on an exact `Nat` model, the causal mask's count, head grouping, packing two 16-bit values a word.
  They do not cover F32 numbers. The link between the F32 code and the `Nat` model, the gradients of the non-linear
  layers, and every number are tested against PyTorch, not proved.
- A few steps inside proofs are read from the code, not proved: that the rest of an attention row keeps its zeros (the
  checker cannot follow writes into an array by index), the bit identities under the 16-bit packing (tested over all
  65,536 values).
- Trusted, not checked: Bend's checker and kernel, the translation to C or CUDA, clang, nvcc, the CPU and the GPU.

## Outside the proofs on purpose (`@unsafe` or foreign code)

- `torch/fast.bend` uses bend-ml-tensor-fork, and `torch/gpu.bend` uses bend-ml-tensor-gpu: both share arrays between
  parallel blocks through Base's `@unsafe` `Array.fork`/`Array.join`, which Bend's model does not describe. Nothing
  checks that two blocks never write the same cell; that is by construction, and the tests compare the bits with
  `main.bend`'s. No file with laws imports them; `make laws` pins the defs that reach `@unsafe` code.
- Loading weights uses bend-ml-binary-read, foreign C and JavaScript, because Bend 2.0.37 reads files only as byte lists.
  It is temporary, and only loading code imports it.

## Numerics: where the results differ from PyTorch

- Every test compares with one tolerance, |bend - torch| <= 1e-6 + 1e-5 |torch| in float64 (`reference/oracle.py`),
  never widened to make a test pass. Training runs over many steps are judged against a float64 run instead, because
  rounding grows with every AdamW step in any F32 system, PyTorch's included.
- Sums run in a fixed order (a balanced tree for long reductions, partial sums of four in the products), so the bits are
  the same at any thread count, but they are not PyTorch's: a weight gradient summed over the batch in another order
  can differ by up to ~2 tolerances under heavy cancellation. The release-time corpus (`make corpus`) judges each number
  against float64 scaled by the size of the terms that produced it; under that rule every case passes.
- Long dot products: tensor-array's products keep four running partial sums, which lose small terms when k is very
  large (SmolLM2's head, 49,152 classes: the gradients were 1.8x further from float64 than PyTorch's float32). The
  linear layers' dX now uses the opt-in `matmul_precise` (k in blocks of 1024): 0.03x of PyTorch float32's error, for
  2.7% of a SmolLM2-135M step. The other products (the forward x·Wᵀ, dW) keep the plain four sums; none of them is that
  long in these models, but a model with a very wide input would meet it there.
- AdamW's `1 - b1` and `1 - b2` are given as constants next to the betas (`CAdamW{lr, b1, b2, eps, wd, c1, c2}`): F32
  cannot take `1 - 0.999` accurately, and with it computed the tiny Llama drifted from float64 38x as far as PyTorch over
  100 steps (0.50x with the constants). `OptCfg.check` stops a program whose `c1`, `c2` disagree with its betas, but
  only where the program calls it; `Train.new` cannot stop a pure computation, so without the call a wrong configuration
  shows up as NaN losses from the second step. Adam/AdamW checkpoints written before 2026-10-18 cannot be resumed.
- Edge cases accepted for now, to be compared later:
  - attention gives 0 where PyTorch gives NaN at a position that cannot see a -inf in V (PyTorch's 0 · inf);
  - RMSNorm's dX and LayerNorm's output and dX stay finite where the squares overflow F32, where PyTorch gives NaN.
- Cross-entropy is log-softmax as PyTorch computes it since 2026-10-18; before, it took log(max(p, 1e-12)) and a row's
  loss stopped at 27.63 where PyTorch's kept growing (the gradient was right).
- On the GPU, CUDA takes the float overload of `exp`, whose last bit differs from the host's on some inputs: SiLU
  differs by up to 2 ulps in its output and 4 in its gradient; attention, which also takes `exp`, gave 0 on the tested
  inputs. Accepted because of that cause; another way to compute the exponential on the device is still to try.
- BF16 and F16 conversions are bit-exact against this machine's PyTorch 2.14.1 CPU kernel, which turns every NaN into
  0xFFFF in BF16; PyTorch's scalar code or another backend may encode NaN otherwise.

## Speed

- On the CPU, BendTorch is slower than PyTorch wherever matrix products dominate: 5.5x on MNIST, 6.5x (one thread) and
  3.4x (16 threads) on a SmolLM2-135M-sized block, about 6x on SmolLM2-135M itself (idle machine, 2026-10-18). Every
  number, against eleven systems, is in [`BENCHMARK.md`](../BENCHMARK.md).
- On the GPU (RTX 4050 Laptop), BendTorch is slower than on 16 CPU threads at every size I train. A SmolLM2-135M-sized
  block: 703 ms for 3 steps on the GPU against 416 on 16 CPU threads (idle machine). Where the GPU time goes, from one
  step profiled with nsys on a busy machine (about 460 ms in all, so not the idle average above): about 160 ms filling
  new arrays (Bend's `Array.new` on the device fills its block in one loop on one lane, ~4.6 ns a cell), about 140 ms of
  the work itself, and about 140 ms between bangs (62 gaps of 2.3 ms each there, 0.55 ms on a quieter machine). The work
  alone is already below the CPU's step; the rest is the runtime. Issue drafts with the measurements are in
  bend-ml's `docs/upstream/`.
- Every bang has a fixed cost of about 0.3-1.4 ms, so small products are faster on the CPU.
- The precise product on the GPU runs one bang per block of 1024 steps (tensor-gpu 0.1.2.1), so products with
  k > 1024 pay one more bang each per block: in the SmolLM2-sized block about 1% of a step.
- The leaves of a bang run in a run-time order inside each group of 128 lanes, so loads are not fully coalesced (16
  sectors a request where 4 is possible).
- Keeping tensor-array's four partial sums (for its exact bits) costs registers on the GPU: a one-sum kernel was 2.3x
  faster with other bits.

## Memory

- An array is one block of a power of two cells, so a 2560 x 960 matrix takes a 4,194,304-cell block for 2,457,600
  numbers (1.71x); at whole-model sizes the overhead is 1.24x (135M) to 1.58x (360M).
- SmolLM2-1.7B does not train on a 32 GB machine: AdamW's state is 26.0 GiB in F32, plus about 8 GiB of activations.
  BF16 storage today gives BF16's numbers, not its memory (the values sit in F32 arrays); packed BF16 gradients and
  moments would bring it to about 16 GiB of state, and that is left for after v1. Its F32 weights file is also past the
  reader's 32-bit offsets.
- A 360M AdamW checkpoint is past the reader's 32-bit offsets too.

## Features

- The layers work on matrices: a tensor is a `Mat<r, c>`, with batches and heads folded into rows and columns.
- No exact GELU (the tanh form only: the exact one needs `erf`), no `.pt` or safetensors loading (weights come from a
  flat binary file in the twins' layout).
- int8 is for inference only (weights quantized by rows, activations in F32).
- Mixed precision is BF16 storage with F32 arithmetic and F32 master weights, measured on the tiny Llama only. It
  costs time: 200 steps take 180 ms against 160 in F32 on one thread and 330 against 220 on 16
  (idle machine, whole-program wall time at 10 ms resolution, medians of 5 rounds).
- One GPU, CUDA only (what Bend 2.0.37 builds); no multi-GPU, no Metal or ROCm.

## Machine and versions

- Measured on one laptop: Core Ultra 7 155H, 32 GB, RTX 4050 Laptop 6 GB; Bend 2.0.37, PyTorch 2.14.1. Numbers on other
  machines will differ; the bits on the CPU should not.

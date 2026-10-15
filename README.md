# BendTorch

**BendTorch** trains neural networks in [Bend 2](https://bend-lang.com) with general pieces: layers with typed forward
and backward passes, combinators that compose them, losses, four optimizers and one training step that works for any
model. A 784-128-10 MLP on MNIST and a two-layer Llama (grouped-query attention, RoPE, SwiGLU, RMSNorm) are built from
those pieces and trained in Bend, and both agree with PyTorch twins that start from the same weights and see the same
batches. Every gradient has the type of the value it is the gradient of, so a transposed gradient or two layers that do
not fit together do not compile, and the formulas of the backward pass are laws checked by Bend's kernel. It builds on
[bend-ml](https://github.com/nuxyel/bend-ml)'s matrices and on what [llama.bend](https://github.com/nuxyel/llama.bend)
learned about the Llama layers.

Early research code: CPU only, F32 only, matrices only (higher ranks are folded into rows and columns). Bend **2.0.37**,
MIT.

## 01 · A model is a layer

```python
# Linear(784, 128) -> ReLU -> Linear(128, 10), on batches of 100 rows
def Hidden() -> T.Layer<T.Two<T.Lin<784n, 128n>, Unit>, X, TA.Mat<100n, 128n>, T.Two<X, TA.Mat<100n, 128n>>>:
  T.Seq(~T.Lin<784n, 128n>, ~Unit, ~X, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 128n>, ~X, ~TA.Mat<100n, 128n>, ~T.Linear(100n, 784n, 128n, 3n), ~T.ReLU(100n, 128n))

def Model() -> T.Layer<P, X, Y, S>:
  T.Seq(~T.Two<T.Lin<784n, 128n>, Unit>, ~T.Lin<128n, 10n>, ~X, ~TA.Mat<100n, 128n>, ~Y, ~T.Two<X, TA.Mat<100n, 128n>>, ~TA.Mat<100n, 128n>, ~Hidden(), ~T.Linear(100n, 128n, 10n, 0n))

# one training step, the same for every model and loss
T.Train.step(~P, ~X, ~Y, ~S, ~List<&2, Nat>, ~Model(), ~ce, cfg, state, x, labels)
```

A layer is a record of four functions over its parameters `P`, its input `X`, its output `Y` and what its forward pass
saves, `S`. Its backward pass turns a gradient typed `Y` into one typed `X` and adds the parameters' gradient into a value
typed `P`. Layers are passed as templates (`~`), which Bend copies at compile time, so `Seq`, `Fork`, `Both` and
`Residual` compose any layers whose types meet. The Llama in [`demo/tiny_llama.bend`](demo/tiny_llama.bend) is written
the same way: `Residual(Seq(RMSNorm, attention))`, where attention is `Fork(q, Fork(k, v))`, then RoPE on q and k, the
attention core and the output projection.

A layer whose shapes do not meet is rejected before anything runs ([`tests/`](tests/)):

```
$ bend tests/bad_seq.bend      # Linear(784, 128) followed by Linear(64, 10)
- expected : T.Layer<T.Lin<64n, 10n>, TA.Mat<100n, 128n>, TA.Mat<100n, 10n>, TA.Mat<100n, 64n>>
- observed : T.Layer<T.Lin<64n, 10n>, TA.Mat<100n, 64n>, TA.Mat<100n, 10n>, TA.Mat<100n, 64n>>
```

## 02 · Laws

| Law | In plain language |
|---|---|
| `adjoint` | a linear layer's backward pass `dx = Wᵀ·dy` is the adjoint of its forward pass `y = W·x`: `dy·(W·x) = (Wᵀ·dy)·x` for every `W`, `dy`, `x`, which is what makes `dx` the gradient |
| `weight_adjoint` | the weight gradient `dW = dy ⊗ x` is the adjoint in `W`: `dy·(W·x) = <dW, W>` |
| `adjoint_seq` | `Seq`'s backward pass on two linear layers, `W1ᵀ·(W2ᵀ·dy)`, is the adjoint of `W2·(W1·x)`: the chain rule |
| `causal` | attention gives probability exactly 0 to every later position |
| `gqa_head` | with `g` query heads per key/value head, query head `j·g + i` belongs to key/value head `j`, the grouping both passes of attention use |

They are stated in [`LAWS.bend`](LAWS.bend) and proved in [`PROOF.bend`](PROOF.bend); `bend PROOF.bend --verdict` checks
them again with Bend's kernel proved in Lean, with no `@unsafe`, no `?TODO` and no axioms. [`torch/README.md`](torch/README.md#the-laws-in-plain-language)
explains each proof. The linear laws are about an exact model where numbers are `Nat`; the F32 code computes the same
formulas and is tested, not proved, to do so. 18 deliberate faults in the code and the specification
(`scripts/mutate.py`) are each caught by a law or a test, except an equivalent one and one the laws do not reach yet (the
tests catch it).

## 03 · It trains, and it computes what PyTorch computes

Against `torch.autograd` and `torch.optim` (PyTorch 2.14.1, CPU), with one tolerance for every number:
`|bend - torch| <= 1e-6 + 1e-5·|torch|`, NaN matching NaN.

| | numbers compared | bit-identical | worst error, as a share of the tolerance |
|---|---|---|---|
| every layer, forward and backward, 10 seeds ([`test_ops.py`](reference/test_ops.py)) | 15,550 | 11,308 | 0.39 |
| a 7-6-3 MLP: gradients, accumulation, 3 steps of SGD, momentum, Adam, Adam + L2, AdamW, 12 seeds ([`test_mlp.py`](reference/test_mlp.py)) | 6,588 | 2,196 | 0.10 |
| the two-layer Llama: the loss and all 21 parameter gradients ([`test_tiny.py`](reference/test_tiny.py)) | 22,753 | 3,931 | 0.03 |
| the same Llama after 20 AdamW steps: every loss and every final weight | 22,772 | 1,361 | 0.83 |

- **MNIST** (784-128-10, SGD at lr 0.1, one epoch): loss 0.5148658 against PyTorch's 0.5148658, 9,108 test hits in both.
  50 batches under each optimizer give the same loss within the tolerance and the same hits.
- **The Llama** (65 characters, width 32, 4 query and 2 key/value heads, AdamW): the loss falls from 4.23 to 2.63 in
  500 steps on Tiny Shakespeare, as PyTorch's. The two runs agree to 1e-6 for 60 steps and then drift apart: Adam
  divides by `sqrt(v)`, which magnifies the rounding of gradients near zero ([`NOTES.md`](NOTES.md) traces where).
- The activations also match PyTorch on NaN, infinities, signed zeros, subnormals and overflowing inputs.

## 04 · Speed

On an Intel Core Ultra 7 155H, 5 alternated rounds, medians ([`NOTES.md`](NOTES.md#speed), `make bench`):

| | BendTorch, 1 thread | PyTorch, 1 thread | BendTorch, 16 threads | PyTorch, 16 threads |
|---|---|---|---|---|
| MNIST, one epoch of SGD | 12.7 s | 0.32 s | 8.1 s | 0.38 s |
| two-layer Llama, one AdamW step | 13.1 ms | 9.1 ms | 13.4 ms | 12.8 ms |

BendTorch is about 40x slower than PyTorch on MNIST, where the products dominate, and 1.4x slower on the small Llama,
where PyTorch's per-operation overhead does. The Llama's row-wise layers run on lists and do not use more than one
thread yet.

## 05 · Reproduce

Linux x86_64, clang, Python 3.14, `curl`. The Python venv with PyTorch takes about 1.2 GB.

```bash
make setup   # Bend 2.0.37 (SHA256-checked), Lean 4.34.0, the Python venv of reference/; no sudo
make data    # MNIST and Tiny Shakespeare into data/, the twins' initial weights and batches (seed 0)
make check   # the laws (--verdict), the rejected programs, every layer, the MLP, MNIST and the Llama against PyTorch (~1 min)
make mutate  # the 18 deliberate faults (~1.5 min)
make bench   # the speed table (~3 min)
```

`main` will hold released versions; work happens on `devel`. Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 06 · Limits

- **Matrices only.** A tensor is a `Mat<r, c>`; batches and heads are folded into rows and columns. No general
  broadcasting, slicing or batched matmul API, no rank-N type.
- **Sizes are compile-time literals** in the demos; a batch size read at run time is not wired in yet.
- **CPU and F32 only.** No GPU, no F16/BF16, no quantization.
- **Slow**, as section 04 says; the row-wise layers are list code.
- Missing: dropout, exact GELU (the tanh form is there), tied embeddings, checkpoints. GELU's exact form needs `erf`.
- **What the laws do not cover**: the F32 numbers, the gradients of the non-linear layers, and the link between the F32
  code and the `Nat` model; all of these are tested.

## Where things are

| Path | What |
|---|---|
| [`torch/`](torch/) | the package: layers, combinators, losses, optimizers, training (`main.bend`); the exact model the laws use (`spec.bend`) |
| [`LAWS.bend`](LAWS.bend), [`PROOF.bend`](PROOF.bend) | the laws and their proofs |
| [`demo/`](demo/) | MNIST, the two-layer Llama, and the programs the tests run |
| [`tests/`](tests/) | programs that must not compile |
| [`reference/`](reference/) | the PyTorch twins and the tests |
| [`scripts/`](scripts/) | setup, data, the mutation run, the benchmark |
| [`docs/design.md`](docs/design.md), [`docs/plan.md`](docs/plan.md) | how it is built and why; the milestones |
| [`NOTES.md`](NOTES.md) | every finding, measurement and decision, by date |

<p align="center"><sub>MIT · Renan Vinícius (@nuxyel)</sub></p>

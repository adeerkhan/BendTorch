# Design

How BendTorch is built and why: each decision, the alternatives I weighed and the evidence behind the choice. The
measurements are in [`NOTES.md`](../NOTES.md); the milestones in [`plan.md`](plan.md).

## The goal

Train real networks in Bend 2 with general pieces (layers, losses, optimizers, a training step that does not know the
model), keep the guarantees Bend can give (shapes in the types, laws checked by the kernel), and check every number
against PyTorch. Speed matters, but is measured, never assumed.

## Tensors

**A tensor is `bend-ml-tensor-array`'s `Mat<r, c>`**: `r·c` F32 numbers in one flat `Array<F32>`, row by row, with `r`
and `c` erased type parameters. In the compiled program an `Array` is one block of memory and a read is an index mask
and a load (bend-ml NOTES, 2026-10-13), so a `Mat` is a contiguous buffer with its shape in the type.

- **Higher ranks are folded into rows and columns.** A batch of `b` sequences of `t` positions is `b·t` rows; `h` heads
  of `d` numbers are `h·d` columns. The layers that need the structure (RoPE, attention) take the sizes as arguments,
  with proofs that the folded sizes match (`n == b·t`, `w == kv·g·d`). This covers an MLP and a Llama; it does not give a
  general rank-N API.
- **Why not a dynamic tensor** (shape as a run-time value, checked at each operation)? It is the usual design and works in
  Bend, but it moves every shape error from the checker to run time, which gives up what Bend adds. A rank-polymorphic
  typed tensor, `Tensor<shape: List<Nat>>`, is the direction for a general engine; its costs (unary `Nat` in the checker,
  proofs about products of lists of sizes, the verbosity of the types) are unmeasured, so it is a planned experiment
  (plan.md, M7), not a promise.
- **Storage.** One owner per array: every operation hands back what it reads (`MMul{a, b, c}` returns both operands with
  the product). No views or strides: a reshape that keeps the row-major order is free in principle (the same array under
  another type, with a `product` proof, as bend-ml's `Mat.reshape`), a transpose is a copy. `Array.clone` is the only
  copy, and the code calls it only where a value is needed twice (the output an activation saves, the input `Fork` gives
  to two layers).
- **Rows as lists.** The row-wise layers (norms, softmax, RoPE, attention) turn a matrix into a list of rows. Lists are
  `Data`: they can be kept for the backward pass with no copy and read any number of times. Their cost is a list cell per
  number; fine at the sizes of the demos, a debt for larger ones.
- **Dtypes.** F32 only. Bend has `U32` and `F32` words; F16, BF16 and quantized formats would be packed into `U32` words
  and unpacked by the kernels, and F64 does not exist. None of this is built or measured yet.

## Autograd: typed forward/backward pairs

```python
type Layer<-P: Type, -X: Type, -Y: Type, -S: Type> is Type:
  Layer{fwd: P -> X -> Fwd<P, Y, S>, bwd: P -> S -> Y -> P -> Bwd<P, X>, step: Opt -> P -> P -> Two<P, P> -> Step<P>,
        zeros: Unit -> P, dump: P -> PL<P>, load: P -> List<&2, F32> -> PL<P>}
```

The alternatives:

| | What it is | In Bend 2 |
|---|---|---|
| **Typed pairs** (chosen) | each layer has a forward that returns its saved values `S` and a backward from `dY` to `dX` and `dP`; composition is code | The types of `dX` and `dP` are those of `X` and `P`, checked. What is saved is a visible record. No run-time graph. Needs the composition combinators. |
| Dynamic tape | the forward pass records operations on a list of nodes, the backward pass walks it | Nodes of different shapes need a dynamic tensor type, so shapes are checked at run time. A tape is first-order data, which makes laws about the sweep easy to state. |
| Pullback closures | each operation returns its result and a closure for its backward pass | Closures are affine: a value used twice needs an explicit copy, and a law cannot look inside a closure. |
| Forward mode (dual numbers) | each value carries its derivative along one direction | One forward pass per parameter: unusable for training. Good as an independent check. |

Typed pairs were the plan since 2026-10-05, and a first prototype of a linear layer showed a forward and backward pass
with no clone of the weights, the input or the gradient, matching PyTorch. What made them
general here is Bend's **templates**: a `Layer` record passed as `~l` is substituted at compile time, so
`Seq(~la, ~lb)` can call `la`'s and `lb`'s functions in both passes although each closure may run only once.

- **Composition.** `Seq` (pairs the parameters, saved values and optimizer state of its two layers; its backward runs
  the second layer's first), `Fork` (copies the input to two layers and sums their input gradients), `Both` (two layers
  on a pair), `Residual` (`Seq(Fork(Id, f), Add2)`). A Llama block is written with these, nothing model-specific.
- **Accumulation.** A backward pass adds into the gradient it receives (`P` typed), so repeated uses and two backward
  passes add up, as PyTorch's `.grad`; the optimizer step zeroes it.
- **What it does not do yet.** No graph built at run time, so control flow that depends on values is written in Bend
  around the layers, not recorded; no higher-order derivatives; no gradient checkpointing (a layer that recomputes its
  forward in its backward is possible, attention does it for its probabilities).

## Neural network API and training

- Layers: `Linear`, `LinearNB`, `Embedding`, `RMSNorm`, `LayerNorm`, the activations, `Softmax`, `LogSoftmax`, `RoPE`,
  `Attention`; losses `CrossEntropy`, `NLL`, `MSE`; optimizers SGD, momentum, Adam, AdamW, in PyTorch's order of
  operations.
- `Train.step(~model, ~loss, cfg, state, x, targets)` is the whole training step for any layer and any loss. The state
  holds the parameters, their gradient, two moment buffers and `b1^t`, `b2^t` for Adam's bias corrections.
- **A generic walk over the parameters.** `dump` gives every number of a layer's parameters in a fixed order and `load`
  overwrites them from such a list; the combinators compose both. `Train.save` and `Train.load` use them for
  checkpoints of any model (parameters, both optimizer moments, `b1^t` and `b2^t`); a resumed run gives the same bits.
- **State that is not learned** lives in `P` too when it must survive a step: `Dropout`'s step counter is its parameter,
  incremented by the optimizer step and kept by checkpoints, so the mask is reproducible.
- **Shared parameters.** `Tie(first, mid, last)` runs `first` and `last` with one parameter and adds both gradients into
  one accumulator (tied embeddings).
- **Work nobody reads is skipped**: `LinearFirst` does not compute the gradient of a model's input.

## Verification

### What is proved, tested and trusted

| Claim | Kind | Where |
|---|---|---|
| Every operation gets operands of the right shapes; every gradient has the shape of its value | **types**, checked | every program; rejected examples in `tests/` |
| Attention's and RoPE's folded sizes are consistent (`n == b·t`, `w == kv·g·d`, `w == h·2·half`) | **proof at construction**, erased | `Attention`, `RoPE` |
| A linear layer's backward pass is the adjoint of its forward, for the input and the weights; two linear layers compose by the chain rule | **law** over the exact `Nat` model | `LAWS.bend`: `adjoint`, `weight_adjoint`, `adjoint_seq` |
| Attention gives probability 0 to later positions | **law** about the package's own function | `causal` |
| Query head `j·g + i` belongs to key/value head `j`, through the function both passes of attention call | **law** about the package's `kv_of` (and `Nat.div`) | `kv_of_group`, `gqa_head` |
| Each layer's output and gradients equal PyTorch's within the tolerance | **tested**, 10 seeds, with IEEE edge cases for the activations | `reference/test_ops.py` |
| The optimizers equal `torch.optim` step by step | **tested** | `reference/test_mlp.py` |
| MNIST and a two-layer Llama train like their PyTorch twins | **tested** | `test_mnist.py`, `test_tiny.py` |
| The F32 code computes the formulas of the `Nat` model | **tested only** | the same tests |
| The non-linear layers' gradient formulas | **tested only** | `test_ops.py` |
| Bend's checker and kernel, the translation to C, clang, the CPU | **trusted** | |

### Laws

The laws are in `LAWS.bend` and their proofs in `PROOF.bend`, at the root, the convention of the Bend guide: the law is
the specification a reader checks, the proof is the code that discharges it. `make laws` runs the checker, then
`--verdict` (Bend's kernel proved in Lean), and requires every program in `tests/` to be rejected with its expected error.

The linear laws are about `torch/spec.bend`, where vectors are lists of `Nat`: exact arithmetic, so the statement is the
algebra of the formulas (`dy·(W·x) == (Wᵀ·dy)·x`), with no rounding to reason about. F32 is opaque to the checker
(`F32.mul(3.0, 3.0)` does not reduce), so a law about F32 values can only treat the operations as uninterpreted; no
IEEE 754 property is claimed anywhere.

`scripts/mutate.py` changes the code and the specification on purpose (23 mutants) and checks that a gate fails. A
mutant that survives a gate is listed with the reason (today one: a fault the laws do not reach but the numbers catch).

## Numerical validation

- **Independent oracles.** The Python side (`reference/`) builds every model with PyTorch's own functions
  (`F.layer_norm`, `F.gelu(approximate="tanh")`, `torch.optim.AdamW`, ...); it never calls BendTorch, and the Bend
  programs never see a PyTorch result. Inputs and initial weights are written to files that both read.
- **Exact transport.** The Bend programs print each F32 as the decimal of its 32 bits, so nothing is lost between the
  two sides; the tests also count how many numbers are bit-identical.
- **One comparison policy for every test** (`reference/oracle.py`): `|bend - torch| <= 1e-6 + 1e-5·|torch|` element by
  element in float64; NaN matches NaN and an infinity matches the same infinity. On top of it, an **exact tier**: results
  that are a copy, a selection or one or two F32 operations in PyTorch's order (relu forward and backward, an embedding
  lookup, the gradients of MSE and NLL) must be the same bits, NaN included. Results whose order of additions or whose
  `exp`/`tanh` implementation differs from PyTorch's stay in the tolerance even when they happen to match today. The tolerance is never widened to make a
  test pass: when a comparison failed, the cause was found (relu's gradient at NaN) and the code fixed.
- **Trajectories.** Training is compared step by step for as long as the two runs are expected to stay together (50
  batches of MNIST, 20 steps of the Llama). Longer runs drift for a traced reason (NOTES.md, milestone 3); they are
  compared by statistics, and the drift is reported, not hidden.

## Execution and backends

- **CPU first.** Native code from Bend 2.0.37, clang. Products run their rows in `2^par` parallel blocks (`par` is an
  argument of each linear layer); the row-wise layers are sequential.
- **GPU later.** Bend's `!` calls run on CUDA, but bend-ml measured its memory-bound products slower on the RTX 4050
  than on the CPU (bend-ml NOTES, exp. 10). Nothing here runs on the GPU yet. The kernels are behind the layer
  functions, so a GPU path would change the kernels, not the protocol.
- **JavaScript.** `bend -o x.js` builds a program that runs sequentially under Bun (its file reads use `bun:ffi`, so not
  node). `make check-js` runs the test programs that way against the same oracles and compares the outputs of the
  training and generation programs with the native build's: the same bits (NOTES, 2026-10-16).

## Extensibility

A new layer is four functions and a constructor that builds the `Layer` record; a new optimizer is a constructor of
`Opt` and its per-element loop; a new loss is a function `Y -> T -> LossOut<Y>`. The tests follow one pattern: a section
in a Bend program that prints the results, and the same computation in PyTorch.

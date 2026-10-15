# Plan

Each milestone has a pass criterion that a command checks. Estimates are planning guesses, revised as milestones close;
they are not commitments. Decisions and measurements go to [`NOTES.md`](../NOTES.md), the architecture to
[`design.md`](design.md).

## Goal and success bar

A machine learning framework in Bend 2 whose pieces are general (a training step that does not know the model), whose
shapes are in the types, whose structural guarantees are laws checked by Bend's kernel, and whose numbers agree with
PyTorch. Success: real models (an MLP, a Llama) trained in Bend with the same results as their PyTorch twins, every claim
reproducible with one command, and the speed measured and reported as it is.

## Done (2026-10-15)

| # | Milestone | Pass criterion | Status |
|---|---|---|---|
| M1 | The layer protocol, `Linear`, `ReLU`, `Seq`, cross-entropy, SGD/momentum/Adam/AdamW, `Train.step`; MNIST on them | gradients and 3 steps of each optimizer equal PyTorch's on 12 seeds (`test_mlp.py`); 50 MNIST batches of each optimizer: the same loss within the tolerance and the same hits (`test_mnist.py`) | done |
| M2 | The layers of a Llama and the combinators | every layer's output and gradients within the tolerance on 10 seeds, the activations also on IEEE edge cases (`test_ops.py`) | done |
| M3 | A two-layer character-level Llama trained in Bend | all gradients and 20 AdamW steps within the tolerance (`test_tiny.py`); the loss falls (4.23 to 2.63 in 500 steps) | done |
| M4 | Laws | `adjoint`, `weight_adjoint`, `adjoint_seq`, `causal`, `kv_of_group`, `gqa_head` pass `--verdict`; `tests/bad_*.bend` rejected with their expected errors (`make laws`) | done |
| M5 | Hardening | 20 mutants behave as listed (`make mutate`); IEEE edge cases through the activations; benchmark script and first numbers (`make bench`) | done |

| v1 | What a first release needs | `LinearFirst` (MNIST 30% faster, same bits); attention's sequences in parallel; generation from the trained Llama equal to PyTorch's from the same weights; checkpoints (a resumed run gives the same bits); `Dropout` and `Tie` against PyTorch, bit-exact where possible; the exact tier of the numerical policy; 23 mutants | done |

## Next

### M6 · Row operations on arrays, and parallel (~3-5 days)

The norms, softmax, RoPE and attention run on lists today. Move them to `Array` kernels (a row at a time, reading and
writing in place), parallel over rows and heads with balanced parallel lets.

- Pass: `make check` unchanged; `test_ops.py` and `test_tiny.py` within the same tolerance (bits may move: the order of
  the sums may change, documented); the tiny Llama's step faster on 1 thread and faster again on 16 (`make bench`,
  5 alternated rounds).
- Risk: per-row loops over arrays need the affine threading of llama.bend's attention; sharing the input of a parallel
  let still copies it (bendlang/bend#1376).

### M7 · A rank-N typed tensor: decide (~1 week, an experiment first)

`Tensor<shape: List<Nat>>` over one `Array<F32>`: element-wise operations, broadcasting by a proof that two shapes
broadcast, reductions over an axis, reshape with a product proof, permute, slicing, batched matmul.

- First an afternoon prototype: element-wise add with broadcasting and a sum over one axis, rank 3, with its laws (the
  output shape; the number of elements). Measure the checker's time and the verbosity of a call against `Mat`.
- Decision point: adopt as the package's tensor (and make `Mat<r, c>` the rank-2 case), or keep matrices and fold ranks,
  as today. Written down in NOTES either way.

### M8 · Laws about the code that runs (~1-2 weeks)

The linear laws speak about a `Nat` model. Write the list kernels once as templates over the arithmetic
(`~add`, `~mul`, `~zero`), instantiate them at `F32` to run and at `Nat` to prove, so the adjoint law is about the code's
own structure. Then:

- the attention code reads its key/value head through `kv_of` (`kv_of_group` is about `kv_of`; a mutant that bypasses it
  still survives the laws);
- causal gradients: the gradient into position `p` comes only from queries at positions `t >= p` (on the `Nat` instance);
- the backward pass of `Seq` composes those of its layers for any two layers, on the generic instance.
- Pass: the laws in `LAWS.bend`, `--verdict`; the mutants that survive the laws today are caught by them.

### M9 · Training infrastructure (~1 week)

Done in v1: checkpoints, tied embeddings (`Tie`), dropout, the generic walk over parameters (`dump`/`load`). Left:
batch sizes read at run time (bend-ml's `examples/runtime_batch.bend` pattern), an optimizer state per optimizer (SGD
needs none), loading initial weights through `load` from one file instead of one file per tensor.

- Pass: a model whose batch size comes from the command line trains like its literal-size twin; SGD's state holds no
  buffers.

### M10 · Bigger: SmolLM2-135M fine-tuning (after M6, ~2-3 weeks)

Load SmolLM2-135M with llama.bend's loader into BendTorch's layers and fine-tune it for a few steps against PyTorch.
Needs M6 (lists are too slow at width 576) and the transposed band product for `Bands` (llama.bend's `docs/bendtorch.md`).

- Pass: one step's gradients against `torch.autograd` on the real weights; memory and time per step measured.

### Later, exploratory (6-12 weeks or more)

GPU kernels (after the storage settles); kernel fusion (a linear layer with its activation); packed BF16 weights; mixed
precision; gradient checkpointing for longer contexts. Each needs its own measured case before it enters the plan.

## Going public

The repository is private. Before it opens: a README with the measured numbers, `CONTRIBUTING.md` and
the GitHub templates on `main`, CI green, a decision about the BendHub package name, and a check that nothing private is
in the files or the history.

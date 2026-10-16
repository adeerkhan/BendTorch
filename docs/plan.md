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

bend-ml's products come first, because they are the bottleneck; the milestones below are this repository's part. Each closes with `make check`, `make bits` (or an explained
change of bits), `make mutate` and a NOTES entry.

### M6 · Row-wise layers on arrays, in parallel (~4 days)

Needs M13 first (my call, 2026-10-16): M6 touches every layer, so CI runs on every push before it starts.

RMSNorm, LayerNorm, softmax, log-softmax, RoPE, attention and the losses move from lists to `Array` kernels (a row at a
time, in place), parallel over rows and heads with balanced parallel lets.

- Pass: `test_ops.py` and `test_tiny.py` within the tolerance; the exact tier unchanged; `make bits` identical where the
  order of the sums is kept (and every moved bit explained); the tiny Llama's step faster on 1 thread and again on 16
  (`make bench`).
- Risk: sharing an input across a parallel let still copies it (bendlang/bend#1376).

### M7 · General tensors (~1-2 weeks; a prototype first)

`Tensor<shape: List<Nat>>` over one `Array<F32>`, row-major. Operations, each with its backward pass, its test against
PyTorch and its shape law:

- element-wise with broadcasting, the broadcast proved by an argument (`Broadcast<a, b, out>`), and its gradient reduced
  back to each operand's shape;
- sum and mean over an axis, with and without `keepdim`; the whole-tensor sum;
- reshape and flatten (a product proof), permute and transpose (copies), slice/narrow (a bounds proof), concat (the
  sizes add up);
- batched matmul over a leading batch axis, and its backward;
- ranks 0 to 4 at least; `Mat<r, c>` becomes `Tensor<[r, c]>` without a copy.

First an afternoon prototype (broadcast add and a sum over an axis at rank 3, with their laws), measuring the checker's
time and how long a call's types get. Decision point, written in NOTES: adopt and port the layers, or keep `Mat` and add
the operations on it.

- Pass: every operation within the tolerance against PyTorch on random shapes (including size-1 axes and empty
  tensors), its shape law by `--verdict`, rejected programs for a wrong broadcast, slice and concat.

### M8 · Laws about the code that runs (~1-2 weeks)

The row and list kernels written once as templates over the arithmetic (`~add`, `~mul`, `~zero`), run at `F32` and
proved at `Nat`, so a law is about the executed code's structure:

- the adjoint law for the list products attention uses, on their `Nat` instance;
- attention reads its key/value head through `kv_of` (a law about `attn_head`'s `Nat` instance, so the bypass mutant
  that survives today is caught);
- causal gradients: no gradient reaches a key or value at a position after the query's;
- `Seq`'s backward is the composition of its layers' for any two layers in the linear model;
- M7's shape laws.
- Pass: the laws by `--verdict`; the mutants that survive the laws today are caught by them.

### M9 · Training infrastructure, the rest (~3 days)

- Sizes read at run time (bend-ml's `examples/runtime_batch.bend` pattern): a batch size from the command line.
- An optimizer state per optimizer: SGD keeps no buffers, momentum one, Adam two.
- Initial weights from one file through `Train.load`/`load` instead of one file per tensor.
- A shorter way to write a model's types (helpers that name the composite types).
- Pass: the run-time-sized MNIST trains like the literal one (same bits); SGD's checkpoint holds only the parameters.

### M10 · SmolLM2-135M fine-tuning (~2-3 weeks, after M6 and bend-ml's transposed band product)

Load SmolLM2-135M with llama.bend's loader into BendTorch's layers (tied embeddings through `Tie`) and fine-tune it for a
few steps against PyTorch.

- Pass: one step's gradients against `torch.autograd` on the real weights; loss and memory per step measured.

### M11 · Dtypes (~1-2 weeks)

- BF16 and F16 stored two per `U32`, converted to F32 for compute with PyTorch's rounding (round to nearest even), the
  conversions tested bit for bit against `tensor.to(torch.bfloat16)`.
- Mixed precision: BF16 weights and activations in storage, F32 arithmetic and F32 master weights in the optimizer.
- int8 with a scale per row, for inference.
- F64: not in Bend 2.0.37; a two-F32 ("double-word") accumulator as an experiment for long sums, and an upstream request
  draft.
- Pass: conversions bit-exact; a mixed-precision tiny Llama within a stated tolerance of the F32 one, measured.

### M12 · Backends (~1 week, GPU open-ended)

- A JavaScript lane: the test programs built with `bend -o x.js` and run with Bun (not node: the file reads use
  `bun:ffi`), compared with the same oracles. **Done 2026-10-16** (`make check-js`): every test passes and the training
  and generation outputs are the native build's bits; `exp`/`tanh` happened to match libm's on every input tested.
- WASM: not a Bend 2.0.37 target; only JS inside a bundled page (`bend page.html -o dir`). Recorded, not built.
- GPU: element-wise and product kernels with `!` on the RTX 4050, measured against the CPU on the same work; adopted
  only where faster.
- Pass: `make check-js` green; a GPU measurement in NOTES whatever its result.

### M13 · CI (done 2026-10-16; before M6)

The workflow exists and has never run (Actions is off on the private remote). The free plan gives 2,000 minutes a month
to private repositories, and a full run (setup, `check`, `bits`, `mutate`, `check-js`) should take 10-15 minutes, so not
everything on every push: `make check` and `make bits` on each push to `devel`, cancelling a run still in progress
when a newer push arrives, and the full run on pull requests to `main` and nightly. Measure the first runs and fix
whatever differs on a clean Ubuntu runner. Done: quick 6.4 min (cold caches), full 10.4 min, both green (NOTES).

### M14 · A frozen, held-out numerical corpus (~2 days)

Inputs drawn with a seed never used in development, their PyTorch results computed once and stored with a checksum
(`bench/corpus/`), the policy versioned; the corpus is run at release time, not while developing, so it stays held out.
IEEE edge cases extended to the norms, softmax and the losses.

### M15 · A benchmark against other frameworks (~1 week, after M10)

As bend-ml and llama.bend were measured against PyTorch, NumPy, llama.cpp, candle and llm.c: the same models trained
in other frameworks and languages, from the same initial weights on the same data.

- Workloads: the MNIST MLP (an epoch), the tiny Llama (training steps, then sampling), M10's SmolLM2-135M fine-tuning
  step, and the single products and layers (forward and backward) on their shapes.
- Systems (versions pinned, each built and run as its docs say): PyTorch (eager, and `torch.compile` where it works on
  CPU), JAX, tinygrad, a NumPy reference with hand-written backward, Rust's candle and burn, llm.c for the GPT-style step,
  Julia's Flux. Which ones can run each workload is decided by trying, and written down.
- Measured with my benchmark harness: an idle machine, alternated runs, one thread pinned and 16 threads; time per step,
  start-up, peak memory; every system's loss curve against PyTorch's within the stated tolerance.
- Pass: a `BENCHMARK.md` with the machine, the versions, every number and the losses shown; the method and raw results
  in my private notes. Its numbers go into the README before the repository opens.

## Going public

The repository is private until a v1 I am satisfied with; then it opens. Before that:

- an audit of every file and of the whole history that will be published: nothing private, no analysis or method
  beyond a one-line result in `NOTES.md` and `docs/` (the detailed material belongs in my private repositories), no link
  to private repositories;
- a README with the measured numbers (M15's benchmark among them), `CONTRIBUTING.md` and the GitHub templates on `main`, CI green;
- a decision about the BendHub package name and a check of the name's conflicts.

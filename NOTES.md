# Notes

Every finding, measurement and decision, by date. Newest at the bottom.

## Pinned versions

- Bend **2.0.37** (`bend version`), Lean **4.34.0** for `--verdict`.
- `bend-ml-tensor-array@0.1.11.0`, `bend-ml-nat-lemmas@0.1.1.0` (BendHub).
- Python 3.14.7, torch 2.14.1+cpu, numpy 2.5.4 (`reference/requirements.txt`).
- Data: MNIST from the PyTorch mirror (`scripts/mnist.sh`); Tiny Shakespeare from karpathy/char-rnn, SHA256
  `86c4e6aa…c565ed` (`scripts/tiny.sh`).

## 2026-10-15: start

BendTorch is step 2 of the plans I made after bend-ml v1.2: typed autograd over tensors and a small transformer trained in
Bend against a PyTorch twin. llama.bend's `docs/bendtorch.md` listed what each operation of the Llama forward pass needs
for its backward pass; this repository starts from there.

### Decisions

- **A layer is a typed forward/backward pair, not an entry on a dynamic tape.** `fwd : P -> X -> Fwd<P, Y, S>` and
  `bwd : P -> S -> Y -> P -> Bwd<P, X>`. The gradient of the input has the input's type `X`, and the gradient of the
  parameters is accumulated into a value of the parameters' type `P`. A tape would need a dynamic tensor type (shape known
  only at run time) for its heterogeneous nodes, which gives up the one thing Bend adds over PyTorch: shapes in the types.
  The saved values `S` are an explicit record, so what a backward pass keeps is visible in its type.
- **Layers are templates.** Bend 2.0.37's templates (`~f`) are substituted at compile time and, unlike closures, can be
  used any number of times. A `Layer` is a record of four closures; passed as a template, `Seq(~la, ~lb)` builds a new
  record at each use, and each closure in it is called once. That is what makes generic composition possible with affine
  closures. A template can also take a `Nat` (`~n: Nat`), which `Residual` uses for its matrix shape.
- **Arrays from bend-ml.** Matrices are `bend-ml-tensor-array`'s `Mat<r, c>` (a flat `Array<F32>`, the dimensions erased
  type parameters) and its products (`matmul`, `matmul_nt`, `matmul_tn`, four rotating partial sums, rows in parallel
  blocks). The linear layers use the weights in PyTorch's `out x in` layout: forward `x·Wᵀ` (`matmul_nt`), backward
  `dy·W` (`matmul`) and `dyᵀ·x` (`matmul_tn`). No parameter, input or gradient is cloned by a linear layer; each product
  hands its operands back.
- **Gradients accumulate**, as PyTorch's `.grad`: a backward pass adds into the gradient it is given, and an optimizer
  step zeroes it. Repeated uses of a value (`Fork`, tied ids in `Embedding`) add up.
- **Optimizers in PyTorch's order of operations**, one pass per parameter matrix over the parameter, its gradient and two
  moment buffers: SGD (`w + (-lr)·g`), momentum (`buf·mu + g`), Adam (`m + (1-b1)·(g - m)` as `lerp`, then
  `v·b2 + (1-b2)·g·g`, then `w + (-step·m) / (sqrt(v)/bc2 + eps)`), AdamW's decoupled decay. The optimizer state of
  every layer is two values of its parameters' type, so SGD carries two buffers it never reads (2x the parameters in
  memory). One `Train` record type serves every optimizer; a leaner state per optimizer is a debt.
- **Laws in `LAWS.bend` and `PROOF.bend` at the root**, the convention the Bend 2.0.37 guide describes (a human states
  the laws, the proofs are written next to the code). llama.bend keeps its laws inside its package; here the package
  stays code, the laws live apart, and `torch/spec.bend` holds the exact model the linear laws speak about.

### Milestone 1: the protocol, MNIST

`Linear`, `ReLU`, `Seq`, cross-entropy, the four optimizers and `Train.step`, then `demo/mnist.bend` written only with
them (`Seq(Seq(Linear(784, 128), ReLU), Linear(128, 10))`).

- **MLP test** (`demo/mlp_test.bend`, `reference/test_mlp.py`): a 7-6-3 MLP on batches of 5, 12 seeds: logits, loss,
  `dx`, every parameter gradient, two backward passes accumulated, and three steps of each optimizer (SGD, momentum,
  Adam, Adam with weight decay, AdamW). 6,588 numbers, 2,196 bit-identical to PyTorch, the worst at 0.104 of the
  tolerance.
- **MNIST against its twin** (`reference/mnist_ref.py`, the same `nn.Linear` initial weights under seed 0, the same
  batches): one epoch of SGD at lr 0.1, loss 0.5148658 against 0.5148657560 and 9,108 test hits in both. 50 batches of
  each optimizer agree within 5e-7 in loss and give the same hits (the gate in `make check`).
- **Over a whole epoch, Adam drifts.** Two epochs of Adam at lr 1e-3: losses 0.3976068 / 0.1905289 against PyTorch's
  0.3976275 / 0.1905378, hits 9,320 / 9,521 against 9,317 / 9,514. The two sum the products in different orders; Adam
  divides by `sqrt(v)`, which magnifies the small differences of gradients near zero (see the tiny Llama below).

### Milestone 2: the layers of a Llama

Sigmoid, Tanh, SiLU, GELU (tanh form; the exact one needs `erf`, which Base lacks), Softmax, LogSoftmax, RMSNorm,
LayerNorm, Embedding, Linear without bias, RoPE (rotate-half, as `transformers`), causal attention with grouped-query
heads, MSE and NLL losses, and the combinators `Id`, `Fork`, `Both`, `Add2`, `Mul2`, `Residual`.

- **Each layer against `torch.autograd`** (`demo/ops_test.bend`, `reference/test_ops.py`): the output, `dx` and the
  parameter gradients from a random `dy`, on 10 seeds with zeros, `-0` and saturating values mixed in; SwiGLU is tested
  as built from `Fork`, `Seq`, `SiLU`, `Mul2` and `LinearNB`, and a residual RMSNorm through `Residual`. 15,550
  numbers, 11,308 bit-identical, the worst at 0.388 of the tolerance.
- **IEEE edge cases through the activations**: NaN, both infinities, both zeros, subnormals, inputs where `exp`
  overflows (|x| > 88.7) and 1e6. All five match PyTorch's class (NaN for NaN, the same infinity) once one thing was
  fixed: I had written relu's gradient as "passes where y > 0", which stops it at NaN; PyTorch's `threshold_backward`
  zeroes it only where `y <= 0`, so a NaN passes. Now the same.
- **Row operations on lists.** RMSNorm, LayerNorm, softmax, RoPE and attention turn a matrix into its rows (a list of
  lists) and back. Lists are `Data`, so a forward pass keeps its input rows for the backward pass without a copy of an
  `Array`. This is the simple first version: for the small models here it is fast enough (the tiny Llama below), and the
  products, where the time goes in larger models, stay on arrays. Moving the row operations to arrays is a debt.
- **Attention recomputes its probabilities** in the backward pass from the saved Q and K (rows) instead of keeping them.
  `Act` in llama.bend did not keep them either; for training the choice is now recomputation, by design, not by lack.

### Milestone 3: a two-layer Llama trained in Bend

`demo/tiny_llama.bend`: token embedding 65 x 32, two blocks of `Residual(Seq(RMSNorm, attention))` and
`Residual(Seq(RMSNorm, SwiGLU 32-64-32))`, RMSNorm, a linear head to 65 logits, cross-entropy; attention is
`Fork(q, Fork(k, v))`, then `Both(RoPE, Both(RoPE, Id))`, the attention core (4 query heads, 2 key/value heads of 8,
causal) and the output projection. Character level on Tiny Shakespeare, batches of 2 x 16 positions, AdamW (lr 3e-3,
weight decay 0.01) on every parameter. Its twin, `reference/tiny_ref.py`, writes the initial weights and 2,000 batches.

- The whole model, about 80 composed layer values (combinators included), type-checks in 0.2 s and compiles in 17 s.
- **Gradients on the first batch**: all 21 parameter tensors, 22,753 numbers, 3,931 bit-identical, the worst at 0.026 of
  the tolerance; the loss 4.228060, one ulp from PyTorch's.
- **Training**: 20 AdamW steps agree within the tolerance in every loss and every final weight (worst 0.83 of it; the
  gate in `make check`). Over 500 steps the loss goes from 4.23 to a mean of 2.633 over the last 50 steps, PyTorch's
  2.632; the losses agree within 1e-6 (relative) up to step 60 and drift apart after it (1e-4 at step 72, 2% at step 100).
- **Why they drift, traced**: the first parameter to move apart is row 55 of the embedding, the character `q`, which first
  appears as an input at step 34. At that step the 28th entry of its gradient is -1.39e-7, the remainder of a
  cancellation; Adam's denominator `sqrt(v)/bc2` is then 2.4e-8, of the order of `eps` (1e-8), so the update depends on
  the exact value of that tiny gradient, which the two programs round differently. The update differs by 0.7%, and the
  difference spreads from there. Not a bug in either: a property of Adam on gradients near `eps`. A trajectory test
  longer than ~50 steps would have to compare statistics, not values.

### Laws

Six laws, proved in `PROOF.bend` and checked by `--verdict` (0.4 s): `adjoint` and `weight_adjoint` (the backward pass of
a linear layer is the adjoint of its forward pass, for the input and for the weights), `adjoint_seq` (the chain rule for
two linear layers), `causal` (attention puts probability exactly 0 on later positions, about the package's own
`causal_row`) and `gqa_head` (llama.bend's proof, about the head grouping that both passes of `Attention` use).

- `bend-ml-nat-lemmas`' `add_assoc` states `a + (b + c) == (a + b) + c`; llama.bend's own states the other direction. The
  copy of `div_blocks` needed its `Equal.sym` removed.
- To refute `False == True` (in `len_take`), I rewrite the goal `Empty`, written as `disc(False)`, with the equation turned
  around, and answer `Unit{}`: the guide's recipe for clashes.
- The linear laws speak about a model over `Nat` (`torch/spec.bend`), not about the F32 code: the model's `mv` and `mtv`
  are the formulas the array products compute in another order. That connection is tested, not proved.
- A sixth law, `kv_of_group`, came out of the mutation run (below): the attention code now picks the key/value head in one
  def, `kv_of`, called by both passes, and the law says `gqa_head` of that def. A mutant that makes `kv_of` use `mod` is
  now caught by the laws; a mutant where the attention code stops calling `kv_of` still slips past them (no law says which
  def the attention calls) and is caught by the tests.

### Mutation testing

`scripts/mutate.py` (`make mutate`) applies deliberate faults, each to the code or to the specification, and runs the gate
that should see it. The first run had 18; with `kv_of` there are 20, and all behave as expected: 18 are caught, 2 survive
for a stated reason (one equivalent mutant, one fault the laws cannot see but the tests catch).

- A query head that reads key/value head `hi mod g` instead of `hi / g` first survived the laws (`gqa_head` is about
  `Nat.div` itself, not about the code that calls it) and was caught only by the numbers. Fixed with `kv_of` and
  `kv_of_group` (above).
- Adding the bias gradient in the other order survives: F32 addition is commutative, an equivalent mutant.
- **The mutation run found a gap:** "the bias gradient is not accumulated" (`db` instead of `gb + db`) first survived,
  because every test ran a single backward pass into a zero gradient, where `0 + db == db`. The MLP test now runs two
  backward passes into one gradient and compares with two `backward()` calls in PyTorch; the mutant is caught.

### Speed

`scripts/bench.sh`, 5 alternated rounds, medians (range), on an Intel Core Ultra 7 155H, `performance` profile, the
machine otherwise idle; 1 thread pinned to CPU 3, 16 threads unpinned; PyTorch 2.14.1 CPU with `torch.set_num_threads`.

| | 1 thread | 16 threads |
|---|---|---|
| MNIST, one epoch of SGD (600 batches of 100), BendTorch | 12.7 s (12.6-12.7) | 8.1 s (8.0-8.4) |
| the same, PyTorch | 0.32 s (0.32-0.32) | 0.38 s (0.24-0.96) |
| tiny Llama, one AdamW step (2 x 16 positions), BendTorch | 13.1 ms (13.0-13.1) | 13.4 ms (13.3-14.3) |
| the same, PyTorch | 9.1 ms (9.1-9.1) | 12.8 ms (12.7-18.2) |

- MNIST is about 40x slower than PyTorch on one thread. bend-ml's hand-written MNIST (`demos/mnist/fast.bend`, on
  `bend-ml-tensor-array@0.1.3.0`, weights stored `in x out`) measured in the same session: 18.3 s on one thread, 7.4 s on
  16 (3 rounds). The general layers cost no more than the hand-written version on one thread and 9% more on 16; the two
  differ in kernel version and layout, so this is a sanity check, not an isolated measurement of the abstraction's cost.
- The tiny Llama is 1.4x slower than PyTorch on one thread: at 32 x 32, PyTorch's per-operation overhead dominates. It
  does not use more than one thread in BendTorch (its products are sequential, `par = 0`, and the row operations are
  list code).

### Debts

- **Shapes beyond matrices.** Every tensor is a `Mat<r, c>`; batches and sequences are folded into rows, heads into
  columns. There is no rank-3 or rank-4 type, no general broadcasting, no slicing API, and no batched matmul beyond what
  attention does internally. Whether a rank-polymorphic typed tensor (`Tensor<shape: List<Nat>>`) pays off in Bend is an
  open question (docs/plan.md).
- **Sizes fixed at compile time.** The layer constructors take literal sizes; bend-ml's `examples/runtime_batch.bend`
  shows the pattern for a batch size read at run time, not used here yet.
- Row operations on lists (above), sequential.
- No dropout (needs a reproducible random stream shared with the twin), no exact GELU (`erf`), no tied embeddings, no
  checkpoint save/load.
- Optimizer state: two buffers per parameter for every optimizer.
- The types of a model are long. Small `def X() -> Type` names help; a helper that derives `Seq`'s seven types from its
  two layers would help more.
- No law ties the F32 code to `spec.bend`, and none covers the non-linear layers' gradients.

## 2026-10-15: towards v1

What a v1 needed beyond the milestones: less waste in the products, a model that shows what it learned, checkpoints, the
missing modules (dropout, tied embeddings) and a stricter numerical policy where exactness is possible.

- **`LinearFirst`.** The first layer of a model computed `dX = dY·W`, which nothing reads (the data needs no gradient);
  PyTorch skips it. A variant that skips it (and returns zeros as dX) is a third less work for that layer: MNIST went
  from 12.5 to 8.7 s per epoch on one thread and from 8.2 to 5.9 s on 16 (3 alternated rounds), with the same loss to
  the bit (0.5148658). At 16 threads it is now below bend-ml's hand-written MNIST (7.4 s).
- **Attention over the sequences of a batch in parallel** (balanced halves, forward and backward): the tiny Llama's step
  from 13.1 to 11.3 ms at 16 threads, unchanged on one; outputs bit-identical over 50 steps (the order of every sum is
  the same; only which core runs a sequence changes).
- **The tiny Llama is a template over the batch** (`R(b) = 16·b` rows): `{==}` inside a template is checked before the
  template's arguments are known, so a size written as `Nat.mul(b, 16n)` keeps the proofs reflexive, and each instance
  reduces to its literal. The parameters' type does not depend on `b`, so the weights trained at `b = 2` run at `b = 1`.
- **Generation.** `tiny_llama sample` trains, then continues a prompt greedily one character at a time. PyTorch, given
  the weights Bend trained (read from the program's output), continues the prompt with the same characters: 300 out of
  300 after 2,000 steps; `test_tiny.py` checks 64 after 100 steps. The model is small (22,752 parameters, mean loss 2.36
  over the last 100 of 2,000 steps), and greedy decoding falls into a loop ("e than thar thar thar ..."): it shows that
  both programs compute the same function, not that the model writes well.
- **Checkpoints.** Every layer now has `dump` (its parameters as a flat list, in a fixed order) and `load` (overwrite
  them from such a list, returning what is left); `Seq`, `Fork`, `Both` and `Tie` compose both. `Train.save` writes the
  parameters, both optimizer moments and `b1^t`, `b2^t` as little-endian F32 (273,032 bytes for the tiny Llama:
  3 x 22,752 numbers + 2), `Train.load` reads them into a fresh state. 30 steps, a checkpoint, then 20 steps in another
  process give the same bits as 50 steps without a stop (`test_tiny.py`).
- **Dropout** with a counter-based mask: number `i` at step `t` is kept (times `1/(1-p)`) when
  `(lowbias32(key + i) >> 8) / 2^24 >= p`, `key = lowbias32(seed xor lowbias32(t))`. The step counter is the layer's
  parameter; the optimizer step increments it and checkpoints keep it. The twin computes the same mask in numpy, so the
  test is bit-exact (steps 0, 3, and 4 reached through `Layer.step`); 74.96% kept out of 100,000 at `p = 0.25`.
- **`Tie`** for a parameter used twice (tied embeddings): first and last share `P`, backward adds both gradients into one
  accumulator. Matches PyTorch's table used twice.
- **An exact tier in the numerical policy.** Copies, selections and one or two F32 operations in PyTorch's order (relu,
  an embedding lookup, the MSE and NLL gradients, dropout) must be the same bits; everything else keeps the tolerance.
  Results that match today only by luck of the order of additions or of `libm` (`tanh`, the embedding's gradient) stay
  out of the exact tier.
- **Mutation testing**: 23 mutants (new: two for dropout, one for `Tie`), all as expected.

### Speed, again

`make bench`, 5 alternated rounds, medians (range), the same machine; this time not idle (load average 2.8 when it
started), so the ranges are wider than before.

| | 1 thread | 16 threads |
|---|---|---|
| MNIST, one epoch of SGD, BendTorch | 8.8 s (8.7-9.5) | 6.0 s (5.9-6.6) |
| the same, PyTorch | 0.32 s (0.32-0.34) | 0.24 s (0.22-0.86) |
| tiny Llama, one AdamW step, BendTorch | 13.1 ms (13.1-13.8) | 11.5 ms (11.0-12.2) |
| the same, PyTorch | 9.0 ms (9.0-10.9) | 10.4 ms (9.7-12.8) |

MNIST is ~27x slower than PyTorch on one thread (from ~40x); the gap is the products of `bend-ml-tensor-array`, bound
by one array access per element (bend-ml NOTES exp. 17). The tiny Llama is 1.5x slower on one thread and close on 16.

### Still debts for after v1

Rank-N tensors and general broadcasting (plan M7), the row-wise layers on arrays (M6), laws about the executed F32 code
(M8), sizes read at run time, an optimizer state that SGD does not pay for, exact GELU, a less verbose way to write a
model's types.

## 2026-10-16: bend-ml-tensor-array 0.1.9.0

The pin moves from 0.1.7.0 to 0.1.9.0: bend-ml's products with two or more rows now pack their operands into cells of 8
numbers (0.1.8.0) and run in parallel without copying both operands at every node of the fork tree (0.1.9.0). Nothing
changes in this repository's code. `make check` gives the same counts of bit-identical numbers as before, `make bits`
is identical, the 23 mutants behave as listed.

### Speed on tensor-array 0.1.9.0

`make bench`, 5 alternated rounds, medians (range), the same machine, idle (load average 1.7 with the benchmark itself
running).

| | 1 thread | 16 threads |
|---|---|---|
| MNIST, one epoch of SGD, BendTorch | 3.9 s (3.8-3.9) | 3.6 s (3.5-4.0) |
| the same, PyTorch | 0.32 s (0.32-0.32) | 0.25 s (0.24-1.03) |
| tiny Llama, one AdamW step, BendTorch | 12.4 ms (12.4-12.4) | 10.9 ms (10.3-11.1) |
| the same, PyTorch | 8.9 ms (8.9-9.5) | 9.5 ms (9.4-9.7) |

MNIST goes from 8.8 to 3.9 s on one thread and from 6.0 to 3.6 s on 16: ~12x slower than PyTorch, from ~27x. Measured
against the old pin in the same session (15 alternated rounds, wall time of the whole run), 9.4 s against 4.3 s. The
tiny Llama barely moves (13.1 to 12.4 ms): its products are small and its row-wise layers, on lists, are most of the step.


## 2026-10-16: the JavaScript lane (plan M12)

`make check-js` (`scripts/check_js.sh`) builds the four test programs with `bend -o x.js`, runs them with Bun, and gives
them to the same Python tests as the native ones; then it compares the outputs of the Llama's training (50 steps) and
generation and of 50 MNIST batches under Adam with `bench/bits.txt`. Result: every test passes with the same counts of
bit-identical numbers as the native build, and the three outputs are the native build's bits. `exp` and `tanh` are
JavaScript's `Math` functions there, not libm's; on these inputs they gave the same words.

What it took:

- **Bun, not node.** `-o x.mjs` builds a module of the non-IO definitions, which does not run `main` (the first run
  printed nothing). `-o x.js` builds the program, whose file reads call libc's `read`/`pread` through `bun:ffi`: under
  node, every program that reads a file fails with "Cannot find module 'bun:ffi'".
- **A 1 GB stack.** The generated JavaScript keeps a call frame per step of each recursion that is not a tail call, such
  as a list built head first (`f32s`, the loaders); Bun's default stack holds about 40,000 frames and a weight file is
  100,000 numbers. The wrapper runs `ulimit -s 1048576` and sets `BUN_JSC_maxPerThreadStackUsage`, which allows about
  8 million.
- **The array capacity in bend-ml.** With the pinned 0.1.9.0, 50 MNIST batches take 247 s of training in JavaScript
  (4 s natively): `cap_depth(n)`, the size of a new array, walks `n` steps, each comparing two unary numbers of up to
  `n` steps. bend-ml-tensor-array 0.1.10.0 computes it by doubling; with it the same run takes
  4.3 s and the whole `make check-js` 65 s, against 34 minutes now.

## 2026-10-16: accumulating weight gradients (bend-ml-tensor-array 0.1.10.0)

The three linear layers (`Linear`, `LinearFirst`, `LinearNB`) used to compute a weight gradient into a new matrix and
then add it to the gradient they accumulate (`Mat.add`), and the bias gradient the same way. They now call bend-ml's
accumulating products, `Mat.matmul_tn_acc` and `Mat.col_sums_acc`, which add each cell of the product straight into the
gradient: no new matrix, no second pass. Each cell is computed as before and added once, `g + x`, which is what the
`Mat.add` computed, so `make check` gives the same counts, `make bits` is identical, and the 23 mutants behave as listed
(the two linear ones now remove the accumulation; the equivalent mutant about the order of the bias addition is gone,
since that addition now happens inside bend-ml).

Measured on the same idle machine, alternated rounds, medians:

| | before | accumulating | ratio |
|---|---|---|---|
| `dW += dYᵀ·X`, 32 x 1536 x 576 (a SmolLM2-135M MLP projection, 32 tokens), 1 thread | 260 ms / 40 | 184 ms / 40 | 0.71 |
| the same, 8 tokens | 359 ms / 100 | 159 ms / 100 | 0.44 |
| the same, 128 tokens | 176 ms / 10 | 152 ms / 10 | 0.86 |
| the same, MNIST's first layer (100 x 128 x 784) | 175 ms / 100 | 154 ms / 100 | 0.88 |
| MNIST, one epoch of SGD, whole run, 1 thread (15 rounds) | 4.25 s | 4.12 s | 0.97 |
| the same, 16 threads | 4.07 s | 3.96 s | 0.97 |
| tiny Llama, 200 AdamW steps, 1 thread | 2.49 s | 2.48 s | 1.00 |

The pass it removes is proportional to the weights, the product to the weights times the rows of the batch, so the gain
grows as the batch shrinks relative to the weights: 3% on MNIST, nothing visible on the tiny Llama, 30-56% per weight
gradient at SmolLM2's sizes with a few dozen tokens (plan M10). On those sizes, the same micro-benchmark is also slower
on 16 threads (`par` 3) than on one, for both versions: 455 against 260 ms at 32 tokens.

- 0.1.10.0 is on BendHub (hash `0xbb78793a...`). Against the published package: `make check` the same counts, `make bits`
  identical, `make mutate` 23/23, `make check-js` the same bits as the native build.

## 2026-10-16: bend-ml-tensor-array 0.1.11.0

The products choose their split depth from the work (`par` is an upper bound; 0 below 2^23 multiply-adds or 64 per
cell). Here `make check` gives the same counts, `make bits` is identical, `make mutate` 23/23 and `make check-js` the
same bits as native. The layers keep their `par` values: MNIST's two large products stay above the threshold (an epoch
unchanged, 3.60 s on one thread, 3.30 on 16), and passing 4 instead of 3 to the first layer was slower over a whole run
(measured; the runs are in my private notes). The gain is for M10's shapes: a weight gradient over 8 tokens of
SmolLM2-135M's sizes 3.3x faster on 16 threads.

## 2026-10-16: CI (plan M13)

Actions is on for the private remote. The workflow runs in two modes, so it fits the 2,000 minutes a month a private
repository gets: **quick** on every push to `devel` or `main` (`make check` and `make bits`; a newer push to the same
branch cancels a run in progress) and **full** on pull requests to `main`, every night on `devel` and by hand (also
`make mutate` and `make check-js`, with Bun 1.4.2). The Bend and Lean toolchains, the Python environment and the
datasets are cached.

The first runs, all green on a 4-vCPU `ubuntu-24.04` runner: quick 6.4 min with cold caches (setup 1.0, check 2.9,
bits 1.7); full 10.4 min with warm caches (setup and caches 0.9, check 2.1, bits 1.5, mutate 3.9, JavaScript lane 1.7).
At five pushes a day and one nightly run that is about 600 + 320 minutes a month. The nightly schedule starts once this
workflow is on `main`, since GitHub runs scheduled workflows from the default branch.

## 2026-10-16: general tensors (plan M7)

`torch/tensor.bend`: `Tensor<s>`, the shape a list of sizes in the type, over one flat `Array<F32>`, row-major.

**The prototype first** (broadcast add and a sum over an axis at rank 3, with two laws), to measure what was open in the
design: the checker takes 0.15 s for the module and for a program with literal rank-3 shapes, whose broadcast proof
(`bc_ok(...) == True`) it computes; a wrong broadcast is rejected ("expected True, observed False"); types like
`Tensor<remove(s, k)>` reduce to literal shapes on their own; the laws (`bc_sym`, `sum_numel`) are short inductions over
the shape. **Decision: adopt `Tensor<s>` as the layer of general operations next to `Mat`, without porting the
layers.** `Tensor<[r, c]>` and `Mat<r, c>` are one array under two types, so `of_mat` and `to_mat` cost nothing (law
`mat_numel`); the layers keep bend-ml's fast products, and M6 stays on `Mat`.

**The operations**, each a gather or a scatter over an index map (per-axis sizes and strides, stride 0 where an axis is
broadcast; or "the position with one axis taken out"): `add`, `sub`, `mul`, `div` with broadcasting and their gradients
(PyTorch's formulas, the divisor's as `-dz·((x/y)/y)`); `sum_axis` (and `sum_axis_keep`), `mean_axis`, `sum_all`,
`mean_all`; `reshape` (with the proof that the counts are equal) and `flatten`; `permute` and `transpose`; `narrow`
(with the proof that the window fits); `cat`; `bmm`. `cat` and the backward of `narrow` and `permute` copy into place
(no addition), so `-0` stays `-0`.

**Checks.** `demo/tensor_test.bend` reads the operation and the shapes at run time and checks each shape condition once
where it enters; `reference/test_tensor.py` draws 204 cases (ranks 0-4, size-1 and empty axes, 12 per operation) and
compares the result and every gradient with `torch.autograd`: all within the tolerance, the worst at 0.070 of it (the
order of the sums of reductions and products), and 14 invalid shapes rejected. The same in JavaScript (`make check-js`).
Seven shape laws by `--verdict` (`bc_sym`, `sum_numel`, `narrow_rest`, `narrow_axis`, `cat_numel`, `mat_numel`,
`permute_rank`); four programs that must not compile (`tests/bad_broadcast`, `bad_narrow`, `bad_cat`, `bad_bmm`). Five
new mutants (a wrong sign in div's gradient, narrow's window on the wrong axis, bmm reading its operand transposed,
broadcasting one way round only, cat's shape counting one operand twice), all caught: 28 of 28.

**Left.** The kernels are scalar and use U32 division per position to find coordinates; fast enough for the tests, not
measured against PyTorch, and not parallel. That a general permutation keeps the count of numbers is tested, not proved
(the law covers the rank). The layers do not take tensors of rank 3 or more yet.

## 2026-10-16: the tensors' index maps stay inside their arrays (laws track L1)

`demo/imap_check.bend` and `reference/check_imap.py` walk every position of every index map on every small shape (rank
0-3, sizes 0-3) and on larger random ones: 4,206 cases, 242,871 positions, none out of bounds, natively and in
JavaScript; in `make check` and `make check-js`. Against 11 deliberate faults in the maps it caught one the numeric test
cannot: the whole-tensor sum mapping each element to its own position of a 1-cell array, right only because an Array
index wraps around. It is in `make mutate` as a pair (survives the numbers, caught by the bounds), 30 mutants in all.
A law that the maps stay in bounds is the next step (L2).

## bend-lint's first pass (2026-10-17)

Mathews Fernando's bend-lint, with its 12 performance rules, is a dev dependency (`package.json`, `bend-lint.json`,
`make lint`), run before each milestone closes; a finding is measured before it is taken. First pass over the 19 Bend
files: 25 findings, none taken yet:

- `list_index_loop` on `lm_rows` in `attn_batches`/`attn_bwd_batches` and on `emb_gather`, `loop_invariant_call` on
  `lm_zeros`/`l_zeros`: loops over the b = 2 sequences of a batch, or 32 rows of an embedding; their O(n²) is a few
  hundred list steps a step.
- `eager_branch` in `torch/tensor.bend` (shape checks, once per operation), `append_in_loop` in `demo/tensor_test.bend`
  (a test's output), `unused_result` in `PROOF.bend` (a law), `eager_branch` on `index_of` in the tiny Llama's
  vocabulary lookup (65 characters).

Found next to it, not by a rule: a profile of 50 tiny-Llama training steps on one thread spends 33.5% in `term_drop` and
33.3% in `span_fade`, the runtime taking apart list nodes that are shared (`+` lists through attention, the norms and the
row layers). Two thirds of a step is that, against 2.4% in the matrix products. This is what M6 (row-wise layers on
arrays) removes, and the measure M6 should be judged by.

## bend-lint's attention findings at larger batches (2026-10-17)

The tiny Llama's attention forward and backward over b sequences of 16 positions, one thread, 3 rounds: 2.01 ms per
sequence at b = 2, 2.04 at 8, 2.06 at 16, 2.08 at 32, 2.13 at 64. The quadratic part of `lm_rows` and `lm_zeros` is under
6% at b = 64, and a full fix would cut the lists down `attn_par`'s tree, which M6 replaces. Not taken; the 14 findings
carry a `bend-lint: disable-next` directive with the reason. `term_drop` and `span_fade` are 79% of the same profile.
`emb_gather` is left open: measured separately, the Embedding over arrays is 52-78x faster at SmolLM2's table, waiting
for my call.

## The Embedding over the table's array (2026-10-17)

bend-lint flagged `emb_gather` (`List.drop` per id); the cost was larger than that: the forward turned the whole table
into a list of rows (`Mat.rows(v, d, p)`) on every step, and the backward turned the gradient table into rows and back.
Now the rows of the ids are read from the table's `Array` by index (`TA.Mat.read_row`), and each dY row is added into
the gradient's row in place (`read_row`, `l_add`, `write_row`), in the order of the ids, as before. `row_head`,
`emb_gather`, `add_at` and `emb_scatter` are gone.

At SmolLM2-135M's table (49,152 x 576), forward and backward, one step, 3 rounds alternated: 2,487 → 31.8 ms with 64
ids (78x), 2,870 → 55.7 ms with 512 (52x); peak memory 942 → 273 MB at 64 ids; the same checksum (the runs are in
my private notes). `make check` passes, `make bits` identical, `make mutate` 30/30 as expected (the
embedding mutant now drops the `l_add` in place: caught), `make lint` 0 findings.

## The linear layers over bend-ml-tensor-fork (2026-10-17)

bend-ml-tensor-fork 0.1.0.0 (bend-ml v1.4.0) has tensor-array's six products with the parallel blocks sharing their
operands through Base's `@unsafe` `Array.fork` instead of copying them; same arguments, result types and bits. Renan's
decision: use it, as fast layers next to the proved ones, the demos on the fast ones.

- tensor-array 0.1.11.0 → 0.1.12.0 first (tensor-fork's types are 0.1.12.0's): `make laws`, `check`, `bits` identical.
- `torch/fast.bend`: `LinearF`, `LinearFirstF`, `LinearNBF`, `main.bend`'s layers with their three products from
  tensor-fork; everything else is `main.bend`'s code. `main.bend`, `LAWS.bend` and `PROOF.bend` do not change, so
  `--verdict` still checks every law. `make laws` pins the eight defs of `fast.bend` outside the proofs and checks that
  no file with laws imports it.
- `demo/fast_test.bend`: each fast layer against its proved twin, forward and backward, at MNIST's sizes (LinearFirst
  100x784x128 at par 3 and 4, Linear 100x128x10), the tiny Llama's (LinearNB 32x32x64) and splitting ones (Linear and
  LinearNB 128x576x1536): 0 differing words in y, dX and every gradient, on 1 and 16 threads and in JavaScript (in
  `make check` and `make check-js`). Three mutants on `fast.bend` (each weight gradient not accumulated), caught by it:
  `make mutate` 33/33 as expected.
- The demos (MNIST, the tiny Llama) on the fast layers: `make check` passes, `make bits` identical, `make check-js` the
  same outputs. MNIST's first layer at par 4 instead of 3: with the fast layers par 4 is faster on 16 threads (2.6 against
  3.0 s, alternated) and the same on one (2.9); with the proved layers it had been slower (2026-10-16). The same bits.
- `make bench`, 5 rounds, before (proved layers, par 3) and after: MNIST 3.6 → 2.9 s on one thread, 3.3 → 2.4 s on 16
  (PyTorch 0.32 and 0.26 s: 11x → 9x on one thread); the tiny Llama 12.1 → 12.2 ms on one thread and 10.7 → 10.3 on 16,
  unchanged, as expected: its products are 32 wide, below tensor-array's split threshold, so both layers run the same
  sequential code. Its step is spent taking apart shared lists (two thirds of it), which is M6's work.

## Attention over arrays (M6, phase 1; 2026-10-17)

A profile of the tiny Llama's step (one thread, 30 steps, call stacks) put about half of it in attention's list
products (`lm_dots` under `lm_mul_t`, with the teardown of the shared lists they walk) and 11% more in `List.take`
cutting rows and columns; the other row-wise layers under 5% each. So M6 starts with attention.

- Q, K and V stay in their arrays. Each sequence's rows are copied out once into arrays of its own; each head's P and G
  (t x t) are arrays. Every sum is added in the list version's order (`l_dot`: left to right from +0.0, including the
  zeros past the visible positions in P·V), and each row of P still goes through `causal_row` and `l_softmax`, each row
  of G through the same `l_mul`/`l_shift`/`l_dot`/`l_scale`. The scores past position i are no longer computed. `make
  bits` identical at 1 and 16 threads, and every summary line of `make check` (bit-identical counts, worst errors)
  identical to the list version's.
- The sequences of a batch: a tree of parallel lets from 8 sequences on, one after another below. Attention alone,
  forward and backward, 4 query heads and 2 key/value heads of 8, 16 positions, 16 threads, 5 rounds: 2 sequences 0.37 ms
  in a loop against 0.48 in the tree; 8: 1.52 against 1.32; 32: 5.95 against 4.00; 128: 23.8 against 14.0. One thread
  the same either way.
- Law `causal_keys`, the law about the code that runs (my call: write it over the new code): attention reads keys
  only through `qrow`, and the row it builds for the query at position i has exactly i + 1 scores, keys 0..i. The proof
  is an induction over `drow`'s loop with `len_rcat` and `add_succ`. `causal` still holds over the executed
  `causal_row`. A mutant that reads one key too many survives the numbers (`causal_row` drops the extra score) and is
  caught only by the law. `make mutate` 35/35 as expected (the two attention mutants that aimed at the list code
  rewritten for the new one).
- The list helpers attention alone used are gone (`lm_mul`, `lm_mul_t`, `lm_t`, `lm_dots`, `lm_heads_of`,
  `lm_tails_of`, `lm_cols`, `lm_rows`, `lm_zeros`, `lm_add`, `lm_hcat`).
- `make bench`, 5 rounds (load 1.7 at the start): the tiny Llama 12.2 → 4.1 ms a step on one thread, 10.3 → 5.0 on 16
  (PyTorch 9.0 and 12.9). MNIST unchanged (2.9 s on one thread; 2.6 on 16 against 2.4 the run before, within this run's
  2.4-2.7). A profile after: `term_drop` and `span_fade` 33% of a step three times shorter (67% before); what is left is
  the row-wise layers on lists (phase 2).

## The row-wise layers over arrays (M6, phase 2; 2026-10-17)

A profile after phase 1 (one thread, 400 steps) still put 39% of a tiny-Llama step in `term_drop` and `span_fade`, the
runtime taking apart shared lists, most of it under RMSNorm, the rows of attention's P and G, the matrix-to-rows
conversions and RoPE. Phase 2 moves every row-wise layer onto the matrix's own array, a row at a time, by index.

- Every loop keeps the order of operations of the list version it replaces (sums left to right from +0.0, each product
  in the same order), so the bits did not move: `make bits` identical at 1 and 16 threads and in JavaScript, every summary
  line of `make check` identical to phase 1's, and the raw output of `ops_test` under `test_ops.py` (605 lines, 10 seeds,
  every layer) byte for byte the same as before phase 2.
- In order, 200 training steps pinned to one P-core, each change against the one before it: RMSNorm 0.82 → 0.50 s,
  attention's rows of P and G 0.50 → 0.40 s, RoPE (a cos/sin table per call, the pairs rotated in place) 0.40 → 0.24 s.
  Then softmax, log-softmax, LayerNorm, the embedding, MSE and NLL, which the demos barely use or do not use, so that no
  layer turns a matrix into lists any more; 20 list helpers are gone, and the shared row kernels (dot, softmax in place,
  copy) live in one section of `torch/main.bend`.
- A backward pass now keeps a matrix: RMSNorm and LayerNorm their input, softmax and log-softmax a copy of their output.
  The `S` type of those layers changed from a list of rows to `Mat`, so the demos' type arguments changed with it.
- Law `causal`, rewritten over the new code (my call): `prow` writes query i's scores from the start of its row of P
  and hands the softmax the count of positions it wrote, and the law says that count is i + 1. The proof is the same
  induction as `causal_keys`, through `pw`'s count (`cnt_pw`, `cnt_drow`). That the positions past i keep P's zeros is
  read from the code, not proved: the checker cannot follow writes into an array by index. The old law spoke about
  `causal_row`, which no code calls any more, and is gone with it.
- `make mutate` 37/37 as expected. The row-wise mutants now aim at the array kernels. A query that sees the next position
  (the count starts at 1) is caught by the law and by the numbers. A softmax over one position too few survives the law
  (it speaks about the count, not the loops inside the softmax) and is caught by the numbers. A query that reads one key
  too many, which in phase 1 only `causal_keys` caught, now changes the numbers too: the extra score reaches the softmax.
- `make bench`, 5 rounds: the tiny Llama 4.0 → 1.1 ms a step on one thread, 4.3 → 1.3 on 16 (PyTorch 8.9 and 12.8);
  MNIST 2.9 s on one thread, 2.4 on 16, unchanged. PyTorch's MNIST epoch on 16 threads took 0.23, 1.07, 1.01, 0.90 and
  1.17 s in the five rounds (0.23-1.20 in the run before), so the speed table gives its range.
- A profile after (one thread, 1000 steps): `term_drop` and `span_fade` 8% of the step. The products are now the largest
  part (about a third), then bend-ml-tensor-array's cross-entropy, which still softmaxes each row as a list (about 10%),
  and `cap_depth`, recomputed by every `Array.new` of a kernel's scratch array (about 5%). 16 threads are still slower
  than one on a step this small: phase 3.

## What phase 2 left: cap_depth, cross-entropy, the products (2026-10-17)

The profile after phase 2 named three things left in a tiny-Llama step. Times are 1000 training steps pinned to one
P-core, 3 runs each, against the phase-2 binary.

- `cap_depth` (about 5%): every kernel's scratch array asked bend-ml-tensor-array's `cap_depth` for its capacity, which
  compares Nats one unit at a time at each doubling. `depth` doubles a U32 instead: log2(n) cheap steps, the same
  answer (`demo/depth_check.bend`, in `make check`: equal to `cap_depth` for every n from 0 to 4999). 1.06 → 1.03 s.
  What is left of `cap_depth` (2.7%) is inside the packages: the output arrays of bend-ml-tensor-fork's products.
- Cross-entropy (about 10%): `CrossEntropy` called tensor-array's `Mat.softmax_ce`, which reads each row into a list,
  softmaxes it as a list and writes the gradient back. It now works on the logits' array with `psm` and two short loops,
  tensor-array's operations in its order: `make bits` identical (MNIST and the Llama), the raw `ops_test` outputs byte for
  byte the same. With `depth`, 1.10 → 0.93 s.
- The products (about a third): the tiny Llama's products (32 rows, LinearNB 32 → 16, 32, 64, 65 and 64 → 32; forward,
  dX and dW) through tensor-array's two paths, 1000 rounds on one thread: 0.35 s on the 8-wide packed path the layers
  already take, 0.58 s on the scalar one, the same result. So the products are where the time should be, already on the
  faster path; making them faster is a new kernel in bend-ml, not a fix here.
- `make bench`, 5 rounds: the tiny Llama 1.1 → 1.0 ms a step on one thread, 1.3 → 1.1 on 16 (PyTorch 9.0 and 12.8); MNIST
  unchanged (2.9 s, 2.5). A profile after: `term_drop` 0.6% of a step (67% before M6); the products about 43% (with
  tensor-array's packing), attention's kernels about 26%. `make mutate` 39/39 with a cross-entropy mutant and a scratch
  array at half its capacity, both caught.

## Parallelism where it pays (M6, phase 3; 2026-10-17)

The tiny Llama cannot show it: 1000 steps take 0.92 s on one thread and 0.97-1.01 s on 2 to 16, every piece of work
smaller than the cost of splitting it. So I measured on a new workload, `demo/smol_block.bend`: one decoder block at
SmolLM2's sizes (the tiny Llama's block, RMSNorm, grouped-query attention with RoPE, SwiGLU, two residuals), 8 sequences
of 64 positions, AdamW against a target under MSE, its weights, inputs and targets from a hash of each number's index, and
its PyTorch twin `reference/block_ref.py`. `scripts/block.sh` builds it at 135M's sizes (576 wide, 9 query and 3
key/value heads of 64, feed-forward 1536), 360M's (960, 15 and 5, 2560) and 1.7B's (2048, 32 and 32, 8192). The three
agree with PyTorch within the tolerance (worst 0.01 of it) and give the same bits on 1 and 16 threads
(`reference/test_block.py`; the 135M one is in `make check`).

- **MSE summed as a tree.** The block's first comparison was 1e-4 off: the Bend loss was exactly a running F32 sum of
  PyTorch's own squares, and 295k squares added left to right lose the small ones. The loss is now a balanced tree of
  additions (pairwise, as llama.bend's `tree_sum`): 1 ulp from PyTorch and from a float64 sum. The gradient is unchanged.
- **Where one thread spends a step** (135M, `perf`): the products 86% (tensor-fork's two fork trees 82%, packing 3%),
  attention 7%, every row-wise layer together about 3%. The row-wise layers were never the problem.
- **Where 16 threads wait.** A `perf` profile on 16 threads, cut into 1 ms slices and counting the threads at work in each:
  43% of the wall time had three threads or fewer at work. In order: the packing of A and B (sequential), sizing arrays
  through `TA.cap_depth`, the optimizer step, SiLU, the copies and sums of `Fork`/`Residual`, RMSNorm, RoPE, attention's
  copies.
- **The products' split.** At `par` 4 (16 blocks) the products stopped gaining at 8 threads: equal blocks on P-cores,
  their second hyperthreads and E-cores finish at different times. `par` 8 (256 blocks) took the block from 763 to 577
  ms per 3 steps on 16 threads at 135M, 1840 to 1313 ms at 360M (par 9 no better), one thread unchanged.
- **bend-ml-tensor-fork 0.1.1.0** (bend-ml NOTES): `par1`..`par4`, any kernel over blocks of a range on up to four shared
  arrays; the packing in blocks when an operand holds 16384 cells or more; output arrays sized by `depth`. 562 → 527 ms.
- **`torch/fast.bend`**: the optimizer step (SGD, momentum, AdamW), `SiLU`/`ActIn`, `Mul2`, `Add2`, `Residual` and the
  copies and sums its `Fork` uses, each on `par1`..`par4` with main.bend's own loops, so main.bend's bits
  (`demo/fast_test.bend`, 12 cases, 1 and 16 threads and JavaScript). Below 131072 numbers a matrix runs in one block:
  with MNIST's 100352-number weights in 8 blocks its 16-thread epoch was 0.1-0.2 s slower. 528 → 500 ms.
- **Not adopted**: RMSNorm and RoPE in blocks of rows (the backward pass's gain gradient as a second pass over blocks of
  columns, each column in row order: the same bits) took them off the sequential list but changed nothing measurable,
  so they stay sequential. Leaves of 2 x 2 tiles in the products (bend-ml NOTES): the same bits, 20% slower.
- **Result**, ms per AdamW step, medians of 5 alternated rounds (1.7B: 3), a warm-up step outside the timing, one thread
  pinned to a P-core (my private notes have the runs):

  | | 1 thread before | after | PyTorch | 16 threads before | after | PyTorch |
  |---|---|---|---|---|---|---|
  | block at 135M's sizes | 779 | 779 | 106 | 247 | **155** | 51 |
  | block at 360M's sizes | 2029 | 2026 | 282 | 589 | **351** | 92 |
  | block at 1.7B's sizes | 14116 | 13963 | 1852 | 4412 | **2733** | 545 |

  16 threads 1.6-1.7x faster at every size, one thread unchanged; 16 threads now give 5.0-5.8x over one, against 3.1-3.4x
  before. PyTorch stays 7x faster on one thread and 3-5x on 16: its products are a BLAS kernel, and ours are scalar F32
  chains in a fixed order (the 2 x 2 tiles showed the reads are not the limit).
- What is left on 16 threads (135M): about 23% of the wall time with three threads or fewer at work, now mostly zeroed
  allocations (the packed operands and the products' outputs) and attention's copies and joins; attention runs its 8
  sequences on 16 threads.
- `make bench`, 5 rounds: the block 768 ms a step on one thread, 152 ms on 16 (PyTorch 105 and 50); the tiny Llama 1.0
  and 1.1 ms (PyTorch 9.1 and 12.8) and MNIST 2.8 and 2.4 s (PyTorch 0.32 and 0.25-1.07), unchanged within the noise:
  their matrices are below every new threshold. `make check`, `make bits` (identical), `make check-js` and `make mutate`
  (42/42, with mutants for the fast optimizer's blocks, the fast copies and MSE's tree sum) pass.
- M6's pass criterion asked for the tiny Llama's step to be faster on 16 threads than on one. It is not, and cannot be at
  that size; the 16-thread gain is measured on the block instead.
- Decisions (2026-10-17): I keep the optimizer step and the element-wise layers in `torch/fast.bend` (the exception now
  covers them, my call). RMSNorm and RoPE in blocks stay out; I will measure them again if a later workload makes them
  look useful (the code is kept in my lab notes).

## Laws about the code that runs (M8; 2026-10-17)

Six new laws and one rewritten, all by `--verdict`, about the code the demos execute.

- **`kv_cols`.** The head loops of both attention passes used to compute each head's columns inline, so a mutant could
  make the forward pass alone bypass `kv_of`, and no law saw it. Now one function, `head_cols`, gives a head its columns
  (`HC{hc, kc}`), and `head_fwd`/`head_bwd` take only those: with no head index they cannot compute others. The law: query
  head `j·g + i` reads key/value columns `j·hd`, by `kv_of_group`.
- **`causal_grad`.** The backward pass computed each row of dP and G over all t positions and relied on P being 0 past
  the query. `grow` now reads dP's row against values 0..i only (as `qrow` does for keys), and the sum and G run over the
  positions it wrote; the law says that count is i + 1. The rest of G keeps its zeros. Same bits (`make bits`), one more
  bit-identical number in `ops_test` (13,451), and less work: the tiny Llama 0.93 → 0.91 s per 1000 steps on one thread,
  the 135M block 1590 → 1566 ms per 2 steps (3 alternated runs, the machine not idle; 16 threads within the noise).
- **`causal`, rewritten.** It spoke about the count `prow` handed the softmax, so a softmax that ran over one position
  too few passed it. The softmax's division loop now counts the positions it divides (`psmc`, a `PC{a, c}`; `psm` wraps
  it, so Softmax and cross-entropy are unchanged), and `prow3` hands that count on as the row's. The proof reaches the
  count through the maximum and exponential loops by matching their results, not computing them.
- **`attn_par_fwd`, `attn_par_bwd`.** The trees over sequences (`fpar`, `bpar`) give what `fleaf`/`bleaf` give one
  sequence after another, at every depth and however `cut` splits them. Their lists hold arrays, and Bend lets a proof use
  such a list once, so the obvious proof (the loop is a map, maps distribute over append, cut's halves append back to the
  list) needed each list twice. Each lemma now returns the list it was given together with its fact (records like
  `L1R{ys, same, fact}`), and the induction on the depth calls itself through two closures, so there is no mutual
  recursion. About 200 lines for each pass.
- **`seq_chain`, `fork_sum`.** `spec.bend` now has the model's linear layer as a `T.Layer` (`LinL`), and `SeqL`/`ForkL`
  instantiate the package's own `Seq` and `Fork` with it. The laws run those combinators: `dy·y == dx·x` through `Seq`
  (the proof is `adjoint_seq`, since every step of the combinator reduces) and `dy1·y1 + dy2·y2 == dx·x` through `Fork`
  (`dot_vadd` and `adjoint` twice). A law cannot quantify over layers (they are templates), so these speak about the
  model's linear layers.
- **Not stated**: tensor-array's parallel product. Each block reads its own packed copy of its rows, so "the same rows as
  the sequential product" would need the contents of arrays written by index. Tensor-fork and `fast.bend` cannot have
  such a law at all: in Base's model `Array.join` keeps one handle. They stay tested bit for bit.
- `make mutate`: 46 as expected. The two mutants that survived the laws (`kv_of` bypassed, the softmax one position
  short) are caught by laws, and so are new ones: `head_cols` with `mod`, dP's row reading one value past the query, each
  tree putting its right half first, `Fork` dropping a gradient. The one survivor is the tensor sum that a 1-cell array
  makes right by accident (caught by the bounds check). `make check`, `make bits`, `make check-js` pass.

## Training infrastructure (M9; 2026-10-17)

- MNIST reads an optional batch size (default 100) and handles partial training/evaluation batches. `Seq.at` composes
  owned layer records and `Train.step_at` consumes one across forward, backward and step; their shapes remain checked.
  Batches 37 and 8191 match PyTorch within the existing tolerance with the same test hits. Batch 100 retains the
  recorded bits. Pixel reads are bounded and fill arrays in a tail loop: the first large-batch JavaScript test hit a
  stack overflow when millions of pixels were converted in one list; the bounded loader passes it.
- `OptState<P>` holds no buffers for SGD, one for momentum, two for Adam/AdamW. `Train.new` takes the configuration;
  every layer/combinator uses the matching state. The fast matrix step splits two, three or four arrays respectively;
  momentum's loop returns three arrays without a dummy fourth one. The F32 expressions and their order are unchanged.
- `Layer.load` takes one `Array<F32>` and an element offset, returning the source and next offset with the parameters;
  `Layer.count` gives its serialized size. MNIST and the tiny Llama use one initial-weights file each, in `dump` order.
  `torch/load.bend` checks exact sizes and short reads with the temporary binary reader; it stays outside every law.
  Checkpoint counts are checked before multiplying into a U32 byte length. Array allocation uses the existing U32 depth
  loop, avoiding `cap_depth`'s unary size comparisons on a large file.
- Checkpoints now contain only parameters for SGD, one extra parameter-shaped buffer for momentum, two plus Adam's
  powers for Adam/AdamW. A tied 3x3 parameter with Dropout gives 40, 80 and 128 bytes respectively; all four optimizers
  resume bit for bit, including the mask counter. The tiny Llama's 30-step AdamW checkpoint is byte-identical to the
  pre-M9 file. Old SGD/momentum checkpoints must be regenerated. Writing still uses lists and waits for M10.
- `Pair3` and `ResidualType` replace repeated composite types in the tiny Llama and the SmolLM2-sized block.
- Limits: JavaScript canonicalizes a noncanonical quiet NaN payload in the copy test; native preserves it. Both keep
  signed zero, infinities and subnormals. The numerical policy still matches NaN with NaN; finite-value checks and the
  shared tolerance are unchanged. The owned composition/step are tested, not newly proved; the existing laws remain.
- Checks: `make check`, `make check-js`, the original bit digests, `make laws` by `--verdict`, and 57/57 mutation outcomes
  as expected (one known numerical survivor is caught by the bounds gate). A rejected run-time-size composition keeps
  the shape-error gate. The block's SGD path also matches PyTorch and gives identical bits on 1 and 16 threads.
- `make lint` reports the same 10 eager-branch findings in `torch/tensor.bend`; none is in the new M9 paths. I did
  not change those existing predicates without measurements.
- Measured on an idle machine (alternated rounds, 1 thread pinned to CPU 3, 16 unpinned; Core Ultra 7 155H, Bend
  2.0.37, clang 22):
  - Peak RSS of one decoder block at SmolLM2-360M's sizes (9.8M parameters, 5 steps): 234 MiB under SGD and 354 MiB
    under AdamW on 1 thread, 470 and 584 MiB on 16. AdamW adds 114-120 MiB, more than its two moments' 75 MiB; I have not
    broken the rest down.
  - A SmolLM2-135M-sized file (134,515,008 F32, 538 MB) reads into one array in 0.26 s with the file in the page cache
    and 0.33 s after asking the kernel to drop it (a hint, not a guaranteed cold read), medians of 5. Peak RSS is
    1.0 GiB: the array's capacity rounds up to 2^28 cells, twice the file. That array is released once the layers are
    filled; it is the transient cost of loading, and for 360M it would be 2 GiB.
  - Loading the demos' initial weights (with a walk over every number): MNIST 6.65 -> 3.02 ms, the tiny Llama 1.67 ->
    0.81 ms.
  - A whole MNIST epoch run: 3.20 -> 2.63 s on 1 thread, 2.67 -> 2.02 s on 16, the same output. The 135M- and
    360M-sized block steps did not move (within 0.7%).
- The tiny Llama got slower: 1000 training steps took 6% longer on 1 thread with the same bits. It ran 5.2% more
  instructions, so it was not clang's loop schedule. A profile put the difference in `term_drop`: `count` was a plain
  `U32` field, and `Seq`, `Fork`, `Both` and `Tie` set it to `Layer.count(la) + Layer.count(lb)`. A layer passed as a
  template is built again at every use, so each `fwd`, `bwd` and `step` of a composite built its children's whole
  records to add their counts and dropped them. `count` is now a closure (`Unit -> U32`), like `zeros`: the tiny Llama
  is back to +0.13% instructions and +1.1% time on 1 thread against before M9 (0.897 -> 0.907 s; on 16 threads 1.158 ->
  1.081 s, a noisy lane). MNIST and the block did not change with it. `make check`, `make bits`, `make mutate` (57 of
  57) and `make check-js` pass.
- I also tried 8 and 16 independent partial sums in the packed dot product instead of 4 (numerically fine, the same
  bits on 1 and 16 threads): the whole block step got slower, 0.56-0.58x on 1 thread and 0.74-0.78x on 16, at both
  sizes. Kept 4.
- bend-lint did not see the eager `count`. I wrote a local rule, `bend-lint-rules/performance/template_value_field.bend`:
  a record with a lambda field whose other field is a value that reads a template parameter (not a `Type`, number, flag
  or string). On the code before the fix it reports exactly `Seq`, `Fork`, `Both` and `Tie`; on the current code,
  nothing. `tests/lint_template_value_field.bend` expects the finding, so `make lint` fails if the rule stops seeing it.

## Before M10: where the memory goes (2026-10-17)

- AdamW's extra memory in the 360M-sized block is its two moments and nothing else, but each parameter-shaped buffer
  takes 60.0 MiB for 37.5 MiB of numbers: a Bend array is one block of 2^k cells, so the 2560 x 960 matrices sit in
  blocks 1.71x their size and the 960 x 960 ones 1.14x. AdamW's heap is exactly 122,896 KiB above SGD's, the two
  moments at that capacity; the step allocates nothing. Under SGD the peak is parameters and gradient (61,448 KiB each)
  plus 107,828 KiB of forward and backward values (about half of it saved for the backward pass), and about 23 MiB of it
  is freed blocks that the runtime's size classes cannot reuse for other sizes. Measured with allocation counters added
  to the emitted runtime (massif sees nothing: the runtime does not use malloc); raw data kept outside this repository.
- At whole-model sizes the same rounding makes AdamW's four buffers 2.49 GiB instead of 2.00 at 135M and 8.50 GiB
  instead of 5.39 at 360M (arithmetic over the shapes). Both fit this machine's 32 GB; 1.7B would not without M11.
- I did not make loading exact (dropped from the plan, my decision, 2026-10-17). The loader's source array is the next power of two above the file (1 GiB for the 538 MB
  135M file), but it is released before `Train.new` allocates the gradient and the moments, so the loading peak (1.62
  GiB at 135M, 4.13 at 360M) stays below training's. It would not lower M10's peak.
- Left for later, with sizes: banded storage of the big matrices (cells just under a power of two; about 19.5 MiB per
  buffer at the 360M block), one saved input shared by the q/k/v and gate/up forks (6 MiB at 512 rows), and the size
  classes' stranded blocks (a runtime question).

## SmolLM2-135M and 360M in BendTorch's layers (M10; 2026-10-17)

- `demo/smol.bend` is SmolLM2-135M from the package's layers: the embedding, 30 of `smol_block.bend`'s blocks on
  sequences of 128, the final RMSNorm and the head, tied to the embedding through `Tie`. `scripts/smol.sh 360m` writes
  the 360M's copy. The stack is a balanced tree of `Seq` written by the script, one def per subtree size: a template
  takes only arguments fixed at compile time (no recursion on the depth), and Bend instantiates templates 64 levels
  deep at most (a chain of 30 `Seq` fails). Each block is wrapped in a `Box` layer whose parameters and saved values are
  one boxed word: Bend lays nested records out flat, a word a field, and a def or a continuation takes at most 247
  words; from 8 blocks up the compiler stopped with "an arity over 247". The program compiles in about 50 s.
- `reference/smol_ref.py` is the twin in plain torch; `reference/check_smol_hf.py` shows it equals transformers'
  `LlamaForCausalLM` on the same batch (the same loss and bit-identical logits). `scripts/smol_data.sh` converts the
  pinned checkpoints llama.bend already has (SHA-256 checked) into `init.bin` in dump order
  (`reference/smol_layout.txt`, `smol360_layout.txt`) and makes 64 batches of 2 x 129 ids from Tiny Shakespeare.
- The first loss was 3.4 tolerances off PyTorch's. Cross-entropy summed the 49,152 exponentials of each row as a running
  F32 sum; it now sums them as a balanced tree (`ce_sm`, over `tree_sum`). The loss is 0.41 of the tolerance off at 135M
  and 0.47 at 360M, and closer to float64 than PyTorch's float32 loss. The tiny Llama's bits moved with it (its rows of
  65 classes are summed in another order; `bench/bits.txt` saved again, the same on 1 and 16 threads); MNIST's did not.
  Attention's softmax (`psm`) keeps its order: its rows are short and the law `causal` reads its count.
- Gradients: the shared tolerance cannot hold here, for PyTorch either. Against float64 gradients of the same model and
  batch, 17 of the 135M's 272 tensors from PyTorch float32 are within it (worst element 109x) and 1 from Bend (300x);
  Bend's relative error is a median 1.8x PyTorch's (0.59-3.54x per tensor). At 360M: 83 and 63 of 290 tensors, worst
  26x and 91x, Bend 1.42x PyTorch's error in the median. I did not widen the tolerance. The gradient criterion is now
  relative (my decision, 2026-10-17): per tensor, Bend's rms error against float64 at most 8x PyTorch float32's, and at
  most 2x in the median; `test_smol.py` computes the float64 gradients with the twin (`grads64`) and still prints the
  element-wise comparison for information. What I ruled out as the cause: the head's backward product (simulated, 0.58
  of the tolerance even as one running sum of 49,152 terms), RMSNorm's two row sums (as trees, no change; reverted), and
  fused multiply-add. Bend's build has none (clang gets no `-march`, and the emitted C puns each F32 through a word
  between the multiply and the add, so even `-march=native` fuses nothing); built with `-march=native
  -ffp-contract=fast`, the hot product loop fuses and the gradients' error is unchanged. The missing FMA stays a
  language item (an upstream draft is in bend-ml's `docs/upstream/fma-codegen.md`); the cause of the 1.8x is not found.
- Training, 3 AdamW steps (lr 1e-4): at 360M every loss is within the tolerance of PyTorch's (0.47, 0.58, 0.16). At 135M
  the third is 1.38 of it off PyTorch and 1.84 off float64 (PyTorch float32: 0.46): AdamW's first update moves each
  weight by about lr times the sign of its gradient, so where a gradient is near zero its rounding flips the step.
- Peak memory (16 threads, `/usr/bin/time`): 135M loss 2.4 GiB, 3 AdamW steps 5.1 GiB; 360M loss 5.5 GiB, gradients
  10.0 GiB, 3 AdamW steps 12.8 GiB (repeated runs vary by a few percent). The arrays' power-of-two capacity is part of it (above).
- `Layer` has a `store` field, `load`'s inverse: the parameters copied into an array at an offset. `Params.save` writes
  a model's parameters and `Train.save` a training state from one array, a chunk of 65,536 numbers per byte list
  instead of one list for the whole file (several GB at these sizes). `Train.save` writes the same bytes as before
  (`test_infra.py` checks them for every optimizer); `infra_test params` checks `Params.save`. `demo/smol.bend grads`
  writes all 134,515,008 gradients this way. A 360M AdamW checkpoint (1.09 G numbers) is over the binary reader's
  U32 byte offsets, so it cannot be read back yet.
- Step time, idle machine (2 steps x 3 alternated rounds, one thread pinned to CPU 3, sixteen unpinned; each program's
  own clock after loading; Bend 2.0.37, clang 22.1.8, PyTorch 2.14.1): 135M 17.4 s per AdamW step on 1 thread and
  5.3 s on 16 (PyTorch 2.5 s and 1.1 s); 360M 42.7 s and 11.7 s (PyTorch 6.8 s and 2.4 s). Bend is 5.0-7.0x slower per
  step; 16 threads are 3.3-3.6x faster than one in Bend, 2.3-2.9x in PyTorch. I have not profiled where the step goes.
- Checks: `make check`, `make bits` (saved again, above), `make check-js`, `make mutate` (60 of 60, three new: `store`
  one cell short, Adam's powers swapped in the array checkpoint, cross-entropy's tree sum one class short) and
  `make smol` (the loss and the 1/16-thread bits pass; the gradients fail as described). Between the `Layer.store`
  commit and the one after it that gives `Box` its store, `demo/smol.bend` does not compile.

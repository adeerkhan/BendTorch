# Notes

Every finding, measurement and decision, by date. Newest at the bottom.

## Pinned versions

- Bend **2.0.37** (`bend version`), Lean **4.34.0** for `--verdict`.
- `bend-ml-tensor-array@0.1.10.0`, `bend-ml-nat-lemmas@0.1.1.0` (BendHub).
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
  `n` steps. bend-ml's next version (0.1.10.0, not published yet) computes it by doubling; with it the same run takes
  4.3 s and the whole `make check-js` 65 s, against 34 minutes now.

## 2026-10-17: accumulating weight gradients (bend-ml-tensor-array 0.1.10.0, not published yet)

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

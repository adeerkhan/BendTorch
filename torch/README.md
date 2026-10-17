# torch: BendTorch's package

Layers with typed forward and backward passes, composed generically and trained in Bend 2. A layer is a record of four
functions over its parameters `P`, its input `X`, its output `Y` and what its forward pass saves, `S`:

```python
type Layer<-P: Type, -X: Type, -Y: Type, -S: Type> is Type:
  Layer{fwd: P -> X -> Fwd<P, Y, S>, bwd: P -> S -> Y -> P -> Bwd<P, X>, step: Opt -> P -> P -> Two<P, P> -> Step<P>,
        zeros: Unit -> P, dump: P -> PL<P>, load: P -> List<&2, F32> -> PL<P>}
```

`zeros` makes the gradient and the optimizer state; `dump` and `load` turn the parameters into a flat list of numbers
and back, which is how `Train.save` and `Train.load` write checkpoints for any model.

The backward pass takes the gradient of the output, typed `Y`, and returns the gradient of the input, typed `X`, while
it adds the gradient of the parameters into a gradient of type `P`. So **a gradient always has the type of the value it
is the gradient of**: a backward pass that returns a transposed weight gradient, or composes two layers whose shapes do
not meet, does not compile ([`tests/`](../tests/)). Layers are passed as templates (`~l`), which Bend copies at compile
time, so `Seq`, `Fork`, `Both` and `Residual` compose any layers whose types meet, with no closure called twice and no
cost at run time.

- Bend: **2.0.37** · License: MIT · Depends on `bend-ml-tensor-array@0.1.12.0` (the matrices and their products) and
  `bend-ml-nat-lemmas@0.1.1.0`.
- `bend PROOF.bend --verdict` (from the repository's root) → `ALL PROOFS CHECK`, with no `@unsafe`, no `?TODO` and no
  axioms.
- Not on BendHub yet. Programs import the local copy: `import ../torch/main.bend as T`, as those in `demo/` do, and
  `import ../torch/tensor.bend as TT` for the general tensors (below).

## What it has

| | |
|---|---|
| Layers with parameters | `Linear` (with bias, PyTorch's `out x in` layout), `LinearFirst` (the same, for a model's first layer: it skips the input gradient nobody reads), `LinearNB` (no bias), `Embedding`, `RMSNorm`, `LayerNorm` |
| Layers without | `ReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GELU` (tanh form), `Softmax`, `LogSoftmax`, `RoPE` (rotate-half), `Attention` (causal, grouped-query heads; the sequences of a batch in parallel), `Dropout` (a reproducible counter-based mask), `Add2`, `Mul2`, `Id` |
| Combinators | `Seq` (one after the other), `Fork` (one input to two layers, the input gradients summed), `Both` (a pair of inputs to two layers), `Residual` (`x + f(x)`), `Tie` (first, middle, last, where the first and the last share one parameter: tied embeddings) |
| Losses | `CrossEntropy` (mean, from logits), `NLL` (mean, from log-probabilities), `MSE` (mean) |
| Optimizers | `SGD`, SGD with momentum, `Adam` (with or without weight decay), `AdamW`; one pass per parameter matrix, in PyTorch's order of operations, the gradient zeroed on the way |
| Training | `Train.step`: forward, loss, backward and one optimizer step for any layer under any loss; `Train.save` and `Train.load`: checkpoints of the parameters and the optimizer state (a resumed run gives the same bits) |
| General tensors (`tensor.bend`) | `Tensor<s>`, the shape `s` a list of sizes in the type, any rank (tested 0 to 4); `add`, `sub`, `mul`, `div` with PyTorch's broadcasting and their gradients (`grad2`, `sum_to`); `sum_axis`, `sum_axis_keep`, `mean_axis`, `sum_all`, `mean_all`; `reshape` (with a proof that the count does not change), `flatten`; `permute`, `transpose`; `narrow` (with a proof that the window fits); `cat`; `bmm` (`[b, n, k] · [b, k, m]`), each with its backward function. `of_mat` and `to_mat` turn a `Mat<r, c>` into a `Tensor<[r, c]>` and back without a copy |

A 784-128-10 MLP on MNIST and a two-layer Llama (grouped-query attention, RoPE, SwiGLU, RMSNorm) are built from these
pieces in [`demo/`](../demo/) and trained against PyTorch twins; see the repository's README.

## Fast layers (`fast.bend`)

[`fast.bend`](fast.bend) has `LinearF`, `LinearFirstF` and `LinearNBF`: `Linear`, `LinearFirst` and `LinearNB` with
their three products taken from `bend-ml-tensor-fork` instead of `bend-ml-tensor-array`, and their optimizer step in
parallel blocks; and `SiLU`, `ActIn`, `Mul2`, `Add2`, `Residual`, `Mat.dup` and `Mat.sum2` with their element-wise loops
in parallel blocks (tensor-fork's `par1`..`par4`). The types and the bits are `main.bend`'s: every number is computed by
`main.bend`'s own loop or tensor-array's own cells, only split into blocks that share their arrays instead of copying
them. Small matrices (below 131072 numbers, MNIST's and the tiny Llama's) run in one block. The demos use them.

```python
import ../torch/main.bend as T
import ../torch/fast.bend as F
F.LinearFirstF(100n, 784n, 128n, 4n)   # instead of T.LinearFirst(100n, 784n, 128n, 4n)
```

They are outside the proofs, and say so. bend-ml-tensor-fork shares arrays through Base's `Array.fork`, which Base marks
`@unsafe` (two handles to one array break the single owner the proofs rely on), so `bend fast.bend` names the defs
that reach it. No file with laws imports `fast.bend`, and `make laws` checks both that and the list of defs. What
connects them to the proved layers is a test, not a proof: [`demo/fast_test.bend`](../demo/fast_test.bend) runs each
fast piece and its proved twin on the same inputs (with -0, infinities, NaN and a subnormal mixed in), at MNIST's, the
tiny Llama's and larger sizes: the linear layers forward and backward, the optimizer step under SGD, momentum and AdamW,
SiLU, Mul2 and a Residual. Every output and gradient has the same bits, on 1 and 16 threads and in JavaScript. Five
mutants of the fast code are caught by it (`make mutate`).

Measured (NOTES, 2026-10-17): MNIST's epoch 3.6 → 2.9 s on one thread with the fast products; a training step of one
decoder block at SmolLM2-135M's sizes 247 → 155 ms on 16 threads with everything above, one thread unchanged.

## How to write a model

A model is a layer. Its types are written once, in small defs, and the layers are composed with templates:

```python
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.12.0/main.bend as TA

# Linear(784, 128) -> ReLU on batches of 100 rows
def Hidden() -> T.Layer<T.Two<T.Lin<784n, 128n>, Unit>, TA.Mat<100n, 784n>, TA.Mat<100n, 128n>, T.Two<TA.Mat<100n, 784n>, TA.Mat<100n, 128n>>>:
  T.Seq(~T.Lin<784n, 128n>, ~Unit, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 128n>, ~T.Linear(100n, 784n, 128n, 3n), ~T.ReLU(100n, 128n))
```

`Seq` takes its seven types and its two layers. The types are explicit because Bend does almost no inference; they are
also the documentation of the model: `T.Two<T.Lin<784n, 128n>, Unit>` is the parameters of a linear layer next to a
ReLU's (none). Training is `T.Train.step(~P, ~X, ~Y, ~S, ~T, ~Model(), ~loss, cfg, state, x, targets)`.

## The laws, in plain language

The laws are in [`LAWS.bend`](../LAWS.bend) at the repository's root, the proofs in [`PROOF.bend`](../PROOF.bend). The
five about linear maps (`seq_chain` and `fork_sum` through the package's own combinators) speak about
[`spec.bend`](spec.bend), a model of vectors and matrices as lists of `Nat`, where arithmetic is exact. They say that the formulas of the backward pass are the right ones; whether the F32 code computes
them to F32's precision is tested against PyTorch, not proved.

| Law | What it says |
|---|---|
| `adjoint` | A linear layer computes `y = W·x` forward and `dx = Wᵀ·dy` backward. For every matrix `W` and vectors `dy` and `x`, `dy·(W·x) = (Wᵀ·dy)·x`: the backward pass is the *adjoint* of the forward pass. That equation is what makes `dx` the gradient of the loss with respect to `x` (if the loss changes by `dy·δy` when `y` moves by `δy`, it changes by `dx·δx` when `x` moves by `δx`). The proof goes row by row: each row `r` of `W` with its weight `dy[i]` gives `dy[i]·(r·x)` on one side and `(dy[i]·r)·x` on the other, and the sum of the weighted rows splits into the sum of their products with `x`. |
| `weight_adjoint` | The same for the weights: the weight gradient `dW = dy ⊗ x` (row `i` is `dy[i]·x`) satisfies `dy·(W·x) = <dW, W>`, the sum of the products of matching entries. So `dW` is the gradient with respect to `W`. |
| `adjoint_seq` | `Seq` runs the second layer's backward pass, then the first's. For two linear layers that is `W1ᵀ·(W2ᵀ·dy)`, and the law says it is the adjoint of the composed forward pass `W2·(W1·x)`: the chain rule, for linear maps. It follows from `adjoint` used twice. |
| `causal` | In `Attention`, row `i` of a head's probabilities is a softmax over exactly `i + 1` positions. `prow`, the function the attention runs for each row, writes the scores of query `i` from the start of its row and runs the softmax over as many positions as it wrote; the softmax's division loop counts the positions it divides, and the law says that count is `i + 1`. The probabilities start as zeros and nothing else writes row `i`, so its other `t - (i + 1)` positions stay 0: no probability lands on a later position. That last step is read from the code, not proved (the checker cannot follow writes into an array by index). The proof counts: each score written adds one, `qrow` gives `i + 1` scores, and each step of the division loop adds one (`cnt_div`); the loops before it (the maximum and the exponentials) are matched, not computed, so the count does not depend on any number. A softmax that ran over one position too few is now caught by this law (`make mutate`). |
| `causal_grad` | The same for the backward pass. Row `i` of a head's `G` (the gradient of its scores) is computed over exactly `i + 1` positions: `grow` reads the row of `dP` against values `0 .. i` only, writes it from the row's start and runs the softmax's backward pass over as many positions as it wrote. `G` starts as zeros, so its positions after `i` stay 0, and no gradient reaches a key or a value after the query's position. Before this law the backward pass computed the whole row and relied on the probabilities being 0 there; now it computes half the row and the restriction is a law (the same bits, and the tiny Llama about 2% faster). |
| `causal_keys` | `Attention` reads keys in one place, `qrow`, and the scores it builds for the query at position `i` come from keys `0 .. i`, one each: exactly `i + 1` of them. So no key after position `i` is even read. The proof follows the loop that reads them: each step reads one key and adds one score, so a loop started at `i` gives `i + 1` scores (an induction, with `1 + (a + b) = a + (1 + b)`). A version that read one key too many is caught by this law, by `causal` (the extra score would reach the softmax) and by the numbers (`make mutate`). |
| `kv_of_group` | `kv_of(hi, g)` is the one function both passes of `Attention` call to pick the key/value head of query head `hi`. For every group `j`, the heads `j·g + 0 .. j·g + g - 1` all get `j`. It is `gqa_head` (below) said about the package's own function, so a change to how the code picks the head breaks a law, not only a test. |
| `kv_cols` | The columns each query head reads come from one function, `head_cols`, in both passes: the head loops give the head's work only the columns, never the head's index, so it cannot read other columns. With `g = 1 + i + k` query heads per key/value head, query head `j·g + i` reads the keys and values of head `j`, from column `j·hd`. The proof is `kv_of_group` carried through the column arithmetic. A change that gives a head another key/value head's columns is caught by this law. |
| `attn_par_fwd` | `Attention` runs its sequences in a tree of parallel lets: `fpar` cuts the sequences in halves, each half again, `d` levels deep, and appends the leaves' outputs. Whatever the depth and however the sequences are split, the tree gives exactly what `fleaf` gives running them one after another, in order. So the parallel forward pass is the sequential one, and its output cannot depend on `d` or on the number of threads. The proof is an induction on the depth: a pair of leaves is the loop twice; the loop is "apply the head work to each sequence"; the map of two lists joined is the two maps joined; and `cut`'s halves joined again are the list it cut. The lists hold arrays, which Bend lets a proof use only once, so each fact hands its list back to the next one. |
| `attn_par_bwd` | The same for the backward pass: `bpar`'s tree gives each sequence's `dQ`, `dK` and `dV` exactly as `bleaf`, one sequence after another. |
| `seq_chain` | `Seq`'s own code, run on two linear layers of the exact model (`spec.bend`'s `LinL`, a `Layer` like any other), gives a backward pass that is the adjoint of its forward pass: `dy·y = dx·x`, for the `y` its forward pass returns and the `dx` its backward pass returns from the state the forward pass saved. `adjoint_seq` says this of the formulas; this law says it of the combinator: `Seq` hands the second layer's input gradient to the first, the chain rule. Every step of the combinator reduces, so the proof is `adjoint_seq`. |
| `fork_sum` | `Fork`'s own code, giving one input to two linear layers, returns as input gradient the sum of the two layers' input gradients, and that sum is the adjoint of the pair of outputs: `dy1·y1 + dy2·y2 = dx·x`. A residual block and attention's q, k, v projections are `Fork`s. The proof: the dot of a sum is the sum of the dots, and each is its layer's `adjoint`. A `Fork` that dropped one gradient is caught by it. |
| `bc_sym` | Broadcasting does not depend on the order of the operands: shapes `a` and `b` broadcast to `o` exactly when `b` and `a` do, so `x + y` and `y + x` accept the same shapes. The rule on one axis ("the sizes are equal, or one of them is 1") is written as "one way round, or the other", so swapping the operands only swaps the two halves of an "or"; the proof goes axis by axis. |
| `sum_numel` | Summing over axis `k` partitions the numbers: a tensor of shape `s` holds exactly `size(k)` times as many numbers as its sum over `k`, for every shape and every axis. The proof goes axis by axis and moves `size(k)` to the front with the rules `a·(b·c) = (a·b)·c` and `a·b = b·a`. |
| `narrow_rest` | Narrowing axis `d` (or giving it any size) changes only that axis: with axis `d` taken out, the shape is what it was. |
| `narrow_axis` | Narrowing axis `d` of a tensor that has it gives that axis exactly the length asked for. |
| `cat_numel` | Joining two tensors along an axis they have adds their counts: an axis of `x + y` holds as many numbers as one of `x` and one of `y` together. The proof is `(x + y)·n = x·n + y·n`, carried past the axes before it. |
| `mat_numel` | A `Tensor<[r, c]>` holds the `r·c` numbers of a `Mat<r, c>`, so `of_mat` and `to_mat` can reuse the array as it is. |
| `permute_rank` | Permuting the axes gives one axis per entry of the permutation, so a permutation keeps the rank. |
| `gqa_head` | With `g` query heads per key/value head, query head `j·g + i` (`i < g`) reads key/value head `(j·g + i) / g = j`. The forward pass picks key/value head `hi / g` for query head `hi`; the backward pass adds the gradients of heads `j·g + 0 .. j·g + g - 1` into key/value head `j`. The law says these are the same groups. The proof is llama.bend's: `j·g` is `j` whole blocks of `g`, and `i < g` more numbers do not complete another one. |

`add4`, `mul_dist_l`, `dot_vadd`, `dot_scale`, `dot_comm`, `len_rcat`, `len_drow`, `sub0`, `cnt_div`, `cnt_psm3`,
`cnt_psm2`, `cnt_pw`, `cnt_drow`, `cnt_gw`, `cnt_gdrow`, `div_block`, `div_blocks`, `div_rest`, `or_comm`, `and_comm`,
`and_cong`, `or_cong`, `bc_dim_sym`, `bc_eq_sym`, `bc_rank_sym` and `mul_swap` are the lemmas the proofs use; `l1`,
`map_app`, `go_app`, `cg`, `tree` and their `b_` twins are the trees' (each returns the list it was given, with the
fact).

The tensor operations' shape conditions are `Bool` functions of the shapes, and each operation asks for the proof that
its condition is `True`: `bc_ok` for broadcasting, `narrow_ok`, `cat_ok`, `is_perm`, and for `reshape` the equality of
the two counts. For literal shapes the checker computes the condition, so a wrong broadcast, window or join does not
compile ([`tests/bad_broadcast.bend`](../tests/bad_broadcast.bend), [`bad_narrow`](../tests/bad_narrow.bend),
[`bad_cat`](../tests/bad_cat.bend); [`bad_bmm`](../tests/bad_bmm.bend) for the product, whose sizes are in its type).
For shapes read at run time, a program checks the condition once where they enter (`Bool.check`, `Nat.check_eq`), as
[`demo/tensor_test.bend`](../demo/tensor_test.bend) does.

Shape facts that a layer needs are arguments of its constructor, checked when the program is: `Attention` asks for proofs
that the rows hold `kv·g·head` numbers and that the rows are `b·t` positions, `RoPE` that a row is `h` heads of
`2·half`. For literal sizes the proof is `{==}`; a wrong size fails it ([`tests/bad_attention.bend`](../tests/bad_attention.bend)).

What is **not** proved:

- **The numbers.** `F32` is not a real number. Every layer is tested against `torch.autograd` and every optimizer against
  `torch.optim`, by the scripts in [`reference/`](../reference/).
- **That the F32 code implements the Nat model.** `adjoint` is about `spec.bend`'s `mv` and `mtv`; the package's `Linear`
  uses bend-ml's array products, which compute the same sums in another order. The connection is tested, not proved.
- **The gradients of the non-linear layers** (softmax, the norms, the activations, attention): their formulas are
  checked by the tests, not by a law.
- **The parallel trees that share arrays**: bend-ml-tensor-fork's products and blocks and `fast.bend`'s layers. In
  Base's model `Array.fork` returns two copies and `Array.join` keeps the first, so a law about the shared tree cannot be
  stated; they are tested bit for bit against the proved code. The same holds for tensor-array's parallel product: each
  block reads its own packed copy of its rows, and a law would need the contents of arrays written by index, which the
  checker cannot follow. Attention's trees hold no shared arrays, and those are laws (`attn_par_fwd`, `attn_par_bwd`).
- **The tensor operations' numbers and index maps**: every operation and its gradient is tested against
  `torch.autograd` on random shapes (`reference/test_tensor.py`), not proved; that a general permutation keeps the
  count of numbers is tested too (the law covers the rank).
- **The compiler and the runtime.** The kernel checks the translated Bend terms; it does not check Bend's translation to
  C, clang or the hardware.

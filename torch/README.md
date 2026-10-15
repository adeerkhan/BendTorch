# torch: BendTorch's package

Layers with typed forward and backward passes, composed generically and trained in Bend 2. A layer is a record of four
functions over its parameters `P`, its input `X`, its output `Y` and what its forward pass saves, `S`:

```python
type Layer<-P: Type, -X: Type, -Y: Type, -S: Type> is Type:
  Layer{fwd: P -> X -> Fwd<P, Y, S>, bwd: P -> S -> Y -> P -> Bwd<P, X>, step: Opt -> P -> P -> Two<P, P> -> Step<P>, zeros: Unit -> P}
```

The backward pass takes the gradient of the output, typed `Y`, and returns the gradient of the input, typed `X`, while
it adds the gradient of the parameters into a gradient of type `P`. So **a gradient always has the type of the value it
is the gradient of**: a backward pass that returns a transposed weight gradient, or composes two layers whose shapes do
not meet, does not compile ([`tests/`](../tests/)). Layers are passed as templates (`~l`), which Bend copies at compile
time, so `Seq`, `Fork`, `Both` and `Residual` compose any layers whose types meet, with no closure called twice and no
cost at run time.

- Bend: **2.0.37** · License: MIT · Depends on `bend-ml-tensor-array@0.1.7.0` (the matrices and their products) and
  `bend-ml-nat-lemmas@0.1.1.0`.
- `bend PROOF.bend --verdict` (from the repository's root) → `ALL PROOFS CHECK`, with no `@unsafe`, no `?TODO` and no
  axioms.
- Not on BendHub yet. Programs import the local copy: `import ../torch/main.bend as T`, as those in `demo/` do.

## What it has

| | |
|---|---|
| Layers with parameters | `Linear` (with bias, PyTorch's `out x in` layout), `LinearNB` (no bias), `Embedding`, `RMSNorm`, `LayerNorm` |
| Layers without | `ReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GELU` (tanh form), `Softmax`, `LogSoftmax`, `RoPE` (rotate-half), `Attention` (causal, grouped-query heads), `Add2`, `Mul2`, `Id` |
| Combinators | `Seq` (one after the other), `Fork` (one input to two layers, the input gradients summed), `Both` (a pair of inputs to two layers), `Residual` (`x + f(x)`) |
| Losses | `CrossEntropy` (mean, from logits), `NLL` (mean, from log-probabilities), `MSE` (mean) |
| Optimizers | `SGD`, SGD with momentum, `Adam` (with or without weight decay), `AdamW`; one pass per parameter matrix, in PyTorch's order of operations, the gradient zeroed on the way |
| Training | `Train.step`: forward, loss, backward and one optimizer step for any layer under any loss |

A 784-128-10 MLP on MNIST and a two-layer Llama (grouped-query attention, RoPE, SwiGLU, RMSNorm) are built from these
pieces in [`demo/`](../demo/) and trained against PyTorch twins; see the repository's README.

## How to write a model

A model is a layer. Its types are written once, in small defs, and the layers are composed with templates:

```python
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.7.0/main.bend as TA

# Linear(784, 128) -> ReLU on batches of 100 rows
def Hidden() -> T.Layer<T.Two<T.Lin<784n, 128n>, Unit>, TA.Mat<100n, 784n>, TA.Mat<100n, 128n>, T.Two<TA.Mat<100n, 784n>, TA.Mat<100n, 128n>>>:
  T.Seq(~T.Lin<784n, 128n>, ~Unit, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 128n>, ~T.Linear(100n, 784n, 128n, 3n), ~T.ReLU(100n, 128n))
```

`Seq` takes its seven types and its two layers. The types are explicit because Bend does almost no inference; they are
also the documentation of the model: `T.Two<T.Lin<784n, 128n>, Unit>` is the parameters of a linear layer next to a
ReLU's (none). Training is `T.Train.step(~P, ~X, ~Y, ~S, ~T, ~Model(), ~loss, cfg, state, x, targets)`.

## The laws, in plain language

The laws are in [`LAWS.bend`](../LAWS.bend) at the repository's root, the proofs in [`PROOF.bend`](../PROOF.bend). The
three about linear maps speak about [`spec.bend`](spec.bend), a model of vectors and matrices as lists of `Nat`, where
arithmetic is exact. They say that the formulas of the backward pass are the right ones; whether the F32 code computes
them to F32's precision is tested against PyTorch, not proved.

| Law | What it says |
|---|---|
| `adjoint` | A linear layer computes `y = W·x` forward and `dx = Wᵀ·dy` backward. For every matrix `W` and vectors `dy` and `x`, `dy·(W·x) = (Wᵀ·dy)·x`: the backward pass is the *adjoint* of the forward pass. That equation is what makes `dx` the gradient of the loss with respect to `x` (if the loss changes by `dy·δy` when `y` moves by `δy`, it changes by `dx·δx` when `x` moves by `δx`). The proof goes row by row: each row `r` of `W` with its weight `dy[i]` gives `dy[i]·(r·x)` on one side and `(dy[i]·r)·x` on the other, and the sum of the weighted rows splits into the sum of their products with `x`. |
| `weight_adjoint` | The same for the weights: the weight gradient `dW = dy ⊗ x` (row `i` is `dy[i]·x`) satisfies `dy·(W·x) = <dW, W>`, the sum of the products of matching entries. So `dW` is the gradient with respect to `W`. |
| `adjoint_seq` | `Seq` runs the second layer's backward pass, then the first's. For two linear layers that is `W1ᵀ·(W2ᵀ·dy)`, and the law says it is the adjoint of the composed forward pass `W2·(W1·x)`: the chain rule, for linear maps. It follows from `adjoint` used twice. |
| `causal` | In `Attention`, the query at position `i` of a sequence of `t` positions puts probability exactly 0 on every later position: past its first `i + 1` entries, its row of probabilities is `t - (i + 1)` zeros. This one speaks about the package's own `causal_row`, the function the attention runs. The proof counts: softmax keeps the length of its input, `i + 1` scores go in, so `i + 1` probabilities come out, and dropping them leaves the zeros. |
| `kv_of_group` | `kv_of(hi, g)` is the one function both passes of `Attention` call to pick the key/value head of query head `hi`. For every group `j`, the heads `j·g + 0 .. j·g + g - 1` all get `j`. It is `gqa_head` (below) said about the package's own function, so a change to how the code picks the head breaks a law, not only a test. |
| `gqa_head` | With `g` query heads per key/value head, query head `j·g + i` (`i < g`) reads key/value head `(j·g + i) / g = j`. The forward pass picks key/value head `hi / g` for query head `hi`; the backward pass adds the gradients of heads `j·g + 0 .. j·g + g - 1` into key/value head `j`. The law says these are the same groups. The proof is llama.bend's: `j·g` is `j` whole blocks of `g`, and `i < g` more numbers do not complete another one. |

`add4`, `mul_dist_l`, `dot_vadd`, `dot_scale`, `dot_comm`, `len_softmax`, `len_take`, `drop_app`, `div_block`,
`div_blocks` and `div_rest` are the lemmas the proofs use.

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
- **The compiler and the runtime.** The kernel checks the translated Bend terms; it does not check Bend's translation to
  C, clang or the hardware.

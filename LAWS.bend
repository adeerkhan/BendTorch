# BendTorch's laws. Each is a claim about every input, not a test case; PROOF.bend proves them, and
# `bend PROOF.bend --verdict` checks the proofs again with Bend's kernel proved in Lean. Plain-language explanations
# are in torch/README.md. The laws about linear maps speak about the exact model in torch/spec.bend; the others speak
# about the package's own functions.

import Base
import ./torch/main.bend as T
import ./torch/spec.bend as Spec

# LAW (adjoint): the backward pass of a linear layer, dx = Wᵀ·dy, is the adjoint of its forward pass y = W·x: for any
# W, dy and x, dy·(W·x) == (Wᵀ·dy)·x. This is what makes dx the gradient of the loss with respect to x.
law adjoint:
  for w: List<&2, List<&2, Nat>>
  for g: List<&2, Nat>
  for +v: List<&2, Nat>
  {Spec.dotn(g, Spec.mv(w, v)) == Spec.dotn(Spec.mtv(w, g), v) : Nat}

# LAW (weight_adjoint): the weight gradient dW = dy ⊗ x is the adjoint in W: dy·(W·x) == <dy ⊗ x, W>.
law weight_adjoint:
  for w: List<&2, List<&2, Nat>>
  for g: List<&2, Nat>
  for +v: List<&2, Nat>
  {Spec.dotn(g, Spec.mv(w, v)) == Spec.frob(Spec.outer(g, v), w) : Nat}

# LAW (adjoint_seq): Seq runs the second layer's backward pass, then the first's. For two linear layers that is
# W1ᵀ·(W2ᵀ·dy), and it is the adjoint of the composed forward pass W2·(W1·x): the chain rule, for linear maps.
law adjoint_seq:
  for +w1: List<&2, List<&2, Nat>>
  for +w2: List<&2, List<&2, Nat>>
  for +g: List<&2, Nat>
  for +v: List<&2, Nat>
  {Spec.dotn(g, Spec.mv(w2, Spec.mv(w1, v))) == Spec.dotn(Spec.mtv(w1, Spec.mtv(w2, g)), v) : Nat}

# LAW (causal): in T.Attention, the query at position i of a sequence of t gives probability exactly 0 to every
# position after i: past its first i + 1 entries, its row of probabilities is t - (i + 1) zeros.
law causal:
  for +r: List<&2, F32>
  for +i: Nat
  for +t: Nat
  for e: {Spec.le(1n+i, List.length(&2, F32, r)) == True{} : Bool}
  {List.drop(&2, F32, T.causal_row(r, i, t), 1n+i) == T.l_zeros(Nat.sub(t, 1n+i)) : List<&2, F32>}

# LAW (gqa_head): with g = 1 + i + k query heads per key/value head, query head j·g + i reads key/value head
# (j·g + i) / g = j, the head of its own group: T.Attention's forward pass picks head hi / g, and its backward pass
# sums the gradients of heads j·g + 0 .. j·g + g - 1 into key/value head j, so both see the same groups.
law gqa_head:
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {Nat.div(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)) == j : Nat}

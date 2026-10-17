# BendTorch's laws. Each is a claim about every input, not a test case; PROOF.bend proves them, and
# `bend PROOF.bend --verdict` checks the proofs again with Bend's kernel proved in Lean. Plain-language explanations
# are in torch/README.md. The laws about linear maps speak about the exact model in torch/spec.bend; the others speak
# about the package's own functions.

import Base
import ./torch/main.bend as T
import ./torch/spec.bend as Spec
import ./torch/tensor.bend as TT

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

# LAW (causal_keys): T.Attention reads keys only through T.qrow, and the scores it builds for the query at position i
# come from keys 0..i, one per key: exactly i + 1 of them. So no key after position i is read, and T.causal_row (law
# causal) gets the i + 1 scores its condition asks for: past them, the row of probabilities is zeros.
law causal_keys:
  for +i: Nat
  for +hd: Nat
  for +qs: U32
  for +qc: U32
  for +ks: U32
  for +kc: U32
  for q: Array<F32>
  for k: Array<F32>
  {List.length(&2, F32, T.dr_row(T.qrow(i, hd, qs, qc, ks, kc, q, k))) == 1n+i : Nat}

# LAW (causal): in T.Attention, row i of a head's probabilities is a softmax over exactly i + 1 positions. T.prow
# writes the query's scores from the start of its row and runs the softmax over as many positions as it wrote; the count
# is the one the softmax's own division loop reports (T.psmc), and this law says it is i + 1, the keys 0..i of
# causal_keys. P starts as zeros and nothing else writes the row, so its other t - (i + 1) positions stay 0: no
# probability lands on a later position.
law causal:
  for +i: Nat
  for +t: Nat
  for +hd: Nat
  for +sc: F32
  for +qs: U32
  for +qc: U32
  for +ks: U32
  for +kc: U32
  for q: Array<F32>
  for k: Array<F32>
  for p: Array<F32>
  {T.pr_n(T.prow(i, t, hd, sc, qs, qc, ks, kc, q, k, p)) == 1n+i : Nat}

# LAW (kv_of_group): T.kv_of, the function both passes of T.Attention use to pick the key/value head of query head hi,
# gives j for every head of group j: query heads j·g + 0 .. j·g + g - 1 (g = 1 + i + k, i < g) all read key/value head j.
law kv_of_group:
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {T.kv_of(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)) == j : Nat}

# LAW (gqa_head): with g = 1 + i + k query heads per key/value head, query head j·g + i reads key/value head
# (j·g + i) / g = j, the head of its own group: T.Attention's forward pass picks head hi / g, and its backward pass
# sums the gradients of heads j·g + 0 .. j·g + g - 1 into key/value head j, so both see the same groups.
law gqa_head:
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {Nat.div(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)) == j : Nat}

# LAW (bc_sym): broadcasting does not depend on the order of the operands: a and b broadcast to o exactly when b and a
# do, so x + y and y + x (and x * y, y * x) accept the same shapes.
law bc_sym:
  for +a: List<&2, Nat>
  for +b: List<&2, Nat>
  for +o: List<&2, Nat>
  {TT.bc_ok(a, b, o) == TT.bc_ok(b, a, o) : Bool}

# LAW (sum_numel): summing over axis k partitions the elements: a tensor of shape s has exactly size(k) times as many
# numbers as its sum over axis k, for every shape and every k (past the last axis the size counts as 1).
law sum_numel:
  for +s: List<&2, Nat>
  for +k: Nat
  {TT.numel(s) == Nat.mul(TT.at1(s, k), TT.numel(TT.remove(s, k))) : Nat}

# LAW (narrow_rest): narrow (and setting any axis' size) changes only its own axis: with axis d taken out, the shape is
# what it was.
law narrow_rest:
  for +s: List<&2, Nat>
  for +d: Nat
  for +l: Nat
  {TT.remove(TT.setd(s, d, l), d) == TT.remove(s, d) : List<&2, Nat>}

# LAW (narrow_axis): narrowing axis d of a tensor that has it gives that axis exactly the length asked for.
law narrow_axis:
  for +s: List<&2, Nat>
  for +d: Nat
  for +l: Nat
  for e: {True{} == Nat.is_lt(d, TT.rank(s)) : Bool}
  {TT.at1(TT.setd(s, d, l), d) == l : Nat}

# LAW (cat_numel): concatenating along an axis the tensor has adds the counts: a tensor whose axis d is x + y long holds
# as many numbers as the two with x and y.
law cat_numel:
  for +s: List<&2, Nat>
  for +d: Nat
  for +x: Nat
  for +y: Nat
  for e: {True{} == Nat.is_lt(d, TT.rank(s)) : Bool}
  {TT.numel(TT.setd(s, d, Nat.add(x, y))) == Nat.add(TT.numel(TT.setd(s, d, x)), TT.numel(TT.setd(s, d, y))) : Nat}

# LAW (mat_numel): a Tensor<[r, c]> holds the r·c numbers of a Mat<r, c>, so of_mat and to_mat reuse the array as it is.
law mat_numel:
  for +r: Nat
  for +c: Nat
  {TT.numel([r, c]) == Nat.mul(r, c) : Nat}

# LAW (permute_rank): permuting by p gives as many axes as p has, so a permutation of the axes keeps the rank.
law permute_rank:
  for +s: List<&2, Nat>
  for p: List<&2, Nat>
  {TT.rank(TT.permuted(s, p)) == List.length(&2, Nat, p) : Nat}

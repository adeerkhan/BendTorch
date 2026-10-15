# The proofs of LAWS.bend. Every proof is plain Bend: no @unsafe, no ?TODO, no axioms. The arithmetic lemmas come
# from bend-ml-nat-lemmas; the division lemmas behind gqa_head are llama.bend's (MIT, same author).

import Base
import ./torch/main.bend as T
import ./torch/spec.bend as Spec
import ./LAWS.bend as Laws
import bend-ml-nat-lemmas@0.1.1.0/main.bend as NL

# =====================================================================================================
# Arithmetic
# =====================================================================================================

# (a + b) + (c + d) == (a + c) + (b + d)
def add4(+a: Nat, +b: Nat, +c: Nat, +d: Nat) -> {Nat.add(Nat.add(a, b), Nat.add(c, d)) == Nat.add(Nat.add(a, c), Nat.add(b, d)) : Nat}:
  Equal.trans(Nat, Nat.add(Nat.add(a, b), Nat.add(c, d)), Nat.add(a, Nat.add(b, Nat.add(c, d))), Nat.add(Nat.add(a, c), Nat.add(b, d)),
    Equal.sym(Nat, Nat.add(a, Nat.add(b, Nat.add(c, d))), Nat.add(Nat.add(a, b), Nat.add(c, d)), NL.add_assoc(a, b, Nat.add(c, d))),
    Equal.trans(Nat, Nat.add(a, Nat.add(b, Nat.add(c, d))), Nat.add(a, Nat.add(c, Nat.add(b, d))), Nat.add(Nat.add(a, c), Nat.add(b, d)),
      Equal.cong(Nat, Nat, s => Nat.add(a, s), Nat.add(b, Nat.add(c, d)), Nat.add(c, Nat.add(b, d)), NL.add_swap(b, c, d)),
      NL.add_assoc(a, c, Nat.add(b, d))))

# k * (a + b) == k * a + k * b
def mul_dist_l(+k: Nat, +a: Nat, +b: Nat) -> {Nat.mul(k, Nat.add(a, b)) == Nat.add(Nat.mul(k, a), Nat.mul(k, b)) : Nat}:
  Equal.trans(Nat, Nat.mul(k, Nat.add(a, b)), Nat.mul(Nat.add(a, b), k), Nat.add(Nat.mul(k, a), Nat.mul(k, b)),
    NL.mul_comm(k, Nat.add(a, b)),
    Equal.trans(Nat, Nat.mul(Nat.add(a, b), k), Nat.add(Nat.mul(a, k), Nat.mul(b, k)), Nat.add(Nat.mul(k, a), Nat.mul(k, b)),
      Equal.sym(Nat, Nat.add(Nat.mul(a, k), Nat.mul(b, k)), Nat.mul(Nat.add(a, b), k), NL.mul_dist(a, b, k)),
      Equal.trans(Nat, Nat.add(Nat.mul(a, k), Nat.mul(b, k)), Nat.add(Nat.mul(k, a), Nat.mul(b, k)), Nat.add(Nat.mul(k, a), Nat.mul(k, b)),
        Equal.cong(Nat, Nat, s => Nat.add(s, Nat.mul(b, k)), Nat.mul(a, k), Nat.mul(k, a), NL.mul_comm(a, k)),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(k, a), s), Nat.mul(b, k), Nat.mul(k, b), NL.mul_comm(b, k)))))

# =====================================================================================================
# Dot products
# =====================================================================================================

# dot(a + b, v) == dot(a, v) + dot(b, v): induction on the three vectors; at each entry (x + y)·z splits into x·z + y·z
# (distributivity) and add4 regroups the four terms
def dot_vadd(a: List<&2, Nat>, b: List<&2, Nat>, v: List<&2, Nat>) -> {Spec.dotn(Spec.vadd(a, b), v) == Nat.add(Spec.dotn(a, v), Spec.dotn(b, v)) : Nat}:
  match a b v:
    case Nil{} _ _:
      {==}
    case Con{+x, +xt} Nil{} +v:
      Equal.sym(Nat, Nat.add(Spec.dotn(Con{x, xt}, v), 0n), Spec.dotn(Con{x, xt}, v), NL.add_zero(Spec.dotn(Con{x, xt}, v)))
    case Con{x, xt} Con{y, yt} Nil{}:
      {==}
    case Con{+x, +xt} Con{+y, +yt} Con{+z, +zt}:
      +xz = Nat.mul(x, z)
      +yz = Nat.mul(y, z)
      +pa = Spec.dotn(xt, zt)
      +pb = Spec.dotn(yt, zt)
      Equal.trans(Nat, Nat.add(Nat.mul(Nat.add(x, y), z), Spec.dotn(Spec.vadd(xt, yt), zt)), Nat.add(Nat.mul(Nat.add(x, y), z), Nat.add(pa, pb)), Nat.add(Nat.add(xz, pa), Nat.add(yz, pb)),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(Nat.add(x, y), z), s), Spec.dotn(Spec.vadd(xt, yt), zt), Nat.add(pa, pb), dot_vadd(xt, yt, zt)),
        Equal.trans(Nat, Nat.add(Nat.mul(Nat.add(x, y), z), Nat.add(pa, pb)), Nat.add(Nat.add(xz, yz), Nat.add(pa, pb)), Nat.add(Nat.add(xz, pa), Nat.add(yz, pb)),
          Equal.cong(Nat, Nat, s => Nat.add(s, Nat.add(pa, pb)), Nat.mul(Nat.add(x, y), z), Nat.add(xz, yz), Equal.sym(Nat, Nat.add(xz, yz), Nat.mul(Nat.add(x, y), z), NL.mul_dist(x, y, z))),
          add4(xz, yz, pa, pb)))

# dot(k·r, v) == k · dot(r, v): at each entry (k·x)·z == k·(x·z) (associativity), then k factors out of the sum
def dot_scale(+k: Nat, r: List<&2, Nat>, v: List<&2, Nat>) -> {Spec.dotn(Spec.vscale(k, r), v) == Nat.mul(k, Spec.dotn(r, v)) : Nat}:
  match r v:
    case Nil{} _:
      Equal.sym(Nat, Nat.mul(k, 0n), 0n, NL.mul_zero(k))
    case Con{x, xt} Nil{}:
      Equal.sym(Nat, Nat.mul(k, 0n), 0n, NL.mul_zero(k))
    case Con{+x, +xt} Con{+z, +zt}:
      +d = Spec.dotn(xt, zt)
      Equal.trans(Nat, Nat.add(Nat.mul(Nat.mul(k, x), z), Spec.dotn(Spec.vscale(k, xt), zt)), Nat.add(Nat.mul(Nat.mul(k, x), z), Nat.mul(k, d)), Nat.mul(k, Nat.add(Nat.mul(x, z), d)),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(Nat.mul(k, x), z), s), Spec.dotn(Spec.vscale(k, xt), zt), Nat.mul(k, d), dot_scale(k, xt, zt)),
        Equal.trans(Nat, Nat.add(Nat.mul(Nat.mul(k, x), z), Nat.mul(k, d)), Nat.add(Nat.mul(k, Nat.mul(x, z)), Nat.mul(k, d)), Nat.mul(k, Nat.add(Nat.mul(x, z), d)),
          Equal.cong(Nat, Nat, s => Nat.add(s, Nat.mul(k, d)), Nat.mul(Nat.mul(k, x), z), Nat.mul(k, Nat.mul(x, z)), Equal.sym(Nat, Nat.mul(k, Nat.mul(x, z)), Nat.mul(Nat.mul(k, x), z), NL.mul_assoc(k, x, z))),
          Equal.sym(Nat, Nat.mul(k, Nat.add(Nat.mul(x, z), d)), Nat.add(Nat.mul(k, Nat.mul(x, z)), Nat.mul(k, d)), mul_dist_l(k, Nat.mul(x, z), d))))

# dot(a, b) == dot(b, a)
def dot_comm(a: List<&2, Nat>, b: List<&2, Nat>) -> {Spec.dotn(a, b) == Spec.dotn(b, a) : Nat}:
  match a b:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{y, yt}:
      {==}
    case Con{x, xt} Nil{}:
      {==}
    case Con{+x, +xt} Con{+y, +yt}:
      Equal.trans(Nat, Nat.add(Nat.mul(x, y), Spec.dotn(xt, yt)), Nat.add(Nat.mul(y, x), Spec.dotn(xt, yt)), Nat.add(Nat.mul(y, x), Spec.dotn(yt, xt)),
        Equal.cong(Nat, Nat, s => Nat.add(s, Spec.dotn(xt, yt)), Nat.mul(x, y), Nat.mul(y, x), NL.mul_comm(x, y)),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(y, x), s), Spec.dotn(xt, yt), Spec.dotn(yt, xt), dot_comm(xt, yt)))

# =====================================================================================================
# The linear laws
# =====================================================================================================

# PROOF (adjoint): induction on the rows of W and the entries of dy. Row r with weight g_i contributes g_i·(r·x) on the
# left and (g_i·r)·x on the right (dot_scale); the right side's sum of weighted rows splits by dot_vadd.
def Laws.adjoint(w, g, v):
  match w g:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{gi, gt}:
      {==}
    case Con{r, rt} Nil{}:
      {==}
    case Con{+r, +rt} Con{+gi, +gt}:
      +a = Spec.dotn(Spec.vscale(gi, r), v)
      +m = Spec.mtv(rt, gt)
      Equal.trans(Nat, Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(gt, Spec.mv(rt, v))), Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(m, v)), Spec.dotn(Spec.vadd(Spec.vscale(gi, r), m), v),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(gi, Spec.dotn(r, v)), s), Spec.dotn(gt, Spec.mv(rt, v)), Spec.dotn(m, v), Laws.adjoint(rt, gt, v)),
        Equal.trans(Nat, Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(m, v)), Nat.add(a, Spec.dotn(m, v)), Spec.dotn(Spec.vadd(Spec.vscale(gi, r), m), v),
          Equal.cong(Nat, Nat, s => Nat.add(s, Spec.dotn(m, v)), Nat.mul(gi, Spec.dotn(r, v)), a, Equal.sym(Nat, a, Nat.mul(gi, Spec.dotn(r, v)), dot_scale(gi, r, v))),
          Equal.sym(Nat, Spec.dotn(Spec.vadd(Spec.vscale(gi, r), m), v), Nat.add(a, Spec.dotn(m, v)), dot_vadd(Spec.vscale(gi, r), m, v))))

# PROOF (weight_adjoint): row by row, g_i·(r·x) == (g_i·x)·r, by dot_comm and dot_scale.
def Laws.weight_adjoint(w, g, v):
  match w g:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{gi, gt}:
      {==}
    case Con{r, rt} Nil{}:
      {==}
    case Con{+r, +rt} Con{+gi, +gt}:
      +f = Spec.frob(Spec.outer(gt, v), rt)
      Equal.trans(Nat, Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(gt, Spec.mv(rt, v))), Nat.add(Nat.mul(gi, Spec.dotn(r, v)), f), Nat.add(Spec.dotn(Spec.vscale(gi, v), r), f),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(gi, Spec.dotn(r, v)), s), Spec.dotn(gt, Spec.mv(rt, v)), f, Laws.weight_adjoint(rt, gt, v)),
        Equal.cong(Nat, Nat, s => Nat.add(s, f), Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(Spec.vscale(gi, v), r),
          Equal.trans(Nat, Nat.mul(gi, Spec.dotn(r, v)), Nat.mul(gi, Spec.dotn(v, r)), Spec.dotn(Spec.vscale(gi, v), r),
            Equal.cong(Nat, Nat, s => Nat.mul(gi, s), Spec.dotn(r, v), Spec.dotn(v, r), dot_comm(r, v)),
            Equal.sym(Nat, Spec.dotn(Spec.vscale(gi, v), r), Nat.mul(gi, Spec.dotn(v, r)), dot_scale(gi, v, r)))))

# PROOF (adjoint_seq): adjoint once for W2 (with W1·x as the input), then once for W1.
def Laws.adjoint_seq(w1, w2, g, v):
  Equal.trans(Nat, Spec.dotn(g, Spec.mv(w2, Spec.mv(w1, v))), Spec.dotn(Spec.mtv(w2, g), Spec.mv(w1, v)), Spec.dotn(Spec.mtv(w1, Spec.mtv(w2, g)), v), Laws.adjoint(w2, g, Spec.mv(w1, v)), Laws.adjoint(w1, Spec.mtv(w2, g), v))

# =====================================================================================================
# Causal attention
# =====================================================================================================

# the element-wise maps keep the length of a list
def len_shift(+k: F32, xs: List<&2, F32>) -> {List.length(&2, F32, T.l_shift(k, xs)) == List.length(&2, F32, xs) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{x, t}:
      Equal.cong(Nat, Nat, s => 1n+s, List.length(&2, F32, T.l_shift(k, t)), List.length(&2, F32, t), len_shift(k, t))

def len_exp(xs: List<&2, F32>) -> {List.length(&2, F32, T.l_exp(xs)) == List.length(&2, F32, xs) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{x, t}:
      Equal.cong(Nat, Nat, s => 1n+s, List.length(&2, F32, T.l_exp(t)), List.length(&2, F32, t), len_exp(t))

def len_scale2(+k: F32, xs: List<&2, F32>) -> {List.length(&2, F32, T.l_scale2(k, xs)) == List.length(&2, F32, xs) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{x, t}:
      Equal.cong(Nat, Nat, s => 1n+s, List.length(&2, F32, T.l_scale2(k, t)), List.length(&2, F32, t), len_scale2(k, t))

# softmax keeps the length: it is a shift, an exp and a division, each element by element
def len_softmax(+xs: List<&2, F32>) -> {List.length(&2, F32, T.l_softmax(xs)) == List.length(&2, F32, xs) : Nat}:
  +es = T.l_exp(T.l_shift(T.l_max(xs), xs))
  Equal.trans(Nat, List.length(&2, F32, T.l_scale2(T.l_sum(es, 0.0), es)), List.length(&2, F32, es), List.length(&2, F32, xs),
    len_scale2(T.l_sum(es, 0.0), es),
    Equal.trans(Nat, List.length(&2, F32, es), List.length(&2, F32, T.l_shift(T.l_max(xs), xs)), List.length(&2, F32, xs), len_exp(T.l_shift(T.l_max(xs), xs)), len_shift(T.l_max(xs), xs)))

def disc(b: Bool) -> Type:
  match b:
    case True{}:
      Unit
    case False{}:
      Empty

# False == True is impossible: rewriting the goal Empty = disc(False) with it gives disc(True) = Unit
def false_true(e: {False{} == True{} : Bool}) -> Empty:
  %Equal.sym(Bool, False{}, True{}, e) : disc(_)
  Unit{}

# taking n entries of a list with at least n entries gives n entries
def len_take(xs: List<&2, F32>, n: Nat, e: {Spec.le(n, List.length(&2, F32, xs)) == True{} : Bool}) -> {List.length(&2, F32, List.take(&2, F32, xs, n)) == n : Nat}:
  match xs n:
    case Nil{} 0n:
      {==}
    case Nil{} 1n+p:
      Empty.absurd({0n == 1n+p : Nat}, false_true(e))
    case Con{h, t} 0n:
      {==}
    case Con{h, t} 1n+p:
      Equal.cong(Nat, Nat, s => 1n+s, List.length(&2, F32, List.take(&2, F32, t, p)), p, len_take(t, p, e))

def drop_zero(z: List<&2, F32>) -> {List.drop(&2, F32, z, 0n) == z : List<&2, F32>}:
  match z:
    case Nil{}:
      {==}
    case Con{h, t}:
      {==}

# dropping as many entries as a has from a ++ z leaves z
def drop_app(a: List<&2, F32>, +z: List<&2, F32>) -> {List.drop(&2, F32, List.append(&2, F32, a, z), List.length(&2, F32, a)) == z : List<&2, F32>}:
  match a:
    case Nil{}:
      drop_zero(z)
    case Con{h, t}:
      drop_app(t, z)

# PROOF (causal): the row is softmax(first i + 1 scores) ++ zeros; the softmax part has exactly i + 1 entries
# (len_softmax, len_take), so dropping i + 1 entries leaves the zeros (drop_app).
def Laws.causal(r, i, t, e):
  +a = T.l_softmax(List.take(&2, F32, r, 1n+i))
  +z = T.l_zeros(Nat.sub(t, 1n+i))
  %Equal.trans(Nat, List.length(&2, F32, a), List.length(&2, F32, List.take(&2, F32, r, 1n+i)), 1n+i, len_softmax(List.take(&2, F32, r, 1n+i)), len_take(r, 1n+i, e)) : {List.drop(&2, F32, List.append(&2, F32, a, z), _) == z : List<&2, F32>}
  drop_app(a, z)

# =====================================================================================================
# Grouped-query heads (llama.bend's proof)
# =====================================================================================================

# one block of 1 + m numbers adds 1 to the quotient (Nat.divmod.go counts down the remainder from m)
def div_block(+m: Nat, -n: Nat, -d: Nat, -r: Nat) -> {Nat.divmod.go(Nat.add(1n+m, n), m, d, r) == Nat.divmod.go(n, Nat.add(m, r), 1n+d, 0n) : Nat & Nat}:
  match m:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, 1n+Nat.add(p, r), Nat.add(p, 1n+r), NL.add_succ(p, r)) : {Nat.divmod.go(Nat.add(1n+p, n), p, d, 1n+r) == Nat.divmod.go(n, _, 1n+d, 0n) : Nat & Nat}
      div_block(p, n, d, 1n+r)

# j whole blocks of 1 + s numbers add j to the quotient
def div_blocks(+j: Nat, +s: Nat, +n: Nat, +d: Nat) -> {Nat.divmod.go(Nat.add(Nat.mul(j, 1n+s), n), s, d, 0n) == Nat.divmod.go(n, s, Nat.add(j, d), 0n) : Nat & Nat}:
  match j:
    case 0n:
      {==}
    case 1n+jp:
      %NL.add_assoc(1n+s, Nat.mul(jp, 1n+s), n) : {Nat.divmod.go(_, s, d, 0n) == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(1n+s, Nat.add(Nat.mul(jp, 1n+s), n)), s, d, 0n), Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), Nat.add(s, 0n), 1n+d, 0n), div_block(s, Nat.add(Nat.mul(jp, 1n+s), n), d, 0n)) : {_ == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, Nat.add(s, 0n), s, NL.add_zero(s)) : {Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), _, 1n+d, 0n) == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, 1n+Nat.add(jp, d), Nat.add(jp, 1n+d), NL.add_succ(jp, d)) : {Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), s, 1n+d, 0n) == Nat.divmod.go(n, s, _, 0n) : Nat & Nat}
      div_blocks(jp, s, n, 1n+d)

# fewer than 1 + s numbers after the last whole block leave the quotient alone
def div_rest(+i: Nat, -k: Nat, -d: Nat, -r: Nat) -> {Pair.fst(Nat, Nat, Nat.divmod.go(i, Nat.add(i, k), d, r)) == d : Nat}:
  match i:
    case 0n:
      {==}
    case 1n+p:
      div_rest(p, k, d, 1n+r)

# PROOF (gqa_head): Nat.div counts how many whole blocks of g numbers fit. j·g is j whole blocks (div_blocks), and
# i < g numbers after them do not complete another one (div_rest).
def Laws.gqa_head(j, i, k):
  %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), Nat.add(i, k), 0n, 0n), Nat.divmod.go(i, Nat.add(i, k), Nat.add(j, 0n), 0n), div_blocks(j, Nat.add(i, k), i, 0n)) : {Pair.fst(Nat, Nat, _) == j : Nat}
  %NL.add_zero(j) : {Pair.fst(Nat, Nat, Nat.divmod.go(i, Nat.add(i, k), Nat.add(j, 0n), 0n)) == _ : Nat}
  div_rest(i, k, Nat.add(j, 0n), 0n)

def main() -> Unit:
  Unit{}

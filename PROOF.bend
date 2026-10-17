# The proofs of LAWS.bend. Every proof is plain Bend: no @unsafe, no ?TODO, no axioms. The arithmetic lemmas come
# from bend-ml-nat-lemmas; the division lemmas behind gqa_head are llama.bend's (MIT, same author).

import Base
import ./torch/main.bend as T
import ./torch/spec.bend as Spec
import ./torch/tensor.bend as TT
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

# xs reversed onto acc has the length of both
def len_rcat(+xs: List<&2, F32>, +acc: List<&2, F32>) -> {List.length(&2, F32, T.rcat(xs, acc)) == Nat.add(List.length(&2, F32, xs), List.length(&2, F32, acc)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{+x, +t}:
      Equal.trans(Nat, List.length(&2, F32, T.rcat(t, x <> acc)), Nat.add(List.length(&2, F32, t), 1n+List.length(&2, F32, acc)), 1n+Nat.add(List.length(&2, F32, t), List.length(&2, F32, acc)), len_rcat(t, x <> acc), Equal.sym(Nat, 1n+Nat.add(List.length(&2, F32, t), List.length(&2, F32, acc)), Nat.add(List.length(&2, F32, t), 1n+List.length(&2, F32, acc)), NL.add_succ(List.length(&2, F32, t), List.length(&2, F32, acc))))

# drow's row has the current dot, the m after it, and acc: 1 + m + |acc| numbers
def len_drow(m: Nat, +hd: Nat, +ia: U32, +ib: U32, +bs: U32, +acc: List<&2, F32>, r: T.D2) -> {List.length(&2, F32, T.dr_row(T.drow(m, hd, ia, ib, bs, acc, r))) == Nat.add(1n+m, List.length(&2, F32, acc)) : Nat}:
  match m r:
    case 0n T.D2{a, b, +s}:
      Equal.trans(Nat, List.length(&2, F32, T.rcat(s <> acc, Nil{})), Nat.add(1n+List.length(&2, F32, acc), 0n), 1n+List.length(&2, F32, acc), len_rcat(s <> acc, Nil{}), NL.add_zero(1n+List.length(&2, F32, acc)))
    case 1n++p T.D2{a, b, +s}:
      Equal.trans(Nat, List.length(&2, F32, T.dr_row(T.drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b)))), Nat.add(1n+p, 1n+List.length(&2, F32, acc)), 2n+Nat.add(p, List.length(&2, F32, acc)), len_drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b)), Equal.cong(Nat, Nat, z => 1n+z, Nat.add(p, 1n+List.length(&2, F32, acc)), 1n+Nat.add(p, List.length(&2, F32, acc)), Equal.sym(Nat, 1n+Nat.add(p, List.length(&2, F32, acc)), Nat.add(p, 1n+List.length(&2, F32, acc)), NL.add_succ(p, List.length(&2, F32, acc)))))

# PROOF (causal_keys): qrow is drow(i, ..) from an empty accumulator, so its row has 1 + i + 0 numbers (len_drow).
def Laws.causal_keys(i, hd, qs, qc, ks, kc, q, k):
  Equal.trans(Nat, List.length(&2, F32, T.dr_row(T.qrow(i, hd, qs, qc, ks, kc, q, k))), Nat.add(1n+i, 0n), 1n+i, len_drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks, Nil{}, T.dot0(hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, 1, 1, q, k)), NL.add_zero(1n+i))

# pw writes xs and counts them from n: prow3 hands on |xs| + n
def cnt_pw(+xs: List<&2, F32>, +idx: U32, +sc: F32, +n: Nat, +off: U32, q: Array<F32>, k: Array<F32>, p: Array<F32>) -> {T.pr_n(T.prow3(off, q, k, T.pw(xs, idx, sc, n, p))) == Nat.add(List.length(&2, F32, xs), n) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{+x, +t}:
      Equal.trans(Nat, T.pr_n(T.prow3(off, q, k, T.pw(t, (idx + 1 : U32), sc, 1n+n, Array.set(F32, p, idx, (x * sc : F32))))), Nat.add(List.length(&2, F32, t), 1n+n), 1n+Nat.add(List.length(&2, F32, t), n), cnt_pw(t, (idx + 1 : U32), sc, 1n+n, off, q, k, Array.set(F32, p, idx, (x * sc : F32))), Equal.sym(Nat, 1n+Nat.add(List.length(&2, F32, t), n), Nat.add(List.length(&2, F32, t), 1n+n), NL.add_succ(List.length(&2, F32, t), n)))

# the count prow2 hands on for drow's row: 1 + m + |acc|, as len_drow, through cnt_pw
def cnt_drow(m: Nat, +hd: Nat, +ia: U32, +ib: U32, +bs: U32, +acc: List<&2, F32>, r: T.D2, +sc: F32, +off: U32, pa: Array<F32>) -> {T.pr_n(T.prow2(sc, off, pa, T.drow(m, hd, ia, ib, bs, acc, r))) == Nat.add(1n+m, List.length(&2, F32, acc)) : Nat}:
  match m r:
    case 0n T.D2{a, b, +s}:
      +rw = T.rcat(s <> acc, Nil{})
      Equal.trans(Nat, T.pr_n(T.prow3(off, a, b, T.pw(rw, off, sc, 0n, pa))), Nat.add(List.length(&2, F32, rw), 0n), 1n+List.length(&2, F32, acc), cnt_pw(rw, off, sc, 0n, off, a, b, pa),
        Equal.trans(Nat, Nat.add(List.length(&2, F32, rw), 0n), List.length(&2, F32, rw), 1n+List.length(&2, F32, acc), NL.add_zero(List.length(&2, F32, rw)),
          Equal.trans(Nat, List.length(&2, F32, rw), Nat.add(1n+List.length(&2, F32, acc), 0n), 1n+List.length(&2, F32, acc), len_rcat(s <> acc, Nil{}), NL.add_zero(1n+List.length(&2, F32, acc)))))
    case 1n++p T.D2{a, b, +s}:
      Equal.trans(Nat, T.pr_n(T.prow2(sc, off, pa, T.drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b)))), Nat.add(1n+p, 1n+List.length(&2, F32, acc)), 2n+Nat.add(p, List.length(&2, F32, acc)), cnt_drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b), sc, off, pa), Equal.cong(Nat, Nat, z => 1n+z, Nat.add(p, 1n+List.length(&2, F32, acc)), 1n+Nat.add(p, List.length(&2, F32, acc)), Equal.sym(Nat, 1n+Nat.add(p, List.length(&2, F32, acc)), Nat.add(p, 1n+List.length(&2, F32, acc)), NL.add_succ(p, List.length(&2, F32, acc)))))

# PROOF (causal): prow runs prow2 on qrow's row, drow(i, ..) from an empty accumulator, so the count it hands the
# softmax is 1 + i + 0 (cnt_drow).
def Laws.causal(i, t, hd, sc, qs, qc, ks, kc, q, k, p):
  Equal.trans(Nat, T.pr_n(T.prow(i, t, hd, sc, qs, qc, ks, kc, q, k, p)), Nat.add(1n+i, 0n), 1n+i, cnt_drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks, Nil{}, T.dot0(hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, 1, 1, q, k), sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p), NL.add_zero(1n+i))

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

# PROOF (kv_of_group): kv_of is Nat.div, so this is gqa_head.
def Laws.kv_of_group(j, i, k):
  Laws.gqa_head(j, i, k)

# =====================================================================================================
# Tensors: shapes
# =====================================================================================================

# a || b == b || a, and a && b == b && a: every case computes
def or_comm(+a: Bool, +b: Bool) -> {Bool.or(a, b) == Bool.or(b, a) : Bool}:
  match a b:
    case True{} True{}:
      {==}
    case True{} False{}:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      {==}

def and_comm(+a: Bool, +b: Bool) -> {Bool.and(a, b) == Bool.and(b, a) : Bool}:
  match a b:
    case True{} True{}:
      {==}
    case True{} False{}:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      {==}

# equal operands give equal conjunctions and disjunctions
def and_cong(+p: Bool, +p2: Bool, +q: Bool, +q2: Bool, e: {p == p2 : Bool}, f: {q == q2 : Bool}) -> {Bool.and(p, q) == Bool.and(p2, q2) : Bool}:
  Equal.trans(Bool, Bool.and(p, q), Bool.and(p2, q), Bool.and(p2, q2), Equal.cong(Bool, Bool, x => Bool.and(x, q), p, p2, e), Equal.cong(Bool, Bool, x => Bool.and(p2, x), q, q2, f))

def or_cong(+p: Bool, +p2: Bool, +q: Bool, +q2: Bool, e: {p == p2 : Bool}, f: {q == q2 : Bool}) -> {Bool.or(p, q) == Bool.or(p2, q2) : Bool}:
  Equal.trans(Bool, Bool.or(p, q), Bool.or(p2, q), Bool.or(p2, q2), Equal.cong(Bool, Bool, x => Bool.or(x, q), p, p2, e), Equal.cong(Bool, Bool, x => Bool.or(p2, x), q, q2, f))

# one axis: bc_dim is written as pair(x, y) || pair(y, x), so swapping x and y only swaps the two sides of the ||
def bc_dim_sym(+x: Nat, +y: Nat, +z: Nat) -> {TT.bc_dim(x, y, z) == TT.bc_dim(y, x, z) : Bool}:
  or_comm(TT.bc_pair(x, y, z), TT.bc_pair(y, x, z))

# axis by axis: the same on each axis (bc_dim_sym), and by induction on the rest; shapes of different ranks are False
# either way round
def bc_eq_sym(a: List<&2, Nat>, b: List<&2, Nat>, o: List<&2, Nat>) -> {TT.bc_eq(a, b, o) == TT.bc_eq(b, a, o) : Bool}:
  match a b o:
    case Nil{} Nil{} Nil{}:
      {==}
    case Nil{} Nil{} Con{z, ro}:
      {==}
    case Nil{} Con{y, rb} Nil{}:
      {==}
    case Nil{} Con{y, rb} Con{z, ro}:
      {==}
    case Con{x, ra} Nil{} Nil{}:
      {==}
    case Con{x, ra} Nil{} Con{z, ro}:
      {==}
    case Con{x, ra} Con{y, rb} Nil{}:
      {==}
    case Con{+x, +ra} Con{+y, +rb} Con{+z, +ro}:
      and_cong(TT.bc_dim(x, y, z), TT.bc_dim(y, x, z), TT.bc_eq(ra, rb, ro), TT.bc_eq(rb, ra, ro), bc_dim_sym(x, y, z), bc_eq_sym(ra, rb, ro))

# the ranks: the same two comparisons and the same two equalities, in the other order
def bc_rank_sym(+la: Nat, +lb: Nat, +lo: Nat) -> {TT.bc_rank(la, lb, lo) == TT.bc_rank(lb, la, lo) : Bool}:
  and_cong(Bool.and(Nat.is_le(la, lo), Nat.is_le(lb, lo)), Bool.and(Nat.is_le(lb, lo), Nat.is_le(la, lo)), Bool.or(Nat.is_eq(la, lo), Nat.is_eq(lb, lo)), Bool.or(Nat.is_eq(lb, lo), Nat.is_eq(la, lo)),
    and_comm(Nat.is_le(la, lo), Nat.is_le(lb, lo)), or_comm(Nat.is_eq(la, lo), Nat.is_eq(lb, lo)))

def Laws.bc_sym(a, b, o):
  and_cong(TT.bc_rank(List.length(&2, Nat, a), List.length(&2, Nat, b), List.length(&2, Nat, o)), TT.bc_rank(List.length(&2, Nat, b), List.length(&2, Nat, a), List.length(&2, Nat, o)),
    TT.bc_eq(TT.pad(a, List.length(&2, Nat, o)), TT.pad(b, List.length(&2, Nat, o)), o), TT.bc_eq(TT.pad(b, List.length(&2, Nat, o)), TT.pad(a, List.length(&2, Nat, o)), o),
    bc_rank_sym(List.length(&2, Nat, a), List.length(&2, Nat, b), List.length(&2, Nat, o)), bc_eq_sym(TT.pad(a, List.length(&2, Nat, o)), TT.pad(b, List.length(&2, Nat, o)), o))

# h * (a * r) == a * (h * r): associativity, then commutativity of the first two factors, then associativity back
def mul_swap(+h: Nat, +a: Nat, +r: Nat) -> {Nat.mul(h, Nat.mul(a, r)) == Nat.mul(a, Nat.mul(h, r)) : Nat}:
  Equal.trans(Nat, Nat.mul(h, Nat.mul(a, r)), Nat.mul(Nat.mul(h, a), r), Nat.mul(a, Nat.mul(h, r)), NL.mul_assoc(h, a, r),
    Equal.trans(Nat, Nat.mul(Nat.mul(h, a), r), Nat.mul(Nat.mul(a, h), r), Nat.mul(a, Nat.mul(h, r)),
      Equal.cong(Nat, Nat, x => Nat.mul(x, r), Nat.mul(h, a), Nat.mul(a, h), NL.mul_comm(h, a)),
      Equal.sym(Nat, Nat.mul(a, Nat.mul(h, r)), Nat.mul(Nat.mul(a, h), r), NL.mul_assoc(a, h, r))))

# by induction on the shape and the axis: axis 0 computes; past it, the first size h stays in front on both sides, and
# mul_swap moves it past size(k)
def Laws.sum_numel(s, k):
  match s k:
    case Nil{} _:
      {==}
    case Con{h, t} 0n:
      {==}
    case Con{+h, +t} 1n+p:
      Equal.trans(Nat, Nat.mul(h, NL.product(t)), Nat.mul(h, Nat.mul(TT.at1(t, p), NL.product(TT.remove(t, p)))), Nat.mul(TT.at1(t, p), Nat.mul(h, NL.product(TT.remove(t, p)))),
        Equal.cong(Nat, Nat, x => Nat.mul(h, x), NL.product(t), Nat.mul(TT.at1(t, p), NL.product(TT.remove(t, p))), Laws.sum_numel(t, p)),
        mul_swap(h, TT.at1(t, p), NL.product(TT.remove(t, p))))

# =====================================================================================================
# Tensors: narrow, cat, of_mat, permute
# =====================================================================================================

# by induction: axis 0 is replaced and then taken out; past it, the first size stays in front on both sides
def Laws.narrow_rest(s, d, l):
  match s d:
    case Nil{} _:
      {==}
    case Con{h, t} 0n:
      {==}
    case Con{+h, +t} 1n+p:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, x => h <> x, TT.remove(TT.setd(t, p, l), p), TT.remove(t, p), Laws.narrow_rest(t, p, l))

# a shape with no axis d contradicts the premise (d < rank is False); axis 0 is l; past it, the induction
def Laws.narrow_axis(s, d, l, e):
  match s d:
    case Nil{} 0n:
      Empty.absurd({TT.at1(TT.setd([], 0n, l), 0n) == l : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Nil{} 1n+p:
      Empty.absurd({TT.at1(TT.setd([], 1n+p, l), 1n+p) == l : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Con{h, t} 0n:
      {==}
    case Con{h, +t} 1n+p:
      Laws.narrow_axis(t, p, l, e)

# axis 0: (x + y)·n == x·n + y·n (distributivity); past it, the induction and k·(a + b) == k·a + k·b
def Laws.cat_numel(s, d, x, y, e):
  match s d:
    case Nil{} 0n:
      Empty.absurd({TT.numel(TT.setd([], 0n, Nat.add(x, y))) == Nat.add(TT.numel(TT.setd([], 0n, x)), TT.numel(TT.setd([], 0n, y))) : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Nil{} 1n+p:
      Empty.absurd({TT.numel(TT.setd([], 1n+p, Nat.add(x, y))) == Nat.add(TT.numel(TT.setd([], 1n+p, x)), TT.numel(TT.setd([], 1n+p, y))) : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Con{h, +t} 0n:
      Equal.sym(Nat, Nat.add(Nat.mul(x, NL.product(t)), Nat.mul(y, NL.product(t))), Nat.mul(Nat.add(x, y), NL.product(t)), NL.mul_dist(x, y, NL.product(t)))
    case Con{+h, +t} 1n+p:
      Equal.trans(Nat, Nat.mul(h, NL.product(TT.setd(t, p, Nat.add(x, y)))), Nat.mul(h, Nat.add(NL.product(TT.setd(t, p, x)), NL.product(TT.setd(t, p, y)))), Nat.add(Nat.mul(h, NL.product(TT.setd(t, p, x))), Nat.mul(h, NL.product(TT.setd(t, p, y)))),
        Equal.cong(Nat, Nat, v => Nat.mul(h, v), NL.product(TT.setd(t, p, Nat.add(x, y))), Nat.add(NL.product(TT.setd(t, p, x)), NL.product(TT.setd(t, p, y))), Laws.cat_numel(t, p, x, y, e)),
        mul_dist_l(h, NL.product(TT.setd(t, p, x)), NL.product(TT.setd(t, p, y))))

# r·(c·1) == r·c: c·1 == c
def Laws.mat_numel(r, c):
  Equal.cong(Nat, Nat, v => Nat.mul(r, v), Nat.mul(c, 1n), c, NL.mul_one_r(c))

# one axis of the result per entry of p
def Laws.permute_rank(s, p):
  match p:
    case Nil{}:
      {==}
    case Con{k, r}:
      Equal.cong(Nat, Nat, v => 1n+v, TT.rank(TT.permuted(s, r)), List.length(&2, Nat, r), Laws.permute_rank(s, r))

def main() -> Unit:
  Unit{}

# BendTorch's layers over bend-ml-tensor-fork: the same layers, types and bits as torch/main.bend's, faster where the
# work splits into parallel blocks.
#
#   import ../torch/main.bend as T
#   import ../torch/fast.bend as F
#   F.LinearFirstF(100n, 784n, 128n, 3n)     # instead of T.LinearFirst(100n, 784n, 128n, 3n)
#
# LinearF, LinearFirstF and LinearNBF are main.bend's Linear, LinearFirst and LinearNB with their three products
# (x·Wᵀ, dY·W, dYᵀ·X added into dW) taken from bend-ml-tensor-fork instead of bend-ml-tensor-array, and their optimizer
# step in blocks. SiLU, ActIn, Mul2, Add2, Residual, Mat.dup and Mat.sum2 are main.bend's with their element-wise loops
# in blocks (tensor-fork's par1..par4), the optimizer step (mat_step, lin_step) too. Every number is computed by
# main.bend's own loop or tensor-array's own cells, so the bits are main.bend's at every depth and thread count (tests:
# demo/fast_test.bend). Small matrices run in one block (ew_depth), as in main.bend.
#
# Outside the proofs, on purpose: tensor-fork shares arrays through Base's @unsafe Array.fork, so `bend` names the defs
# below that reach it, and no file with laws imports this one (`make laws` checks both). All of it is Bend.

import Base
import bend-ml-tensor-array@0.1.12.0/main.bend as TA
import bend-ml-tensor-fork@0.1.1.0/main.bend as TF
import ./main.bend as T

# =====================================================================================================
# Element-wise work and the optimizer step in parallel blocks (bend-ml-tensor-fork's par1..par4): every number is
# computed by main.bend's own loop, so the bits are main.bend's at every depth and thread count. Small matrices run in
# one block (ew_depth).
# =====================================================================================================

# log2 of the blocks for cnt numbers: one block below 131072 (MNIST's and the tiny Llama's matrices; at 100352 the
# blocks cost MNIST more than they gave on 16 threads), else one per 16384, at most 64
def ew_d2(small: Bool, big: Bool, +cnt: Nat) -> Nat:
  match small big:
    case True{} _:
      0n
    case False{} True{}:
      6n
    case False{} False{}:
      TF.depth(Nat.div(cnt, 16384n))

def ew_depth(+cnt: Nat) -> Nat:
  ew_d2(Nat.is_lt(cnt, 131072n), Nat.is_le(1048576n, cnt), cnt)

# ---- the optimizer step on a matrix ----

def sgd_b(-r: Nat, -c: Nat, m: Array<F32>, v: Array<F32>, q: TA.P2) -> TF.B4<F32, F32, F32, F32>:
  match q:
    case TA.P2{w, g}:
      TF.B4{w, g, m, v}

def mom_b(v: Array<F32>, q: T.Q4) -> TF.B4<F32, F32, F32, F32>:
  match q:
    case T.Q4{w, g, m, dummy}:
      TF.B4{w, g, m, v}

def q4_b(q: T.Q4) -> TF.B4<F32, F32, F32, F32>:
  match q:
    case T.Q4{w, g, m, v}:
      TF.B4{w, g, m, v}

# numbers i0 .. i0+cnt-1 of the parameter, its gradient and its moments: main.bend's loops from i0
def opt_leaf(+o: T.Opt, +i0: U32, +cnt: Nat, r: TF.B4<F32, F32, F32, F32>) -> TF.B4<F32, F32, F32, F32>:
  match o r:
    case T.SGD{+lr} TF.B4{w, g, m, v}:
      sgd_b(0n, 0n, m, v, T.sgd_l(cnt, i0, lr, Array.get(F32, w, i0), Array.get(F32, g, i0)))
    case T.Momentum{+lr, +mu} TF.B4{w, g, m, v}:
      mom_b(v, T.mom_l(cnt, i0, lr, mu, Array.get(F32, w, i0), Array.get(F32, g, i0), Array.get(F32, m, i0)))
    case T.Adam{+lr, +b1, +b2, +eps, +wd, +dec, +step, +bc2} TF.B4{w, g, m, v}:
      q4_b(T.adam_l(cnt, i0, lr, b1, b2, eps, wd, dec, step, bc2, Array.get(F32, w, i0), Array.get(F32, g, i0), Array.get(F32, m, i0), Array.get(F32, v, i0)))

def opt_fin(-r: Nat, -c: Nat, q: TF.B4<F32, F32, F32, F32>) -> T.MatStep<r, c>:
  match q:
    case TF.B4{w, g, m, v}:
      T.MatStep{TA.Mat{w}, TA.Mat{g}, TA.Mat{m}, TA.Mat{v}}

# T.Opt.mat in blocks
def opt_mat(+r: Nat, +c: Nat, +o: T.Opt, w: TA.Mat<r, c>, g: TA.Mat<r, c>, m: TA.Mat<r, c>, v: TA.Mat<r, c>) -> T.MatStep<r, c>:
  match w g m v:
    case TA.Mat{wd} TA.Mat{gd} TA.Mat{md} TA.Mat{vd}:
      opt_fin(r, c, TF.par4(~T.Opt, ~F32, ~F32, ~F32, ~F32, ~opt_leaf, o, ew_depth(Nat.mul(r, c)), Nat.mul(r, c), wd, gd, md, vd))

def mat_step(+r: Nat, +c: Nat, +op: T.Opt, p: TA.Mat<r, c>, g: TA.Mat<r, c>, st: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>) -> T.Step<TA.Mat<r, c>>:
  match st:
    case T.Two{m, v}:
      T.mat_step_fin(r, c, opt_mat(r, c, op, p, g, m, v))

def lin_step(+i: Nat, +o: Nat, +op: T.Opt, p: T.Lin<i, o>, g: T.Lin<i, o>, st: T.Two<T.Lin<i, o>, T.Lin<i, o>>) -> T.Step<T.Lin<i, o>>:
  match p g st:
    case T.Lin{w, b} T.Lin{gw, gb} T.Two{T.Lin{mw, mb}, T.Lin{vw, vb}}:
      T.lin_s2(i, o, opt_mat(o, i, op, w, gw, mw, vw), opt_mat(1n, o, op, b, gb, mb, vb))

# ---- element-wise maps, zips and copies ----

def ew1_leaf(~f: @+a: F32 -> F32, +e: Unit, +i0: U32, +cnt: Nat, a: Array<F32>) -> Array<F32>:
  T.ew1_l(~f, cnt, i0, Array.get(F32, a, i0))

def map(~f: @+a: F32 -> F32, +r: Nat, +c: Nat, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  match m:
    case TA.Mat{d}:
      TA.Mat{TF.par1(~Unit, ~F32, ~(e => i0 => k => a => ew1_leaf(~f, e, i0, k, a)), Unit{}, ew_depth(Nat.mul(r, c)), Nat.mul(r, c), d)}

def p2_b2(q: TA.P2) -> TF.B2<F32, F32>:
  match q:
    case TA.P2{a, b}:
      TF.B2{a, b}

def ew2_leaf(~f: @+a: F32 -> @+b: F32 -> F32, +e: Unit, +i0: U32, +cnt: Nat, r: TF.B2<F32, F32>) -> TF.B2<F32, F32>:
  match r:
    case TF.B2{a, b}:
      p2_b2(T.ew2_l(~f, cnt, i0, Array.get(F32, a, i0), Array.get(F32, b, i0)))

def zip_fin(-r: Nat, -c: Nat, q: TF.B2<F32, F32>) -> TA.MPair<r, c>:
  match q:
    case TF.B2{a, b}:
      TA.MPair{TA.Mat{a}, TA.Mat{b}}

# a[i] = f(a[i], b[i]); b comes back unchanged
def zip(~f: @+a: F32 -> @+b: F32 -> F32, +r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.MPair<r, c>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      zip_fin(r, c, TF.par2(~Unit, ~F32, ~F32, ~(e => i0 => k => q => ew2_leaf(~f, e, i0, k, q)), Unit{}, ew_depth(Nat.mul(r, c)), Nat.mul(r, c), x, y))

def a2_b2(q: T.A2) -> TF.B2<F32, F32>:
  match q:
    case T.A2{a, b}:
      TF.B2{a, b}

# numbers i0 .. i0+cnt-1 of a into b
def copy_leaf(+e: Unit, +i0: U32, +cnt: Nat, r: TF.B2<F32, F32>) -> TF.B2<F32, F32>:
  match cnt r:
    case 0n r:
      r
    case 1n+k TF.B2{a, b}:
      a2_b2(T.copy_n(k, i0, i0, b, Array.get(F32, a, i0)))

def dup_fin(-r: Nat, -c: Nat, q: TF.B2<F32, F32>) -> T.Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  match q:
    case TF.B2{a, b}:
      T.Two{TA.Mat{a}, TA.Mat{b}}

# two copies of a matrix (T.Mat.dup)
def Mat.dup(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> T.Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  match m:
    case TA.Mat{a}:
      dup_fin(r, c, TF.par2(~Unit, ~F32, ~F32, ~copy_leaf, Unit{}, ew_depth(Nat.mul(r, c)), Nat.mul(r, c), a, T.zeros_a(Nat.mul(r, c))))

def mpair_a(-r: Nat, -c: Nat, q: TA.MPair<r, c>) -> TA.Mat<r, c>:
  match q:
    case TA.MPair{a, b}:
      a

# a + b, dropping b (T.Mat.sum2)
def Mat.sum2(+r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.Mat<r, c>:
  mpair_a(r, c, zip(~T.f32_add, r, c, a, b))

# ---- Linear: y = x·Wᵀ + b ----

def lin_fwd(+n: Nat, +i: Nat, +o: Nat, par: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>) -> T.Fwd<T.Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match p:
    case T.Lin{w, b}:
      T.lin_f2(n, i, o, b, TF.matmul_nt(n, i, o, par, x, w))

def lin_b2(+n: Nat, +i: Nat, +o: Nat, par: Nat, b: TA.Mat<1n, o>, x: TA.Mat<n, i>, g: T.Lin<i, o>, r: TA.MMul<n, o, i>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match g r:
    case T.Lin{gw, gb} TA.MMul{dy, w, dx}:
      T.lin_b3(n, i, o, w, b, dx, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))

def lin_bwd(+n: Nat, +i: Nat, +o: Nat, +par: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: T.Lin<i, o>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match p:
    case T.Lin{w, b}:
      lin_b2(n, i, o, par, b, x, g, TF.matmul(n, o, i, par, dy, w))

def LinearF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<T.Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{p => x => lin_fwd(n, i, o, par, p, x), p => s => dy => g => lin_bwd(n, i, o, par, p, s, dy, g), op => p => g => st => lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => T.lin_load(i, o, p, xs)}

# ---- LinearFirst: no dX (the first layer's input needs no gradient) ----

def linf_bwd(+n: Nat, +i: Nat, +o: Nat, +par: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: T.Lin<i, o>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match p g:
    case T.Lin{w, b} T.Lin{gw, gb}:
      T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))

def LinearFirstF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<T.Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{p => x => lin_fwd(n, i, o, par, p, x), p => s => dy => g => linf_bwd(n, i, o, par, p, s, dy, g), op => p => g => st => lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => T.lin_load(i, o, p, xs)}

# ---- LinearNB: y = x·Wᵀ, no bias (Llama's projections) ----

def lnb_b2(+n: Nat, +i: Nat, +o: Nat, +par: Nat, x: TA.Mat<n, i>, g: TA.Mat<o, i>, r: TA.MMul<n, o, i>) -> T.Bwd<TA.Mat<o, i>, TA.Mat<n, i>>:
  match r:
    case TA.MMul{dy, w, dx}:
      T.lnb_b3(n, i, o, w, dx, TF.matmul_tn_acc(o, n, i, par, dy, x, g))

def LinearNBF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<TA.Mat<o, i>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{w => x => T.lnb_f2(n, i, o, TF.matmul_nt(n, i, o, par, x, w)), w => s => dy => g => lnb_b2(n, i, o, par, s, g, TF.matmul(n, o, i, par, dy, w)), op => p => g => st => mat_step(o, i, op, p, g, st), u => TA.Mat.zeros(o, i), p => T.Mat.dump(o, i, p), p => xs => T.Mat.restore(o, i, p, xs)}

# ---- element-wise layers ----

def acti_f3(~f: @+a: F32 -> F32, +n: Nat, +c: Nat, m: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match m:
    case T.Two{x, s}:
      T.Fwd{Unit{}, map(~f, n, c, x), s}

# T.ActIn: f on a copy, the input saved
def ActIn(~f: @+a: F32 -> F32, ~d: @+a: F32 -> @+b: F32 -> F32, +n: Nat, +c: Nat) -> T.Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  T.Layer{p => x => acti_f3(~f, n, c, Mat.dup(n, c, x)), p => s => dy => g => T.act_b2(n, c, zip(~d, n, c, dy, s)), op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => T.PL{p, xs}}

def SiLU(+n: Nat, +c: Nat) -> T.Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActIn(~T.silu_v, ~T.silu_d, n, c)

def add2_f(+n: Nat, +c: Nat, x: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Fwd<Unit, TA.Mat<n, c>, Unit>:
  match x:
    case T.Two{a, b}:
      T.Fwd{Unit{}, Mat.sum2(n, c, a, b), Unit{}}

def Add2(+n: Nat, +c: Nat) -> T.Layer<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, Unit>:
  T.Layer{p => x => add2_f(n, c, x), p => s => dy => g => T.Bwd{p, Mat.dup(n, c, dy), g}, op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => T.PL{p, xs}}

def mul2_f2(+n: Nat, +c: Nat, da: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, db: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Fwd<Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match da db:
    case T.Two{a1, a2} T.Two{b1, b2}:
      T.mul2_f3(n, c, a2, b2, zip(~T.f32_mul, n, c, a1, b1))

def mul2_f(+n: Nat, +c: Nat, x: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Fwd<Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match x:
    case T.Two{a, b}:
      mul2_f2(n, c, Mat.dup(n, c, a), Mat.dup(n, c, b))

def mul2_b2(+n: Nat, +c: Nat, s: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, d: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Bwd<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match s d:
    case T.Two{a, b} T.Two{d1, d2}:
      T.Bwd{Unit{}, T.Two{mpair_a(n, c, zip(~T.f32_mul, n, c, d1, b)), mpair_a(n, c, zip(~T.f32_mul, n, c, d2, a))}, Unit{}}

def Mul2(+n: Nat, +c: Nat) -> T.Layer<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  T.Layer{p => x => mul2_f(n, c, x), p => s => dy => g => mul2_b2(n, c, s, Mat.dup(n, c, dy)), op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => T.PL{p, xs}}

# T.Residual with these copies and sums
def Residual(~n: Nat, ~c: Nat, ~P: Type, ~S: Type, ~f: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, c>, S>) -> T.Layer<T.Two<T.Two<Unit, P>, Unit>, TA.Mat<n, c>, TA.Mat<n, c>, T.Two<T.Two<Unit, S>, Unit>>:
  T.Seq(~T.Two<Unit, P>, ~Unit, ~TA.Mat<n, c>, ~T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, ~TA.Mat<n, c>, ~T.Two<Unit, S>, ~Unit, ~T.Fork(~Unit, ~P, ~TA.Mat<n, c>, ~TA.Mat<n, c>, ~TA.Mat<n, c>, ~Unit, ~S, ~(m => Mat.dup(n, c, m)), ~(a => b => Mat.sum2(n, c, a, b)), ~T.Id(~TA.Mat<n, c>), ~f), ~Add2(n, c))

# BendTorch's layers with their work on the GPU (bend-ml-tensor-gpu): the same layers, types and bits as
# torch/main.bend's and torch/fast.bend's.
#
#   import ../torch/main.bend as T
#   import ../torch/gpu.bend as F
#   F.LinearFirstF(100n, 784n, 128n, 4n)     # this layer's products and optimizer step run as bangs
#
# Every layer here runs its work as bangs (one `!` call each), which run on the GPU in a binary built with CUDA and on
# the CPU threads otherwise:
# - LinearF, LinearFirstF and LinearNBF: their three products (x·Wᵀ, dY·W, dYᵀ·X added into dW) from tensor-gpu, and
#   their optimizer step.
# - SiLU, Mul2, Add2, Residual, Mat.dup and Mat.sum2: their element-wise loops.
# - RMSNorm, RoPE and Attention (main.bend's names; fast.bend's are main.bend's, on the CPU).
# - MSE, CrossEntropy and Embedding (its rows, its gradient and its step), so that a training step's work, the loss
#   included, stays on the device; the host reads back a few numbers a row for the loss.
# The names are fast.bend's, so a program switches by its import line alone (the Makefile builds the GPU demos so).
#
# Each bang is one of tensor-gpu's run1..run4 (outputs in place) or run1_new..run4_new (outputs made on the device):
# a tree of leaves, every node sharing the arrays. The code of a leaf is below, in plain Bend, and computes each number
# with main.bend's own expression or row code (rmsf_row, rope_one, qrow, prow2, drow, gr2, dot0, adam_m..), so the bits
# are main.bend's, except where an exponential is taken on the device (SiLU, attention's softmax, cross-entropy): CUDA
# takes the float overload of exp there and its last bit can differ from the host's (bend-ml
# docs/upstream/gpu-math-overloads.md); 1 ulp is accepted there, my decision of 2026-10-18 (tests: demo/gpu_test.bend
# and demo/gpu_layers_test.bend). Sizes, depths and capacities are
# computed on the host and passed in. The arrays stay on the device from one bang to the next as long as the host only
# passes their handles; a bang that follows the host's parallel lets in one evaluation runs on the CPU threads.
#
# Outside the proofs, on purpose, like fast.bend: tensor-gpu shares arrays between the leaves of a bang through Base's
# @unsafe Array.fork, so `bend` names the defs below that reach it, and no file with laws imports this one (`make laws`
# checks both). This file writes no @unsafe of its own. All of it is Bend.

import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import bend-ml-tensor-gpu@0.1.2.1/main.bend as TG
import ./main.bend as T

# ---- Linear: y = x·Wᵀ + b ----

def lin_fwd(+n: Nat, +i: Nat, +o: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>) -> T.Fwd<T.Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match p:
    case T.Lin{w, b}:
      T.lin_f2(n, i, o, b, TG.matmul_nt(n, i, o, 0n, x, w))

def lin_b2(+n: Nat, +i: Nat, +o: Nat, b: TA.Mat<1n, o>, x: TA.Mat<n, i>, g: T.Lin<i, o>, r: TA.MMul<n, o, i>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match g r:
    case T.Lin{gw, gb} TA.MMul{dy, w, dx}:
      T.lin_b3(n, i, o, w, b, dx, gb, TG.matmul_tn_acc(o, n, i, 0n, dy, x, gw))

def lin_bwd(+n: Nat, +i: Nat, +o: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: T.Lin<i, o>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match p:
    case T.Lin{w, b}:
      lin_b2(n, i, o, b, x, g, TG.matmul_precise(n, o, i, 0n, dy, w))

# =====================================================================================================
# The leaves' shared pieces. A leaf gets its index t among l leaves, a record of scalars and four input arrays, and
# returns its outputs; an input it does not use is tensor-gpu's none(Unit{}). Element-wise leaves take the cells t,
# t + l, .. (TG.share), so neighbouring lanes read neighbouring cells; row and column leaves take one row or column
# each and leave their outputs unchanged past the last one.
# =====================================================================================================

type EN is Data:
  EN{n: U32}

def nn(u: Unit) -> Array<F32>:
  TG.none(u)

# out[i] = f(x[i]) for the leaf's cells
def ew1_loop(~f: @+a: F32 -> F32, k: Nat, +i: U32, +l: U32, o: Array<F32>, r: Array<F32> & F32) -> Array<F32>:
  match k r:
    case 0n Tuple{a, +x}: o
    case 1n+q Tuple{a, +x}: ew1_loop(~f, q, (i + l : U32), l, Array.set(F32, o, i, f(x)), Array.get(F32, a, (i + l : U32)))

def ew1(~f: @+a: F32 -> F32, +t: U32, +l: U32, +e: EN, x: Array<F32>, o: Array<F32>) -> Array<F32>:
  match e:
    case EN{+n}: ew1_loop(~f, TG.share(t, n, l), t, l, o, Array.get(F32, x, t))

# out[i] = f(x[i], y[i]) for the leaf's cells
def ew2_loop(~f: @+a: F32 -> @+b: F32 -> F32, k: Nat, +i: U32, +l: U32, o: Array<F32>, ra: Array<F32> & F32, rb: Array<F32> & F32) -> Array<F32>:
  match k ra rb:
    case 0n Tuple{a, +x} Tuple{b, +y}: o
    case 1n+q Tuple{a, +x} Tuple{b, +y}: ew2_loop(~f, q, (i + l : U32), l, Array.set(F32, o, i, f(x, y)), Array.get(F32, a, (i + l : U32)), Array.get(F32, b, (i + l : U32)))

def ew2(~f: @+a: F32 -> @+b: F32 -> F32, +t: U32, +l: U32, +e: EN, x: Array<F32>, y: Array<F32>, o: Array<F32>) -> Array<F32>:
  match e:
    case EN{+n}: ew2_loop(~f, TG.share(t, n, l), t, l, o, Array.get(F32, x, t), Array.get(F32, y, t))

# o[i] = f(o[i], y[i]) for the leaf's cells, in place (o's old value is the first argument)
def ewi_loop(~f: @+a: F32 -> @+b: F32 -> F32, k: Nat, +i: U32, +l: U32, ro: Array<F32> & F32, ry: Array<F32> & F32) -> Array<F32>:
  match k ro ry:
    case 0n Tuple{o, +x} Tuple{y, +v}: o
    case 1n+q Tuple{o, +x} Tuple{y, +v}: ewi_loop(~f, q, (i + l : U32), l, Array.get(F32, Array.set(F32, o, i, f(x, v)), (i + l : U32)), Array.get(F32, y, (i + l : U32)))

def ewi(~f: @+a: F32 -> @+b: F32 -> F32, +t: U32, +l: U32, +e: EN, o: Array<F32>, y: Array<F32>) -> Array<F32>:
  match e:
    case EN{+n}: ewi_loop(~f, TG.share(t, n, l), t, l, Array.get(F32, o, t), Array.get(F32, y, t))

def f32_id(+x: F32) -> F32:
  x

def lf_copy(+t: U32, +l: U32, +e: EN, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  ew1(~f32_id, t, l, e, x1, o)

def lf_silu(+t: U32, +l: U32, +e: EN, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  ew1(~T.silu_v, t, l, e, x1, o)

# o = dy, written in place, x1 = x: main.bend's silu_d(dy, x)
def lf_silud(+t: U32, +l: U32, +e: EN, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  ewi(~T.silu_d, t, l, e, o, x1)

# o = a, written in place, x1 = b: a + b
def lf_add(+t: U32, +l: U32, +e: EN, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  ewi(~T.f32_add, t, l, e, o, x1)

def lf_mul(+t: U32, +l: U32, +e: EN, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  ew2(~T.f32_mul, t, l, e, x1, x2, o)

# x1 = dy; the saved a and b, written in place: a becomes db = dy * a and b becomes da = dy * b, in one bang
def mulb_loop(k: Nat, +i: U32, +l: U32, rd: Array<F32> & F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> TG.O2:
  match k rd ra rb:
    case 0n Tuple{d, +x} Tuple{a, +y} Tuple{b, +z}: TG.O2{a, b}
    case 1n+q Tuple{d, +x} Tuple{a, +y} Tuple{b, +z}:
      +u = (i + l : U32)
      mulb_loop(q, u, l, Array.get(F32, d, u), Array.get(F32, Array.set(F32, a, i, (x * y : F32)), u), Array.get(F32, Array.set(F32, b, i, (x * z : F32)), u))

def lf_mulb(+t: U32, +l: U32, +e: EN, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O2) -> TG.O2:
  match e o:
    case EN{+n} TG.O2{a, b}: mulb_loop(TG.share(t, n, l), t, l, Array.get(F32, x1, t), Array.get(F32, a, t), Array.get(F32, b, t))

def en(+cnt: Nat) -> EN:
  EN{U32.from_nat(cnt)}

def r1_a(r: TG.R1) -> Array<F32>:
  match r:
    case TG.R1{x1, x2, x3, x4, a}: a

# ---- element-wise layers ----

def dup_fin(-r: Nat, -c: Nat, q: TG.R1) -> T.Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  match q:
    case TG.R1{x1, x2, x3, x4, a}: T.Two{TA.Mat{x1}, TA.Mat{a}}

# two copies of a matrix, the second made on the device
def Mat.dup(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> T.Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  match m:
    case TA.Mat{a}: dup_fin(r, c, TG.run1_new(~EN, ~lf_copy, TG.depth(Nat.mul(r, c)), en(Nat.mul(r, c)), a, nn(Unit{}), nn(Unit{}), nn(Unit{}), TA.cap_depth(Nat.mul(r, c))))

# a + b, as main.bend's Mat.sum2 adds each cell, written over a (both are consumed)
def Mat.sum2(+r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.Mat<r, c>:
  match a b:
    case TA.Mat{x} TA.Mat{y}: TA.Mat{r1_a(TG.run1(~EN, ~lf_add, TG.depth(Nat.mul(r, c)), en(Nat.mul(r, c)), y, nn(Unit{}), nn(Unit{}), nn(Unit{}), x))}

def silu_f2(-n: Nat, -c: Nat, q: TG.R1) -> T.Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match q:
    case TG.R1{x1, x2, x3, x4, a}: T.Fwd{Unit{}, TA.Mat{a}, TA.Mat{x1}}

def silu_f(+n: Nat, +c: Nat, x: TA.Mat<n, c>) -> T.Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match x:
    case TA.Mat{a}: silu_f2(n, c, TG.run1_new(~EN, ~lf_silu, TG.depth(Nat.mul(n, c)), en(Nat.mul(n, c)), a, nn(Unit{}), nn(Unit{}), nn(Unit{}), TA.cap_depth(Nat.mul(n, c))))

def silu_b(+n: Nat, +c: Nat, s: TA.Mat<n, c>, dy: TA.Mat<n, c>) -> T.Bwd<Unit, TA.Mat<n, c>>:
  match s dy:
    case TA.Mat{x} TA.Mat{d}: T.Bwd{Unit{}, TA.Mat{r1_a(TG.run1(~EN, ~lf_silud, TG.depth(Nat.mul(n, c)), en(Nat.mul(n, c)), x, nn(Unit{}), nn(Unit{}), nn(Unit{}), d))}, Unit{}}

# SiLU: y = silu(x), the input saved; dx = silu'(x) * dy
def SiLU(+n: Nat, +c: Nat) -> T.Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  T.Layer{p => x => silu_f(n, c, x), p => s => dy => g => silu_b(n, c, s, dy), op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

def mul2_f2(-n: Nat, -c: Nat, q: TG.R1) -> T.Fwd<Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match q:
    case TG.R1{x1, x2, x3, x4, y}: T.Fwd{Unit{}, TA.Mat{y}, T.Two{TA.Mat{x1}, TA.Mat{x2}}}

def mul2_f(+n: Nat, +c: Nat, x: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Fwd<Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match x:
    case T.Two{TA.Mat{a}, TA.Mat{b}}: mul2_f2(n, c, TG.run1_new(~EN, ~lf_mul, TG.depth(Nat.mul(n, c)), en(Nat.mul(n, c)), a, b, nn(Unit{}), nn(Unit{}), TA.cap_depth(Nat.mul(n, c))))

def mul2_b2(-n: Nat, -c: Nat, q: TG.R2) -> T.Bwd<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match q:
    case TG.R2{x1, x2, x3, x4, TG.O2{db, da}}: T.Bwd{Unit{}, T.Two{TA.Mat{da}, TA.Mat{db}}, Unit{}}

def mul2_b(+n: Nat, +c: Nat, s: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, dy: TA.Mat<n, c>) -> T.Bwd<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match s dy:
    case T.Two{TA.Mat{a}, TA.Mat{b}} TA.Mat{d}: mul2_b2(n, c, TG.run2(~EN, ~lf_mulb, TG.depth(Nat.mul(n, c)), en(Nat.mul(n, c)), d, nn(Unit{}), nn(Unit{}), nn(Unit{}), a, b))

# Mul2: y = a * b, the inputs saved; da = dy * b, db = dy * a
def Mul2(+n: Nat, +c: Nat) -> T.Layer<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  T.Layer{p => x => mul2_f(n, c, x), p => s => dy => g => mul2_b(n, c, s, dy), op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

def add2_f(+n: Nat, +c: Nat, x: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Fwd<Unit, TA.Mat<n, c>, Unit>:
  match x:
    case T.Two{a, b}: T.Fwd{Unit{}, Mat.sum2(n, c, a, b), Unit{}}

def Add2(+n: Nat, +c: Nat) -> T.Layer<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, Unit>:
  T.Layer{p => x => add2_f(n, c, x), p => s => dy => g => T.Bwd{p, Mat.dup(n, c, dy), g}, op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

# T.Residual with these copies and sums
def Residual(~n: Nat, ~c: Nat, ~P: Type, ~S: Type, ~f: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, c>, S>) -> T.Layer<T.Two<T.Two<Unit, P>, Unit>, TA.Mat<n, c>, TA.Mat<n, c>, T.Two<T.Two<Unit, S>, Unit>>:
  T.Seq(~T.Two<Unit, P>, ~Unit, ~TA.Mat<n, c>, ~T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, ~TA.Mat<n, c>, ~T.Two<Unit, S>, ~Unit, ~T.Fork(~Unit, ~P, ~TA.Mat<n, c>, ~TA.Mat<n, c>, ~TA.Mat<n, c>, ~Unit, ~S, ~(m => Mat.dup(n, c, m)), ~(a => b => Mat.sum2(n, c, a, b)), ~T.Id(~TA.Mat<n, c>), ~f), ~Add2(n, c))

# ---- the optimizer step in place, each cell as main.bend's sgd_l, mom_l and adam_l compute it ----

type ESGD is Data:
  ESGD{n: U32, lr: F32}

def sgd_loop(k: Nat, +i: U32, +l: U32, +lr: F32, rw: Array<F32> & F32, rg: Array<F32> & F32) -> TG.O2:
  match k rw rg:
    case 0n Tuple{w, +x} Tuple{g, +y}: TG.O2{w, g}
    case 1n+q Tuple{w, +x} Tuple{g, +y}:
      sgd_loop(q, (i + l : U32), l, lr, Array.get(F32, Array.set(F32, w, i, (x + (F32.neg(lr) * y : F32) : F32)), (i + l : U32)), Array.get(F32, Array.set(F32, g, i, 0.0), (i + l : U32)))

def lf_sgd(+t: U32, +l: U32, +e: ESGD, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O2) -> TG.O2:
  match e o:
    case ESGD{+n, +lr} TG.O2{w, g}: sgd_loop(TG.share(t, n, l), t, l, lr, Array.get(F32, w, t), Array.get(F32, g, t))

type EMOM is Data:
  EMOM{n: U32, lr: F32, mu: F32}

def mom_loop(k: Nat, +i: U32, +l: U32, +lr: F32, +mu: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<F32> & F32) -> TG.O3:
  match k rw rg rm:
    case 0n Tuple{w, +x} Tuple{g, +y} Tuple{m, +b}: TG.O3{w, g, m}
    case 1n+q Tuple{w, +x} Tuple{g, +y} Tuple{m, +b}:
      +nb = ((b * mu : F32) + y : F32)
      +u = (i + l : U32)
      mom_loop(q, u, l, lr, mu, Array.get(F32, Array.set(F32, w, i, (x + (F32.neg(lr) * nb : F32) : F32)), u), Array.get(F32, Array.set(F32, g, i, 0.0), u), Array.get(F32, Array.set(F32, m, i, nb), u))

def lf_mom(+t: U32, +l: U32, +e: EMOM, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O3) -> TG.O3:
  match e o:
    case EMOM{+n, +lr, +mu} TG.O3{w, g, m}: mom_loop(TG.share(t, n, l), t, l, lr, mu, Array.get(F32, w, t), Array.get(F32, g, t), Array.get(F32, m, t))

type EADAM is Data:
  EADAM{n: U32, lr: F32, c1: F32, b2: F32, c2: F32, eps: F32, wd: F32, dec: Bool, step: F32, bc2: F32}

def adam_loop(k: Nat, +i: U32, +l: U32, +lr: F32, +c1: F32, +b2: F32, +c2: F32, +eps: F32, +wd: F32, +dec: Bool, +step: F32, +bc2: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<F32> & F32, rv: Array<F32> & F32) -> TG.O4:
  match k rw rg rm rv:
    case 0n Tuple{w, +x} Tuple{g, +y} Tuple{m, +a} Tuple{v, +b}: TG.O4{w, g, m, v}
    case 1n+q Tuple{w, +x} Tuple{g, +y} Tuple{m, +a} Tuple{v, +b}:
      +gg = T.adam_grad(dec, wd, x, y)
      +nm = T.adam_m(c1, a, gg)
      +nv = T.adam_v(b2, c2, b, gg)
      +nw = T.adam_w(step, bc2, eps, T.adam_decay(dec, lr, wd, x), nm, nv)
      +u = (i + l : U32)
      adam_loop(q, u, l, lr, c1, b2, c2, eps, wd, dec, step, bc2, Array.get(F32, Array.set(F32, w, i, nw), u), Array.get(F32, Array.set(F32, g, i, 0.0), u), Array.get(F32, Array.set(F32, m, i, nm), u), Array.get(F32, Array.set(F32, v, i, nv), u))

def lf_adam(+t: U32, +l: U32, +e: EADAM, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O4) -> TG.O4:
  match e o:
    case EADAM{+n, +lr, +c1, +b2, +c2, +eps, +wd, +dec, +step, +bc2} TG.O4{w, g, m, v}: adam_loop(TG.share(t, n, l), t, l, lr, c1, b2, c2, eps, wd, dec, step, bc2, Array.get(F32, w, t), Array.get(F32, g, t), Array.get(F32, m, t), Array.get(F32, v, t))

def sgd_fin(-r: Nat, -c: Nat, q: TG.R2) -> T.MatStep<r, c>:
  match q:
    case TG.R2{x1, x2, x3, x4, TG.O2{w, g}}: T.MatStep{TA.Mat{w}, TA.Mat{g}, T.NoMom{}}

def mom_fin(-r: Nat, -c: Nat, q: TG.R3) -> T.MatStep<r, c>:
  match q:
    case TG.R3{x1, x2, x3, x4, TG.O3{w, g, m}}: T.MatStep{TA.Mat{w}, TA.Mat{g}, T.OneMom{TA.Mat{m}}}

def opt_fin(-r: Nat, -c: Nat, q: TG.R4) -> T.MatStep<r, c>:
  match q:
    case TG.R4{x1, x2, x3, x4, TG.O4{w, g, m, v}}: T.MatStep{TA.Mat{w}, TA.Mat{g}, T.TwoMom{TA.Mat{m}, TA.Mat{v}}}

# one matrix's step, in place: SGD writes two arrays, momentum three, Adam four
def opt_mat(+r: Nat, +c: Nat, +o: T.Opt, w: TA.Mat<r, c>, g: TA.Mat<r, c>, st: T.OptState<TA.Mat<r, c>>) -> T.MatStep<r, c>:
  match o w g st:
    case T.SGD{+lr} TA.Mat{wa} TA.Mat{ga} _:
      sgd_fin(r, c, TG.run2(~ESGD, ~lf_sgd, TG.depth(Nat.mul(r, c)), ESGD{U32.from_nat(Nat.mul(r, c)), lr}, nn(Unit{}), nn(Unit{}), nn(Unit{}), nn(Unit{}), wa, ga))
    case T.Momentum{+lr, +mu} TA.Mat{wa} TA.Mat{ga} T.OneMom{TA.Mat{ma}}:
      mom_fin(r, c, TG.run3(~EMOM, ~lf_mom, TG.depth(Nat.mul(r, c)), EMOM{U32.from_nat(Nat.mul(r, c)), lr, mu}, nn(Unit{}), nn(Unit{}), nn(Unit{}), nn(Unit{}), wa, ga, ma))
    case T.Adam{+lr, +b1, +b2, +eps, +wd, +dec, +step, +bc2, +c1, +c2} TA.Mat{wa} TA.Mat{ga} T.TwoMom{TA.Mat{ma}, TA.Mat{va}}:
      opt_fin(r, c, TG.run4(~EADAM, ~lf_adam, TG.depth(Nat.mul(r, c)), EADAM{U32.from_nat(Nat.mul(r, c)), lr, c1, b2, c2, eps, wd, dec, step, bc2}, nn(Unit{}), nn(Unit{}), nn(Unit{}), nn(Unit{}), wa, ga, ma, va))
    case _ w g st: T.MatStep{w, g, st}

def mat_step(+r: Nat, +c: Nat, +op: T.Opt, p: TA.Mat<r, c>, g: TA.Mat<r, c>, st: T.OptState<TA.Mat<r, c>>) -> T.Step<TA.Mat<r, c>>:
  T.mat_step_fin(r, c, opt_mat(r, c, op, p, g, st))

def lin_step2(+i: Nat, +o: Nat, +op: T.Opt, p: T.Lin<i, o>, g: T.Lin<i, o>, st: T.Two<T.OptState<TA.Mat<o, i>>, T.OptState<TA.Mat<1n, o>>>) -> T.Step<T.Lin<i, o>>:
  match p g st:
    case T.Lin{w, b} T.Lin{gw, gb} T.Two{sw, sb}:
      T.lin_s2(i, o, opt_mat(o, i, op, w, gw, sw), opt_mat(1n, o, op, b, gb, sb))

def lin_step(+i: Nat, +o: Nat, +op: T.Opt, p: T.Lin<i, o>, g: T.Lin<i, o>, st: T.OptState<T.Lin<i, o>>) -> T.Step<T.Lin<i, o>>:
  lin_step2(i, o, op, p, g, T.OptState.split(TA.Mat<o, i>, TA.Mat<1n, o>, T.OptState.map(T.Lin<i, o>, T.Two<TA.Mat<o, i>, TA.Mat<1n, o>>, st, x => T.lin_pair(i, o, x), x => T.lin_pair(i, o, x))))

# ---- the linear layers ----

def LinearF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<T.Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{p => x => lin_fwd(n, i, o, p, x), p => s => dy => g => lin_bwd(n, i, o, p, s, dy, g), op => p => g => st => lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => off => T.lin_load(i, o, p, xs, off), p => xs => off => T.lin_store(i, o, p, xs, off), u => U32.from_nat(Nat.mul(o, 1n+i))}

# LinearFirst: no dX (the first layer's input needs no gradient)
def linf_bwd(+n: Nat, +i: Nat, +o: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: T.Lin<i, o>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match p g:
    case T.Lin{w, b} T.Lin{gw, gb}:
      T.linf_b3(n, i, o, w, b, gb, TG.matmul_tn_acc(o, n, i, 0n, dy, x, gw))

def LinearFirstF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<T.Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{p => x => lin_fwd(n, i, o, p, x), p => s => dy => g => linf_bwd(n, i, o, p, s, dy, g), op => p => g => st => lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => off => T.lin_load(i, o, p, xs, off), p => xs => off => T.lin_store(i, o, p, xs, off), u => U32.from_nat(Nat.mul(o, 1n+i))}

# LinearNB: y = x·Wᵀ, no bias (Llama's projections)
def lnb_b2(+n: Nat, +i: Nat, +o: Nat, x: TA.Mat<n, i>, g: TA.Mat<o, i>, r: TA.MMul<n, o, i>) -> T.Bwd<TA.Mat<o, i>, TA.Mat<n, i>>:
  match r:
    case TA.MMul{dy, w, dx}:
      T.lnb_b3(n, i, o, w, dx, TG.matmul_tn_acc(o, n, i, 0n, dy, x, g))

def LinearNBF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<TA.Mat<o, i>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{w => x => T.lnb_f2(n, i, o, TG.matmul_nt(n, i, o, 0n, x, w)), w => s => dy => g => lnb_b2(n, i, o, s, g, TG.matmul_precise(n, o, i, 0n, dy, w)), op => p => g => st => mat_step(o, i, op, p, g, st), u => TA.Mat.zeros(o, i), p => T.Mat.dump(o, i, p), p => xs => off => T.Mat.restore(o, i, p, xs, off), p => xs => off => T.Mat.store(o, i, p, xs, off), u => U32.from_nat(Nat.mul(o, i))}

# ---- RMSNorm: a leaf a row with main.bend's own row code (rmsf_row) ----

type ERF is Data:
  ERF{n: U32, d: Nat, d32: U32, df: F32, eps: F32}

def r3_y(s: T.R3) -> Array<F32>:
  match s:
    case T.R3{x, g, y}: y

def rmsf_if(ok: Bool, +t: U32, +d: Nat, +d32: U32, +df: F32, +eps: F32, x: Array<F32>, g: Array<F32>, y: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: r3_y(T.rmsf_row(d, (t * d32 : U32), df, eps, T.R3{x, g, y}))
    case False{}: y

# x1 = x, x2 = the gain g; y a row a leaf
def lf_rmsf(+t: U32, +l: U32, +e: ERF, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, y: Array<F32>) -> Array<F32>:
  match e:
    case ERF{+n, +d, +d32, +df, +eps}: rmsf_if(U32.is_lt(t, n), t, d, d32, df, eps, x1, x2, y)

def erf(+n: Nat, +d: Nat, +eps: F32) -> ERF:
  ERF{U32.from_nat(n), d, U32.from_nat(d), F32.from_nat(d), eps}

def rms_f2(-n: Nat, -d: Nat, q: TG.R1) -> T.Fwd<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  match q:
    case TG.R1{x1, x2, x3, x4, y}: T.Fwd{TA.Mat{x2}, TA.Mat{y}, TA.Mat{x1}}

def rms_f(+n: Nat, +d: Nat, +eps: F32, p: TA.Mat<1n, d>, x: TA.Mat<n, d>) -> T.Fwd<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  match p x:
    case TA.Mat{g} TA.Mat{a}: rms_f2(n, d, TG.run1_new(~ERF, ~lf_rmsf, TA.cap_depth(n), erf(n, d, eps), a, g, nn(Unit{}), nn(Unit{}), TA.cap_depth(Nat.mul(n, d))))

# the backward pass in two bangs: a leaf a row writes dx and the row's 1/rms (main.bend's rmsb_row without dg), then a
# leaf a column sums dg down the rows in order from +0.0 and adds it to the gradient it is given, as main.bend adds each
# row's term into dg

def gdx_w(k: Nat, +j: U32, +off: U32, +r: F32, +m: F32, dx: Array<F32>, rx: Array<F32> & F32, rd: Array<F32> & F32, rg: Array<F32> & F32) -> Array<F32>:
  match k rx rd rg:
    case 0n Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      +xh = (a * r : F32)
      Array.set(F32, dx, (off + j : U32), (((b * c : F32) - (xh * m : F32) : F32) * r : F32))
    case 1n+p Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      +xh = (a * r : F32)
      gdx_w(p, (j + 1 : U32), off, r, m, Array.set(F32, dx, (off + j : U32), (((b * c : F32) - (xh * m : F32) : F32) * r : F32)), Array.get(F32, x, ((off + j : U32) + 1 : U32)), Array.get(F32, dy, ((off + j : U32) + 1 : U32)), Array.get(F32, g, (j + 1 : U32)))

def gdx_3(+d: Nat, +off: U32, +df: F32, +r: F32, +t: U32, dx: Array<F32>, rr: Array<F32>, q: T.R3S) -> TG.O2:
  match q:
    case T.R3S{x, dy, g, +s}:
      TG.O2{gdx_w(Nat.sub(d, 1n), 0, off, r, (s / df : F32), dx, Array.get(F32, x, off), Array.get(F32, dy, off), Array.get(F32, g, 0)), Array.set(F32, rr, t, r)}

def gdx_2(+d: Nat, +off: U32, +df: F32, +eps: F32, +t: U32, dy: Array<F32>, g: Array<F32>, dx: Array<F32>, rr: Array<F32>, q: T.AS) -> TG.O2:
  match q:
    case T.AS{x, +s}:
      +r = T.rms_inv(s, df, eps)
      gdx_3(d, off, df, r, t, dx, rr, T.rmsb_dot(Nat.sub(d, 1n), 0, off, r, 0.0, Array.get(F32, x, off), Array.get(F32, dy, off), Array.get(F32, g, 0)))

def gdx_if(ok: Bool, +t: U32, +d: Nat, +d32: U32, +df: F32, +eps: F32, x: Array<F32>, dy: Array<F32>, g: Array<F32>, dx: Array<F32>, rr: Array<F32>) -> TG.O2:
  match ok:
    case True{}: gdx_2(d, (t * d32 : U32), df, eps, t, dy, g, dx, rr, T.ssq(Nat.sub(d, 1n), (t * d32 : U32), 0.0, Array.get(F32, x, (t * d32 : U32))))
    case False{}: TG.O2{dx, rr}

# x1 = x, x2 = dy, x3 = g; outputs dx and 1/rms of each row
def lf_gdx(+t: U32, +l: U32, +e: ERF, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O2) -> TG.O2:
  match e o:
    case ERF{+n, +d, +d32, +df, +eps} TG.O2{dx, rr}: gdx_if(U32.is_lt(t, n), t, d, d32, df, eps, x1, x2, x3, dx, rr)

# acc + dy[i]·(x[i]·r[row]) down a column, i from j by d
def gdg_loop(k: Nat, +i: U32, +d: U32, +row: U32, acc: F32, rx: Array<F32> & F32, rd: Array<F32> & F32, rr: Array<F32> & F32) -> F32:
  match k rx rd rr:
    case 0n Tuple{x, +a} Tuple{dy, +b} Tuple{r, +c}: (acc + (b * (a * c : F32) : F32) : F32)
    case 1n+p Tuple{x, +a} Tuple{dy, +b} Tuple{r, +c}:
      gdg_loop(p, (i + d : U32), d, (row + 1 : U32), (acc + (b * (a * c : F32) : F32) : F32), Array.get(F32, x, (i + d : U32)), Array.get(F32, dy, (i + d : U32)), Array.get(F32, r, (row + 1 : U32)))

def gdg_put(+j: U32, +s: F32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{ga, +v}: Array.set(F32, ga, j, (v + s : F32))

type EDG is Data:
  EDG{d: U32, n: Nat}

def gdg_if(ok: Bool, +j: U32, +n: Nat, +d: U32, x: Array<F32>, dy: Array<F32>, rr: Array<F32>, ga: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: gdg_put(j, gdg_loop(Nat.sub(n, 1n), j, d, 0, 0.0, Array.get(F32, x, j), Array.get(F32, dy, j), Array.get(F32, rr, 0)), Array.get(F32, ga, j))
    case False{}: ga

# x1 = x, x2 = dy, x3 = 1/rms; the gradient of the gain, in place: a leaf a column
def lf_gdg(+t: U32, +l: U32, +e: EDG, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, ga: Array<F32>) -> Array<F32>:
  match e:
    case EDG{+d, +n}: gdg_if(U32.is_lt(t, d), t, n, d, x1, x2, x3, ga)

def rms_b3(-n: Nat, -d: Nat, g: Array<F32>, dx: Array<F32>, q: TG.R1) -> T.Bwd<TA.Mat<1n, d>, TA.Mat<n, d>>:
  match q:
    case TG.R1{x1, x2, x3, x4, ga}: T.Bwd{TA.Mat{g}, TA.Mat{dx}, TA.Mat{ga}}

def rms_b2(+n: Nat, +d: Nat, ga: Array<F32>, q: TG.R2) -> T.Bwd<TA.Mat<1n, d>, TA.Mat<n, d>>:
  match q:
    case TG.R2{x, dy, g, x4, TG.O2{dx, rr}}: rms_b3(n, d, g, dx, TG.run1(~EDG, ~lf_gdg, TA.cap_depth(d), EDG{U32.from_nat(d), n}, x, dy, rr, x4, ga))

def rms_b(+n: Nat, +d: Nat, +eps: F32, p: TA.Mat<1n, d>, x: TA.Mat<n, d>, dy: TA.Mat<n, d>, gacc: TA.Mat<1n, d>) -> T.Bwd<TA.Mat<1n, d>, TA.Mat<n, d>>:
  match p x dy gacc:
    case TA.Mat{g} TA.Mat{xa} TA.Mat{da} TA.Mat{ga}: rms_b2(n, d, ga, TG.run2_new(~ERF, ~lf_gdx, TA.cap_depth(n), erf(n, d, eps), xa, da, g, nn(Unit{}), TA.cap_depth(Nat.mul(n, d)), TA.cap_depth(n)))

# RMSNorm over the last dimension, its gain trained (main.bend's RMSNorm)
def RMSNorm(+n: Nat, +d: Nat, +eps: F32) -> T.Layer<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  T.Layer{p => x => rms_f(n, d, eps, p, x), p => s => dy => g => rms_b(n, d, eps, p, s, dy, g), op => p => g => st => mat_step(1n, d, op, p, g, st), u => TA.Mat.zeros(1n, d), p => T.Mat.dump(1n, d, p), p => xs => off => T.Mat.restore(1n, d, p, xs, off), p => xs => off => T.Mat.store(1n, d, p, xs, off), u => U32.from_nat(d)}

# ---- RoPE: the tables on the host (main.bend's rope_tab, so the host's cos and sin), each pair rotated in place by
# main.bend's rope_one, the pairs dealt to the leaves round-robin ----

type ERO is Data:
  ERO{n: U32, fwd: Bool, hh: U32, half: U32, w: U32, t: U32}

def rope_loop(k: Nat, +e: U32, +l: U32, +fwd: Bool, +hh: U32, +half: U32, +w: U32, +t: U32, st: T.RS3) -> T.RS3:
  match k:
    case 0n: st
    case 1n+q: rope_loop(q, (e + l : U32), l, fwd, hh, half, w, t, T.rope_one(e, fwd, hh, half, w, t, st))

def rs3_x(st: T.RS3) -> Array<F32>:
  match st:
    case T.RS3{x, c, s}: x

# x1 = cos, x2 = sin; x rotated in place
def lf_rope(+t: U32, +l: U32, +e: ERO, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, x: Array<F32>) -> Array<F32>:
  match e:
    case ERO{+n, +fwd, +hh, +half, +w, +tt}: rs3_x(rope_loop(TG.share(t, n, l), t, l, fwd, hh, half, w, tt, T.RS3{x, x1, x2}))

def rope_go2(+n: Nat, +w: Nat, +fwd: Bool, +t: Nat, +h: Nat, +half: Nat, x: Array<F32>, tb: T.CS) -> TA.Mat<n, w>:
  match tb:
    case T.CS{cs, sn}: TA.Mat{r1_a(TG.run1(~ERO, ~lf_rope, TG.depth(Nat.mul(n, Nat.mul(h, half))), ERO{U32.from_nat(Nat.mul(n, Nat.mul(h, half))), fwd, U32.from_nat(Nat.mul(h, half)), U32.from_nat(half), U32.from_nat(w), U32.from_nat(t)}, cs, sn, nn(Unit{}), nn(Unit{}), x))}

def rope_go(+n: Nat, +w: Nat, +fwd: Bool, +t: Nat, +h: Nat, +half: Nat, +base: F32, m: TA.Mat<n, w>) -> TA.Mat<n, w>:
  match m:
    case TA.Mat{x}: rope_go2(n, w, fwd, t, h, half, x, T.rope_tab(Nat.sub(Nat.mul(t, half), 1n), 0, U32.from_nat(half), base, T.CS{T.zeros_a(Nat.mul(t, half)), T.zeros_a(Nat.mul(t, half))}))

# RoPE on n = b·t rows of h heads of 2·half (main.bend's RoPE)
def RoPE(+n: Nat, +w: Nat, +t: Nat, +h: Nat, +half: Nat, +base: F32, -e: {w == Nat.mul(h, Nat.add(half, half)) : Nat}) -> T.Layer<Unit, TA.Mat<n, w>, TA.Mat<n, w>, Unit>:
  T.Layer{p => x => T.Fwd{p, rope_go(n, w, True{}, t, h, half, base, x), Unit{}}, p => s => dy => g => T.Bwd{p, rope_go(n, w, False{}, t, h, half, base, dy), g}, op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

# ---- attention. A leaf per (sequence, query head, position) for the forward pass and the first half of the backward
# pass, with main.bend's own row code (qrow, prow2, drow, gr2, dot0), each head's P and G rows kept at their place in
# one array of b·h·t·t; then a leaf per (sequence, key/value head, position) for dK and dV, each head's sum from dot0,
# added over the group's heads in head order from +0.0, as main.bend's pacc. ----

type EAT is Data:
  EAT{n: U32, t: Nat, t32: U32, h: U32, g: U32, hd: Nat, hd32: U32, w: U32, kvw: U32, sc: F32}

# the leaf's place: sequence bi, head hi, position i; P's row from off, q's from qb (row stride w), k's and v's from
# kb (stride kvw), o's from ob
type AP is Data:
  AP{i: Nat, off: U32, qb: U32, kb: U32, ob: U32}

def gap(+e: U32, +t: U32, +h: U32, +g: U32, +hd: U32, +w: U32, +kvw: U32) -> AP:
  +bi = U32.div(e, (h * t : U32))
  +hi = U32.mod(U32.div(e, t), h)
  +i = U32.mod(e, t)
  AP{U32.to_nat(i), (e * t : U32), (((bi * t : U32) * w : U32) + (hi * hd : U32) : U32), (((bi * t : U32) * kvw : U32) + (U32.div(hi, g) * hd : U32) : U32), ((((bi * t : U32) + i : U32) * w : U32) + (hi * hd : U32) : U32)}

# o[ob + dd] = P's row (t from off) · V's column kb + dd, dd = 0..k
def gaf_o(k: Nat, +dd: U32, +t: Nat, +off: U32, +kb: U32, +kvw: U32, +ob: U32, o: Array<F32>, r: T.D2) -> TG.O2:
  match k r:
    case 0n T.D2{p, v, +s}: TG.O2{p, Array.set(F32, o, (ob + dd : U32), s)}
    case 1n+q T.D2{p, v, +s}: gaf_o(q, (dd + 1 : U32), t, off, kb, kvw, ob, Array.set(F32, o, (ob + dd : U32), s), T.dot0(t, off, (kb + (dd + 1 : U32) : U32), 1, kvw, p, v))

def gaf_2(+t: Nat, +hd: Nat, +off: U32, +kb: U32, +kvw: U32, +ob: U32, o: Array<F32>, v: Array<F32>, r: T.PR) -> TG.O2:
  match r:
    case T.PR{q, k, p, n}: gaf_o(Nat.sub(hd, 1n), 0, t, off, kb, kvw, ob, o, T.dot0(t, off, kb, 1, kvw, p, v))

def gaf_if(ok: Bool, +a: AP, +t: Nat, +hd: Nat, +w: U32, +kvw: U32, +sc: F32, q: Array<F32>, k: Array<F32>, v: Array<F32>, p: Array<F32>, o: Array<F32>) -> TG.O2:
  match ok a:
    case True{} AP{+i, +off, +qb, +kb, +ob}: gaf_2(t, hd, off, kb, kvw, ob, o, v, T.prow2(sc, off, p, T.qrow(i, hd, w, qb, kvw, kb, q, k)))
    case False{} AP{+i, +off, +qb, +kb, +ob}: TG.O2{p, o}

# x1 = q, x2 = k, x3 = v; outputs P (every head's rows) and o
def lf_attf(+e: U32, +l: U32, +c: EAT, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O2) -> TG.O2:
  match c o:
    case EAT{+n, +t, +t32, +h, +g, +hd, +hd32, +w, +kvw, +sc} TG.O2{p, y}: gaf_if(U32.is_lt(e, n), gap(e, t32, h, g, hd32, w, kvw), t, hd, w, kvw, sc, x1, x2, x3, p, y)

# dq[ob + dd] = G's row (t from off) · K's column kb + dd, dd = 0..k
def gab_q(k: Nat, +dd: U32, +t: Nat, +off: U32, +kb: U32, +kvw: U32, +ob: U32, dq: Array<F32>, r: T.D2) -> TG.O2:
  match k r:
    case 0n T.D2{g, kk, +s}: TG.O2{Array.set(F32, dq, (ob + dd : U32), s), g}
    case 1n+q T.D2{g, kk, +s}: gab_q(q, (dd + 1 : U32), t, off, kb, kvw, ob, Array.set(F32, dq, (ob + dd : U32), s), T.dot0(t, off, (kb + (dd + 1 : U32) : U32), 1, kvw, g, kk))

def gab_o3(p: Array<F32>, r: TG.O2) -> TG.O3:
  match r:
    case TG.O2{dq, g}: TG.O3{p, g, dq}

def gab_3(+t: Nat, +hd: Nat, +off: U32, +kb: U32, +kvw: U32, +ob: U32, k: Array<F32>, dq: Array<F32>, r: T.GR) -> TG.O3:
  match r:
    case T.GR{d, v, T.A2{p, g}, n}: gab_o3(p, gab_q(Nat.sub(hd, 1n), 0, t, off, kb, kvw, ob, dq, T.dot0(t, off, kb, 1, kvw, g, k)))

def gab_2(+i: Nat, +t: Nat, +hd: Nat, +sc: F32, +off: U32, +kb: U32, +kvw: U32, +ob: U32, d: Array<F32>, v: Array<F32>, gb: Array<F32>, dq: Array<F32>, r: T.PR) -> TG.O3:
  match r:
    case T.PR{q, k, p, n}: gab_3(t, hd, off, kb, kvw, ob, k, dq, T.gr2(sc, off, p, gb, T.drow(i, hd, ob, kb, kvw, Nil{}, T.dot0(hd, ob, kb, 1, 1, d, v))))

def gab_if(ok: Bool, +a: AP, +t: Nat, +hd: Nat, +w: U32, +kvw: U32, +sc: F32, q: Array<F32>, k: Array<F32>, v: Array<F32>, d: Array<F32>, p: Array<F32>, gb: Array<F32>, dq: Array<F32>) -> TG.O3:
  match ok a:
    case True{} AP{+i, +off, +qb, +kb, +ob}: gab_2(i, t, hd, sc, off, kb, kvw, ob, d, v, gb, dq, T.prow2(sc, off, p, T.qrow(i, hd, w, qb, kvw, kb, q, k)))
    case False{} AP{+i, +off, +qb, +kb, +ob}: TG.O3{p, gb, dq}

# x1 = q, x2 = k, x3 = v, x4 = dO; outputs P and G (every head's rows) and dQ
def lf_attb(+e: U32, +l: U32, +c: EAT, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O3) -> TG.O3:
  match c o:
    case EAT{+n, +t, +t32, +h, +g, +hd, +hd32, +w, +kvw, +sc} TG.O3{p, gb, dq}: gab_if(U32.is_lt(e, n), gap(e, t32, h, g, hd32, w, kvw), t, hd, w, kvw, sc, x1, x2, x3, x4, p, gb, dq)

# acc + the group's heads' sums, in head order: a's column (t·t apart a head) · b's column (hd apart a head)
def gsum_h(k: Nat, +ia: U32, +ib: U32, +tt: Nat, +t: U32, +w: U32, +sa: U32, +sb: U32, acc: F32, r: T.D2) -> T.D2:
  match k r:
    case 0n T.D2{a, b, +s}: T.D2{a, b, (acc + s : F32)}
    case 1n+q T.D2{a, b, +s}: gsum_h(q, (ia + sa : U32), (ib + sb : U32), tt, t, w, sa, sb, (acc + s : F32), T.dot0(tt, (ia + sa : U32), (ib + sb : U32), t, w, a, b))

# o[ob + dd], dd = 0..k: dK = Σ Gᵀ·Q (a = G, b = Q) or dV = Σ Pᵀ·dO (a = P, b = dO) over the group's heads
def gkv_d(k: Nat, +dd: U32, +ia: U32, +ib: U32, +tt: Nat, +t: U32, +w: U32, +sa: U32, +sb: U32, +g1: Nat, +ob: U32, o: Array<F32>, r: T.D2) -> Array<F32>:
  match k r:
    case 0n T.D2{a, b, +s}: Array.set(F32, o, (ob + dd : U32), s)
    case 1n+q T.D2{a, b, +s}: gkv_d(q, (dd + 1 : U32), ia, ib, tt, t, w, sa, sb, g1, ob, Array.set(F32, o, (ob + dd : U32), s), gsum_h(g1, ia, (ib + (dd + 1 : U32) : U32), tt, t, w, sa, sb, 0.0, T.dot0(tt, ia, (ib + (dd + 1 : U32) : U32), t, w, a, b)))

type EKV is Data:
  EKV{n: U32, tt: Nat, t32: U32, kv: U32, g: Nat, g32: U32, hd: Nat, hd32: U32, w: U32, kvw: U32}

# leaf e: sequence bi, key/value head J, position j
type KP is Data:
  KP{ia: U32, ib: U32, ob: U32}

def gkp(+e: U32, +t: U32, +kv: U32, +g: U32, +hd: U32, +w: U32, +kvw: U32) -> KP:
  +bi = U32.div(e, (kv * t : U32))
  +jj = U32.mod(U32.div(e, t), kv)
  +j = U32.mod(e, t)
  KP{((((((bi * kv : U32) * g : U32) + (jj * g : U32) : U32) * t : U32) * t : U32) + j : U32), (((bi * t : U32) * w : U32) + ((jj * g : U32) * hd : U32) : U32), ((((bi * t : U32) + j : U32) * kvw : U32) + (jj * hd : U32) : U32)}

def gkv_if(ok: Bool, +c: KP, +tt: Nat, +hd: Nat, +g: Nat, +t: U32, +w: U32, +hd32: U32, a: Array<F32>, b: Array<F32>, o: Array<F32>) -> Array<F32>:
  match ok c:
    case True{} KP{+ia, +ib, +ob}: gkv_d(Nat.sub(hd, 1n), 0, ia, ib, tt, t, w, (t * t : U32), hd32, Nat.sub(g, 1n), ob, o, gsum_h(Nat.sub(g, 1n), ia, ib, tt, t, w, (t * t : U32), hd32, 0.0, T.dot0(tt, ia, ib, t, w, a, b)))
    case False{} KP{+ia, +ib, +ob}: o

# x1 = G or P, x2 = Q or dO; the output dK or dV
def lf_kv(+e: U32, +l: U32, +c: EKV, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  match c:
    case EKV{+n, +tt, +t32, +kv, +g, +g32, +hd, +hd32, +w, +kvw}: gkv_if(U32.is_lt(e, n), gkp(e, t32, kv, g32, hd32, w, kvw), tt, hd, g, t32, w, hd32, x1, x2, o)

def eat(+w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, +sc: F32) -> EAT:
  EAT{U32.from_nat(Nat.mul(Nat.mul(b, Nat.mul(kv, g)), t)), t, U32.from_nat(t), U32.from_nat(Nat.mul(kv, g)), U32.from_nat(g), hd, U32.from_nat(hd), U32.from_nat(w), U32.from_nat(kvw), sc}

def ekv(+w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat) -> EKV:
  EKV{U32.from_nat(Nat.mul(Nat.mul(b, kv), t)), t, U32.from_nat(t), U32.from_nat(kv), g, U32.from_nat(g), hd, U32.from_nat(hd), U32.from_nat(w), U32.from_nat(kvw)}

# the dK or dV bang, written over o (K or V, no longer read; every cell is written)
def gkv(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, a: Array<F32>, x: Array<F32>, o: Array<F32>) -> Array<F32>:
  r1_a(TG.run1(~EKV, ~lf_kv, TA.cap_depth(Nat.mul(Nat.mul(b, kv), t)), ekv(w, kvw, b, t, kv, g, hd), a, x, nn(Unit{}), nn(Unit{}), o))

def attn_f2(-n: Nat, -w: Nat, -kvw: Nat, q: TG.R2) -> T.Fwd<Unit, TA.Mat<n, w>, T.AttnS>:
  match q:
    case TG.R2{x1, x2, x3, x4, TG.O2{p, y}}: T.Fwd{Unit{}, TA.Mat{y}, T.AttnS{x1, x2, x3}}

def attn_f(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, +sc: F32, x: T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>) -> T.Fwd<Unit, TA.Mat<n, w>, T.AttnS>:
  match x:
    case T.Two{TA.Mat{q}, T.Two{TA.Mat{k}, TA.Mat{v}}}: attn_f2(n, w, kvw, TG.run2_new(~EAT, ~lf_attf, TA.cap_depth(Nat.mul(Nat.mul(b, Nat.mul(kv, g)), t)), eat(w, kvw, b, t, kv, g, hd, sc), q, k, v, nn(Unit{}), TA.cap_depth(Nat.mul(Nat.mul(Nat.mul(b, Nat.mul(kv, g)), t), t)), TA.cap_depth(Nat.mul(n, w))))

def attn_b2(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, r: TG.R3) -> T.Bwd<Unit, T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>>:
  match r:
    case TG.R3{q, k, v, d, TG.O3{p, gb, dq}}: T.Bwd{Unit{}, T.Two{TA.Mat{dq}, T.Two{TA.Mat{gkv(n, w, kvw, b, t, kv, g, hd, gb, q, k)}, TA.Mat{gkv(n, w, kvw, b, t, kv, g, hd, p, d, v)}}}, Unit{}}

def attn_b(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, +sc: F32, s: T.AttnS, dy: TA.Mat<n, w>) -> T.Bwd<Unit, T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>>:
  match s dy:
    case T.AttnS{q, k, v} TA.Mat{d}: attn_b2(n, w, kvw, b, t, kv, g, hd, TG.run3_new(~EAT, ~lf_attb, TA.cap_depth(Nat.mul(Nat.mul(b, Nat.mul(kv, g)), t)), eat(w, kvw, b, t, kv, g, hd, sc), q, k, v, d, TA.cap_depth(Nat.mul(Nat.mul(Nat.mul(b, Nat.mul(kv, g)), t), t)), TA.cap_depth(Nat.mul(Nat.mul(Nat.mul(b, Nat.mul(kv, g)), t), t)), TA.cap_depth(Nat.mul(n, w))))

# causal attention with grouped heads over b sequences of t positions (main.bend's Attention)
def Attention(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, +sc: F32, -e1: {n == Nat.mul(b, t) : Nat}, -e2: {w == Nat.mul(Nat.mul(kv, g), hd) : Nat}, -e3: {kvw == Nat.mul(kv, hd) : Nat}) -> T.Layer<Unit, T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>, TA.Mat<n, w>, T.AttnS>:
  T.Layer{p => x => attn_f(n, w, kvw, b, t, kv, g, hd, sc, x), p => s => dy => g2 => attn_b(n, w, kvw, b, t, kv, g, hd, sc, s, dy), op => p => g2 => st => T.no_step(Unit, op, p, g2, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

# ---- MSE on the device. main.bend's MSE writes each gradient cell 2 (y - t) / (n·c) over y and sums the squares with
# tree_sum, whose passes add neighbouring pairs; its first p passes carry no odd cell when 2^p divides the count, so a
# leaf that sums a block of 2^p squares as a balanced tree of pairs gets exactly the number those passes leave at the
# block's place. The leaves do that on the device (y written in place), and the host finishes tree_sum over the
# count / 2^p block sums: the same bits as main.bend's MSE, with only the block sums read back. ----

type EMS is Data:
  EMS{nb: U32, p: Nat, bs: U32, kk: F32}

type MS is Type:
  MS{y: Array<F32>, t: Array<F32>, s: F32}

def ms_add(+s: F32, r: MS) -> MS:
  match r:
    case MS{y, t, +s2}: MS{y, t, (s + s2 : F32)}

def ms_then(r: MS, k: @+s: F32 -> Array<F32> -> Array<F32> -> MS) -> MS:
  match r:
    case MS{y, t, +s}: k(s, y, t)

def ms_cell(+i: U32, +kk: F32, ry: Array<F32> & F32, rt: Array<F32> & F32) -> MS:
  match ry rt:
    case Tuple{y, +a} Tuple{t, +b}: MS{Array.set(F32, y, i, ((a - b : F32) * kk : F32)), t, ((a - b : F32) * (a - b : F32) : F32)}

# the squares of the block of 2^p cells from i summed as a balanced tree of pairs (h = 2^(p-1)), gradients written
def ms_blk(p: Nat, +i: U32, +h: U32, +kk: F32, y: Array<F32>, t: Array<F32>) -> MS:
  match p:
    case 0n: ms_cell(i, kk, Array.get(F32, y, i), Array.get(F32, t, i))
    case 1n++q: ms_then(ms_blk(q, i, U32.shr(h), kk, y, t), +s => y2 => t2 => ms_add(s, ms_blk(q, (i + h : U32), U32.shr(h), kk, y2, t2)))

def ms_put(+t: U32, ps: Array<F32>, r: MS) -> TG.O2:
  match r:
    case MS{y, tt, +s}: TG.O2{y, Array.set(F32, ps, t, s)}

def ms_if(ok: Bool, +t: U32, +p: Nat, +bs: U32, +kk: F32, tg: Array<F32>, y: Array<F32>, ps: Array<F32>) -> TG.O2:
  match ok:
    case True{}: ms_put(t, ps, ms_blk(p, (t * bs : U32), U32.shr(bs), kk, y, tg))
    case False{}: TG.O2{y, ps}

# x1 = the target; y written in place with the gradient, ps[t] = leaf t's block sum (one leaf a block)
def lf_mse(+t: U32, +l: U32, +e: EMS, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O2) -> TG.O2:
  match e o:
    case EMS{+nb, +p, +bs, +kk} TG.O2{y, ps}: ms_if(U32.is_lt(t, nb), t, p, bs, kk, x1, y, ps)

# the largest p <= k with 2^p dividing m (m > 0)
def ms_p(k: Nat, +m: U32) -> Nat:
  match k:
    case 0n: 0n
    case 1n+q: Bool.pick(Nat, U32.is_eq(U32.mod(m, 2), 0), 1n+ms_p(q, U32.div(m, 2)), 0n)

def mse_fin(-n: Nat, -c: Nat, +nb: U32, +nc: F32, q: TG.R2) -> T.LossOut<TA.Mat<n, c>>:
  match q:
    case TG.R2{x1, x2, x3, x4, TG.O2{y, ps}}: T.LossOut{(T.tree_sum(nb, ps) / nc : F32), TA.Mat{y}}

def mse_go(+n: Nat, +c: Nat, +m: U32, +p: Nat, y: Array<F32>, t: Array<F32>) -> T.LossOut<TA.Mat<n, c>>:
  mse_fin(n, c, U32.shrn(m, p), F32.from_nat(Nat.mul(n, c)), TG.run2(~EMS, ~lf_mse, TA.cap_depth(U32.to_nat(U32.shrn(m, p))), EMS{U32.shrn(m, p), p, U32.pow(2, p), (2.0 / F32.from_nat(Nat.mul(n, c)) : F32)}, t, nn(Unit{}), nn(Unit{}), nn(Unit{}), y, T.zeros_a(U32.to_nat(U32.shrn(m, p)))))

# mean squared error over every number (main.bend's MSE, the same bits)
def MSE(+n: Nat, +c: Nat, y: TA.Mat<n, c>, t: TA.Mat<n, c>) -> T.LossOut<TA.Mat<n, c>>:
  match y t:
    case TA.Mat{ya} TA.Mat{ta}: mse_go(n, c, U32.from_nat(Nat.mul(n, c)), ms_p(6n, U32.from_nat(Nat.mul(n, c))), ya, ta)

# ---- ids and labels on the device: a list of n Nats as an array of n cells, each holding the id's U32 word as an F32's
# bits (a bit copy, not a conversion, so every U32 id is exact); the count is how many of the list's first n there are ----

def u2f(x: U32) -> F32:
  match x:
    case U32{w}: F32{w}

def f2u(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

type IA is Type:
  IA{a: Array<F32>, k: Nat}

def ia_go(left: Nat, ids: List<&2, Nat>, +i: U32, k: Nat, a: Array<F32>) -> IA:
  match left ids:
    case 1n+q Con{+id, t}: ia_go(q, t, (i + 1 : U32), 1n+k, Array.set(F32, a, i, u2f(U32.from_nat(id))))
    case _ _: IA{a, k}

def ids_arr(+n: Nat, ids: List<&2, Nat>) -> IA:
  ia_go(n, ids, 0, 0n, T.zeros_a(n))

# ---- Embedding: a leaf a column j of the table (d of them), each looping over the ids in order: the forward pass
# copies e[id_i, j] into row i, the backward adds dy[i, j] into g[id_i, j] as main.bend's emb_add does (g + dy), one id
# after the other, so every cell of the gradient gets its rows added in main.bend's order, with no two leaves writing
# one cell. Neighbouring lanes read neighbouring cells. ----

type EEM is Data:
  EEM{k: Nat, d: U32}

type ES3 is Type:
  ES3{ids: Array<F32>, a: Array<F32>, o: Array<F32>}

def ef_3(+i: U32, +j: U32, +d: U32, ids: Array<F32>, o: Array<F32>, r: Array<F32> & F32) -> ES3:
  match r:
    case Tuple{e, +x}: ES3{ids, e, Array.set(F32, o, ((i * d : U32) + j : U32), x)}

def ef_2(+i: U32, +j: U32, +d: U32, e: Array<F32>, o: Array<F32>, r: Array<F32> & F32) -> ES3:
  match r:
    case Tuple{ids, +v}: ef_3(i, j, d, ids, o, Array.get(F32, e, ((f2u(v) * d : U32) + j : U32)))

# o[i, j] = e[ids[i], j]
def ef_cell(+i: U32, +j: U32, +d: U32, st: ES3) -> ES3:
  match st:
    case ES3{ids, e, o}: ef_2(i, j, d, e, o, Array.get(F32, ids, i))

def ef_loop(k: Nat, +i: U32, +j: U32, +d: U32, st: ES3) -> ES3:
  match k:
    case 0n: st
    case 1n+q: ef_loop(q, (i + 1 : U32), j, d, ef_cell(i, j, d, st))

def es3_o(st: ES3) -> Array<F32>:
  match st:
    case ES3{ids, a, o}: o

def ef_if(ok: Bool, +t: U32, +k: Nat, +d: U32, e: Array<F32>, ids: Array<F32>, o: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: es3_o(ef_loop(k, 0, t, d, ES3{ids, e, o}))
    case False{}: o

# x1 = the table, x2 = the ids; o (made on the device, zeros) gets the rows
def lf_embf(+t: U32, +l: U32, +e: EEM, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  match e:
    case EEM{+k, +d}: ef_if(U32.is_lt(t, d), t, k, d, x1, x2, o)

def eb_4(+gi: U32, ids: Array<F32>, g: Array<F32>, +a: F32, r: Array<F32> & F32) -> ES3:
  match r:
    case Tuple{dy, +b}: ES3{ids, dy, Array.set(F32, g, gi, (a + b : F32))}

def eb_3(+i: U32, +j: U32, +d: U32, +gi: U32, ids: Array<F32>, dy: Array<F32>, r: Array<F32> & F32) -> ES3:
  match r:
    case Tuple{g, +a}: eb_4(gi, ids, g, a, Array.get(F32, dy, ((i * d : U32) + j : U32)))

def eb_2(+i: U32, +j: U32, +d: U32, dy: Array<F32>, g: Array<F32>, r: Array<F32> & F32) -> ES3:
  match r:
    case Tuple{ids, +v}: eb_3(i, j, d, ((f2u(v) * d : U32) + j : U32), ids, dy, Array.get(F32, g, ((f2u(v) * d : U32) + j : U32)))

# g[ids[i], j] = g[ids[i], j] + dy[i, j]
def eb_cell(+i: U32, +j: U32, +d: U32, st: ES3) -> ES3:
  match st:
    case ES3{ids, dy, g}: eb_2(i, j, d, dy, g, Array.get(F32, ids, i))

def eb_loop(k: Nat, +i: U32, +j: U32, +d: U32, st: ES3) -> ES3:
  match k:
    case 0n: st
    case 1n+q: eb_loop(q, (i + 1 : U32), j, d, eb_cell(i, j, d, st))

def eb_if(ok: Bool, +t: U32, +k: Nat, +d: U32, dy: Array<F32>, ids: Array<F32>, g: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: es3_o(eb_loop(k, 0, t, d, ES3{ids, dy, g}))
    case False{}: g

# x1 = dy, x2 = the ids; the gradient g, in place
def lf_embb(+t: U32, +l: U32, +e: EEM, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, g: Array<F32>) -> Array<F32>:
  match e:
    case EEM{+k, +d}: eb_if(U32.is_lt(t, d), t, k, d, x1, x2, g)

def emb_f3(-n: Nat, -v: Nat, -d: Nat, +ids: List<&2, Nat>, q: TG.R1) -> T.Fwd<TA.Mat<v, d>, TA.Mat<n, d>, List<&2, Nat>>:
  match q:
    case TG.R1{x1, x2, x3, x4, o}: T.Fwd{TA.Mat{x1}, TA.Mat{o}, ids}

def emb_f2(+n: Nat, +v: Nat, +d: Nat, +ids: List<&2, Nat>, ea: Array<F32>, r: IA) -> T.Fwd<TA.Mat<v, d>, TA.Mat<n, d>, List<&2, Nat>>:
  match r:
    case IA{ia, +k}: emb_f3(n, v, d, ids, TG.run1_new(~EEM, ~lf_embf, TA.cap_depth(d), EEM{k, U32.from_nat(d)}, ea, ia, nn(Unit{}), nn(Unit{}), TA.cap_depth(Nat.mul(n, d))))

def emb_f(+n: Nat, +v: Nat, +d: Nat, +ids: List<&2, Nat>, e: TA.Mat<v, d>) -> T.Fwd<TA.Mat<v, d>, TA.Mat<n, d>, List<&2, Nat>>:
  match e:
    case TA.Mat{ea}: emb_f2(n, v, d, ids, ea, ids_arr(n, ids))

def emb_b3(-v: Nat, -d: Nat, e: TA.Mat<v, d>, q: TG.R1) -> T.Bwd<TA.Mat<v, d>, List<&2, Nat>>:
  match q:
    case TG.R1{x1, x2, x3, x4, g}: T.Bwd{e, Nil{}, TA.Mat{g}}

def emb_b2(+v: Nat, +d: Nat, e: TA.Mat<v, d>, da: Array<F32>, ga: Array<F32>, r: IA) -> T.Bwd<TA.Mat<v, d>, List<&2, Nat>>:
  match r:
    case IA{ia, +k}: emb_b3(v, d, e, TG.run1(~EEM, ~lf_embb, TA.cap_depth(d), EEM{k, U32.from_nat(d)}, da, ia, nn(Unit{}), nn(Unit{}), ga))

def emb_bwd(+n: Nat, +v: Nat, +d: Nat, e: TA.Mat<v, d>, ids: List<&2, Nat>, dy: TA.Mat<n, d>, g: TA.Mat<v, d>) -> T.Bwd<TA.Mat<v, d>, List<&2, Nat>>:
  match dy g:
    case TA.Mat{da} TA.Mat{ga}: emb_b2(v, d, e, da, ga, ids_arr(n, ids))

# a table of v rows of d numbers, looked up by n ids (main.bend's Embedding, the same bits); its step on the device
def Embedding(+n: Nat, +v: Nat, +d: Nat) -> T.Layer<TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>>:
  T.Layer{p => x => emb_f(n, v, d, x, p), p => s => dy => g => emb_bwd(n, v, d, p, s, dy, g), op => p => g => st => mat_step(v, d, op, p, g, st), u => TA.Mat.zeros(v, d), p => T.Mat.dump(v, d, p), p => xs => off => T.Mat.restore(v, d, p, xs, off), p => xs => off => T.Mat.store(v, d, p, xs, off), u => U32.from_nat(Nat.mul(v, d))}

# ---- cross-entropy on the device, the same numbers as main.bend's CrossEntropy (but CUDA's exp on the GPU), in four
# bangs; only 3 numbers a row come back to the host, and the gradient stays on the device:
# 1. a leaf a row: its max m by main.bend's pmax0 (the same order of comparisons) and x[y] - m;
# 2. a leaf for each block of 2^p cells of a row (2^p dividing c, p <= 9): e = exp(x - m) written over each cell, as
#    pexp writes it, and the block's sum as a balanced tree of pairs. tree_sum's first p passes over a row add exactly
#    these pairs (no odd cell is carried while 2^p divides the count), so a block's sum is the number they leave there;
# 3. a leaf a row: tree_sum's remaining passes over its c / 2^p block sums (tnode below), which is s, cesm's sum;
# 4. a leaf for 2^14 cells at a time: ((e / s) - [j == y]) / n, pdiv's and ce_g's two expressions.
# The host takes log(s) and adds the rows' terms in order from 0.0, as ce_rows does. Rows past the labels are left as
# they are, as in main.bend; labels past the n rows are not read (main.bend would write past the matrix). ----

type TS is Type:
  TS{a: Array<F32>, s: F32}

def ts_leaf(r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}: TS{a, x}

def ts_plus(+s: F32, r: TS) -> TS:
  match r:
    case TS{a, +s2}: TS{a, (s + s2 : F32)}

def ts_then(r: TS, k: @+s: F32 -> Array<F32> -> TS) -> TS:
  match r:
    case TS{a, +s}: k(s, a)

# tree_sum of the len cells from lo, read in place: a node l passes above the cells sums its left child (the first h =
# 2^(l-1) cells) and its right child (the rest) when there is a rest (big = h < len), and is its left child otherwise
# (tree_sum's odd last cell carried up a pass)
def tnode(l: Nat, big: Bool, +lo: U32, +len: U32, +h: U32, a: Array<F32>) -> TS:
  match l big:
    case 0n _: ts_leaf(Array.get(F32, a, lo))
    case 1n++q True{}: ts_then(tnode(q, U32.is_lt(U32.shr(h), h), lo, h, U32.shr(h), a), +s => a2 => ts_plus(s, tnode(q, U32.is_lt(U32.shr(h), (len - h : U32)), (lo + h : U32), (len - h : U32), U32.shr(h), a2)))
    case 1n+q False{}: tnode(q, U32.is_lt(U32.shr(h), len), lo, len, U32.shr(h), a)

# the passes tree_sum makes over m cells: the smallest l with 2^l >= m (f is fuel)
def ts_lv(f: Nat, +m: U32, +p: U32) -> Nat:
  match f:
    case 0n: 0n
    case 1n+q: Bool.pick(Nat, U32.is_lt(p, m), 1n+ts_lv(q, m, (p * 2 : U32)), 0n)

# the host's part, row by row in order: loss - ((x[y] - m) - log(s)), or main.bend's clamped term for a label past the
# last class
type LS is Type:
  LS{a: Array<F32>, l: F32}

def ce_t2(ok: Bool, +loss: F32, +xm: F32, r: Array<F32> & F32) -> LS:
  match ok r:
    case True{} Tuple{a, +s}: LS{a, (loss - (xm - F32.log(s) : F32) : F32)}
    case False{} Tuple{a, s}: LS{a, (loss - F32.log(F32.max(0.0, 0.000000000001)) : F32)}

def ce_g4(+n: Nat, -c: Nat, z: Array<F32>, r: LS) -> T.LossOut<TA.Mat<n, c>>:
  match r:
    case LS{a, +l}: T.LossOut{(l / F32.from_nat(n) : F32), TA.Mat{z}}

# 1. a leaf a row: st[3t] = x[y] - m, st[3t + 1] = m (x[y] read only when y is a class)
type EC1 is Data:
  EC1{k: U32, c1: Nat, c32: U32}

def c1_3(+t: U32, +m: F32, st: Array<F32>, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{z, +xy}: Array.set(F32, Array.set(F32, st, (t * 3 : U32), (xy - m : F32)), ((t * 3 : U32) + 1 : U32), m)

def c1_2(ok: Bool, +t: U32, +off: U32, +y: U32, st: Array<F32>, r: T.AS) -> Array<F32>:
  match ok r:
    case True{} T.AS{z, +m}: c1_3(t, m, st, Array.get(F32, z, (off + y : U32)))
    case False{} T.AS{z, +m}: c1_3(t, m, st, Tuple{z, 0.0})

def c1_1(+t: U32, +off: U32, +c1: Nat, +c32: U32, z: Array<F32>, st: Array<F32>, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{la, +yv}: c1_2(U32.is_lt(f2u(yv), c32), t, off, f2u(yv), st, T.pmax0(c1, off, Array.get(F32, z, off)))

def c1_if(ok: Bool, +t: U32, +c1: Nat, +c32: U32, lab: Array<F32>, z: Array<F32>, st: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: c1_1(t, (t * c32 : U32), c1, c32, z, st, Array.get(F32, lab, t))
    case False{}: st

def lf_c1(+t: U32, +l: U32, +e: EC1, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, st: Array<F32>) -> Array<F32>:
  match e:
    case EC1{+k, +c1, +c32}: c1_if(U32.is_lt(t, k), t, c1, c32, x1, x2, st)

# 2. the 2^l cells from lo: e = exp(x - m) written over each, and their sum as a balanced tree of pairs (h = 2^(l-1))
def tsx_leaf(+lo: U32, +m: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      TS{Array.set(F32, a, lo, e), e}

def tsx(l: Nat, +lo: U32, +h: U32, +m: F32, a: Array<F32>) -> TS:
  match l:
    case 0n: tsx_leaf(lo, m, Array.get(F32, a, lo))
    case 1n++q: ts_then(tsx(q, lo, U32.shr(h), m, a), +s => a2 => ts_plus(s, tsx(q, (lo + h : U32), U32.shr(h), m, a2)))

# the same over 8 cells in straight-line code (no closure a node): e0..e7 written, then ((e0 + e1) + (e2 + e3)) +
# ((e4 + e5) + (e6 + e7)), the balanced tree's three levels
def x8_7(+lo: U32, +m: F32, +e0: F32, +e1: F32, +e2: F32, +e3: F32, +e4: F32, +e5: F32, +e6: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e7 = F32.exp((x - m : F32))
      TS{Array.set(F32, a, (lo + 7 : U32), e7), ((((e0 + e1 : F32) + (e2 + e3 : F32) : F32) + ((e4 + e5 : F32) + (e6 + e7 : F32) : F32)) : F32)}

def x8_6(+lo: U32, +m: F32, +e0: F32, +e1: F32, +e2: F32, +e3: F32, +e4: F32, +e5: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      x8_7(lo, m, e0, e1, e2, e3, e4, e5, e, Array.get(F32, Array.set(F32, a, (lo + 6 : U32), e), (lo + 7 : U32)))

def x8_5(+lo: U32, +m: F32, +e0: F32, +e1: F32, +e2: F32, +e3: F32, +e4: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      x8_6(lo, m, e0, e1, e2, e3, e4, e, Array.get(F32, Array.set(F32, a, (lo + 5 : U32), e), (lo + 6 : U32)))

def x8_4(+lo: U32, +m: F32, +e0: F32, +e1: F32, +e2: F32, +e3: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      x8_5(lo, m, e0, e1, e2, e3, e, Array.get(F32, Array.set(F32, a, (lo + 4 : U32), e), (lo + 5 : U32)))

def x8_3(+lo: U32, +m: F32, +e0: F32, +e1: F32, +e2: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      x8_4(lo, m, e0, e1, e2, e, Array.get(F32, Array.set(F32, a, (lo + 3 : U32), e), (lo + 4 : U32)))

def x8_2(+lo: U32, +m: F32, +e0: F32, +e1: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      x8_3(lo, m, e0, e1, e, Array.get(F32, Array.set(F32, a, (lo + 2 : U32), e), (lo + 3 : U32)))

def x8_1(+lo: U32, +m: F32, +e0: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      x8_2(lo, m, e0, e, Array.get(F32, Array.set(F32, a, (lo + 1 : U32), e), (lo + 2 : U32)))

def x8_0(+lo: U32, +m: F32, r: Array<F32> & F32) -> TS:
  match r:
    case Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      x8_1(lo, m, e, Array.get(F32, Array.set(F32, a, lo, e), (lo + 1 : U32)))

# tsx over 2^(l+3) cells, its last three levels in blocks of 8
def tsx8(l: Nat, +lo: U32, +h: U32, +m: F32, a: Array<F32>) -> TS:
  match l:
    case 0n: x8_0(lo, m, Array.get(F32, a, lo))
    case 1n++q: ts_then(tsx8(q, lo, U32.shr(h), m, a), +s => a2 => ts_plus(s, tsx8(q, (lo + h : U32), U32.shr(h), m, a2)))

# a leaf takes the blocks b, b + l, .. of the nb = rows * per; block b is row b / per, cells (b % per) * 2^p on
type EC2 is Data:
  EC2{nb: U32, per: U32, p: Nat, bsz: U32, c32: U32}

type C3S is Type:
  C3S{a: Array<F32>, b: Array<F32>, c: Array<F32>}

def c2_put(+b: U32, st: Array<F32>, bs: Array<F32>, r: TS) -> C3S:
  match r:
    case TS{z, +s}: C3S{st, z, Array.set(F32, bs, b, s)}

def c2_m(b8: Bool, +b: U32, +lo: U32, +p: Nat, +h: U32, z: Array<F32>, bs: Array<F32>, r: Array<F32> & F32) -> C3S:
  match b8 r:
    case True{} Tuple{st, +m}: c2_put(b, st, bs, tsx8(Nat.sub(p, 3n), lo, h, m, z))
    case False{} Tuple{st, +m}: c2_put(b, st, bs, tsx(p, lo, h, m, z))

def c2_cell(+b: U32, +per: U32, +p: Nat, +bsz: U32, +c32: U32, s: C3S) -> C3S:
  match s:
    case C3S{st, z, bs}: c2_m(Nat.is_le(3n, p), b, ((U32.div(b, per) * c32 : U32) + (U32.mod(b, per) * bsz : U32) : U32), p, U32.shr(bsz), z, bs, Array.get(F32, st, ((U32.div(b, per) * 3 : U32) + 1 : U32)))

def c2_loop(k: Nat, +b: U32, +l: U32, +per: U32, +p: Nat, +bsz: U32, +c32: U32, s: C3S) -> C3S:
  match k:
    case 0n: s
    case 1n+q: c2_loop(q, (b + l : U32), l, per, p, bsz, c32, c2_cell(b, per, p, bsz, c32, s))

def c2_o(s: C3S) -> TG.O2:
  match s:
    case C3S{st, z, bs}: TG.O2{z, bs}

def lf_c2(+t: U32, +l: U32, +e: EC2, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O2) -> TG.O2:
  match e o:
    case EC2{+nb, +per, +p, +bsz, +c32} TG.O2{z, bs}: c2_o(c2_loop(TG.share(t, nb, l), t, l, per, p, bsz, c32, C3S{x1, z, bs}))

# 3. a leaf a row: st[3t + 2] = s, tree_sum's remaining passes over the row's per block sums
type EC3 is Data:
  EC3{k: U32, per: U32, tl: Nat, th: U32}

def c3_put(+t: U32, st: Array<F32>, r: TS) -> Array<F32>:
  match r:
    case TS{bs, +s}: Array.set(F32, st, ((t * 3 : U32) + 2 : U32), s)

def c3_if(ok: Bool, +t: U32, +per: U32, +tl: Nat, +th: U32, bs: Array<F32>, st: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: c3_put(t, st, tnode(tl, U32.is_lt(th, per), (t * per : U32), per, th, bs))
    case False{}: st

def lf_c3(+t: U32, +l: U32, +e: EC3, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, st: Array<F32>) -> Array<F32>:
  match e:
    case EC3{+k, +per, +tl, +th}: c3_if(U32.is_lt(t, k), t, per, tl, th, x1, st)

# 4. element-wise over the first k rows: z[i] = ((e / s) - [j == y]) / nf, j = i % c, s and y of row i / c
type EC4 is Data:
  EC4{n: U32, c32: U32, nf: F32}

def c4_4(+i: U32, +j: U32, +y: U32, +s: F32, +nf: F32, lab: Array<F32>, st: Array<F32>, r: Array<F32> & F32) -> C3S:
  match r:
    case Tuple{z, +x}: C3S{lab, st, Array.set(F32, z, i, (((x / s : F32) - T.hit(U32.is_eq(j, y)) : F32) / nf : F32))}

def c4_3(+i: U32, +j: U32, +y: U32, +nf: F32, lab: Array<F32>, z: Array<F32>, r: Array<F32> & F32) -> C3S:
  match r:
    case Tuple{st, +s}: c4_4(i, j, y, s, nf, lab, st, Array.get(F32, z, i))

def c4_2(+i: U32, +row: U32, +j: U32, +nf: F32, st: Array<F32>, z: Array<F32>, r: Array<F32> & F32) -> C3S:
  match r:
    case Tuple{lab, +yv}: c4_3(i, j, f2u(yv), nf, lab, z, Array.get(F32, st, ((row * 3 : U32) + 2 : U32)))

def c4_cell(+i: U32, +c32: U32, +nf: F32, s: C3S) -> C3S:
  match s:
    case C3S{lab, st, z}: c4_2(i, U32.div(i, c32), (i - (U32.div(i, c32) * c32 : U32) : U32), nf, st, z, Array.get(F32, lab, U32.div(i, c32)))

def c4_loop(k: Nat, +i: U32, +l: U32, +c32: U32, +nf: F32, s: C3S) -> C3S:
  match k:
    case 0n: s
    case 1n+q: c4_loop(q, (i + l : U32), l, c32, nf, c4_cell(i, c32, nf, s))

def c3s_c(s: C3S) -> Array<F32>:
  match s:
    case C3S{a, b, c}: c

def lf_c4(+t: U32, +l: U32, +e: EC4, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, z: Array<F32>) -> Array<F32>:
  match e:
    case EC4{+n, +c32, +nf}: c3s_c(c4_loop(TG.share(t, n, l), t, l, c32, nf, C3S{x1, x2, z}))

# the host's part over st, row by row
def cx_t1(ok: Bool, +i: U32, +loss: F32, r: Array<F32> & F32) -> LS:
  match r:
    case Tuple{a, +xm}: ce_t2(ok, loss, xm, Array.get(F32, a, ((i * 3 : U32) + 2 : U32)))

def cx_ls(left: Nat, ls: List<&2, Nat>, +i: U32, +c: Nat, r: LS) -> LS:
  match left ls r:
    case 1n+q Con{+y, t} LS{a, +loss}: cx_ls(q, t, (i + 1 : U32), c, cx_t1(Nat.is_le(1n+y, c), i, loss, Array.get(F32, a, (i * 3 : U32))))
    case _ _ r: r

def cx_5(+n: Nat, +c: Nat, +labels: List<&2, Nat>, +k: Nat, q: TG.R1) -> T.LossOut<TA.Mat<n, c>>:
  match q:
    case TG.R1{x1, x2, x3, x4, z}: ce_g4(n, c, z, cx_ls(k, labels, 0, c, LS{x2, 0.0}))

def cx_4(+n: Nat, +c: Nat, +labels: List<&2, Nat>, +k: Nat, la: Array<F32>, z: Array<F32>, q: TG.R1) -> T.LossOut<TA.Mat<n, c>>:
  match q:
    case TG.R1{x1, x2, x3, x4, st}: cx_5(n, c, labels, k, TG.run1(~EC4, ~lf_c4, TG.depth(Nat.mul(k, c)), EC4{U32.from_nat(Nat.mul(k, c)), U32.from_nat(c), F32.from_nat(n)}, la, st, nn(Unit{}), nn(Unit{}), z))

def cx_3(+n: Nat, +c: Nat, +labels: List<&2, Nat>, +k: Nat, +per: U32, la: Array<F32>, q: TG.R2) -> T.LossOut<TA.Mat<n, c>>:
  match q:
    case TG.R2{st, x2, x3, x4, TG.O2{z, bs}}: cx_4(n, c, labels, k, la, z, TG.run1(~EC3, ~lf_c3, TA.cap_depth(k), EC3{U32.from_nat(k), per, ts_lv(32n, per, 1), U32.shr(U32.pow(2, ts_lv(32n, per, 1)))}, bs, nn(Unit{}), nn(Unit{}), nn(Unit{}), st))

def cx_2(+n: Nat, +c: Nat, +labels: List<&2, Nat>, +k: Nat, +p: Nat, q: TG.R1) -> T.LossOut<TA.Mat<n, c>>:
  match q:
    case TG.R1{la, z, x3, x4, st}: cx_3(n, c, labels, k, U32.shrn(U32.from_nat(c), p), la, TG.run2(~EC2, ~lf_c2, TG.depth(Nat.mul(k, U32.to_nat(U32.shrn(U32.from_nat(c), p)))), EC2{(U32.from_nat(k) * U32.shrn(U32.from_nat(c), p) : U32), U32.shrn(U32.from_nat(c), p), p, U32.pow(2, p), U32.from_nat(c)}, st, nn(Unit{}), nn(Unit{}), nn(Unit{}), z, T.zeros_a(Nat.mul(k, U32.to_nat(U32.shrn(U32.from_nat(c), p))))))

def cx_1(+n: Nat, +c: Nat, +labels: List<&2, Nat>, z: Array<F32>, r: IA) -> T.LossOut<TA.Mat<n, c>>:
  match r:
    case IA{la, +k}: cx_2(n, c, labels, k, ms_p(9n, U32.from_nat(c)), TG.run1(~EC1, ~lf_c1, TA.cap_depth(n), EC1{U32.from_nat(k), Nat.sub(c, 1n), U32.from_nat(c)}, la, z, nn(Unit{}), nn(Unit{}), T.zeros_a(Nat.mul(3n, n))))

# cross-entropy over n rows of c classes against the labels (main.bend's CrossEntropy; its bits but CUDA's exp)
def CrossEntropy(+n: Nat, +c: Nat, z: TA.Mat<n, c>, +labels: List<&2, Nat>) -> T.LossOut<TA.Mat<n, c>>:
  match z:
    case TA.Mat{d}: cx_1(n, c, labels, d, ids_arr(n, labels))

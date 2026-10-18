# BendTorch's linear layers with their products on the GPU (bend-ml-tensor-gpu): the same layers, types and bits as
# torch/main.bend's and torch/fast.bend's.
#
#   import ../torch/main.bend as T
#   import ../torch/gpu.bend as F
#   F.LinearFirstF(100n, 784n, 128n, 4n)     # the products of this layer run as bangs (on the GPU when there is one)
#
# LinearF, LinearFirstF and LinearNBF are fast.bend's with their three products (x·Wᵀ, dY·W, dYᵀ·X added into dW) taken
# from bend-ml-tensor-gpu: one `!` call each, which runs on the GPU in a binary built with CUDA and on the CPU threads
# otherwise. tensor-gpu gives tensor-array's bits, so these layers give main.bend's (tests: demo/gpu_test.bend); par is
# not used. The optimizer step, SiLU, Mul2, Residual, Mat.dup and Mat.sum2 are fast.bend's, on the CPU: the names are
# the same as fast.bend's so that a program switches by its import line alone (the Makefile builds the GPU demos so).
#
# Outside the proofs, on purpose, like fast.bend: tensor-gpu shares arrays between the leaves of a bang through Base's
# @unsafe Array.fork, so `bend` names the defs below that reach it, and no file with laws imports this one (`make laws`
# checks both). All of it is Bend. tensor-gpu is a local copy (vendor/) until it is published on BendHub.

import Base
import bend-ml-tensor-array@0.1.12.0/main.bend as TA
import ../vendor/tensor-gpu.bend as TG
import ./main.bend as T
import ./fast.bend as FF

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
      lin_b2(n, i, o, b, x, g, TG.matmul(n, o, i, 0n, dy, w))

def LinearF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<T.Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{p => x => lin_fwd(n, i, o, p, x), p => s => dy => g => lin_bwd(n, i, o, p, s, dy, g), op => p => g => st => FF.lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => off => T.lin_load(i, o, p, xs, off), p => xs => off => T.lin_store(i, o, p, xs, off), u => U32.from_nat(Nat.mul(o, 1n+i))}

# ---- LinearFirst: no dX (the first layer's input needs no gradient) ----

def linf_bwd(+n: Nat, +i: Nat, +o: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: T.Lin<i, o>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match p g:
    case T.Lin{w, b} T.Lin{gw, gb}:
      T.linf_b3(n, i, o, w, b, gb, TG.matmul_tn_acc(o, n, i, 0n, dy, x, gw))

def LinearFirstF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<T.Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{p => x => lin_fwd(n, i, o, p, x), p => s => dy => g => linf_bwd(n, i, o, p, s, dy, g), op => p => g => st => FF.lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => off => T.lin_load(i, o, p, xs, off), p => xs => off => T.lin_store(i, o, p, xs, off), u => U32.from_nat(Nat.mul(o, 1n+i))}

# ---- LinearNB: y = x·Wᵀ, no bias (Llama's projections) ----

def lnb_b2(+n: Nat, +i: Nat, +o: Nat, x: TA.Mat<n, i>, g: TA.Mat<o, i>, r: TA.MMul<n, o, i>) -> T.Bwd<TA.Mat<o, i>, TA.Mat<n, i>>:
  match r:
    case TA.MMul{dy, w, dx}:
      T.lnb_b3(n, i, o, w, dx, TG.matmul_tn_acc(o, n, i, 0n, dy, x, g))

def LinearNBF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<TA.Mat<o, i>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{w => x => T.lnb_f2(n, i, o, TG.matmul_nt(n, i, o, 0n, x, w)), w => s => dy => g => lnb_b2(n, i, o, s, g, TG.matmul(n, o, i, 0n, dy, w)), op => p => g => st => FF.mat_step(o, i, op, p, g, st), u => TA.Mat.zeros(o, i), p => T.Mat.dump(o, i, p), p => xs => off => T.Mat.restore(o, i, p, xs, off), p => xs => off => T.Mat.store(o, i, p, xs, off), u => U32.from_nat(Nat.mul(o, i))}

# ---- fast.bend's, under the same names (on the CPU) ----

def Mat.dup(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> T.Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  FF.Mat.dup(r, c, m)

def Mat.sum2(+r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.Mat<r, c>:
  FF.Mat.sum2(r, c, a, b)

def SiLU(+n: Nat, +c: Nat) -> T.Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  FF.SiLU(n, c)

def Mul2(+n: Nat, +c: Nat) -> T.Layer<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  FF.Mul2(n, c)

def Residual(~n: Nat, ~c: Nat, ~P: Type, ~S: Type, ~f: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, c>, S>) -> T.Layer<T.Two<T.Two<Unit, P>, Unit>, TA.Mat<n, c>, TA.Mat<n, c>, T.Two<T.Two<Unit, S>, Unit>>:
  FF.Residual(~n, ~c, ~P, ~S, ~f)

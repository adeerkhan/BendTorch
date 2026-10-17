# BendTorch's linear layers over bend-ml-tensor-fork: the same layers, types and bits as torch/main.bend's, faster where
# a product splits into parallel blocks.
#
#   import ../torch/main.bend as T
#   import ../torch/fast.bend as F
#   F.LinearFirst(100n, 784n, 128n, 3n)      # instead of T.LinearFirst(100n, 784n, 128n, 3n)
#
# LinearF, LinearFirstF and LinearNBF are main.bend's Linear, LinearFirst and LinearNB with their three products
# (x·Wᵀ, dY·W, dYᵀ·X added into dW) taken from bend-ml-tensor-fork instead of bend-ml-tensor-array. Everything else
# (the bias, the optimizer step, saving and loading) is main.bend's own code. tensor-fork's products give
# tensor-array's bits at every depth and thread count, so these layers give main.bend's (tests: demo/fast_test.bend).
#
# Outside the proofs, on purpose: tensor-fork shares arrays through Base's @unsafe Array.fork, so `bend` names the defs
# below that reach it, and no file with laws imports this one (`make laws` checks both). All of it is Bend.

import Base
import bend-ml-tensor-array@0.1.12.0/main.bend as TA
import bend-ml-tensor-fork@0.1.0.0/main.bend as TF
import ./main.bend as T

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
  T.Layer{p => x => lin_fwd(n, i, o, par, p, x), p => s => dy => g => lin_bwd(n, i, o, par, p, s, dy, g), op => p => g => st => T.lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => T.lin_load(i, o, p, xs)}

# ---- LinearFirst: no dX (the first layer's input needs no gradient) ----

def linf_bwd(+n: Nat, +i: Nat, +o: Nat, +par: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: T.Lin<i, o>) -> T.Bwd<T.Lin<i, o>, TA.Mat<n, i>>:
  match p g:
    case T.Lin{w, b} T.Lin{gw, gb}:
      T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))

def LinearFirstF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<T.Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{p => x => lin_fwd(n, i, o, par, p, x), p => s => dy => g => linf_bwd(n, i, o, par, p, s, dy, g), op => p => g => st => T.lin_step(i, o, op, p, g, st), u => T.lin_zeros(i, o), p => T.lin_dump(i, o, p), p => xs => T.lin_load(i, o, p, xs)}

# ---- LinearNB: y = x·Wᵀ, no bias (Llama's projections) ----

def lnb_b2(+n: Nat, +i: Nat, +o: Nat, +par: Nat, x: TA.Mat<n, i>, g: TA.Mat<o, i>, r: TA.MMul<n, o, i>) -> T.Bwd<TA.Mat<o, i>, TA.Mat<n, i>>:
  match r:
    case TA.MMul{dy, w, dx}:
      T.lnb_b3(n, i, o, w, dx, TF.matmul_tn_acc(o, n, i, par, dy, x, g))

def LinearNBF(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> T.Layer<TA.Mat<o, i>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  T.Layer{w => x => T.lnb_f2(n, i, o, TF.matmul_nt(n, i, o, par, x, w)), w => s => dy => g => lnb_b2(n, i, o, par, s, g, TF.matmul(n, o, i, par, dy, w)), op => p => g => st => T.mat_step(o, i, op, p, g, st), u => TA.Mat.zeros(o, i), p => T.Mat.dump(o, i, p), p => xs => T.Mat.restore(o, i, p, xs)}

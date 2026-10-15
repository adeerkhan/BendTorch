# BendTorch: layers with typed forward and backward passes, composed generically, trained in Bend 2.
#
#   import ../torch/main.bend as T
#
# A layer is a record of four functions over its parameters P, its input X, its output Y and what its forward pass
# saves for the backward pass, S:
#
#   fwd   : P -> X -> Fwd<P, Y, S>               the output, the parameters handed back, the saved values
#   bwd   : P -> S -> Y -> P -> Bwd<P, X>        from dY (typed Y) to dX (typed X), adding dP into a gradient of type P
#   step  : Opt -> P -> P -> Two<P, P> -> Step<P>  one optimizer step on the parameters; the gradient comes back zeroed
#   zeros : Unit -> P                            a zero P (gradients and optimizer state start from it)
#
# The gradient of the input has the input's type and the gradient of the parameters has the parameters' type, so a
# backward pass that returns a transposed or a wrongly sized gradient does not compile. Layers are passed as templates
# (`~l`), so `Seq` composes any two layers whose types meet, with no closure called twice and no runtime cost.
#
# Matrices are bend-ml's `Mat<r, c>` (bend-ml-tensor-array 0.1.7.0): r*c F32 numbers in one flat Array, row by row,
# the dimensions erased type parameters. An Array has one owner, so every operation hands back what it reads.
#
# What is proved and what is tested: the shapes are in the types (checked); the laws at the end of this file are
# checked by Bend's kernel; every F32 result is tested against PyTorch by the scripts in reference/, never proved.

import Base
import bend-ml-tensor-array@0.1.7.0/main.bend as TA

# =====================================================================================================
# The layer protocol
# =====================================================================================================

# what a forward pass returns: the parameters, the output, what the backward pass will need
type Fwd<-P: Type, -Y: Type, -S: Type> is Type:
  Fwd{p: P, y: Y, s: S}

# what a backward pass returns: the parameters, the gradient of the input, the accumulated gradient of the parameters
type Bwd<-P: Type, -X: Type> is Type:
  Bwd{p: P, dx: X, g: P}

# a pair of owned values (parameters of two layers, their saved values, an optimizer's two moments)
type Two<-A: Type, -B: Type> is Type:
  Two{a: A, b: B}

# what an optimizer step returns: the new parameters, the zeroed gradient, the optimizer state
type Step<-P: Type> is Type:
  Step{p: P, g: P, st: Two<P, P>}

# The optimizer, with the scalars of the current step. Adam's `step` is lr / (1 - b1^t) and `bc2` is
# sqrt(1 - b2^t), as PyTorch computes them; `Opt.at` fills them from an `OptCfg` and the step count.
# `dec` picks AdamW's decoupled weight decay (w *= 1 - lr*wd) over Adam's (g += wd*w).
type Opt is Data:
  SGD{lr: F32}
  Momentum{lr: F32, mu: F32}
  Adam{lr: F32, b1: F32, b2: F32, eps: F32, wd: F32, dec: Bool, step: F32, bc2: F32}

type Layer<-P: Type, -X: Type, -Y: Type, -S: Type> is Type:
  Layer{fwd: P -> X -> Fwd<P, Y, S>, bwd: P -> S -> Y -> P -> Bwd<P, X>, step: Opt -> P -> P -> Two<P, P> -> Step<P>, zeros: Unit -> P}

def Layer.fwd(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, p: P, x: X) -> Fwd<P, Y, S>:
  match l:
    case Layer{fwd, bwd, step, zeros}:
      fwd(p, x)

def Layer.bwd(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, p: P, s: S, dy: Y, g: P) -> Bwd<P, X>:
  match l:
    case Layer{fwd, bwd, step, zeros}:
      bwd(p, s, dy, g)

def Layer.step(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, +o: Opt, p: P, g: P, st: Two<P, P>) -> Step<P>:
  match l:
    case Layer{fwd, bwd, step, zeros}:
      step(o, p, g, st)

def Layer.zeros(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>) -> P:
  match l:
    case Layer{fwd, bwd, step, zeros}:
      zeros(Unit{})

# =====================================================================================================
# Element-wise kernels (templates: one copy per function, no closure at run time)
# =====================================================================================================

# a[t] = f(a[t]) for the n numbers from t; ra holds the array and a[t], already read
def ew1_l(~f: @+a: F32 -> F32, nleft: Nat, +t: U32, ra: Array<F32> & F32) -> Array<F32>:
  match nleft ra:
    case 0n Tuple{a, v}:
      a
    case 1n+p Tuple{a, +v}:
      ew1_l(~f, p, (t + 1 : U32), Array.get(F32, Array.set(F32, a, t, f(v)), (t + 1 : U32)))

# a[t] = f(a[t], b[t]): the result replaces a; b comes back unchanged
def ew2_l(~f: @+a: F32 -> @+b: F32 -> F32, nleft: Nat, +t: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> TA.P2:
  match nleft ra rb:
    case 0n Tuple{a, v} Tuple{b, w}:
      TA.P2{a, b}
    case 1n+p Tuple{a, +v} Tuple{b, +w}:
      ew2_l(~f, p, (t + 1 : U32), Array.get(F32, Array.set(F32, a, t, f(v, w)), (t + 1 : U32)), Array.get(F32, b, (t + 1 : U32)))

def Ew.map(~f: @+a: F32 -> F32, +r: Nat, +c: Nat, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  match m:
    case TA.Mat{d}:
      TA.Mat{ew1_l(~f, Nat.mul(r, c), 0, Array.get(F32, d, 0))}

def ew2_fin(-r: Nat, -c: Nat, p: TA.P2) -> TA.MPair<r, c>:
  match p:
    case TA.P2{a, b}:
      TA.MPair{TA.Mat{a}, TA.Mat{b}}

# (f(a, b), b), element by element
def Ew.zip(~f: @+a: F32 -> @+b: F32 -> F32, +r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.MPair<r, c>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      ew2_fin(r, c, ew2_l(~f, Nat.mul(r, c), 0, Array.get(F32, x, 0), Array.get(F32, y, 0)))

def f32_add(+a: F32, +b: F32) -> F32:
  (a + b : F32)

# (a + b, b)
def Mat.add(+r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.MPair<r, c>:
  Ew.zip(~f32_add, r, c, a, b)

def zero_of(+x: F32) -> F32:
  0.0

# =====================================================================================================
# Optimizers: one pass over a parameter, its gradient and the two moment buffers; the gradient is zeroed
# =====================================================================================================

type Q4 is Type:
  Q4{w: Array<F32>, g: Array<F32>, m: Array<F32>, v: Array<F32>}

type MatStep<-r: Nat, -c: Nat> is Type:
  MatStep{w: TA.Mat<r, c>, g: TA.Mat<r, c>, m: TA.Mat<r, c>, v: TA.Mat<r, c>}

# w <- w + (-lr)*g, as PyTorch's p.add_(g, alpha=-lr)
def sgd_l(nleft: Nat, +t: U32, +lr: F32, rw: Array<F32> & F32, rg: Array<F32> & F32) -> TA.P2:
  match nleft rw rg:
    case 0n Tuple{w, x} Tuple{g, y}:
      TA.P2{w, g}
    case 1n+p Tuple{w, +x} Tuple{g, +y}:
      sgd_l(p, (t + 1 : U32), lr, Array.get(F32, Array.set(F32, w, t, (x + (F32.neg(lr) * y : F32) : F32)), (t + 1 : U32)), Array.get(F32, Array.set(F32, g, t, 0.0), (t + 1 : U32)))

# buf <- buf*mu + g; w <- w + (-lr)*buf, as PyTorch's SGD with momentum (no dampening, not Nesterov)
def mom_l(nleft: Nat, +t: U32, +lr: F32, +mu: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<F32> & F32) -> Q4:
  match nleft rw rg rm:
    case 0n Tuple{w, x} Tuple{g, y} Tuple{m, b}:
      Q4{w, g, m, [0.0 : F32*1n]}
    case 1n+p Tuple{w, +x} Tuple{g, +y} Tuple{m, +b}:
      +nb = ((b * mu : F32) + y : F32)
      +u = (t + 1 : U32)
      mom_l(p, u, lr, mu, Array.get(F32, Array.set(F32, w, t, (x + (F32.neg(lr) * nb : F32) : F32)), u), Array.get(F32, Array.set(F32, g, t, 0.0), u), Array.get(F32, Array.set(F32, m, t, nb), u))

# the Adam family, in PyTorch's order of operations (torch.optim.adam._single_tensor_adam):
#   g' = g + wd*w (Adam) or w' = w*(1 - lr*wd) (AdamW); m = m + (1-b1)*(g' - m) (lerp); v = v*b2 + (1-b2)*g'*g';
#   w = w' + (-step*m) / (sqrt(v)/bc2 + eps)
def adam_grad(dec: Bool, +wd: F32, +w: F32, +g: F32) -> F32:
  match dec:
    case True{}:
      g
    case False{}:
      (g + (w * wd : F32) : F32)

def adam_decay(dec: Bool, +lr: F32, +wd: F32, +w: F32) -> F32:
  match dec:
    case True{}:
      (w * (1.0 - (lr * wd : F32) : F32) : F32)
    case False{}:
      w

def adam_m(+b1: F32, +m: F32, +g: F32) -> F32:
  (m + ((1.0 - b1 : F32) * (g - m : F32) : F32) : F32)

def adam_v(+b2: F32, +v: F32, +g: F32) -> F32:
  ((v * b2 : F32) + (((1.0 - b2 : F32) * g : F32) * g : F32) : F32)

def adam_w(+step: F32, +bc2: F32, +eps: F32, +w: F32, +m: F32, +v: F32) -> F32:
  (w + ((F32.neg(step) * m : F32) / ((F32.sqrt(v) / bc2 : F32) + eps : F32) : F32) : F32)

def adam_l(nleft: Nat, +t: U32, +lr: F32, +b1: F32, +b2: F32, +eps: F32, +wd: F32, +dec: Bool, +step: F32, +bc2: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<F32> & F32, rv: Array<F32> & F32) -> Q4:
  match nleft rw rg rm rv:
    case 0n Tuple{w, x} Tuple{g, y} Tuple{m, a} Tuple{v, b}:
      Q4{w, g, m, v}
    case 1n+p Tuple{w, +x} Tuple{g, +y} Tuple{m, +a} Tuple{v, +b}:
      +gg = adam_grad(dec, wd, x, y)
      +nm = adam_m(b1, a, gg)
      +nv = adam_v(b2, b, gg)
      +nw = adam_w(step, bc2, eps, adam_decay(dec, lr, wd, x), nm, nv)
      +u = (t + 1 : U32)
      adam_l(p, u, lr, b1, b2, eps, wd, dec, step, bc2, Array.get(F32, Array.set(F32, w, t, nw), u), Array.get(F32, Array.set(F32, g, t, 0.0), u), Array.get(F32, Array.set(F32, m, t, nm), u), Array.get(F32, Array.set(F32, v, t, nv), u))

# one optimizer step on a matrix, its gradient (zeroed on the way) and its two moment buffers (SGD reads neither,
# momentum only the first)
def Opt.mat2(-r: Nat, -c: Nat, v: Array<F32>, q: TA.P2) -> MatStep<r, c>:
  match q:
    case TA.P2{w, g}:
      MatStep{TA.Mat{w}, TA.Mat{g}, TA.Mat{[0.0 : F32*1n]}, TA.Mat{v}}

def Opt.fin(-r: Nat, -c: Nat, q: Q4) -> MatStep<r, c>:
  match q:
    case Q4{w, g, m, v}:
      MatStep{TA.Mat{w}, TA.Mat{g}, TA.Mat{m}, TA.Mat{v}}

def Opt.fin_m(-r: Nat, -c: Nat, v: Array<F32>, q: Q4) -> MatStep<r, c>:
  match q:
    case Q4{w, g, m, dummy}:
      MatStep{TA.Mat{w}, TA.Mat{g}, TA.Mat{m}, TA.Mat{v}}

def Opt.sgd_fin(-r: Nat, -c: Nat, m: Array<F32>, v: Array<F32>, q: TA.P2) -> MatStep<r, c>:
  match q:
    case TA.P2{w, g}:
      MatStep{TA.Mat{w}, TA.Mat{g}, TA.Mat{m}, TA.Mat{v}}

def Opt.mat(+r: Nat, +c: Nat, +o: Opt, w: TA.Mat<r, c>, g: TA.Mat<r, c>, m: TA.Mat<r, c>, v: TA.Mat<r, c>) -> MatStep<r, c>:
  match o w g m v:
    case SGD{lr} TA.Mat{wd} TA.Mat{gd} TA.Mat{md} TA.Mat{vd}:
      Opt.sgd_fin(r, c, md, vd, sgd_l(Nat.mul(r, c), 0, lr, Array.get(F32, wd, 0), Array.get(F32, gd, 0)))
    case Momentum{lr, mu} TA.Mat{wd} TA.Mat{gd} TA.Mat{md} TA.Mat{vd}:
      Opt.fin_m(r, c, vd, mom_l(Nat.mul(r, c), 0, lr, mu, Array.get(F32, wd, 0), Array.get(F32, gd, 0), Array.get(F32, md, 0)))
    case Adam{lr, b1, b2, eps, wd2, dec, step, bc2} TA.Mat{wd} TA.Mat{gd} TA.Mat{md} TA.Mat{vd}:
      Opt.fin(r, c, adam_l(Nat.mul(r, c), 0, lr, b1, b2, eps, wd2, dec, step, bc2, Array.get(F32, wd, 0), Array.get(F32, gd, 0), Array.get(F32, md, 0), Array.get(F32, vd, 0)))

# What the user picks; `Opt.at` turns it into the step's `Opt` from b1^t and b2^t (t = 1 on the first step).
type OptCfg is Data:
  CSGD{lr: F32}
  CMomentum{lr: F32, mu: F32}
  CAdam{lr: F32, b1: F32, b2: F32, eps: F32, wd: F32}
  CAdamW{lr: F32, b1: F32, b2: F32, eps: F32, wd: F32}

def Opt.at(+cfg: OptCfg, +p1: F32, +p2: F32) -> Opt:
  match cfg:
    case CSGD{lr}:
      SGD{lr}
    case CMomentum{lr, mu}:
      Momentum{lr, mu}
    case CAdam{lr, b1, b2, eps, wd}:
      Adam{lr, b1, b2, eps, wd, False{}, (lr / (1.0 - p1 : F32) : F32), F32.sqrt((1.0 - p2 : F32))}
    case CAdamW{lr, b1, b2, eps, wd}:
      Adam{lr, b1, b2, eps, wd, True{}, (lr / (1.0 - p1 : F32) : F32), F32.sqrt((1.0 - p2 : F32))}

def OptCfg.b1(+cfg: OptCfg) -> F32:
  match cfg:
    case CAdam{lr, b1, b2, eps, wd}:
      b1
    case CAdamW{lr, b1, b2, eps, wd}:
      b1
    case _:
      0.0

def OptCfg.b2(+cfg: OptCfg) -> F32:
  match cfg:
    case CAdam{lr, b1, b2, eps, wd}:
      b2
    case CAdamW{lr, b1, b2, eps, wd}:
      b2
    case _:
      0.0

# =====================================================================================================
# Linear: y = x·Wᵀ + b, with W stored out x in (PyTorch's nn.Linear layout); saves its input
# =====================================================================================================

type Lin<-i: Nat, -o: Nat> is Type:
  Lin{w: TA.Mat<o, i>, b: TA.Mat<1n, o>}

def lin_f3(-n: Nat, -i: Nat, -o: Nat, w: TA.Mat<o, i>, x: TA.Mat<n, i>, r: TA.MBias<n, o>) -> Fwd<Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match r:
    case TA.MBias{y, b}:
      Fwd{Lin{w, b}, y, x}

def lin_f2(+n: Nat, -i: Nat, +o: Nat, b: TA.Mat<1n, o>, r: TA.MMulNT<n, i, o>) -> Fwd<Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match r:
    case TA.MMulNT{x, w, z}:
      lin_f3(n, i, o, w, x, TA.Mat.add_row(n, o, z, b))

def lin_fwd(+n: Nat, +i: Nat, +o: Nat, par: Nat, p: Lin<i, o>, x: TA.Mat<n, i>) -> Fwd<Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match p:
    case Lin{w, b}:
      lin_f2(n, i, o, b, TA.Mat.matmul_nt(n, i, o, par, x, w))

def lin_b5(-n: Nat, -i: Nat, -o: Nat, w: TA.Mat<o, i>, b: TA.Mat<1n, o>, dx: TA.Mat<n, i>, rw: TA.MPair<o, i>, rb: TA.MPair<1n, o>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match rw rb:
    case TA.MPair{gw, dw} TA.MPair{gb, db}:
      Bwd{Lin{w, b}, dx, Lin{gw, gb}}

def lin_b4(-n: Nat, +i: Nat, +o: Nat, w: TA.Mat<o, i>, b: TA.Mat<1n, o>, dx: TA.Mat<n, i>, g: Lin<i, o>, dw: TA.Mat<o, i>, r: TA.MColSum<n, o>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match g r:
    case Lin{gw, gb} TA.MColSum{dy, db}:
      lin_b5(n, i, o, w, b, dx, Mat.add(o, i, gw, dw), Mat.add(1n, o, gb, db))

def lin_b3(+n: Nat, +i: Nat, +o: Nat, w: TA.Mat<o, i>, b: TA.Mat<1n, o>, dx: TA.Mat<n, i>, g: Lin<i, o>, r: TA.MMulTN<o, n, i>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match r:
    case TA.MMulTN{dy, x, dw}:
      lin_b4(n, i, o, w, b, dx, g, dw, TA.Mat.col_sums(n, o, dy))

def lin_b2(+n: Nat, +i: Nat, +o: Nat, par: Nat, b: TA.Mat<1n, o>, x: TA.Mat<n, i>, g: Lin<i, o>, r: TA.MMul<n, o, i>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match r:
    case TA.MMul{dy, w, dx}:
      lin_b3(n, i, o, w, b, dx, g, TA.Mat.matmul_tn(o, n, i, par, dy, x))

# dX = dY·W, dW += dYᵀ·X, db += the column sums of dY
def lin_bwd(+n: Nat, +i: Nat, +o: Nat, +par: Nat, p: Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: Lin<i, o>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match p:
    case Lin{w, b}:
      lin_b2(n, i, o, par, b, x, g, TA.Mat.matmul(n, o, i, par, dy, w))

def lin_s2(-i: Nat, -o: Nat, a: MatStep<o, i>, c: MatStep<1n, o>) -> Step<Lin<i, o>>:
  match a c:
    case MatStep{w, gw, mw, vw} MatStep{b, gb, mb, vb}:
      Step{Lin{w, b}, Lin{gw, gb}, Two{Lin{mw, mb}, Lin{vw, vb}}}

def lin_step(+i: Nat, +o: Nat, +op: Opt, p: Lin<i, o>, g: Lin<i, o>, st: Two<Lin<i, o>, Lin<i, o>>) -> Step<Lin<i, o>>:
  match p g st:
    case Lin{w, b} Lin{gw, gb} Two{Lin{mw, mb}, Lin{vw, vb}}:
      lin_s2(i, o, Opt.mat(o, i, op, w, gw, mw, vw), Opt.mat(1n, o, op, b, gb, mb, vb))

def lin_zeros(+i: Nat, +o: Nat) -> Lin<i, o>:
  Lin{TA.Mat.zeros(o, i), TA.Mat.zeros(1n, o)}

# a linear layer from i to o features on batches of n rows; its products run in 2^par blocks of rows
def Linear(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> Layer<Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  Layer{p => x => lin_fwd(n, i, o, par, p, x), p => s => dy => g => lin_bwd(n, i, o, par, p, s, dy, g), op => p => g => st => lin_step(i, o, op, p, g, st), u => lin_zeros(i, o)}

# =====================================================================================================
# Layers without parameters
# =====================================================================================================

def pick(b: Bool, +x: F32, +y: F32) -> F32:
  match b:
    case True{}:
      x
    case False{}:
      y

def no_step(-P: Type, +op: Opt, p: P, g: P, st: Two<P, P>) -> Step<P>:
  Step{p, g, st}

# an element-wise activation that saves a copy of its output (relu, sigmoid, tanh: their derivative is a function of
# the output). `f` is the activation, `d(dy, y)` the gradient of its input.
def act_f2(-n: Nat, -c: Nat, r: TA.MM2<n, c>) -> Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match r:
    case TA.MM2{y, s}:
      Fwd{Unit{}, y, s}

def act_b2(-n: Nat, -c: Nat, r: TA.MPair<n, c>) -> Bwd<Unit, TA.Mat<n, c>>:
  match r:
    case TA.MPair{dx, y}:
      Bwd{Unit{}, dx, Unit{}}

def ActOut(~f: @+a: F32 -> F32, ~d: @+a: F32 -> @+b: F32 -> F32, +n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  Layer{p => x => act_f2(n, c, TA.Mat.clone(n, c, Ew.map(~f, n, c, x))), p => s => dy => g => act_b2(n, c, Ew.zip(~d, n, c, dy, s)), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}}

# relu(x) = x unless x < 0 (so relu(NaN) = NaN and relu(-0) = -0); its gradient passes where the output is > 0,
# and is +0 elsewhere (PyTorch's threshold_backward)
def relu_v(+x: F32) -> F32:
  pick(F32.is_lt(x, 0.0), 0.0, x)

def relu_d(+dy: F32, +y: F32) -> F32:
  pick(F32.is_gt(y, 0.0), dy, 0.0)

def ReLU(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActOut(~relu_v, ~relu_d, n, c)

# =====================================================================================================
# Seq: la then lb. Parameters, gradients, saved values and optimizer state are pairs of the two layers'.
# =====================================================================================================

def seq_f3(-P1: Type, -P2: Type, -Z: Type, -S1: Type, -S2: Type, p1: P1, s1: S1, r: Fwd<P2, Z, S2>) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match r:
    case Fwd{p2, z, s2}:
      Fwd{Two{p1, p2}, z, Two{s1, s2}}

def seq_f2(~P2: Type, ~Y: Type, ~Z: Type, ~S2: Type, ~lb: Layer<P2, Y, Z, S2>, -P1: Type, -S1: Type, p2: P2, r: Fwd<P1, Y, S1>) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match r:
    case Fwd{p1, y, s1}:
      seq_f3(P1, P2, Z, S1, S2, p1, s1, Layer.fwd(P2, Y, Z, S2, lb, p2, y))

def seq_fwd(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>, p: Two<P1, P2>, x: X) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match p:
    case Two{p1, p2}:
      seq_f2(~P2, ~Y, ~Z, ~S2, ~lb, P1, S1, p2, Layer.fwd(P1, X, Y, S1, la, p1, x))

def seq_b3(-P1: Type, -P2: Type, -X: Type, p2: P2, g2: P2, r: Bwd<P1, X>) -> Bwd<Two<P1, P2>, X>:
  match r:
    case Bwd{p1, dx, g1}:
      Bwd{Two{p1, p2}, dx, Two{g1, g2}}

def seq_b2(~P1: Type, ~X: Type, ~Y: Type, ~S1: Type, ~la: Layer<P1, X, Y, S1>, -P2: Type, p1: P1, s1: S1, g1: P1, r: Bwd<P2, Y>) -> Bwd<Two<P1, P2>, X>:
  match r:
    case Bwd{p2, dy, g2}:
      seq_b3(P1, P2, X, p2, g2, Layer.bwd(P1, X, Y, S1, la, p1, s1, dy, g1))

def seq_bwd(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>, p: Two<P1, P2>, s: Two<S1, S2>, dz: Z, g: Two<P1, P2>) -> Bwd<Two<P1, P2>, X>:
  match p s g:
    case Two{p1, p2} Two{s1, s2} Two{g1, g2}:
      seq_b2(~P1, ~X, ~Y, ~S1, ~la, P2, p1, s1, g1, Layer.bwd(P2, Y, Z, S2, lb, p2, s2, dz, g2))

def seq_s3(-P1: Type, -P2: Type, r1: Step<P1>, r2: Step<P2>) -> Step<Two<P1, P2>>:
  match r1 r2:
    case Step{p1, g1, Two{m1, v1}} Step{p2, g2, Two{m2, v2}}:
      Step{Two{p1, p2}, Two{g1, g2}, Two{Two{m1, m2}, Two{v1, v2}}}

def seq_s2(~P2: Type, ~Y: Type, ~Z: Type, ~S2: Type, ~lb: Layer<P2, Y, Z, S2>, -P1: Type, +op: Opt, p2: P2, g2: P2, m2: P2, v2: P2, r1: Step<P1>) -> Step<Two<P1, P2>>:
  seq_s3(P1, P2, r1, Layer.step(P2, Y, Z, S2, lb, op, p2, g2, Two{m2, v2}))

def seq_step(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>, +op: Opt, p: Two<P1, P2>, g: Two<P1, P2>, st: Two<Two<P1, P2>, Two<P1, P2>>) -> Step<Two<P1, P2>>:
  match p g st:
    case Two{p1, p2} Two{g1, g2} Two{Two{m1, m2}, Two{v1, v2}}:
      seq_s2(~P2, ~Y, ~Z, ~S2, ~lb, P1, op, p2, g2, m2, v2, Layer.step(P1, X, Y, S1, la, op, p1, g1, Two{m1, v1}))

def Seq(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>) -> Layer<Two<P1, P2>, X, Z, Two<S1, S2>>:
  Layer{p => x => seq_fwd(~P1, ~P2, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, p, x), p => s => dz => g => seq_bwd(~P1, ~P2, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, p, s, dz, g), op => p => g => st => seq_step(~P1, ~P2, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, op, p, g, st), u => Two{Layer.zeros(P1, X, Y, S1, la), Layer.zeros(P2, Y, Z, S2, lb)}}

# =====================================================================================================
# Losses: the mean loss over the batch, and its gradient with respect to the output
# =====================================================================================================

type LossOut<-Y: Type> is Type:
  LossOut{loss: F32, dy: Y}

def ce_fin(-n: Nat, -c: Nat, r: TA.MCE<n, c>) -> LossOut<TA.Mat<n, c>>:
  match r:
    case TA.MCE{l, g}:
      LossOut{l, g}

# cross-entropy of the logits z (n x c) against one label per row, mean over the rows (PyTorch's
# F.cross_entropy, reduction="mean"); its gradient is (softmax(z) - onehot) / n
def CrossEntropy(+n: Nat, +c: Nat, z: TA.Mat<n, c>, labels: List<&2, Nat>) -> LossOut<TA.Mat<n, c>>:
  ce_fin(n, c, TA.Mat.softmax_ce(n, c, z, labels))

# =====================================================================================================
# Training: one step of any layer under any loss
# =====================================================================================================

# the parameters, their gradient (zero between steps), the optimizer's two moments, b1^t and b2^t, the last loss
type Train<-P: Type> is Type:
  Train{p: P, g: P, st: Two<P, P>, p1: F32, p2: F32, loss: F32}

def Train.new(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, p: P) -> Train<P>:
  Train{p, Layer.zeros(P, X, Y, S, m), Two{Layer.zeros(P, X, Y, S, m), Layer.zeros(P, X, Y, S, m)}, 1.0, 1.0, 0.0}

def ts5(-P: Type, +l: F32, +p1: F32, +p2: F32, r: Step<P>) -> Train<P>:
  match r:
    case Step{p, g, st}:
      Train{p, g, st, p1, p2, l}

def ts4(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +cfg: OptCfg, st: Two<P, P>, +p1: F32, +p2: F32, +l: F32, r: Bwd<P, X>) -> Train<P>:
  match r:
    case Bwd{p, dx, g}:
      +q1 = (p1 * OptCfg.b1(cfg) : F32)
      +q2 = (p2 * OptCfg.b2(cfg) : F32)
      ts5(P, l, q1, q2, Layer.step(P, X, Y, S, m, Opt.at(cfg, q1, q2), p, g, st))

def ts3(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +cfg: OptCfg, p: P, s: S, g: P, st: Two<P, P>, +p1: F32, +p2: F32, r: LossOut<Y>) -> Train<P>:
  match r:
    case LossOut{l, dy}:
      ts4(~P, ~X, ~Y, ~S, ~m, cfg, st, p1, p2, l, Layer.bwd(P, X, Y, S, m, p, s, dy, g))

def ts2(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~T: Type, ~m: Layer<P, X, Y, S>, ~loss: Y -> T -> LossOut<Y>, +cfg: OptCfg, g: P, st: Two<P, P>, +p1: F32, +p2: F32, t: T, r: Fwd<P, Y, S>) -> Train<P>:
  match r:
    case Fwd{p, y, s}:
      ts3(~P, ~X, ~Y, ~S, ~m, cfg, p, s, g, st, p1, p2, loss(y, t))

# forward, loss, backward and one optimizer step: the same code for every model
def Train.step(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~T: Type, ~m: Layer<P, X, Y, S>, ~loss: Y -> T -> LossOut<Y>, +cfg: OptCfg, tr: Train<P>, x: X, t: T) -> Train<P>:
  match tr:
    case Train{p, g, st, p1, p2, l}:
      ts2(~P, ~X, ~Y, ~S, ~T, ~m, ~loss, cfg, g, st, p1, p2, t, Layer.fwd(P, X, Y, S, m, p, x))

# =====================================================================================================
# Reading and printing numbers (bend-ml's and llama.bend's loaders, MIT)
# =====================================================================================================

# U32 and F32 hold the same Word(32n): these are bitcasts
def bitcast(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def bits(x: F32) -> U32:
  match x:
    case F32{w}:
      U32{w}

def f32_le(+b0: U32, +b1: U32, +b2: U32, +b3: U32) -> F32:
  bitcast((b0 + (b1 * 256 : U32) + (b2 * 65536 : U32) + (b3 * 16777216 : U32) : U32))

def f32s(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Con{b0, Con{b1, Con{b2, Con{b3, t}}}}:
      f32_le(b0, b1, b2, b3) <> f32s(t)
    case _:
      Nil{}

def rd_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  match m:
    case Tuple{f, r}:
      do IO<List<&2, U32>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        File.close(f)
        return bs

# n bytes from offset off
def read_at(path: String, +off: U32, +n: U32) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, off, n)
    rd_done(m)

def mat_or_die(-r: Nat, -c: Nat, +path: String, m: Maybe<&1, TA.Mat<r, c>>) -> IO(TA.Mat<r, c>):
  match m:
    case None{}:
      IO.die(TA.Mat<r, c>, 1, path ++ ": wrong number of values for this shape")
    case Some{x}:
      IO.pure(TA.Mat<r, c>, x)

# an r x c matrix stored as r*c little-endian F32 from byte off of a file
def Mat.load(+r: Nat, +c: Nat, +path: String, +off: U32) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    bs : List<&2, U32> <- read_at(path, off, (U32.from_nat(Nat.mul(r, c)) * 4 : U32))
    mat_or_die(r, c, path, TA.Mat.of_list(r, c, f32s(bs)))

def words(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      " " ++ U32.show(bits(h)) ++ words(t)

def ml_words(-r: Nat, -c: Nat, m: TA.ML<r, c>) -> String:
  match m:
    case TA.ML{d, xs}:
      words(xs)

# one line: the name, then each number as the decimal of its 32 bits (exact, for the tests in reference/)
def Mat.print(+r: Nat, +c: Nat, +name: String, m: TA.Mat<r, c>) -> IO(Unit):
  IO.print(name ++ ml_words(r, c, TA.Mat.to_list(r, c, m)))

def F32.print(+name: String, x: F32) -> IO(Unit):
  IO.print(name ++ " " ++ U32.show(bits(x)))

# =====================================================================================================
# Rows as lists: the row-wise operations (norms, softmax, attention) work on a matrix as a list of rows. Lists are
# Data, so a forward pass can keep them for its backward pass without a copy. This is the simple first version: it
# is correct and fast enough for small models; the products of the linear layers, where the time goes, stay on
# flat arrays. (NOTES.md, "Row operations on lists".)
# =====================================================================================================

def l_add(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      (x + y : F32) <> l_add(xt, yt)
    case _ _:
      Nil{}

def l_sub(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      (x - y : F32) <> l_sub(xt, yt)
    case _ _:
      Nil{}

def l_mul(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      (x * y : F32) <> l_mul(xt, yt)
    case _ _:
      Nil{}

def l_scale(+k: F32, xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      (x * k : F32) <> l_scale(k, t)

def l_shift(+k: F32, xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      (x - k : F32) <> l_shift(k, t)

# a running sum from acc, left to right
def l_sum(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{x, t}:
      l_sum(t, (acc + x : F32))

def l_dot(xs: List<&2, F32>, ys: List<&2, F32>, acc: F32) -> F32:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      l_dot(xt, yt, (acc + (x * y : F32) : F32))
    case _ _:
      acc

def l_max2(xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case Con{x, t}:
      l_max2(t, F32.max(cur, x))

def l_max(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{x, t}:
      l_max2(t, x)

def l_exp(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      F32.exp(x) <> l_exp(t)

def l_zeros(n: Nat) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      0.0 <> l_zeros(p)

def l_slice(+xs: List<&2, F32>, +start: Nat, +len: Nat) -> List<&2, F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, start), len)

# xs / s, element by element
def l_scale2(+s: F32, xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      (x / s : F32) <> l_scale2(s, t)

# softmax(x) = exp(x - max) / sum, as PyTorch
def l_softmax2(+es: List<&2, F32>) -> List<&2, F32>:
  l_scale2(l_sum(es, 0.0), es)

def l_softmax(+xs: List<&2, F32>) -> List<&2, F32>:
  l_softmax2(l_exp(l_shift(l_max(xs), xs)))

# ---- matrices as lists of rows ----

def lm_chunk(r: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, c) <> lm_chunk(p, c, List.drop(&2, F32, xs, c))

def lm_flat(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  List.concat(&2, F32, rows)

def lm_add(xs: List<&2, List<&2, F32>>, ys: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      l_add(x, y) <> lm_add(xt, yt)
    case _ _:
      Nil{}

def lm_scale(+k: F32, xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      l_scale(k, x) <> lm_scale(k, t)

def lm_rows(+rows: List<&2, List<&2, F32>>, +start: Nat, +n: Nat) -> List<&2, List<&2, F32>>:
  List.take(&2, List<&2, F32>, List.drop(&2, List<&2, F32>, rows, start), n)

def lm_cols(rows: List<&2, List<&2, F32>>, +start: Nat, +n: Nat) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{+r, t}:
      l_slice(r, start, n) <> lm_cols(t, start, n)

# rows side by side: row i of the result is row i of a followed by row i of b
def lm_hcat(xs: List<&2, List<&2, F32>>, ys: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      List.append(&2, F32, x, y) <> lm_hcat(xt, yt)
    case _ _:
      Nil{}

def lm_dots(+a: List<&2, F32>, bs: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      l_dot(a, b, 0.0) <> lm_dots(a, t)

def lm_mul_t(as: List<&2, List<&2, F32>>, +bs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match as:
    case Nil{}:
      Nil{}
    case Con{+a, t}:
      lm_dots(a, bs) <> lm_mul_t(t, bs)

def lm_heads_of(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      List.append(&2, F32, List.take(&2, F32, r, 1n), lm_heads_of(t))

def lm_tails_of(rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      List.drop(&2, F32, r, 1n) <> lm_tails_of(t)

# the transpose of a matrix with c columns
def lm_t(c: Nat, +rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match c:
    case 0n:
      Nil{}
    case 1n+p:
      lm_heads_of(rows) <> lm_t(p, lm_tails_of(rows))

# a · b, with b of c columns
def lm_mul(+c: Nat, as: List<&2, List<&2, F32>>, +bs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  lm_mul_t(as, lm_t(c, bs))

def lm_zeros(r: Nat, +c: Nat) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      l_zeros(c) <> lm_zeros(p, c)

# a matrix's rows (handing the matrix back) and a matrix from rows
type MRows<-r: Nat, -c: Nat> is Type:
  MRows{m: TA.Mat<r, c>, rows: List<&2, List<&2, F32>>}

def mrows_fin2(+r: Nat, +c: Nat, ml: TA.ML<r, c>) -> MRows<r, c>:
  match ml:
    case TA.ML{m, +xs}:
      MRows{m, lm_chunk(r, c, xs)}

def Mat.rows(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> MRows<r, c>:
  mrows_fin2(r, c, TA.Mat.to_list(r, c, m))

def Mat.of_rows(+r: Nat, +c: Nat, rows: List<&2, List<&2, F32>>) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, lm_flat(rows))

# =====================================================================================================
# More element-wise activations
# =====================================================================================================

# an activation whose derivative needs its input (silu, gelu): saves a copy of the input
def acti_f3(~f: @+a: F32 -> F32, +n: Nat, +c: Nat, m: TA.MM2<n, c>) -> Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match m:
    case TA.MM2{x, s}:
      Fwd{Unit{}, Ew.map(~f, n, c, x), s}

def ActIn(~f: @+a: F32 -> F32, ~d: @+a: F32 -> @+b: F32 -> F32, +n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  Layer{p => x => acti_f3(~f, n, c, TA.Mat.clone(n, c, x)), p => s => dy => g => act_b2(n, c, Ew.zip(~d, n, c, dy, s)), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}}

def sigmoid_v(+x: F32) -> F32:
  (1.0 / (1.0 + F32.exp(F32.neg(x)) : F32) : F32)

# PyTorch's sigmoid_backward: dy * (1 - y) * y
def sigmoid_d(+dy: F32, +y: F32) -> F32:
  ((dy * (1.0 - y : F32) : F32) * y : F32)

def Sigmoid(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActOut(~sigmoid_v, ~sigmoid_d, n, c)

def tanh_v(+x: F32) -> F32:
  F32.tanh(x)

def tanh_d(+dy: F32, +y: F32) -> F32:
  (dy * (1.0 - (y * y : F32) : F32) : F32)

def Tanh(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActOut(~tanh_v, ~tanh_d, n, c)

# silu(x) = x * sigmoid(x), computed as PyTorch's CPU kernel: x / (1 + exp(-x));
# its gradient dy * s * (1 + x * (1 - s)) with s = sigmoid(x)
def silu_v(+x: F32) -> F32:
  (x / (1.0 + F32.exp(F32.neg(x)) : F32) : F32)

def silu_d(+dy: F32, +x: F32) -> F32:
  +s = sigmoid_v(x)
  ((dy * s : F32) * (1.0 + (x * (1.0 - s : F32) : F32) : F32) : F32)

def SiLU(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActIn(~silu_v, ~silu_d, n, c)

# GELU in its tanh form (PyTorch's approximate="tanh"): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
# The exact form needs erf, which Base does not have yet.
def gelu_inner(+x: F32) -> F32:
  (0.7978845608028654 * (x + (0.044715 * ((x * x : F32) * x : F32) : F32) : F32) : F32)

def gelu_v(+x: F32) -> F32:
  (((0.5 * x : F32) * (1.0 + F32.tanh(gelu_inner(x)) : F32)) : F32)

def gelu_d(+dy: F32, +x: F32) -> F32:
  +th = F32.tanh(gelu_inner(x))
  +left = (0.5 * x : F32)
  +dleft = (0.5 * (1.0 + th : F32) : F32)
  +dright = ((left * (1.0 - (th * th : F32) : F32) : F32) * (0.7978845608028654 * (1.0 + ((3.0 * 0.044715 : F32) * (x * x : F32) : F32) : F32) : F32) : F32)
  (dy * (dleft + dright : F32) : F32)

def GELU(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActIn(~gelu_v, ~gelu_d, n, c)

# =====================================================================================================
# Row-wise layers: softmax, log-softmax, RMSNorm, LayerNorm (each row of an n x c matrix on its own)
# =====================================================================================================

def rows_fin(+r: Nat, +c: Nat, -S: Type, s: S, rows: List<&2, List<&2, F32>>) -> Fwd<Unit, TA.Mat<r, c>, S>:
  Fwd{Unit{}, Mat.of_rows(r, c, rows), s}

def sm_rows(xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+x, t}:
      l_softmax(x) <> sm_rows(t)

# softmax backward, as PyTorch's: dx = y * (dy - sum(dy * y))
def sm_bwd_rows(ys: List<&2, List<&2, F32>>, dys: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match ys dys:
    case Con{+y, yt} Con{+dy, dt}:
      l_mul(y, l_shift(l_dot(dy, y, 0.0), dy)) <> sm_bwd_rows(yt, dt)
    case _ _:
      Nil{}

def sm_f2(+n: Nat, +c: Nat, r: MRows<n, c>) -> Fwd<Unit, TA.Mat<n, c>, List<&2, List<&2, F32>>>:
  match r:
    case MRows{m, +rows}:
      rows_fin(n, c, List<&2, List<&2, F32>>, sm_rows(rows), sm_rows(rows))

def rows_bwd_fin(+r: Nat, +c: Nat, rows: List<&2, List<&2, F32>>) -> Bwd<Unit, TA.Mat<r, c>>:
  Bwd{Unit{}, Mat.of_rows(r, c, rows), Unit{}}

def sm_b2(+n: Nat, +c: Nat, ys: List<&2, List<&2, F32>>, r: MRows<n, c>) -> Bwd<Unit, TA.Mat<n, c>>:
  match r:
    case MRows{m, dys}:
      rows_bwd_fin(n, c, sm_bwd_rows(ys, dys))

# softmax over each row; saves its output
def Softmax(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, List<&2, List<&2, F32>>>:
  Layer{p => x => sm_f2(n, c, Mat.rows(n, c, x)), p => s => dy => g => sm_b2(n, c, s, Mat.rows(n, c, dy)), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}}

# log_softmax(x) = (x - max) - log(sum(exp(x - max))), as PyTorch's CPU kernel
def lsm_row2(+t: List<&2, F32>) -> List<&2, F32>:
  l_shift(F32.log(l_sum(l_exp(t), 0.0)), t)

def lsm_rows(xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+x, t}:
      lsm_row2(l_shift(l_max(x), x)) <> lsm_rows(t)

# dx = dy - exp(y) * sum(dy)
def lsm_bwd_rows(ys: List<&2, List<&2, F32>>, dys: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match ys dys:
    case Con{y, yt} Con{+dy, dt}:
      l_sub(dy, l_scale(l_sum(dy, 0.0), l_exp(y))) <> lsm_bwd_rows(yt, dt)
    case _ _:
      Nil{}

def lsm_f2(+n: Nat, +c: Nat, r: MRows<n, c>) -> Fwd<Unit, TA.Mat<n, c>, List<&2, List<&2, F32>>>:
  match r:
    case MRows{m, +rows}:
      rows_fin(n, c, List<&2, List<&2, F32>>, lsm_rows(rows), lsm_rows(rows))

def lsm_b2(+n: Nat, +c: Nat, ys: List<&2, List<&2, F32>>, r: MRows<n, c>) -> Bwd<Unit, TA.Mat<n, c>>:
  match r:
    case MRows{m, dys}:
      rows_bwd_fin(n, c, lsm_bwd_rows(ys, dys))

def LogSoftmax(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, List<&2, List<&2, F32>>>:
  Layer{p => x => lsm_f2(n, c, Mat.rows(n, c, x)), p => s => dy => g => lsm_b2(n, c, s, Mat.rows(n, c, dy)), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}}

# ---- RMSNorm: y = g * (x / sqrt(mean(x^2) + eps)); its parameter is the gain g (1 x d) ----

def rms_r(+d: F32, +eps: F32, +x: List<&2, F32>) -> F32:
  (1.0 / F32.sqrt(((l_dot(x, x, 0.0) / d : F32) + eps : F32)) : F32)

def rms_rows(+d: F32, +eps: F32, +g: List<&2, F32>, xs: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+x, t}:
      l_mul(g, l_scale(rms_r(d, eps, x), x)) <> rms_rows(d, eps, g, t)

type RowGrad is Data:
  RowGrad{dx: List<&2, List<&2, F32>>, dg: List<&2, F32>, db: List<&2, F32>}

def rg_cons(dx: List<&2, F32>, r: RowGrad) -> RowGrad:
  match r:
    case RowGrad{rows, dg, db}:
      RowGrad{dx <> rows, dg, db}

# with r = 1/rms, xh = x*r, dxh = dy*g: dx = r * (dxh - xh * mean(dxh * xh)); dg += dy * xh
def rms_bwd_rows(+d: F32, +eps: F32, +g: List<&2, F32>, xs: List<&2, List<&2, F32>>, dys: List<&2, List<&2, F32>>, dg: List<&2, F32>) -> RowGrad:
  match xs dys:
    case Con{+x, xt} Con{+dy, dt}:
      +r = rms_r(d, eps, x)
      +xh = l_scale(r, x)
      +dxh = l_mul(dy, g)
      rg_cons(l_scale(r, l_sub(dxh, l_scale((l_dot(dxh, xh, 0.0) / d : F32), xh))), rms_bwd_rows(d, eps, g, xt, dt, l_add(dg, l_mul(dy, xh))))
    case _ _:
      RowGrad{Nil{}, dg, Nil{}}

# torch/gpu.bend's element-wise layers, RMSNorm, RoPE, attention and optimizer step against torch/main.bend's, on the
# same inputs at SmolLM2-135M's sizes (8 sequences of 64 positions: 512 rows; width 576, 9 query heads and 3 key/value
# heads of 64; MLP width 1536). Prints, for each case, the largest distance in ulps between the two results over every
# output (forward y, backward dX and the gradient; for the optimizer, the weights, the zeroed gradient and the moments):
#   SiLU, Mul2, Mat.sum2 and Mat.dup, RMSNorm, RoPE, Attention, SGD, momentum, AdamW, MSE (loss and gradient)
# With `--gpu off` or without CUDA every case is 0 (the same bits). On the GPU, SiLU and attention take CUDA's float exp,
# whose last bit can differ from the host's: 1 ulp is accepted for an exponential (my decision, 2026-10-18), and the
# differences that follow from it downstream are what this test shows; every other case is 0.
import Base
import ../torch/main.bend as T
import ../torch/gpu.bend as G
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

# deterministic values in [-1, 1), no specials (softmax and the norms would carry a NaN through a whole row)
def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  ((U32.to_f32(h) / 500.0 : F32) - 1.0 : F32)

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def arr(+cnt: Nat, +seed: U32) -> Array<F32>:
  fill(cnt, 0, seed, Array.new(F32, TA.cap_depth(cnt), 0.0))

def mat(+r: Nat, +c: Nat, +seed: U32) -> TA.Mat<r, c>:
  TA.Mat{arr(Nat.mul(r, c), seed)}

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def u32_max(+a: U32, +b: U32) -> U32:
  Bool.pick(U32, U32.is_lt(a, b), b, a)

# the distance in ulps between two floats: 0 for equal bits or two NaNs, across zero when the signs differ
def ulps(+x: F32, +y: F32) -> U32:
  +a = bits(x)
  +b = bits(y)
  +ma = U32.and(a, 2147483647)
  +mb = U32.and(b, 2147483647)
  Bool.pick(U32, Bool.or(U32.is_eq(a, b), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, Bool.pick(U32, U32.is_eq(U32.shrn(a, 31n), U32.shrn(b, 31n)), Bool.pick(U32, U32.is_lt(ma, mb), (mb - ma : U32), (ma - mb : U32)), (ma + mb : U32)))

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), u32_max(acc, ulps(x, y)), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def du(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

def dm(+r: Nat, +c: Nat, x: TA.Mat<r, c>, y: TA.Mat<r, c>) -> U32:
  match x y:
    case TA.Mat{a} TA.Mat{b}: du(Nat.mul(r, c), a, b)

def max3(+a: U32, +b: U32, +c: U32) -> U32:
  u32_max(u32_max(a, b), c)

def max4(+a: U32, +b: U32, +c: U32, +d: U32) -> U32:
  u32_max(u32_max(a, b), u32_max(c, d))

# one forward and one backward of a layer: its output, its dX and its gradient
type RR<-P: Type, -X: Type, -Y: Type> is Type:
  RR{y: Y, dx: X, g: P}

def run_b(-P: Type, -X: Type, -Y: Type, y: Y, r: T.Bwd<P, X>) -> RR<P, X, Y>:
  match r:
    case T.Bwd{p, dx, g}: RR{y, dx, g}

def run_f(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~l: T.Layer<P, X, Y, S>, dy: Y, g: P, r: T.Fwd<P, Y, S>) -> RR<P, X, Y>:
  match r:
    case T.Fwd{p, y, s}: run_b(P, X, Y, y, T.Layer.bwd(P, X, Y, S, l, p, s, dy, g))

def run(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~l: T.Layer<P, X, Y, S>, p: P, x: X, dy: Y, g: P) -> RR<P, X, Y>:
  run_f(~P, ~X, ~Y, ~S, ~l, dy, g, T.Layer.fwd(P, X, Y, S, l, p, x))

# ---- the element-wise layers ----

def c_m(+n: Nat, +c: Nat, a: RR<Unit, TA.Mat<n, c>, TA.Mat<n, c>>, b: RR<Unit, TA.Mat<n, c>, TA.Mat<n, c>>) -> U32:
  match a b:
    case RR{y1, d1, g1} RR{y2, d2, g2}: u32_max(dm(n, c, y1, y2), dm(n, c, d1, d2))

def silu() -> U32:
  c_m(512n, 1536n, run(~Unit, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~T.SiLU(512n, 1536n), Unit{}, mat(512n, 1536n, 1), mat(512n, 1536n, 2), Unit{}), run(~Unit, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~G.SiLU(512n, 1536n), Unit{}, mat(512n, 1536n, 1), mat(512n, 1536n, 2), Unit{}))

def c_mul(+n: Nat, +c: Nat, a: RR<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>>, b: RR<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>>) -> U32:
  match a b:
    case RR{y1, T.Two{a1, b1}, g1} RR{y2, T.Two{a2, b2}, g2}: max3(dm(n, c, y1, y2), dm(n, c, a1, a2), dm(n, c, b1, b2))

def two(+r: Nat, +c: Nat, +s: U32) -> T.Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  T.Two{mat(r, c, s), mat(r, c, (s + 1 : U32))}

def mul2() -> U32:
  c_mul(512n, 1536n, run(~Unit, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~TA.Mat<512n, 1536n>, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~T.Mul2(512n, 1536n), Unit{}, two(512n, 1536n, 11), mat(512n, 1536n, 13), Unit{}), run(~Unit, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~TA.Mat<512n, 1536n>, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~G.Mul2(512n, 1536n), Unit{}, two(512n, 1536n, 11), mat(512n, 1536n, 13), Unit{}))

def c_two(+r: Nat, +c: Nat, a: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>, b: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>) -> U32:
  match a b:
    case T.Two{a1, b1} T.Two{a2, b2}: u32_max(dm(r, c, a1, a2), dm(r, c, b1, b2))

def sumdup() -> U32:
  u32_max(dm(512n, 576n, T.Mat.sum2(512n, 576n, mat(512n, 576n, 21), mat(512n, 576n, 22)), G.Mat.sum2(512n, 576n, mat(512n, 576n, 21), mat(512n, 576n, 22))), c_two(512n, 576n, T.Mat.dup(512n, 576n, mat(512n, 576n, 23)), G.Mat.dup(512n, 576n, mat(512n, 576n, 23))))

# ---- RMSNorm: y, dx and the gain's gradient (added into a gradient that is not zero) ----

def c_rms(+n: Nat, +d: Nat, a: RR<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>, b: RR<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>) -> U32:
  match a b:
    case RR{y1, d1, g1} RR{y2, d2, g2}: max3(dm(n, d, y1, y2), dm(n, d, d1, d2), dm(1n, d, g1, g2))

def rms() -> U32:
  c_rms(512n, 576n, run(~TA.Mat<1n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~T.RMSNorm(512n, 576n, 0.00001), mat(1n, 576n, 31), mat(512n, 576n, 32), mat(512n, 576n, 33), mat(1n, 576n, 34)), run(~TA.Mat<1n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~G.RMSNorm(512n, 576n, 0.00001), mat(1n, 576n, 31), mat(512n, 576n, 32), mat(512n, 576n, 33), mat(1n, 576n, 34)))

# ---- RoPE on the 9 query heads ----

def c_rope(+n: Nat, +w: Nat, a: RR<Unit, TA.Mat<n, w>, TA.Mat<n, w>>, b: RR<Unit, TA.Mat<n, w>, TA.Mat<n, w>>) -> U32:
  match a b:
    case RR{y1, d1, g1} RR{y2, d2, g2}: u32_max(dm(n, w, y1, y2), dm(n, w, d1, d2))

def rope() -> U32:
  c_rope(512n, 576n, run(~Unit, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~Unit, ~T.RoPE(512n, 576n, 64n, 9n, 32n, 100000.0, {==}), Unit{}, mat(512n, 576n, 41), mat(512n, 576n, 42), Unit{}), run(~Unit, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~Unit, ~G.RoPE(512n, 576n, 64n, 9n, 32n, 100000.0, {==}), Unit{}, mat(512n, 576n, 41), mat(512n, 576n, 42), Unit{}))

# ---- attention: y, dQ, dK and dV ----

def qkv(+n: Nat, +w: Nat, +kvw: Nat, +s: U32) -> T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>:
  T.Two{mat(n, w, s), T.Two{mat(n, kvw, (s + 1 : U32)), mat(n, kvw, (s + 2 : U32))}}

def c_att(+n: Nat, +w: Nat, +kvw: Nat, a: RR<Unit, T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>, TA.Mat<n, w>>, b: RR<Unit, T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>, TA.Mat<n, w>>) -> U32:
  match a b:
    case RR{y1, T.Two{q1, T.Two{k1, v1}}, g1} RR{y2, T.Two{q2, T.Two{k2, v2}}, g2}: max4(dm(n, w, y1, y2), dm(n, w, q1, q2), dm(n, kvw, k1, k2), dm(n, kvw, v1, v2))

def attn() -> U32:
  c_att(512n, 576n, 192n, run(~Unit, ~T.Two<TA.Mat<512n, 576n>, T.Two<TA.Mat<512n, 192n>, TA.Mat<512n, 192n>>>, ~TA.Mat<512n, 576n>, ~T.AttnS, ~T.Attention(512n, 576n, 192n, 8n, 64n, 3n, 3n, 64n, 0.125, {==}, {==}, {==}), Unit{}, qkv(512n, 576n, 192n, 51), mat(512n, 576n, 54), Unit{}), run(~Unit, ~T.Two<TA.Mat<512n, 576n>, T.Two<TA.Mat<512n, 192n>, TA.Mat<512n, 192n>>>, ~TA.Mat<512n, 576n>, ~T.AttnS, ~G.Attention(512n, 576n, 192n, 8n, 64n, 3n, 3n, 64n, 0.125, {==}, {==}, {==}), Unit{}, qkv(512n, 576n, 192n, 51), mat(512n, 576n, 54), Unit{}))

# ---- the optimizer step on a 1536 x 576 matrix: weights, gradient (zeroed) and moments ----

def c_st(+r: Nat, +c: Nat, a: T.OptState<TA.Mat<r, c>>, b: T.OptState<TA.Mat<r, c>>) -> U32:
  match a b:
    case T.OneMom{m1} T.OneMom{m2}: dm(r, c, m1, m2)
    case T.TwoMom{m1, v1} T.TwoMom{m2, v2}: u32_max(dm(r, c, m1, m2), dm(r, c, v1, v2))
    case _ _: 0

def c_step(+r: Nat, +c: Nat, a: T.Step<TA.Mat<r, c>>, b: T.Step<TA.Mat<r, c>>) -> U32:
  match a b:
    case T.Step{p1, g1, s1} T.Step{p2, g2, s2}: max3(dm(r, c, p1, p2), dm(r, c, g1, g2), c_st(r, c, s1, s2))

# second moments are not negative
def sq(k: Nat, +i: U32, r: Array<F32> & F32) -> Array<F32>:
  match k r:
    case 0n Tuple{a, x}: a
    case 1n+p Tuple{a, +x}: sq(p, (i + 1 : U32), Array.get(F32, Array.set(F32, a, i, (x * x : F32)), (i + 1 : U32)))

def arr2(+cnt: Nat, +seed: U32) -> Array<F32>:
  sq(cnt, 0, Array.get(F32, arr(cnt, seed), 0))

def st_of(+r: Nat, +c: Nat, +k: Nat, +s: U32) -> T.OptState<TA.Mat<r, c>>:
  match k:
    case 0n: T.NoMom{}
    case 1n: T.OneMom{mat(r, c, s)}
    case _: T.TwoMom{mat(r, c, s), TA.Mat{arr2(Nat.mul(r, c), (s + 1 : U32))}}

def step(+op: T.Opt, +k: Nat) -> U32:
  c_step(1536n, 576n, T.mat_step(1536n, 576n, op, mat(1536n, 576n, 61), mat(1536n, 576n, 62), st_of(1536n, 576n, k, 63)), G.mat_step(1536n, 576n, op, mat(1536n, 576n, 61), mat(1536n, 576n, 62), st_of(1536n, 576n, k, 63)))

# ---- MSE: the loss and the gradient, at the block's size and at counts with fewer powers of two ----

def c_loss(+n: Nat, +c: Nat, a: T.LossOut<TA.Mat<n, c>>, b: T.LossOut<TA.Mat<n, c>>) -> U32:
  match a b:
    case T.LossOut{+l1, d1} T.LossOut{+l2, d2}: u32_max(ulps(l1, l2), dm(n, c, d1, d2))

def mse1(+n: Nat, +c: Nat, +s: U32) -> U32:
  c_loss(n, c, T.MSE(n, c, mat(n, c, s), mat(n, c, (s + 1 : U32))), G.MSE(n, c, mat(n, c, s), mat(n, c, (s + 1 : U32))))

def mse() -> U32:
  max3(mse1(512n, 576n, 71), mse1(7n, 5n, 73), mse1(100n, 10n, 75))

def main() -> IO(Unit):
  IO.print(U32.show(silu()) ++ " " ++ U32.show(mul2()) ++ " " ++ U32.show(sumdup()) ++ " " ++ U32.show(rms()) ++ " " ++ U32.show(rope()) ++ " " ++ U32.show(attn()) ++ " " ++ U32.show(step(T.SGD{0.01}, 0n)) ++ " " ++ U32.show(step(T.Momentum{0.01, 0.9}, 1n)) ++ " " ++ U32.show(step(T.Opt.at(T.CAdamW{0.001, 0.9, 0.999, 0.00000001, 0.01}, 0.9, 0.999), 2n)) ++ " " ++ U32.show(mse()))

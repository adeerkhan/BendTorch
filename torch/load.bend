# Temporary loading-only exception: bend-ml-binary-read is foreign C/JavaScript. No file with laws imports this module.
# Replace it with native Bend when the language supports dense reads (Renan, 2026-10-17).
# The reader is a local copy of bend-ml-binary-read@0.1.0.0 with 64-bit offsets and BF16 files added
# (vendor/binary-read/main.bend says what changed).
import Base
import ../vendor/binary-read/main.bend as R
import ./main.bend as T

def require(ok: Bool, +message: String) -> IO(Unit):
  match ok:
    case True{}: IO.pure(Unit, Unit{})
    case False{}: IO.die(Unit, 1, message)

# =====================================================================================================
# W64: a 64-bit byte count or offset as two U32s, hi * 2^32 + lo. Bend 2.0.37 has no 64-bit integer, and a file past
# 4 GiB needs one (SmolLM2-1.7B's F32 weights, a 360M AdamW checkpoint). Plain Bend: none of these reaches the reader.
# =====================================================================================================

type W64 is Data:
  W64{hi: U32, lo: U32}

def W64.of(x: U32) -> W64:
  W64{0, x}

def W64.of_pair(p: U32 & U32) -> W64:
  match p:
    case Tuple{hi, lo}: W64{hi, lo}

def W64.bit(b: Bool) -> U32:
  match b:
    case True{}: 1
    case False{}: 0

def W64.eq(a: W64, b: W64) -> Bool:
  match a b:
    case W64{ah, al} W64{bh, bl}: Bool.and(U32.is_eq(ah, bh), U32.is_eq(al, bl))

def W64.add_go(+ah: U32, +bh: U32, +s: U32, +al: U32) -> W64:
  W64{((ah + bh : U32) + W64.bit(U32.is_lt(s, al)) : U32), s}

# a + b, wrapping at 2^64: the low words' sum carries one when it wrapped (it is then below either term)
def W64.add(a: W64, b: W64) -> W64:
  match a b:
    case W64{ah, +al} W64{bh, bl}: W64.add_go(ah, bh, (al + bl : U32), al)

def W64.mul_go(+h: U32, +k: U32, +p0: U32, +p1: U32) -> W64:
  W64.add(W64{((h * k : U32) + U32.shrn(p1, 16n) : U32), p0}, W64{0, U32.shln(p1, 16n)})

# a * k for k < 2^16 (enough for bytes a number and decimal digits), wrapping at 2^64: lo = l1 2^16 + l0, so
# lo * k = (l1 k) 2^16 + l0 k, where both products fit a U32
def W64.mul_small(a: W64, +k: U32) -> W64:
  match a:
    case W64{h, +l}: W64.mul_go(h, k, (U32.and(l, 65535) * k : U32), (U32.shrn(l, 16n) * k : U32))

def W64.div10_go(+qh: U32, +r1: U32, +q1: U32, +l: U32) -> W64 & U32:
  +t = (U32.shln(r1, 16n) + U32.and(l, 65535) : U32)
  Tuple{W64{qh, U32.or(U32.shln(q1, 16n), U32.div(t, 10))}, U32.mod(t, 10)}

# (a / 10, a % 10), by long division in 16-bit steps so no partial remainder overflows a U32
def W64.div10(a: W64) -> W64 & U32:
  match a:
    case W64{+h, +l}:
      +t = (U32.shln(U32.mod(h, 10), 16n) + U32.shrn(l, 16n) : U32)
      W64.div10_go(U32.div(h, 10), U32.mod(t, 10), U32.div(t, 10), l)

def W64.is_zero(a: W64) -> Bool:
  match a:
    case W64{h, l}: Bool.and(U32.is_zero(h), U32.is_zero(l))

def W64.show_go(f: Nat, z: Bool, qr: W64 & U32, acc: String) -> String:
  match f z qr:
    case _ True{} Tuple{q, r}: acc
    case 0n False{} Tuple{q, r}: acc
    case 1n+g False{} Tuple{+q, r}: W64.show_go(g, W64.is_zero(q), W64.div10(q), U32.show(r) ++ acc)

def W64.show_if(z: Bool, +a: W64) -> String:
  match z:
    case True{}: "0"
    case False{}: W64.show_go(20n, False{}, W64.div10(a), "")

# the decimal digits, as U32.show writes them
def W64.show(+a: W64) -> String:
  W64.show_if(W64.is_zero(a), a)

def W64.digits(s: String, f: Nat) -> Bool:
  match s f:
    case SNil{} _: True{}
    case SCon{c, t} 0n: False{}
    case SCon{c, t} 1n+g: Bool.and(Char.is_digit(c), W64.digits(t, g))

def W64.read_go(s: String, +acc: W64) -> W64:
  match s:
    case SNil{}: acc
    case SCon{Chr{x}, t}: W64.read_go(t, W64.add(W64.mul_small(acc, 10), W64.of(U32.sub(x, 48))))

def W64.read_if(ok: Bool, s: String) -> Maybe<&2, W64>:
  match ok:
    case True{}: Some{W64.read_go(s, W64{0, 0})}
    case False{}: None{}

# A decimal number of 1 to 19 digits (every such number fits 64 bits); anything else is None
def W64.read(s: String) -> Maybe<&2, W64>:
  match s:
    case SNil{}: None{}
    case SCon{+h, +t}: W64.read_if(W64.digits(SCon{h, t}, 19n), SCon{h, t})

# =====================================================================================================
# Reading
# =====================================================================================================

def read_done(+path: String, +n: U32, r: File & (Array<F32> & Result<&1, &1, U32 & String, U32>)) -> IO(Array<F32>):
  match r:
    case Tuple{f, Tuple{a, res}}:
      do IO<Array<F32>>:
        +got : U32 <- IO.pass(U32, res)
        File.close(f)
        require(U32.is_eq(got, n), path ++ ": short read (" ++ U32.show(got) ++ " of " ++ U32.show(n) ++ " numbers)")
        return a

def read_go(+path: String, +n: U32, f: File, +size: W64) -> IO(Array<F32>):
  do IO<Array<F32>>:
    require(W64.eq(size, W64.mul_small(W64.of(n), 4)), path ++ ": expected " ++ W64.show(W64.mul_small(W64.of(n), 4)) ++ " bytes, got " ++ W64.show(size))
    rr : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_f32_at(f, 0, 0, n, Array.new(F32, T.depth(U32.to_nat(n)), 0.0))
    read_done(path, n, rr)

def read_sized(+path: String, +n: U32, r: File & Result<&1, &1, U32 & String, U32 & U32>) -> IO(Array<F32>):
  match r:
    case Tuple{f, res}:
      do IO<Array<F32>>:
        sz : U32 & U32 <- IO.pass(U32 & U32, res)
        read_go(path, n, f, W64.of_pair(sz))

# A raw, exact-size little-endian F32 file, read once into one source array: up to 2^32 - 1 numbers (16 GiB).
def read(+path: String, +n: U32) -> IO(Array<F32>):
  do IO<Array<F32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    rr : File & Result<&1, &1, U32 & String, U32 & U32> <- R.size64(f)
    read_sized(path, n, rr)

# n little-endian F32 numbers at byte `off` of a file of any size (a tensor inside a larger file)
def read_f32_at(+path: String, off: W64, +n: U32) -> IO(Array<F32>):
  match off:
    case W64{hi, lo}:
      do IO<Array<F32>>:
        f : File <- IO.try(File, File.open(path, "r"))
        rr : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_f32_at(f, hi, lo, n, Array.new(F32, T.depth(U32.to_nat(n)), 0.0))
        read_done(path, n, rr)

# n little-endian BF16 numbers at byte `off` of a file of any size, each widened exactly to F32 (bf16.to(torch.float32))
def read_bf16_at(+path: String, off: W64, +n: U32) -> IO(Array<F32>):
  match off:
    case W64{hi, lo}:
      do IO<Array<F32>>:
        f : File <- IO.try(File, File.open(path, "r"))
        rr : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_bf16_at(f, hi, lo, n, Array.new(F32, T.depth(U32.to_nat(n)), 0.0))
        read_done(path, n, rr)

def size_done(r: File & Result<&1, &1, U32 & String, U32 & U32>) -> IO(W64):
  match r:
    case Tuple{f, res}:
      do IO<W64>:
        sz : U32 & U32 <- IO.pass(U32 & U32, res)
        File.close(f)
        return W64.of_pair(sz)

# A file's size in bytes, at any size
def size(+path: String) -> IO(W64):
  do IO<W64>:
    f : File <- IO.try(File, File.open(path, "r"))
    rr : File & Result<&1, &1, U32 & String, U32 & U32> <- R.size64(f)
    size_done(rr)

def weights_done(-P: Type, +n: U32, r: T.PA<P>) -> IO(P):
  match r:
    case T.PA{p, xs, +off}:
      do IO<P>:
        require(U32.is_eq(off, n), "Layer.count differs from the loading cursor")
        return p

# All parameters in Layer.dump order. The source array is released after the model's arrays are filled.
def weights(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, p: P) -> IO(P):
  do IO<P>:
    +n : U32 = T.Layer.count(P, X, Y, S, m)
    a : Array<F32> <- read(path, n)
    weights_done(P, n, T.Layer.load(P, X, Y, S, m, p, a, 0))

# The checkpoint layout is determined by the already initialized optimizer state, without a header:
# SGD: parameters; momentum: parameters, buffer; Adam/AdamW: parameters, m, v, 1 - b1^t, 1 - b2^t.
def train_read(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, +base: U32, +copies: U32, +tail: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  do IO<T.Train<P>>:
    require(U32.is_le(base, U32.div(U32.sub(4294967295, tail), copies)), "checkpoint exceeds 2^32 - 1 numbers")
    +n : U32 = ((copies * base : U32) + tail : U32)
    a : Array<F32> <- read(path, n)
    return T.Train.restore(~P, ~X, ~Y, ~S, ~m, tr, a)

def train(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, tr: T.Train<P>) -> IO(T.Train<P>):
  match tr:
    case T.Train{p, g, T.NoMom{}, p1, p2, loss}:
      train_read(~P, ~X, ~Y, ~S, ~m, path, T.Layer.count(P, X, Y, S, m), 1, 0, T.Train{p, g, T.NoMom{}, p1, p2, loss})
    case T.Train{p, g, T.OneMom{mm}, p1, p2, loss}:
      train_read(~P, ~X, ~Y, ~S, ~m, path, T.Layer.count(P, X, Y, S, m), 2, 0, T.Train{p, g, T.OneMom{mm}, p1, p2, loss})
    case T.Train{p, g, T.TwoMom{mm, vv}, p1, p2, loss}:
      train_read(~P, ~X, ~Y, ~S, ~m, path, T.Layer.count(P, X, Y, S, m), 3, 2, T.Train{p, g, T.TwoMom{mm, vv}, p1, p2, loss})

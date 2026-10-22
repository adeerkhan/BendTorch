# Temporary loading-only exception: bend-ml-binary-read is foreign C/JavaScript. No file with laws imports this module.
# Replace it with native Bend when the language supports dense reads (Renan, 2026-10-17).
import Base
import bend-ml-binary-read@0.1.0.0/main.bend as R
import ./main.bend as T

def require(ok: Bool, +message: String) -> IO(Unit):
  match ok:
    case True{}: IO.pure(Unit, Unit{})
    case False{}: IO.die(Unit, 1, message)

def read_done(+path: String, +n: U32, r: File & (Array<F32> & Result<&1, &1, U32 & String, U32>)) -> IO(Array<F32>):
  match r:
    case Tuple{f, Tuple{a, res}}:
      do IO<Array<F32>>:
        +got : U32 <- IO.pass(U32, res)
        File.close(f)
        require(U32.is_eq(got, (4 * n : U32)), path ++ ": short F32 read")
        return a

def read_sized(+path: String, +n: U32, r: File & Result<&1, &1, U32 & String, U32>) -> IO(Array<F32>):
  match r:
    case Tuple{f, res}:
      do IO<Array<F32>>:
        +size : U32 <- IO.pass(U32, res)
        require(U32.is_eq(size, (4 * n : U32)), path ++ ": expected " ++ U32.show((4 * n : U32)) ++ " bytes, got " ++ U32.show(size))
        rr : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_f32(f, 0, n, Array.new(F32, T.depth(U32.to_nat(n)), 0.0))
        read_done(path, n, rr)

# A raw, exact-size little-endian F32 file, read once into one source array. Counts and file offsets are U32.
def read(+path: String, +n: U32) -> IO(Array<F32>):
  do IO<Array<F32>>:
    require(U32.is_le(n, 1073741823), "F32 file exceeds the binary reader's U32 byte limit")
    f : File <- IO.try(File, File.open(path, "r"))
    rr : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
    read_sized(path, n, rr)

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
    require(U32.is_le(base, U32.div(U32.sub(1073741823, tail), copies)), "checkpoint exceeds the binary reader's U32 byte limit")
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

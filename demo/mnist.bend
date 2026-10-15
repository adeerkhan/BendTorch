# MNIST with BendTorch's general layers: LinearFirst(784, 128) -> ReLU -> Linear(128, 10), cross-entropy, any optimizer.
# The training step is the package's Train.step, the same code that trains every other model; nothing here is
# specific to MNIST but the shapes and the data.
#
#   bend demo/mnist.bend -o .scratch/mnist && .scratch/mnist <epochs> <batches (0 = all 600)> <sgd|momentum|adam|adamw> <lr>
#
# Data in data/mnist/ (scripts/mnist.sh); initial weights from reference/mnist_ref.py --init, its PyTorch twin.

import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.7.0/main.bend as TA

def X() -> Type:
  TA.Mat<100n, 784n>

def Y() -> Type:
  TA.Mat<100n, 10n>

def P() -> Type:
  T.Two<T.Two<T.Lin<784n, 128n>, Unit>, T.Lin<128n, 10n>>

def S() -> Type:
  T.Two<T.Two<TA.Mat<100n, 784n>, TA.Mat<100n, 128n>>, TA.Mat<100n, 128n>>

def Hidden() -> T.Layer<T.Two<T.Lin<784n, 128n>, Unit>, X, TA.Mat<100n, 128n>, T.Two<X, TA.Mat<100n, 128n>>>:
  T.Seq(~T.Lin<784n, 128n>, ~Unit, ~X, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 128n>, ~X, ~TA.Mat<100n, 128n>, ~T.LinearFirst(100n, 784n, 128n, 3n), ~T.ReLU(100n, 128n))

def Model() -> T.Layer<P, X, Y, S>:
  T.Seq(~T.Two<T.Lin<784n, 128n>, Unit>, ~T.Lin<128n, 10n>, ~X, ~TA.Mat<100n, 128n>, ~Y, ~T.Two<X, TA.Mat<100n, 128n>>, ~TA.Mat<100n, 128n>, ~Hidden(), ~T.Linear(100n, 128n, 10n, 0n))

def ce(z: Y, labels: List<&2, Nat>) -> T.LossOut<Y>:
  T.CrossEntropy(100n, 10n, z, labels)

# ---- data ----

def scale255(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      (U32.to_f32(b) / 255.0 : F32) <> scale255(t)

def labs(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labs(t)

def batch_or_die(m: Maybe<&1, X>) -> IO(X):
  match m:
    case None{}:
      IO.die(X, 1, "data/mnist: a batch is short (run scripts/mnist.sh)")
    case Some{x}:
      IO.pure(X, x)

def load_x(+path: String, +b: U32) -> IO(X):
  do IO<X>:
    bs : List<&2, U32> <- T.read_at(path, (16 + (b * 78400 : U32) : U32), 78400)
    batch_or_die(TA.Mat.of_list(100n, 784n, scale255(bs)))

def load_y(+path: String, +b: U32) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(path, (8 + (b * 100 : U32) : U32), 100)
    return labs(bs)

# ---- training and evaluation ----

# k more batches from batch b; acc sums the losses (the record's own loss is the last step's, 0 before the first), and
# the record that comes out holds the sum
def tloop(k: Nat, +b: U32, +cfg: T.OptCfg, +acc: F32, tr: T.Train<P>) -> IO(T.Train<P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      IO.pure(T.Train<P>, T.Train{p, g, st, p1, p2, (acc + l : F32)})
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        x : X <- load_x("data/mnist/train-images-idx3-ubyte", b)
        ys : List<&2, Nat> <- load_y("data/mnist/train-labels-idx1-ubyte", b)
        tloop(q, (b + 1 : U32), cfg, (acc + l : F32), T.Train.step(~P, ~X, ~Y, ~S, ~List<&2, Nat>, ~Model(), ~ce, cfg, T.Train{p, g, st, p1, p2, l}, x, ys))

type Hits is Type:
  Hits{n: Nat, p: P}

def ev3(+n: Nat, p: P, r: TA.MHits<100n, 10n>) -> Hits:
  match r:
    case TA.MHits{z, k}:
      Hits{Nat.add(n, k), p}

def ev2(+n: Nat, +ys: List<&2, Nat>, r: T.Fwd<P, Y, S>) -> Hits:
  match r:
    case T.Fwd{p, y, s}:
      ev3(n, p, TA.Mat.count_correct(100n, 10n, y, ys))

def eloop(k: Nat, +b: U32, h: Hits) -> IO(Hits):
  match k h:
    case 0n h:
      IO.pure(Hits, h)
    case 1n+q Hits{n, p}:
      do IO<Hits>:
        x : X <- load_x("data/mnist/t10k-images-idx3-ubyte", b)
        ys : List<&2, Nat> <- load_y("data/mnist/t10k-labels-idx1-ubyte", b)
        eloop(q, (b + 1 : U32), ev2(n, ys, T.Layer.fwd(P, X, Y, S, Model(), p, x)))

# ---- epochs ----

def ms_show(+ms: Nat) -> String:
  Nat.show(Nat.div(ms, 1000n)) ++ "." ++ Nat.show(Nat.div(Nat.mod(ms, 1000n), 100n))

def report(+e: Nat, +nb: Nat, +secs: Nat, +sum: F32, h: Hits) -> IO(P):
  match h:
    case Hits{n, p}:
      do IO<P>:
        IO.print("epoch " ++ Nat.show(e) ++ " train_loss=" ++ F32.show((sum / F32.from_nat(nb) : F32)) ++ " test_correct=" ++ Nat.show(n) ++ "/10000 train_seconds=" ++ ms_show(secs))
        return p

# the next epoch starts from these parameters with a fresh loss sum; the gradient and the optimizer state carry over
def epoch_end(+e: Nat, +nb: Nat, +secs: Nat, tr: T.Train<P>) -> IO(T.Train<P>):
  match tr:
    case T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        h : Hits <- eloop(100n, 0, Hits{0n, p})
        p3 : P <- report(e, nb, secs, l, h)
        return T.Train{p3, g, st, p1, p2, 0.0}

def epochs(k: Nat, +e: Nat, +nb: Nat, +cfg: T.OptCfg, tr: T.Train<P>) -> IO(Unit):
  match k:
    case 0n:
      IO.print("done")
    case 1n+q:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        tr2 : T.Train<P> <- tloop(nb, 0, cfg, 0.0, tr)
        t1 : Nat <- IO.now()
        tr3 : T.Train<P> <- epoch_end(e, nb, Nat.sub(t1, t0), tr2)
        epochs(q, 1n+e, nb, cfg, tr3)

def load_params() -> IO(P):
  do IO<P>:
    w1 : TA.Mat<128n, 784n> <- T.Mat.load(128n, 784n, "data/mnist/init/w1.bin", 0)
    b1 : TA.Mat<1n, 128n> <- T.Mat.load(1n, 128n, "data/mnist/init/b1.bin", 0)
    w2 : TA.Mat<10n, 128n> <- T.Mat.load(10n, 128n, "data/mnist/init/w2.bin", 0)
    b2 : TA.Mat<1n, 10n> <- T.Mat.load(1n, 10n, "data/mnist/init/b2.bin", 0)
    return T.Two{T.Two{T.Lin{w1, b1}, Unit{}}, T.Lin{w2, b2}}

def cfg_pick(a: Bool, b: Bool, c: Bool, d: Bool, +lr: F32) -> Maybe<&2, T.OptCfg>:
  match a b c d:
    case True{} _ _ _:
      Some{T.CSGD{lr}}
    case False{} True{} _ _:
      Some{T.CMomentum{lr, 0.9}}
    case False{} False{} True{} _:
      Some{T.CAdam{lr, 0.9, 0.999, 0.00000001, 0.0}}
    case False{} False{} False{} True{}:
      Some{T.CAdamW{lr, 0.9, 0.999, 0.00000001, 0.01}}
    case _ _ _ _:
      None{}

def cfg_of(+name: String, +lr: F32) -> Maybe<&2, T.OptCfg>:
  cfg_pick(String.eq(name, "sgd"), String.eq(name, "momentum"), String.eq(name, "adam"), String.eq(name, "adamw"), lr)

def nb_of(all: Bool, +m: Nat) -> Nat:
  match all:
    case True{}:
      600n
    case False{}:
      m

def go(+e: Nat, +m: Nat, mc: Maybe<&2, T.OptCfg>) -> IO(Unit):
  match mc:
    case None{}:
      IO.print("optimizer: sgd, momentum, adam or adamw")
    case Some{cfg}:
      do IO<Unit>:
        p : P <- load_params()
        epochs(e, 1n, nb_of(Nat.is_eq(m, 0n), m), cfg, T.Train.new(~P, ~X, ~Y, ~S, ~Model(), p))

def go2(me: Maybe<&2, Nat>, mm: Maybe<&2, Nat>, +name: String, ml: Maybe<&2, F32>) -> IO(Unit):
  match me mm ml:
    case Some{e} Some{m} Some{lr}:
      go(e, m, cfg_of(name, lr))
    case _ _ _:
      IO.print("usage: mnist <epochs> <batches (0 = all)> <sgd|momentum|adam|adamw> <lr>")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{e, Con{m, Con{o, Con{lr, Nil{}}}}}}:
      go2(Nat.read(e), Nat.read(m), o, F32.read(lr))
    case _:
      IO.print("usage: mnist <epochs> <batches (0 = all)> <sgd|momentum|adam|adamw> <lr>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)

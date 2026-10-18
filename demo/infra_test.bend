# M9 infrastructure: one tied parameter, a Dropout counter, all optimizer states, loading cursors and checkpoints.
import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import bend-ml-tensor-array@0.1.13.0/main.bend as TA

def M() -> Type:
  TA.Mat<2n, 3n>

def P() -> Type:
  T.Two<TA.Mat<3n, 3n>, T.DropP>

def S() -> Type:
  T.Three<M, M, M>

def Model() -> T.Layer<P, M, M, S>:
  T.Tie(~TA.Mat<3n, 3n>, ~T.DropP, ~M, ~M, ~M, ~M, ~M, ~M, ~M, ~T.LinearNB(2n, 3n, 3n, 0n), ~T.Dropout(2n, 3n, 0.25, 7), ~T.LinearNB(2n, 3n, 3n, 0n))

def adamw() -> T.OptCfg:
  # The dispatcher validates the name before using this selection.
  T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.01}

def choose(a: Bool, b: Bool, c: Bool, d: Bool) -> IO(T.OptCfg):
  match a b c d:
    case True{} _ _ _: IO.pure(T.OptCfg, T.CSGD{0.01})
    case _ True{} _ _: IO.pure(T.OptCfg, T.CMomentum{0.01, 0.9})
    case _ _ True{} _: IO.pure(T.OptCfg, T.CAdam{0.01, 0.9, 0.999, 0.00000001, 0.0})
    case _ _ _ True{}: IO.pure(T.OptCfg, adamw())
    case _ _ _ _: IO.die(T.OptCfg, 1, "unknown optimizer")

def initial(+c: T.OptCfg) -> T.Train<P>:
  T.Train.new(~P, ~M, ~M, ~S, ~Model(), c, T.Two{TA.Mat.from_list(3n, 3n, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), T.DropP{0}})

def loss(x: M, y: M) -> T.LossOut<M>:
  T.MSE(2n, 3n, x, y)

def steps(k: Nat, +c: T.OptCfg, tr: T.Train<P>) -> T.Train<P>:
  match k:
    case 0n: tr
    case 1n+q:
      steps(q, c, T.Train.step(~P, ~M, ~M, ~S, ~M, ~Model(), ~loss, c, tr, TA.Mat.from_list(2n, 3n, [0.2, 0.1, 0.4, 0.3, 0.6, 0.5]), TA.Mat.from_list(2n, 3n, [0.1, 0.3, 0.2, 0.4, 0.3, 0.5])))

def finish(r: T.TrainPL<P>) -> IO(Unit):
  match r:
    case T.TrainPL{tr, xs}: IO.print("state" ++ T.words(xs))

def params_dump(r: T.PL<P>) -> IO(Unit):
  match r:
    case T.PL{p, xs}: IO.print("params" ++ T.words(xs))

def params_fin(p: P) -> IO(Unit):
  params_dump(T.Layer.dump(P, M, M, S, Model(), p))

def params_of(+path: String, tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      do IO<Unit>:
        q : P <- T.Params.save(~P, ~M, ~M, ~S, ~Model(), path, p)
        params_fin(q)

def run(mode: String, +c: T.OptCfg, +k: Nat, +path: String, load: Bool, save: Bool, params: Bool) -> IO(Unit):
  match load save params:
    case _ _ True{}:
      params_of(path, steps(k, c, initial(c)))
    case True{} _ _:
      do IO<Unit>:
        tr : T.Train<P> <- L.train(~P, ~M, ~M, ~S, ~Model(), path, initial(c))
        finish(T.Train.dump(~P, ~M, ~M, ~S, ~Model(), steps(k, c, tr)))
    case _ True{} _:
      do IO<Unit>:
        tr : T.Train<P> <- T.Train.save(~P, ~M, ~M, ~S, ~Model(), path, steps(k, c, initial(c)))
        finish(T.Train.dump(~P, ~M, ~M, ~S, ~Model(), tr))
    case _ _ _:
      finish(T.Train.dump(~P, ~M, ~M, ~S, ~Model(), steps(k, c, initial(c))))

def go(+mode: String, +opt: String, mk: Maybe<&2, Nat>, +path: String) -> IO(Unit):
  match mk:
    case None{}: IO.die(Unit, 1, "invalid step count")
    case Some{+k}:
      do IO<Unit>:
        c : T.OptCfg <- choose(String.eq(opt, "sgd"), String.eq(opt, "momentum"), String.eq(opt, "adam"), String.eq(opt, "adamw"))
        run(mode, c, k, path, String.eq(mode, "resume"), String.eq(mode, "save"), String.eq(mode, "params"))

def cursor_done(r: T.PA<TA.Mat<3n, 3n>>) -> IO(Unit):
  match r:
    case T.PA{m, a, off}:
      do IO<Unit>:
        IO.print("offset " ++ U32.show(off))
        T.Mat.print(3n, 3n, "weights", m)

def cursor(+path: String) -> IO(Unit):
  do IO<Unit>:
    a : Array<F32> <- L.read(path, 11)
    cursor_done(T.Mat.restore(3n, 3n, TA.Mat.zeros(3n, 3n), a, 1))

def read_last(r: Array<F32> & F32, +first: F32) -> IO(Unit):
  match r:
    case Tuple{a, last}: IO.print("ends " ++ U32.show(T.bits(first)) ++ " " ++ U32.show(T.bits(last)))

def read_first(+n: U32, r: Array<F32> & F32) -> IO(Unit):
  match r:
    case Tuple{a, +first}: read_last(Array.get(F32, a, U32.sub(n, 1)), first)

def read_file(+path: String, mn: Maybe<&2, U32>) -> IO(Unit):
  match mn:
    case None{}: IO.die(Unit, 1, "invalid count")
    case Some{+n}:
      do IO<Unit>:
        L.require(U32.is_gt(n, 0), "count must be positive")
        a : Array<F32> <- L.read(path, n)
        read_first(n, Array.get(F32, a, 0))

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{mode, Con{opt, Con{k, Con{path, Nil{}}}}}}: go(mode, opt, Nat.read(k), path)
    case Con{prog, Con{mode, Con{path, Nil{}}}}: cursor(path)
    case Con{prog, Con{mode, Con{path, Con{n, Nil{}}}}}: read_file(path, U32.read(n))
    case _: IO.die(Unit, 1, "usage: infra_test run|save|resume|params opt steps file | cursor file | read file count")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)

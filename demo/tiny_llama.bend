# A two-layer character-level Llama trained in Bend on Tiny Shakespeare, against its PyTorch twin
# (reference/tiny_ref.py, which also writes the initial weights and the batches into data/tiny/).
#
#   bend demo/tiny_llama.bend -o .scratch/tiny_llama
#   .scratch/tiny_llama grads           # the loss and every gradient on the first batch
#   .scratch/tiny_llama train <steps>   # AdamW steps: the loss before each update, then the final weights
#
# Every block is composed from the package's layers: Residual(Seq(RMSNorm, attention)) then
# Residual(Seq(RMSNorm, SwiGLU)); attention is Fork(q, Fork(k, v)) -> Both(RoPE, Both(RoPE, Id)) -> Attention -> wo.
# Rows are the 2 x 16 positions of a batch (N = 32); vocabulary 65, width 32, 4 query heads and 2 key/value heads of 8,
# feed-forward 64. Each number is printed as the decimal of its 32 bits.

import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.7.0/main.bend as TA

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def Rows() -> Type:
  List<&2, List<&2, F32>>

# ---- attention: q, k, v projections, RoPE on q and k, the attention core, the output projection ----

def KVP() -> Type:
  T.Two<M(16n, 32n), M(16n, 32n)>

def KVY() -> Type:
  T.Two<M(32n, 16n), M(32n, 16n)>

def KVS() -> Type:
  T.Two<M(32n, 32n), M(32n, 32n)>

def KVL() -> T.Layer<KVP, M(32n, 32n), KVY, KVS>:
  T.Fork(~M(16n, 32n), ~M(16n, 32n), ~M(32n, 32n), ~M(32n, 16n), ~M(32n, 16n), ~M(32n, 32n), ~M(32n, 32n), ~(m => T.Mat.dup(32n, 32n, m)), ~(a => b => T.Mat.sum2(32n, 32n, a, b)), ~T.LinearNB(32n, 32n, 16n, 0n), ~T.LinearNB(32n, 32n, 16n, 0n))

def AX() -> Type:
  T.Two<M(32n, 32n), KVY>

def QKVP() -> Type:
  T.Two<M(32n, 32n), KVP>

def QKVS() -> Type:
  T.Two<M(32n, 32n), KVS>

def QKV() -> T.Layer<QKVP, M(32n, 32n), AX, QKVS>:
  T.Fork(~M(32n, 32n), ~KVP, ~M(32n, 32n), ~M(32n, 32n), ~KVY, ~M(32n, 32n), ~KVS, ~(m => T.Mat.dup(32n, 32n, m)), ~(a => b => T.Mat.sum2(32n, 32n, a, b)), ~T.LinearNB(32n, 32n, 32n, 0n), ~KVL())

def RopeK() -> T.Layer<T.Two<Unit, Unit>, KVY, KVY, T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~M(32n, 16n), ~M(32n, 16n), ~M(32n, 16n), ~M(32n, 16n), ~Unit, ~Unit, ~T.RoPE(32n, 16n, 16n, 2n, 4n, 10000.0, {==}), ~T.Id(~M(32n, 16n)))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes() -> T.Layer<RU, AX, AX, RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~M(32n, 32n), ~KVY, ~M(32n, 32n), ~KVY, ~Unit, ~T.Two<Unit, Unit>, ~T.RoPE(32n, 32n, 16n, 4n, 4n, 10000.0, {==}), ~RopeK())

def CoreP() -> Type:
  T.Two<Unit, M(32n, 32n)>

def CoreS() -> Type:
  T.Two<T.AttnS, M(32n, 32n)>

def Core() -> T.Layer<CoreP, AX, M(32n, 32n), CoreS>:
  T.Seq(~Unit, ~M(32n, 32n), ~AX, ~M(32n, 32n), ~M(32n, 32n), ~T.AttnS, ~M(32n, 32n), ~T.Attention(32n, 32n, 16n, 2n, 16n, 2n, 2n, 8n, 0.35355339059327373, {==}, {==}, {==}), ~T.LinearNB(32n, 32n, 32n, 0n))

def RCP() -> Type:
  T.Two<RU, CoreP>

def RCS() -> Type:
  T.Two<RU, CoreS>

def RopeCore() -> T.Layer<RCP, AX, M(32n, 32n), RCS>:
  T.Seq(~RU, ~CoreP, ~AX, ~AX, ~M(32n, 32n), ~RU, ~CoreS, ~Ropes(), ~Core())

def AttP() -> Type:
  T.Two<QKVP, RCP>

def AttS() -> Type:
  T.Two<QKVS, RCS>

def Att() -> T.Layer<AttP, M(32n, 32n), M(32n, 32n), AttS>:
  T.Seq(~QKVP, ~RCP, ~M(32n, 32n), ~AX, ~M(32n, 32n), ~QKVS, ~RCS, ~QKV(), ~RopeCore())

def NAttP() -> Type:
  T.Two<M(1n, 32n), AttP>

def NAttS() -> Type:
  T.Two<Rows, AttS>

def NAtt() -> T.Layer<NAttP, M(32n, 32n), M(32n, 32n), NAttS>:
  T.Seq(~M(1n, 32n), ~AttP, ~M(32n, 32n), ~M(32n, 32n), ~M(32n, 32n), ~Rows, ~AttS, ~T.RMSNorm(32n, 32n, 0.00001), ~Att())

# ---- the feed-forward block: down(silu(gate(x)) * up(x)) ----

def Gate() -> T.Layer<T.Two<M(64n, 32n), Unit>, M(32n, 32n), M(32n, 64n), T.Two<M(32n, 32n), M(32n, 64n)>>:
  T.Seq(~M(64n, 32n), ~Unit, ~M(32n, 32n), ~M(32n, 64n), ~M(32n, 64n), ~M(32n, 32n), ~M(32n, 64n), ~T.LinearNB(32n, 32n, 64n, 0n), ~T.SiLU(32n, 64n))

def GUP() -> Type:
  T.Two<T.Two<M(64n, 32n), Unit>, M(64n, 32n)>

def GUY() -> Type:
  T.Two<M(32n, 64n), M(32n, 64n)>

def GUS() -> Type:
  T.Two<T.Two<M(32n, 32n), M(32n, 64n)>, M(32n, 32n)>

def GU() -> T.Layer<GUP, M(32n, 32n), GUY, GUS>:
  T.Fork(~T.Two<M(64n, 32n), Unit>, ~M(64n, 32n), ~M(32n, 32n), ~M(32n, 64n), ~M(32n, 64n), ~T.Two<M(32n, 32n), M(32n, 64n)>, ~M(32n, 32n), ~(m => T.Mat.dup(32n, 32n, m)), ~(a => b => T.Mat.sum2(32n, 32n, a, b)), ~Gate(), ~T.LinearNB(32n, 32n, 64n, 0n))

def GLUS() -> Type:
  T.Two<GUS, GUY>

def GLU() -> T.Layer<T.Two<GUP, Unit>, M(32n, 32n), M(32n, 64n), GLUS>:
  T.Seq(~GUP, ~Unit, ~M(32n, 32n), ~GUY, ~M(32n, 64n), ~GUS, ~GUY, ~GU(), ~T.Mul2(32n, 64n))

def MlpP() -> Type:
  T.Two<T.Two<GUP, Unit>, M(32n, 64n)>

def MlpS() -> Type:
  T.Two<GLUS, M(32n, 64n)>

def Mlp() -> T.Layer<MlpP, M(32n, 32n), M(32n, 32n), MlpS>:
  T.Seq(~T.Two<GUP, Unit>, ~M(32n, 64n), ~M(32n, 32n), ~M(32n, 64n), ~M(32n, 32n), ~GLUS, ~M(32n, 64n), ~GLU(), ~T.LinearNB(32n, 64n, 32n, 0n))

def NMlpP() -> Type:
  T.Two<M(1n, 32n), MlpP>

def NMlpS() -> Type:
  T.Two<Rows, MlpS>

def NMlp() -> T.Layer<NMlpP, M(32n, 32n), M(32n, 32n), NMlpS>:
  T.Seq(~M(1n, 32n), ~MlpP, ~M(32n, 32n), ~M(32n, 32n), ~M(32n, 32n), ~Rows, ~MlpS, ~T.RMSNorm(32n, 32n, 0.00001), ~Mlp())

# ---- a block, the model ----

def Res(P: Type) -> Type:
  T.Two<T.Two<Unit, P>, Unit>

def BlockP() -> Type:
  T.Two<Res(NAttP), Res(NMlpP)>

def BlockS() -> Type:
  T.Two<Res(NAttS), Res(NMlpS)>

def Block() -> T.Layer<BlockP, M(32n, 32n), M(32n, 32n), BlockS>:
  T.Seq(~Res(NAttP), ~Res(NMlpP), ~M(32n, 32n), ~M(32n, 32n), ~M(32n, 32n), ~Res(NAttS), ~Res(NMlpS), ~T.Residual(~32n, ~32n, ~NAttP, ~NAttS, ~NAtt()), ~T.Residual(~32n, ~32n, ~NMlpP, ~NMlpS, ~NMlp()))

def HeadP() -> Type:
  T.Two<M(1n, 32n), M(65n, 32n)>

def HeadS() -> Type:
  T.Two<Rows, M(32n, 32n)>

def Head() -> T.Layer<HeadP, M(32n, 32n), M(32n, 65n), HeadS>:
  T.Seq(~M(1n, 32n), ~M(65n, 32n), ~M(32n, 32n), ~M(32n, 32n), ~M(32n, 65n), ~Rows, ~M(32n, 32n), ~T.RMSNorm(32n, 32n, 0.00001), ~T.LinearNB(32n, 32n, 65n, 0n))

def TopP() -> Type:
  T.Two<BlockP, HeadP>

def TopS() -> Type:
  T.Two<BlockS, HeadS>

def Top() -> T.Layer<TopP, M(32n, 32n), M(32n, 65n), TopS>:
  T.Seq(~BlockP, ~HeadP, ~M(32n, 32n), ~M(32n, 32n), ~M(32n, 65n), ~BlockS, ~HeadS, ~Block(), ~Head())

def BodyP() -> Type:
  T.Two<BlockP, TopP>

def BodyS() -> Type:
  T.Two<BlockS, TopS>

def Body() -> T.Layer<BodyP, M(32n, 32n), M(32n, 65n), BodyS>:
  T.Seq(~BlockP, ~TopP, ~M(32n, 32n), ~M(32n, 32n), ~M(32n, 65n), ~BlockS, ~TopS, ~Block(), ~Top())

def P() -> Type:
  T.Two<M(65n, 32n), BodyP>

def S() -> Type:
  T.Two<List<&2, Nat>, BodyS>

def X() -> Type:
  List<&2, Nat>

def Y() -> Type:
  M(32n, 65n)

def Model() -> T.Layer<P, X, Y, S>:
  T.Seq(~M(65n, 32n), ~BodyP, ~X, ~M(32n, 32n), ~Y, ~List<&2, Nat>, ~BodyS, ~T.Embedding(32n, 65n, 32n), ~Body())

def ce(z: Y, labels: List<&2, Nat>) -> T.LossOut<Y>:
  T.CrossEntropy(32n, 65n, z, labels)

# ---- weights: data/tiny/init/<name>.bin, in reference/tiny_ref.py's names ----

def mk_block(ln1: M(1n, 32n), wq: M(32n, 32n), wk: M(16n, 32n), wv: M(16n, 32n), wo: M(32n, 32n), ln2: M(1n, 32n), wg: M(64n, 32n), wu: M(64n, 32n), wd: M(32n, 64n)) -> BlockP:
  T.Two{T.Two{T.Two{Unit{}, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{Unit{}, T.Two{Unit{}, Unit{}}}, T.Two{Unit{}, wo}}}}}, Unit{}}, T.Two{T.Two{Unit{}, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, Unit{}}, wu}, Unit{}}, wd}}}, Unit{}}}

def ld(+r: Nat, +c: Nat, +name: String) -> IO(M(r, c)):
  T.Mat.load(r, c, "data/tiny/init/" ++ name ++ ".bin", 0)

def load_block(+b: String) -> IO(BlockP):
  do IO<BlockP>:
    ln1 : M(1n, 32n) <- ld(1n, 32n, b ++ ".ln1")
    wq : M(32n, 32n) <- ld(32n, 32n, b ++ ".wq")
    wk : M(16n, 32n) <- ld(16n, 32n, b ++ ".wk")
    wv : M(16n, 32n) <- ld(16n, 32n, b ++ ".wv")
    wo : M(32n, 32n) <- ld(32n, 32n, b ++ ".wo")
    ln2 : M(1n, 32n) <- ld(1n, 32n, b ++ ".ln2")
    wg : M(64n, 32n) <- ld(64n, 32n, b ++ ".wg")
    wu : M(64n, 32n) <- ld(64n, 32n, b ++ ".wu")
    wd : M(32n, 64n) <- ld(32n, 64n, b ++ ".wd")
    return mk_block(ln1, wq, wk, wv, wo, ln2, wg, wu, wd)

def load_params() -> IO(P):
  do IO<P>:
    emb : M(65n, 32n) <- ld(65n, 32n, "emb")
    b0 : BlockP <- load_block("b0")
    b1 : BlockP <- load_block("b1")
    lnf : M(1n, 32n) <- ld(1n, 32n, "lnf")
    head : M(65n, 32n) <- ld(65n, 32n, "head")
    return T.Two{emb, T.Two{b0, T.Two{b1, T.Two{lnf, head}}}}

def print_block(+tag: String, b: BlockP) -> IO(Unit):
  match b:
    case T.Two{T.Two{T.Two{u1, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{u2, T.Two{u3, u4}}, T.Two{u5, wo}}}}}, u6}, T.Two{T.Two{u7, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, u8}, wu}, u9}, wd}}}, u10}}:
      do IO<Unit>:
        T.Mat.print(1n, 32n, tag ++ ".ln1", ln1)
        T.Mat.print(32n, 32n, tag ++ ".wq", wq)
        T.Mat.print(16n, 32n, tag ++ ".wk", wk)
        T.Mat.print(16n, 32n, tag ++ ".wv", wv)
        T.Mat.print(32n, 32n, tag ++ ".wo", wo)
        T.Mat.print(1n, 32n, tag ++ ".ln2", ln2)
        T.Mat.print(64n, 32n, tag ++ ".wg", wg)
        T.Mat.print(64n, 32n, tag ++ ".wu", wu)
        T.Mat.print(32n, 64n, tag ++ ".wd", wd)

def print_params(+tag: String, p: P) -> IO(Unit):
  match p:
    case T.Two{emb, T.Two{b0, T.Two{b1, T.Two{lnf, head}}}}:
      do IO<Unit>:
        T.Mat.print(65n, 32n, tag ++ ".emb", emb)
        print_block(tag ++ ".b0", b0)
        print_block(tag ++ ".b1", b1)
        T.Mat.print(1n, 32n, tag ++ ".lnf", lnf)
        T.Mat.print(65n, 32n, tag ++ ".head", head)

# ---- batches: data/tiny/batches.bin, 2 x 17 ids per step; the inputs are the first 16 of each, the targets the last 16 ----

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> nats(t)

def xs_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, ids, 16n), List.take(&2, Nat, List.drop(&2, Nat, ids, 17n), 16n))

def ys_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, List.drop(&2, Nat, ids, 1n), 16n), List.drop(&2, Nat, ids, 18n))

def load_batch(+step: U32) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at("data/tiny/batches.bin", (step * 34 : U32), 34)
    return nats(bs)

# ---- one forward and backward pass ----

def grads3(r: T.Bwd<P, X>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      print_params("grad", g)

def grads2(p: P, s: S, r: T.LossOut<Y>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print("loss", l)
        grads3(T.Layer.bwd(P, X, Y, S, Model(), p, s, dy, T.Layer.zeros(P, X, Y, S, Model())))

def grads1(+ys: List<&2, Nat>, r: T.Fwd<P, Y, S>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      grads2(p, s, ce(y, ys))

def grads(+ids: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    grads1(ys_of(ids), T.Layer.fwd(P, X, Y, S, Model(), p, xs_of(ids)))

# ---- training ----

def step_with(+cfg: T.OptCfg, +ids: List<&2, Nat>, tr: T.Train<P>) -> T.Train<P>:
  T.Train.step(~P, ~X, ~Y, ~S, ~List<&2, Nat>, ~Model(), ~ce, cfg, tr, xs_of(ids), ys_of(ids))

# k more steps from batch `step`; prints the loss of the step before (0 before the first)
def tloop(k: Nat, +step: U32, +cfg: T.OptCfg, tr: T.Train<P>) -> IO(T.Train<P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        return T.Train{p, g, st, p1, p2, l}
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        ids : List<&2, Nat> <- load_batch(step)
        tloop(q, (step + 1 : U32), cfg, step_with(cfg, ids, T.Train{p, g, st, p1, p2, l}))

def finish(tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      print_params("final", p)

def train(+k: Nat) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    tr : T.Train<P> <- tloop(k, 0, T.CAdamW{0.003, 0.9, 0.999, 0.00000001, 0.01}, T.Train.new(~P, ~X, ~Y, ~S, ~Model(), p))
    finish(tr)

def train_m(m: Maybe<&2, Nat>) -> IO(Unit):
  match m:
    case Some{k}:
      train(k)
    case None{}:
      IO.print("usage: tiny_llama train <steps>")

def dispatch2(g: Bool, rest: List<String>) -> IO(Unit):
  match g rest:
    case True{} _:
      do IO<Unit>:
        ids : List<&2, Nat> <- load_batch(0)
        grads(ids)
    case False{} Con{k, Nil{}}:
      train_m(Nat.read(k))
    case _ _:
      IO.print("usage: tiny_llama grads | tiny_llama train <steps>")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{mode, rest}}:
      dispatch2(String.eq(mode, "grads"), rest)
    case _:
      IO.print("usage: tiny_llama grads | tiny_llama train <steps>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)

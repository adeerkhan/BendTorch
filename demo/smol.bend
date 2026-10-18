# SmolLM2-135M in BendTorch's layers, fine-tuned for a few steps against its PyTorch twin (reference/smol_ref.py, which
# also writes the weights and the batches into data/smol135/; scripts/smol.sh 360m writes the 360M's copy).
#
#   bend demo/smol.bend -o .scratch/smol
#   .scratch/smol --threads 16 loss              # the loss on the first batch
#   .scratch/smol --threads 16 train <steps>     # AdamW steps: the loss before each update; train_ms for the steps
#   .scratch/smol --threads 16 grads <file>      # the loss on the first batch; every gradient into <file>, dump order
#   .scratch/smol --threads 16 finetune <k> <e>  # k steps on ft_train.bin; the held-out loss (ft_eval.bin) every e
#
# The model: the token embedding, LAYERS() decoder blocks (smol_block.bend's: Residual(Seq(RMSNorm, attention)) then
# Residual(Seq(RMSNorm, SwiGLU))), the final RMSNorm, and the output head, which shares the embedding's matrix (Tie,
# as SmolLM2 ties them). Batches of 2 sequences of TT() = 128 positions (256 rows). The linear layers are
# torch/fast.bend's. Weights are in Layer.dump order (reference/smol_layout.txt names each tensor).

import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

# where the weights and the batches are
def DATA() -> String:
  "data/smol135/"

# positions per sequence
def TT() -> Nat:
  128n

def V() -> Nat:
  49152n

def LAYERS() -> Nat:
  30n

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# the rows of a batch of b sequences of TT() positions
def R(b: Nat) -> Nat:
  Nat.mul(b, TT())

def A(b: Nat, c: Nat) -> Type:
  TA.Mat<R(b), c>

# log2 of the most parallel blocks of each product: 256 blocks balance the P- and E-cores better than 16 (16 threads,
# per 3 steps: 763 -> 577 ms at 135M's sizes, 1840 -> 1313 ms at 360M's; NOTES 2026-10-17)
def PAR() -> Nat:
  8n

# the sizes: SmolLM2-135M's (scripts/smol.sh writes 360M's into a copy of these defs and of the stack below)
def D() -> Nat:
  576n

def KVW() -> Nat:
  192n

def FF() -> Nat:
  1536n

def HKV() -> Nat:
  3n

def GRP() -> Nat:
  3n

def HQ() -> Nat:
  Nat.mul(HKV(), GRP())

# ---- attention ----

def KVP() -> Type:
  T.Two<M(KVW(), D()), M(KVW(), D())>

def KVY(b: Nat) -> Type:
  T.Two<A(b, KVW()), A(b, KVW())>

def KVS(b: Nat) -> Type:
  T.Two<A(b, D()), A(b, D())>

def KVL(~b: Nat) -> T.Layer<KVP, A(b, D()), KVY(b), KVS(b)>:
  T.Fork(~M(KVW(), D()), ~M(KVW(), D()), ~A(b, D()), ~A(b, KVW()), ~A(b, KVW()), ~A(b, D()), ~A(b, D()), ~(m => F.Mat.dup(R(b), D(), m)), ~(x => y => F.Mat.sum2(R(b), D(), x, y)), ~F.LinearNBF(R(b), D(), KVW(), PAR()), ~F.LinearNBF(R(b), D(), KVW(), PAR()))

def AX(b: Nat) -> Type:
  T.Two<A(b, D()), KVY(b)>

def QKVP() -> Type:
  T.Two<M(D(), D()), KVP>

def QKVS(b: Nat) -> Type:
  T.Two<A(b, D()), KVS(b)>

def QKV(~b: Nat) -> T.Layer<QKVP, A(b, D()), AX(b), QKVS(b)>:
  T.Fork(~M(D(), D()), ~KVP, ~A(b, D()), ~A(b, D()), ~KVY(b), ~A(b, D()), ~KVS(b), ~(m => F.Mat.dup(R(b), D(), m)), ~(x => y => F.Mat.sum2(R(b), D(), x, y)), ~F.LinearNBF(R(b), D(), D(), PAR()), ~KVL(~b))

def RopeK(~b: Nat) -> T.Layer<T.Two<Unit, Unit>, KVY(b), KVY(b), T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~A(b, KVW()), ~A(b, KVW()), ~A(b, KVW()), ~A(b, KVW()), ~Unit, ~Unit, ~T.RoPE(R(b), KVW(), TT(), HKV(), 32n, 100000.0, {==}), ~T.Id(~A(b, KVW())))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes(~b: Nat) -> T.Layer<RU, AX(b), AX(b), RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(b, D()), ~KVY(b), ~A(b, D()), ~KVY(b), ~Unit, ~T.Two<Unit, Unit>, ~T.RoPE(R(b), D(), TT(), HQ(), 32n, 100000.0, {==}), ~RopeK(~b))

def CoreP() -> Type:
  T.Two<Unit, M(D(), D())>

def CoreS(b: Nat) -> Type:
  T.Two<T.AttnS, A(b, D())>

def Core(~b: Nat) -> T.Layer<CoreP, AX(b), A(b, D()), CoreS(b)>:
  T.Seq(~Unit, ~M(D(), D()), ~AX(b), ~A(b, D()), ~A(b, D()), ~T.AttnS, ~A(b, D()), ~T.Attention(R(b), D(), KVW(), b, TT(), HKV(), GRP(), 64n, 0.125, {==}, {==}, {==}), ~F.LinearNBF(R(b), D(), D(), PAR()))

def RCP() -> Type:
  T.Two<RU, CoreP>

def RCS(b: Nat) -> Type:
  T.Two<RU, CoreS(b)>

def RopeCore(~b: Nat) -> T.Layer<RCP, AX(b), A(b, D()), RCS(b)>:
  T.Seq(~RU, ~CoreP, ~AX(b), ~AX(b), ~A(b, D()), ~RU, ~CoreS(b), ~Ropes(~b), ~Core(~b))

def AttP() -> Type:
  T.Two<QKVP, RCP>

def AttS(b: Nat) -> Type:
  T.Two<QKVS(b), RCS(b)>

def Att(~b: Nat) -> T.Layer<AttP, A(b, D()), A(b, D()), AttS(b)>:
  T.Seq(~QKVP, ~RCP, ~A(b, D()), ~AX(b), ~A(b, D()), ~QKVS(b), ~RCS(b), ~QKV(~b), ~RopeCore(~b))

def NAttP() -> Type:
  T.Two<M(1n, D()), AttP>

def NAttS(b: Nat) -> Type:
  T.Two<A(b, D()), AttS(b)>

def NAtt(~b: Nat) -> T.Layer<NAttP, A(b, D()), A(b, D()), NAttS(b)>:
  T.Seq(~M(1n, D()), ~AttP, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~AttS(b), ~T.RMSNorm(R(b), D(), 0.00001), ~Att(~b))

# ---- the feed-forward block: down(silu(gate(x)) * up(x)) ----

def GateS(b: Nat) -> Type:
  T.Two<A(b, D()), A(b, FF())>

def Gate(~b: Nat) -> T.Layer<T.Two<M(FF(), D()), Unit>, A(b, D()), A(b, FF()), GateS(b)>:
  T.Seq(~M(FF(), D()), ~Unit, ~A(b, D()), ~A(b, FF()), ~A(b, FF()), ~A(b, D()), ~A(b, FF()), ~F.LinearNBF(R(b), D(), FF(), PAR()), ~F.SiLU(R(b), FF()))

def GUP() -> Type:
  T.Two<T.Two<M(FF(), D()), Unit>, M(FF(), D())>

def GUY(b: Nat) -> Type:
  T.Two<A(b, FF()), A(b, FF())>

def GUS(b: Nat) -> Type:
  T.Two<GateS(b), A(b, D())>

def GU(~b: Nat) -> T.Layer<GUP, A(b, D()), GUY(b), GUS(b)>:
  T.Fork(~T.Two<M(FF(), D()), Unit>, ~M(FF(), D()), ~A(b, D()), ~A(b, FF()), ~A(b, FF()), ~GateS(b), ~A(b, D()), ~(m => F.Mat.dup(R(b), D(), m)), ~(x => y => F.Mat.sum2(R(b), D(), x, y)), ~Gate(~b), ~F.LinearNBF(R(b), D(), FF(), PAR()))

def GLUS(b: Nat) -> Type:
  T.Two<GUS(b), GUY(b)>

def GLU(~b: Nat) -> T.Layer<T.Two<GUP, Unit>, A(b, D()), A(b, FF()), GLUS(b)>:
  T.Seq(~GUP, ~Unit, ~A(b, D()), ~GUY(b), ~A(b, FF()), ~GUS(b), ~GUY(b), ~GU(~b), ~F.Mul2(R(b), FF()))

def MlpP() -> Type:
  T.Pair3(GUP, Unit, M(D(), FF()))

def MlpS(b: Nat) -> Type:
  T.Two<GLUS(b), A(b, FF())>

def Mlp(~b: Nat) -> T.Layer<MlpP, A(b, D()), A(b, D()), MlpS(b)>:
  T.Seq(~T.Two<GUP, Unit>, ~M(D(), FF()), ~A(b, D()), ~A(b, FF()), ~A(b, D()), ~GLUS(b), ~A(b, FF()), ~GLU(~b), ~F.LinearNBF(R(b), FF(), D(), PAR()))

def NMlpP() -> Type:
  T.Two<M(1n, D()), MlpP>

def NMlpS(b: Nat) -> Type:
  T.Two<A(b, D()), MlpS(b)>

def NMlp(~b: Nat) -> T.Layer<NMlpP, A(b, D()), A(b, D()), NMlpS(b)>:
  T.Seq(~M(1n, D()), ~MlpP, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~MlpS(b), ~T.RMSNorm(R(b), D(), 0.00001), ~Mlp(~b))

# ---- the block ----

def BP() -> Type:
  T.Two<T.ResidualType(NAttP), T.ResidualType(NMlpP)>

def BS(b: Nat) -> Type:
  T.Two<T.ResidualType(NAttS(b)), T.ResidualType(NMlpS(b))>

def Block(~b: Nat) -> T.Layer<BP, A(b, D()), A(b, D()), BS(b)>:
  T.Seq(~T.ResidualType(NAttP), ~T.ResidualType(NMlpP), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~T.ResidualType(NAttS(b)), ~T.ResidualType(NMlpS(b)), ~F.Residual(~R(b), ~D(), ~NAttP, ~NAttS(b), ~NAtt(~b)), ~F.Residual(~R(b), ~D(), ~NMlpP, ~NMlpS(b), ~NMlp(~b)))

# ---- Box: a layer whose parameters and saved values are one boxed word each ----

# Bend lays a value of nested records out flat, a word a field, and a def or a continuation holds at most 247 words
# (bend guide, SHADERS.md): eight blocks' parameters and saved values held across a call are more. A datatype that
# mentions itself is boxed (one word), so Boxed<A> carries A behind one pointer; Chain never occurs.
type Boxed<-A: Type> is Type:
  Boxed{a: A}
  Chain{next: Boxed<A>}

def Boxed.get(-A: Type, b: Boxed<A>) -> A:
  match b:
    case Boxed{a}: a
    case Chain{n}: Boxed.get(A, n)

def Boxed.put(-A: Type, a: A) -> Boxed<A>:
  Boxed{a}

def box_f2(-P: Type, -Y: Type, -S: Type, r: T.Fwd<P, Y, S>) -> T.Fwd<Boxed<P>, Y, Boxed<S>>:
  match r:
    case T.Fwd{p, y, s}: T.Fwd{Boxed{p}, y, Boxed{s}}

def box_b2(-P: Type, -X: Type, r: T.Bwd<P, X>) -> T.Bwd<Boxed<P>, X>:
  match r:
    case T.Bwd{p, dx, g}: T.Bwd{Boxed{p}, dx, Boxed{g}}

def box_s2(-P: Type, r: T.Step<P>) -> T.Step<Boxed<P>>:
  match r:
    case T.Step{p, g, st}: T.Step{Boxed{p}, Boxed{g}, T.OptState.map(P, Boxed<P>, st, x => Boxed{x}, x => Boxed{x})}

def box_d2(-P: Type, r: T.PL<P>) -> T.PL<Boxed<P>>:
  match r:
    case T.PL{p, xs}: T.PL{Boxed{p}, xs}

def box_l2(-P: Type, r: T.PA<P>) -> T.PA<Boxed<P>>:
  match r:
    case T.PA{p, xs, off}: T.PA{Boxed{p}, xs, off}

def box_t2(-P: Type, r: T.PA<P>) -> T.PA<Boxed<P>>:
  box_l2(P, r)

def Box(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~l: T.Layer<P, X, Y, S>) -> T.Layer<Boxed<P>, X, Y, Boxed<S>>:
  T.Layer{p => x => box_f2(P, Y, S, T.Layer.fwd(P, X, Y, S, l, Boxed.get(P, p), x)), p => s => dy => g => box_b2(P, X, T.Layer.bwd(P, X, Y, S, l, Boxed.get(P, p), Boxed.get(S, s), dy, Boxed.get(P, g))), op => p => g => st => box_s2(P, T.Layer.step(P, X, Y, S, l, op, Boxed.get(P, p), Boxed.get(P, g), T.OptState.map(Boxed<P>, P, st, x => Boxed.get(P, x), x => Boxed.get(P, x)))), u => Boxed{T.Layer.zeros(P, X, Y, S, l)}, p => box_d2(P, T.Layer.dump(P, X, Y, S, l, Boxed.get(P, p))), p => xs => off => box_l2(P, T.Layer.load(P, X, Y, S, l, Boxed.get(P, p), xs, off)), p => xs => off => box_t2(P, T.Layer.store(P, X, Y, S, l, Boxed.get(P, p), xs, off)), u => T.Layer.count(P, X, Y, S, l)}

# ---- the stack of blocks: a balanced tree of Seq over LAYERS() blocks, read in order (block 0 first) ----

# A template takes only arguments fixed when the program compiles, and Bend instantiates templates 64 levels deep at
# most, so the stack is neither a recursion on its depth nor a chain of 30 Seq: scripts/smol.sh writes one def per
# subtree size (StackN = Seq(StackN/2, StackN - N/2), Stack1 a block), between the two markers.
# ---- stack: begin ----
def StP1() -> Type:
  Boxed<BP>

def StS1(b: Nat) -> Type:
  Boxed<BS(b)>

def Stack1(~b: Nat) -> T.Layer<StP1, A(b, D()), A(b, D()), StS1(b)>:
  Box(~BP, ~A(b, D()), ~A(b, D()), ~BS(b), ~Block(~b))

def StP2() -> Type:
  T.Two<StP1, StP1>

def StS2(b: Nat) -> Type:
  T.Two<StS1(b), StS1(b)>

def Stack2(~b: Nat) -> T.Layer<StP2, A(b, D()), A(b, D()), StS2(b)>:
  T.Seq(~StP1, ~StP1, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS1(b), ~StS1(b), ~Stack1(~b), ~Stack1(~b))

def StP3() -> Type:
  T.Two<StP1, StP2>

def StS3(b: Nat) -> Type:
  T.Two<StS1(b), StS2(b)>

def Stack3(~b: Nat) -> T.Layer<StP3, A(b, D()), A(b, D()), StS3(b)>:
  T.Seq(~StP1, ~StP2, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS1(b), ~StS2(b), ~Stack1(~b), ~Stack2(~b))

def StP4() -> Type:
  T.Two<StP2, StP2>

def StS4(b: Nat) -> Type:
  T.Two<StS2(b), StS2(b)>

def Stack4(~b: Nat) -> T.Layer<StP4, A(b, D()), A(b, D()), StS4(b)>:
  T.Seq(~StP2, ~StP2, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS2(b), ~StS2(b), ~Stack2(~b), ~Stack2(~b))

def StP7() -> Type:
  T.Two<StP3, StP4>

def StS7(b: Nat) -> Type:
  T.Two<StS3(b), StS4(b)>

def Stack7(~b: Nat) -> T.Layer<StP7, A(b, D()), A(b, D()), StS7(b)>:
  T.Seq(~StP3, ~StP4, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS3(b), ~StS4(b), ~Stack3(~b), ~Stack4(~b))

def StP8() -> Type:
  T.Two<StP4, StP4>

def StS8(b: Nat) -> Type:
  T.Two<StS4(b), StS4(b)>

def Stack8(~b: Nat) -> T.Layer<StP8, A(b, D()), A(b, D()), StS8(b)>:
  T.Seq(~StP4, ~StP4, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS4(b), ~StS4(b), ~Stack4(~b), ~Stack4(~b))

def StP15() -> Type:
  T.Two<StP7, StP8>

def StS15(b: Nat) -> Type:
  T.Two<StS7(b), StS8(b)>

def Stack15(~b: Nat) -> T.Layer<StP15, A(b, D()), A(b, D()), StS15(b)>:
  T.Seq(~StP7, ~StP8, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS7(b), ~StS8(b), ~Stack7(~b), ~Stack8(~b))

def StP30() -> Type:
  T.Two<StP15, StP15>

def StS30(b: Nat) -> Type:
  T.Two<StS15(b), StS15(b)>

def Stack30(~b: Nat) -> T.Layer<StP30, A(b, D()), A(b, D()), StS30(b)>:
  T.Seq(~StP15, ~StP15, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS15(b), ~StS15(b), ~Stack15(~b), ~Stack15(~b))

def StackP() -> Type:
  StP30

def StackS(b: Nat) -> Type:
  StS30(b)

def Stack(~b: Nat) -> T.Layer<StackP, A(b, D()), A(b, D()), StackS(b)>:
  Stack30(~b)
# ---- stack: end ----

# ---- the model: Tie(embedding, Seq(stack, final norm), head), the head sharing the embedding's matrix ----

def BodyP() -> Type:
  T.Two<StackP, M(1n, D())>

def BodyS(b: Nat) -> Type:
  T.Two<StackS(b), A(b, D())>

def Body(~b: Nat) -> T.Layer<BodyP, A(b, D()), A(b, D()), BodyS(b)>:
  T.Seq(~StackP, ~M(1n, D()), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StackS(b), ~A(b, D()), ~Stack(~b), ~T.RMSNorm(R(b), D(), 0.00001))

def P() -> Type:
  T.Two<M(V(), D()), BodyP>

def S(b: Nat) -> Type:
  T.Three<List<&2, Nat>, BodyS(b), A(b, D())>

def X() -> Type:
  List<&2, Nat>

def Y(b: Nat) -> Type:
  A(b, V())

def Model(~b: Nat) -> T.Layer<P, X, Y(b), S(b)>:
  T.Tie(~M(V(), D()), ~BodyP, ~X, ~A(b, D()), ~A(b, D()), ~Y(b), ~List<&2, Nat>, ~BodyS(b), ~A(b, D()), ~T.Embedding(R(b), V(), D()), ~Body(~b), ~F.LinearNBF(R(b), D(), V(), PAR()))

def ce(z: Y(2n), labels: List<&2, Nat>) -> T.LossOut<Y(2n)>:
  T.CrossEntropy(R(2n), V(), z, labels)

def load_params() -> IO(P):
  L.weights(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), DATA() ++ "init.bin", T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n)))

# ---- batches: DATA()/batches.bin, little-endian U32 token ids, 2 sequences of TT() + 1 per step; the inputs are the
# first TT() of each, the targets the last TT() ----

def ids_of(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Con{+b0, Con{+b1, Con{+b2, Con{+b3, t}}}}:
      U32.to_nat((((b0 + (b1 * 256 : U32) : U32) + (b2 * 65536 : U32) : U32) + (b3 * 16777216 : U32) : U32)) <> ids_of(t)
    case _:
      Nil{}

def xs_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, ids, TT()), List.take(&2, Nat, List.drop(&2, Nat, ids, 1n+TT()), TT()))

def ys_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, List.drop(&2, Nat, ids, 1n), TT()), List.drop(&2, Nat, ids, 2n+TT()))

def load_from(+name: String, +step: U32) -> IO(List<&2, Nat>):
  +per = U32.from_nat(Nat.mul(2n, 1n+TT()))
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(DATA() ++ name, ((step * per : U32) * 4 : U32), (per * 4 : U32))
    return ids_of(bs)

def load_batch(+step: U32) -> IO(List<&2, Nat>):
  load_from("batches.bin", step)

# ---- the loss on the first batch ----

def loss2(r: T.LossOut<Y(2n)>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}: T.F32.print("loss", l)

def loss1(+ys: List<&2, Nat>, r: T.Fwd<P, Y(2n), S(2n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}: loss2(ce(y, ys))

def loss_with(p: P, +ids: List<&2, Nat>) -> IO(Unit):
  loss1(ys_of(ids), T.Layer.fwd(P, X, Y(2n), S(2n), Model(~2n), p, xs_of(ids)))

def loss() -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    ids : List<&2, Nat> <- load_batch(0)
    loss_with(p, ids)

# ---- the loss and every gradient on the first batch, the gradients into a file in dump order ----

def grads3(+path: String, r: T.Bwd<P, X>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        g2 : P <- T.Params.save(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), path, g)
        IO.print("grads " ++ path)

def grads2(+path: String, p: P, s: S(2n), r: T.LossOut<Y(2n)>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print("loss", l)
        grads3(path, T.Layer.bwd(P, X, Y(2n), S(2n), Model(~2n), p, s, dy, T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n))))

def grads1(+path: String, +ys: List<&2, Nat>, r: T.Fwd<P, Y(2n), S(2n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}: grads2(path, p, s, ce(y, ys))

def grads_with(+path: String, p: P, +ids: List<&2, Nat>) -> IO(Unit):
  grads1(path, ys_of(ids), T.Layer.fwd(P, X, Y(2n), S(2n), Model(~2n), p, xs_of(ids)))

def grads(+path: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    ids : List<&2, Nat> <- load_batch(0)
    grads_with(path, p, ids)

# ---- training ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.0001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}

def step_with(+ids: List<&2, Nat>, tr: T.Train<P>) -> T.Train<P>:
  T.Train.step(~P, ~X, ~Y(2n), ~S(2n), ~List<&2, Nat>, ~Model(~2n), ~ce, cfg(), tr, xs_of(ids), ys_of(ids))

# k more steps from batch `step`; prints the loss of the step before (0 before the first)
def tloop(k: Nat, +step: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        return T.Train{p, g, st, p1, p2, l}
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        ids : List<&2, Nat> <- load_batch(step)
        tloop(q, (step + 1 : U32), step_with(ids, T.Train{p, g, st, p1, p2, l}))

# ---- fine-tuning: training on ft_train.bin, the mean loss on ft_eval.bin's held-out batches every few steps ----

def FT_EVAL() -> Nat:
  16n

# the parameters handed back by a forward pass, and its loss
type PE is Type:
  PE{p: P, l: F32}

def ev2(p: P, r: T.LossOut<Y(2n)>) -> PE:
  match r:
    case T.LossOut{l, dy}: PE{p, l}

def ev1(+ys: List<&2, Nat>, r: T.Fwd<P, Y(2n), S(2n)>) -> PE:
  match r:
    case T.Fwd{p, y, s}: ev2(p, ce(y, ys))

def ev_one(p: P, +ids: List<&2, Nat>) -> PE:
  ev1(ys_of(ids), T.Layer.fwd(P, X, Y(2n), S(2n), Model(~2n), p, xs_of(ids)))

def ev_then(r: PE, k: P -> F32 -> IO(PE)) -> IO(PE):
  match r:
    case PE{p, l}: k(p, l)

# the sum of the losses on k held-out batches from batch e, added to acc in order; the parameters come back
def ev_loop(k: Nat, +e: U32, +acc: F32, p: P) -> IO(PE):
  match k:
    case 0n:
      IO.pure(PE, PE{p, acc})
    case 1n+q:
      do IO<PE>:
        ids : List<&2, Nat> <- load_from("ft_eval.bin", e)
        ev_then(ev_one(p, ids), p2 => +l => ev_loop(q, (e + 1 : U32), (acc + l : F32), p2))

def ev_print(r: PE, g: P, st: T.OptState<P>, +p1: F32, +p2: F32, +l: F32, k: T.Train<P> -> IO(T.Train<P>)) -> IO(T.Train<P>):
  match r:
    case PE{p, s}:
      do IO<T.Train<P>>:
        T.F32.print("eval", (s / F32.from_nat(FT_EVAL()) : F32))
        k(T.Train{p, g, st, p1, p2, l})

# after a step: its loss, then the held-out loss when do_eval
def ft_after(do_eval: Bool, tr: T.Train<P>, k: T.Train<P> -> IO(T.Train<P>)) -> IO(T.Train<P>):
  match do_eval tr:
    case True{} T.Train{p, g, st, +p1, +p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        r : PE <- ev_loop(FT_EVAL(), 0, 0.0, p)
        ev_print(r, g, st, p1, p2, l, k)
    case False{} T.Train{p, g, st, +p1, +p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        k(T.Train{p, g, st, p1, p2, l})

def ft_loop(k: Nat, +step: U32, +every: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  match k:
    case 0n:
      IO.pure(T.Train<P>, tr)
    case 1n+q:
      do IO<T.Train<P>>:
        ids : List<&2, Nat> <- load_from("ft_train.bin", step)
        ft_after(U32.is_eq(U32.mod((step + 1 : U32), every), 0), step_with(ids, tr), tr2 => ft_loop(q, (step + 1 : U32), every, tr2))

def ft_start(+k: Nat, +every: U32, r: PE) -> IO(Unit):
  match r:
    case PE{p, s}:
      do IO<Unit>:
        T.F32.print("eval", (s / F32.from_nat(FT_EVAL()) : F32))
        t0 : Nat <- IO.now()
        tr : T.Train<P> <- ft_loop(k, 0, every, T.Train.new(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), cfg(), p))
        t1 : Nat <- IO.now()
        IO.print("train_ms " ++ Nat.show(Nat.sub(t1, t0)))

# the held-out loss before any step, then k steps, each loss, and the held-out loss after every `every` steps
def finetune(+k: Nat, +every: U32) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    r : PE <- ev_loop(FT_EVAL(), 0, 0.0, p)
    ft_start(k, every, r)

def train(+k: Nat) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    t0 : Nat <- IO.now()
    tr : T.Train<P> <- tloop(k, 0, T.Train.new(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), cfg(), p))
    t1 : Nat <- IO.now()
    IO.print("train_ms " ++ Nat.show(Nat.sub(t1, t0)))

def usage() -> IO(Unit):
  IO.die(Unit, 2, "usage: smol loss | train <steps> | grads <file> | finetune <steps> <every>")

def train_m(mk: Maybe<&2, Nat>) -> IO(Unit):
  match mk:
    case Some{k}: train(k)
    case None{}: usage()

def dispatch_l(is_loss: Bool) -> IO(Unit):
  match is_loss:
    case True{}: loss()
    case False{}: usage()

def dispatch_t(is_train: Bool, is_grads: Bool, +arg: String) -> IO(Unit):
  match is_train is_grads:
    case True{} _: train_m(Nat.read(arg))
    case _ True{}: grads(arg)
    case _ _: usage()

def finetune_m(mk: Maybe<&2, Nat>, me: Maybe<&2, U32>) -> IO(Unit):
  match mk me:
    case Some{k} Some{e}: finetune(k, e)
    case _ _: usage()

def dispatch_f(is_ft: Bool, +k: String, +e: String) -> IO(Unit):
  match is_ft:
    case True{}: finetune_m(Nat.read(k), U32.read(e))
    case False{}: usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{mode, Con{k, Con{e, Nil{}}}}}:
      dispatch_f(String.eq(mode, "finetune"), k, e)
    case Con{prog, Con{mode, Nil{}}}:
      dispatch_l(String.eq(mode, "loss"))
    case Con{prog, Con{+mode, Con{k, Nil{}}}}:
      dispatch_t(String.eq(mode, "train"), String.eq(mode, "grads"), k)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)

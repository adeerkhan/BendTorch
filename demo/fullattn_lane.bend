# Bit probe (temporary): the same q, k, v through the CPU lane's full forward and the
# device lane's dense one.  b=1, t=4, kv=1, g=1, hd=4, n=4, scale 0.5.
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import ../torch/main.bend as T
import ../torch/gpu.bend as G

# 16 distinct values: ((start + i) mod 7 - 3) / 4, so rows differ and signs differ
def vals(+left: Nat, +i: Nat, +start: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{(F32.from_nat(Nat.sub(Nat.mod(Nat.add(start, i), 7n), 3n)) / 4.0 : F32),
        vals(rest, Nat.add(i, 1n), start)}

def mat(+start: Nat) -> TA.Mat<4n, 4n>:
  TA.Mat.from_list(4n, 4n, vals(16n, 0n, start))

def un(m: TA.Mat<4n, 4n>) -> Array<F32>:
  match m:
    case TA.Mat{d}:
      d

def show_cpu(r: T.Fwd<Unit, TA.Mat<4n, 4n>, T.AttnS>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, _}:
      T.Mat.print(4n, 4n, "cpu", y)

def show_gpu(r: T.Fwd<Unit, TA.Mat<4n, 4n>, T.AttnS>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, _}:
      T.Mat.print(4n, 4n, "gpu", y)

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- show_cpu(T.attn_fwd_full(4n, 4n, 4n, T.AttnDims{1n, 4n, 1n, 1n, 4n, 0.5},
      T.Two{mat(1n), T.Two{mat(2n), mat(3n)}}))
    show_gpu(G.attn_f_f(4n, 4n, 4n, 1n, 4n, 1n, 1n, 4n, 0.5,
      T.Two{mat(1n), T.Two{mat(2n), mat(3n)}}))

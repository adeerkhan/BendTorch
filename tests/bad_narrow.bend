# Must not compile: positions 3 and 4 of an axis of size 4 do not exist, so narrow's proof that start + length fits,
# narrow_ok([4], 0, 3, 2) == True, computes to False.
# expect: - observed : False{}
import Base
import ../torch/tensor.bend as TT

def bad(x: TT.Tensor<[4n]>) -> TT.TP<[4n], [2n]>:
  TT.narrow([4n], 0n, 3n, 2n, {==}, x)

def main() -> Unit:
  Unit{}

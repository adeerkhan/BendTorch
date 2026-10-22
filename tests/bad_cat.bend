# Must not compile: [2, 3] and [2, 4] differ on axis 1, so they cannot be joined along axis 0; cat's proof,
# cat_ok([2, 3], [2, 4], 0) == True, computes to False.
# expect: - observed : False{}
import Base
import ../torch/tensor.bend as TT

def bad(x: TT.Tensor<[2n, 3n]>, y: TT.Tensor<[2n, 4n]>) -> TT.T3<[2n, 3n], [2n, 4n], [4n, 3n]>:
  TT.cat([2n, 3n], [2n, 4n], 0n, {==}, x, y)

def main() -> Unit:
  Unit{}

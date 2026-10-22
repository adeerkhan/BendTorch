# Must not compile: [2, 3] and [4] do not broadcast (the last axes are 3 and 4, and neither is 1), so the proof that
# TT.add asks for, bc_ok([2, 3], [4], [2, 3]) == True, computes to False.
# expect: - observed : False{}
import Base
import ../torch/tensor.bend as TT

def bad(x: TT.Tensor<[2n, 3n]>, y: TT.Tensor<[4n]>) -> TT.T3<[2n, 3n], [4n], [2n, 3n]>:
  TT.add([2n, 3n], [4n], [2n, 3n], {==}, x, y)

def main() -> Unit:
  Unit{}

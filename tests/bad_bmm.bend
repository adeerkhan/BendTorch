# Must not compile: a batch of 3 x 4 matrices times a batch of 5 x 6 ones; the inner sizes (4 and 5) must be one size k.
# expect: - observed : TT.Tensor<[2n, 5n, 6n]>
import Base
import ../torch/tensor.bend as TT

def bad(x: TT.Tensor<[2n, 3n, 4n]>, y: TT.Tensor<[2n, 5n, 6n]>) -> TT.T3<[2n, 3n, 4n], [2n, 4n, 6n], [2n, 3n, 6n]>:
  TT.bmm(2n, 3n, 4n, 6n, x, y)

def main() -> Unit:
  Unit{}

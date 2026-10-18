# bend-ml-tensor-gpu: tensor-array's products on the GPU, with tensor-array's bits.
#
#   import bend-ml-tensor-array@0.1.12.0/main.bend as TA
#   import bend-ml-tensor-gpu@0.1.0.0/main.bend as TG
#   TG.matmul(n, k, m, par, a, b)      # the same arguments, result type and bits as TA.Mat.matmul
#
# The six products (matmul, matmul_nt, matmul_tn and their accumulating forms) take tensor-array's Mat values and return
# its result types; par is ignored (the GPU decides the split). Each product is one `!` call (a bang): a balanced fork
# tree whose leaves each compute a 4 x 4 tile of C, every node sharing A, B and C through Base's Array.fork (two
# handles to one block, no copy), as upstream's one-block pattern does (bendlang/bend tests/run/stencil3d.bend). On a
# machine without a GPU, or with `--gpu off`, the same bang runs on the CPU threads.
#
# The bits: a cell of C is computed in tensor-array's order. Product p of a dot goes to partial sum p mod 4, in
# increasing p, each sum starting at +0.0, and the four sums are combined by TA.t8_comb with rot = k mod 4, as
# tensor-array's dotr and t8_dot do; an accumulating product adds the cell to C once with TA.cwr. The loop takes four
# steps of k a turn; the steps past k in the last turn add +0.0, which leaves a sum unchanged (a sum that starts at +0.0
# is never -0.0). Bend builds the device code with --fmad=false, so no multiply-add is fused. tests/bits.bend checks the
# bits against tensor-array on many shapes, on the GPU and the CPU.
#
# Not proved, and marked so: Array.fork and Array.join are @unsafe in Base, so the defs below that call them, or take
# a shared array, are @unsafe, and `bend main.bend` names every def that relies on them. Nothing checks that a leaf
# writes only its own tile of C, or never writes A or B; the tiles are disjoint by construction (leaf t writes tile t).
# All of it is Bend: no foreign code. Written by bend-ml's scripts/gen_tensor_gpu.py (4 x 4 tiles).
import Base
import bend-ml-tensor-array@0.1.12.0/main.bend as TA

# the sizes and strides of one product: C[i][j] = sum_p A[i*arow + p*sa] * B[p*sb + j*bcol], n x m, k steps
type E is Data:
  E{n: U32, m: U32, k: U32, tcols: U32, tiles: U32, arow: U32, sa: U32, sb: U32, bcol: U32, kg: Nat, rot: Nat, rem: U32, ac: Bool}

# a product that counts only when its step is inside k (the last turn's steps past k add +0.0)
def pk(ok: Bool, +x: F32) -> F32:
  match ok:
    case True{}: x
    case False{}: 0.0

# the values of one turn: A's 4 rows and B's 4 columns at four consecutive steps, read through the arrays in order
type GR is Type:
  GR{a: Array<F32>, b: Array<F32>, x0_0: F32, x0_1: F32, x0_2: F32, x0_3: F32, x1_0: F32, x1_1: F32, x1_2: F32, x1_3: F32, x2_0: F32, x2_1: F32, x2_2: F32, x2_3: F32, x3_0: F32, x3_1: F32, x3_2: F32, x3_3: F32, y0_0: F32, y0_1: F32, y0_2: F32, y0_3: F32, y1_0: F32, y1_1: F32, y1_2: F32, y1_3: F32, y2_0: F32, y2_1: F32, y2_2: F32, y2_3: F32, y3_0: F32, y3_1: F32, y3_2: F32, y3_3: F32}

def gr_31(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, +y3_0: F32, +y3_1: F32, +y3_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_3}:
      GR{a, b, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, y3_1, y3_2, y3_3}
def gr_30(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, +y3_0: F32, +y3_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_2}:
      gr_31(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, y3_1, y3_2, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_29(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, +y3_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_1}:
      gr_30(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, y3_1, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_28(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, +y2_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y3_0}:
      gr_29(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, y3_0, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_27(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, +y2_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_3}:
      gr_28(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, y2_3, a, Array.get(F32, b, (ib + ((sb * 3 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_26(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, +y2_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_2}:
      gr_27(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, y2_2, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_25(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, +y2_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_1}:
      gr_26(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, y2_1, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_24(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, +y1_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y2_0}:
      gr_25(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, y2_0, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_23(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, +y1_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_3}:
      gr_24(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, y1_3, a, Array.get(F32, b, (ib + ((sb * 2 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_22(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, +y1_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_2}:
      gr_23(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, y1_2, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_21(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, +y1_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_1}:
      gr_22(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, y1_1, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_20(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, +y0_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y1_0}:
      gr_21(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, y1_0, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_19(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, +y0_2: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_3}:
      gr_20(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, y0_3, a, Array.get(F32, b, (ib + ((sb * 1 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_18(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, +y0_1: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_2}:
      gr_19(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, y0_2, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 3 : U32) : U32) : U32)))
def gr_17(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, +y0_0: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_1}:
      gr_18(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, y0_1, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 2 : U32) : U32) : U32)))
def gr_16(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, +x3_3: F32, a: Array<F32>, rv: Array<F32> & F32) -> GR:
  match rv:
    case Tuple{b, +y0_0}:
      gr_17(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, y0_0, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 1 : U32) : U32) : U32)))
def gr_15(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, +x3_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_3}:
      gr_16(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, x3_3, a, Array.get(F32, b, (ib + ((sb * 0 : U32) + (bcol * 0 : U32) : U32) : U32)))
def gr_14(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, +x3_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_2}:
      gr_15(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, x3_2, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_13(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, +x3_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_1}:
      gr_14(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, x3_1, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_12(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, +x2_3: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x3_0}:
      gr_13(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, x3_0, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 1 : U32) : U32) : U32)), b)
def gr_11(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, +x2_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_3}:
      gr_12(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3, Array.get(F32, a, (ia + ((sa * 3 : U32) + (arow * 0 : U32) : U32) : U32)), b)
def gr_10(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, +x2_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_2}:
      gr_11(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_9(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, +x2_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_1}:
      gr_10(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_8(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, +x1_3: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x2_0}:
      gr_9(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, x2_0, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 1 : U32) : U32) : U32)), b)
def gr_7(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, +x1_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_3}:
      gr_8(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, Array.get(F32, a, (ia + ((sa * 2 : U32) + (arow * 0 : U32) : U32) : U32)), b)
def gr_6(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, +x1_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_2}:
      gr_7(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_5(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, +x1_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_1}:
      gr_6(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_4(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, +x0_3: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x1_0}:
      gr_5(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, x1_0, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 1 : U32) : U32) : U32)), b)
def gr_3(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, +x0_2: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_3}:
      gr_4(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, x0_3, Array.get(F32, a, (ia + ((sa * 1 : U32) + (arow * 0 : U32) : U32) : U32)), b)
def gr_2(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, +x0_1: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_2}:
      gr_3(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, x0_2, Array.get(F32, a, (ia + ((sa * 0 : U32) + (arow * 3 : U32) : U32) : U32)), b)
def gr_1(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +x0_0: F32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_1}:
      gr_2(ia, ib, arow, sa, sb, bcol, x0_0, x0_1, Array.get(F32, a, (ia + ((sa * 0 : U32) + (arow * 2 : U32) : U32) : U32)), b)
def gr_0(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, rv: Array<F32> & F32, b: Array<F32>) -> GR:
  match rv:
    case Tuple{a, +x0_0}:
      gr_1(ia, ib, arow, sa, sb, bcol, x0_0, Array.get(F32, a, (ia + ((sa * 0 : U32) + (arow * 1 : U32) : U32) : U32)), b)
# the turn whose A values start at ia and B values at ib
def turn(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> GR:
  gr_0(ia, ib, arow, sa, sb, bcol, Array.get(F32, a, ia), b)

# the tile's cells after the four sums are combined, row by row
type T is Type:
  T{a: Array<F32>, b: Array<F32>, c0_0: F32, c0_1: F32, c0_2: F32, c0_3: F32, c1_0: F32, c1_1: F32, c1_2: F32, c1_3: F32, c2_0: F32, c2_1: F32, c2_2: F32, c2_3: F32, c3_0: F32, c3_1: F32, c3_2: F32, c3_3: F32}

# kg more full turns; g holds the current turn's values. The last call (kg = 0) adds the turn that holds the steps
# past the last full turn (rem = k mod 4 of them), counting only those, and combines the sums.
@unsafe
def mm(kg: Nat, +ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +rem: U32, +rot: Nat, +s0_0_0: F32, +s1_0_0: F32, +s2_0_0: F32, +s3_0_0: F32, +s0_0_1: F32, +s1_0_1: F32, +s2_0_1: F32, +s3_0_1: F32, +s0_0_2: F32, +s1_0_2: F32, +s2_0_2: F32, +s3_0_2: F32, +s0_0_3: F32, +s1_0_3: F32, +s2_0_3: F32, +s3_0_3: F32, +s0_1_0: F32, +s1_1_0: F32, +s2_1_0: F32, +s3_1_0: F32, +s0_1_1: F32, +s1_1_1: F32, +s2_1_1: F32, +s3_1_1: F32, +s0_1_2: F32, +s1_1_2: F32, +s2_1_2: F32, +s3_1_2: F32, +s0_1_3: F32, +s1_1_3: F32, +s2_1_3: F32, +s3_1_3: F32, +s0_2_0: F32, +s1_2_0: F32, +s2_2_0: F32, +s3_2_0: F32, +s0_2_1: F32, +s1_2_1: F32, +s2_2_1: F32, +s3_2_1: F32, +s0_2_2: F32, +s1_2_2: F32, +s2_2_2: F32, +s3_2_2: F32, +s0_2_3: F32, +s1_2_3: F32, +s2_2_3: F32, +s3_2_3: F32, +s0_3_0: F32, +s1_3_0: F32, +s2_3_0: F32, +s3_3_0: F32, +s0_3_1: F32, +s1_3_1: F32, +s2_3_1: F32, +s3_3_1: F32, +s0_3_2: F32, +s1_3_2: F32, +s2_3_2: F32, +s3_3_2: F32, +s0_3_3: F32, +s1_3_3: F32, +s2_3_3: F32, +s3_3_3: F32, g: GR) -> T:
  match kg g:
    case 0n GR{a, b, +x0_0, +x0_1, +x0_2, +x0_3, +x1_0, +x1_1, +x1_2, +x1_3, +x2_0, +x2_1, +x2_2, +x2_3, +x3_0, +x3_1, +x3_2, +x3_3, +y0_0, +y0_1, +y0_2, +y0_3, +y1_0, +y1_1, +y1_2, +y1_3, +y2_0, +y2_1, +y2_2, +y2_3, +y3_0, +y3_1, +y3_2, +y3_3}:
      T{a, b, TA.t8_comb(rot, (s0_0_0 + pk(U32.is_lt(0, rem), (x0_0 * y0_0 : F32)) : F32), (s1_0_0 + pk(U32.is_lt(1, rem), (x1_0 * y1_0 : F32)) : F32), (s2_0_0 + pk(U32.is_lt(2, rem), (x2_0 * y2_0 : F32)) : F32), (s3_0_0 + pk(U32.is_lt(3, rem), (x3_0 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_0_1 + pk(U32.is_lt(0, rem), (x0_0 * y0_1 : F32)) : F32), (s1_0_1 + pk(U32.is_lt(1, rem), (x1_0 * y1_1 : F32)) : F32), (s2_0_1 + pk(U32.is_lt(2, rem), (x2_0 * y2_1 : F32)) : F32), (s3_0_1 + pk(U32.is_lt(3, rem), (x3_0 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_0_2 + pk(U32.is_lt(0, rem), (x0_0 * y0_2 : F32)) : F32), (s1_0_2 + pk(U32.is_lt(1, rem), (x1_0 * y1_2 : F32)) : F32), (s2_0_2 + pk(U32.is_lt(2, rem), (x2_0 * y2_2 : F32)) : F32), (s3_0_2 + pk(U32.is_lt(3, rem), (x3_0 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_0_3 + pk(U32.is_lt(0, rem), (x0_0 * y0_3 : F32)) : F32), (s1_0_3 + pk(U32.is_lt(1, rem), (x1_0 * y1_3 : F32)) : F32), (s2_0_3 + pk(U32.is_lt(2, rem), (x2_0 * y2_3 : F32)) : F32), (s3_0_3 + pk(U32.is_lt(3, rem), (x3_0 * y3_3 : F32)) : F32)), TA.t8_comb(rot, (s0_1_0 + pk(U32.is_lt(0, rem), (x0_1 * y0_0 : F32)) : F32), (s1_1_0 + pk(U32.is_lt(1, rem), (x1_1 * y1_0 : F32)) : F32), (s2_1_0 + pk(U32.is_lt(2, rem), (x2_1 * y2_0 : F32)) : F32), (s3_1_0 + pk(U32.is_lt(3, rem), (x3_1 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_1_1 + pk(U32.is_lt(0, rem), (x0_1 * y0_1 : F32)) : F32), (s1_1_1 + pk(U32.is_lt(1, rem), (x1_1 * y1_1 : F32)) : F32), (s2_1_1 + pk(U32.is_lt(2, rem), (x2_1 * y2_1 : F32)) : F32), (s3_1_1 + pk(U32.is_lt(3, rem), (x3_1 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_1_2 + pk(U32.is_lt(0, rem), (x0_1 * y0_2 : F32)) : F32), (s1_1_2 + pk(U32.is_lt(1, rem), (x1_1 * y1_2 : F32)) : F32), (s2_1_2 + pk(U32.is_lt(2, rem), (x2_1 * y2_2 : F32)) : F32), (s3_1_2 + pk(U32.is_lt(3, rem), (x3_1 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_1_3 + pk(U32.is_lt(0, rem), (x0_1 * y0_3 : F32)) : F32), (s1_1_3 + pk(U32.is_lt(1, rem), (x1_1 * y1_3 : F32)) : F32), (s2_1_3 + pk(U32.is_lt(2, rem), (x2_1 * y2_3 : F32)) : F32), (s3_1_3 + pk(U32.is_lt(3, rem), (x3_1 * y3_3 : F32)) : F32)), TA.t8_comb(rot, (s0_2_0 + pk(U32.is_lt(0, rem), (x0_2 * y0_0 : F32)) : F32), (s1_2_0 + pk(U32.is_lt(1, rem), (x1_2 * y1_0 : F32)) : F32), (s2_2_0 + pk(U32.is_lt(2, rem), (x2_2 * y2_0 : F32)) : F32), (s3_2_0 + pk(U32.is_lt(3, rem), (x3_2 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_2_1 + pk(U32.is_lt(0, rem), (x0_2 * y0_1 : F32)) : F32), (s1_2_1 + pk(U32.is_lt(1, rem), (x1_2 * y1_1 : F32)) : F32), (s2_2_1 + pk(U32.is_lt(2, rem), (x2_2 * y2_1 : F32)) : F32), (s3_2_1 + pk(U32.is_lt(3, rem), (x3_2 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_2_2 + pk(U32.is_lt(0, rem), (x0_2 * y0_2 : F32)) : F32), (s1_2_2 + pk(U32.is_lt(1, rem), (x1_2 * y1_2 : F32)) : F32), (s2_2_2 + pk(U32.is_lt(2, rem), (x2_2 * y2_2 : F32)) : F32), (s3_2_2 + pk(U32.is_lt(3, rem), (x3_2 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_2_3 + pk(U32.is_lt(0, rem), (x0_2 * y0_3 : F32)) : F32), (s1_2_3 + pk(U32.is_lt(1, rem), (x1_2 * y1_3 : F32)) : F32), (s2_2_3 + pk(U32.is_lt(2, rem), (x2_2 * y2_3 : F32)) : F32), (s3_2_3 + pk(U32.is_lt(3, rem), (x3_2 * y3_3 : F32)) : F32)), TA.t8_comb(rot, (s0_3_0 + pk(U32.is_lt(0, rem), (x0_3 * y0_0 : F32)) : F32), (s1_3_0 + pk(U32.is_lt(1, rem), (x1_3 * y1_0 : F32)) : F32), (s2_3_0 + pk(U32.is_lt(2, rem), (x2_3 * y2_0 : F32)) : F32), (s3_3_0 + pk(U32.is_lt(3, rem), (x3_3 * y3_0 : F32)) : F32)), TA.t8_comb(rot, (s0_3_1 + pk(U32.is_lt(0, rem), (x0_3 * y0_1 : F32)) : F32), (s1_3_1 + pk(U32.is_lt(1, rem), (x1_3 * y1_1 : F32)) : F32), (s2_3_1 + pk(U32.is_lt(2, rem), (x2_3 * y2_1 : F32)) : F32), (s3_3_1 + pk(U32.is_lt(3, rem), (x3_3 * y3_1 : F32)) : F32)), TA.t8_comb(rot, (s0_3_2 + pk(U32.is_lt(0, rem), (x0_3 * y0_2 : F32)) : F32), (s1_3_2 + pk(U32.is_lt(1, rem), (x1_3 * y1_2 : F32)) : F32), (s2_3_2 + pk(U32.is_lt(2, rem), (x2_3 * y2_2 : F32)) : F32), (s3_3_2 + pk(U32.is_lt(3, rem), (x3_3 * y3_2 : F32)) : F32)), TA.t8_comb(rot, (s0_3_3 + pk(U32.is_lt(0, rem), (x0_3 * y0_3 : F32)) : F32), (s1_3_3 + pk(U32.is_lt(1, rem), (x1_3 * y1_3 : F32)) : F32), (s2_3_3 + pk(U32.is_lt(2, rem), (x2_3 * y2_3 : F32)) : F32), (s3_3_3 + pk(U32.is_lt(3, rem), (x3_3 * y3_3 : F32)) : F32))}
    case 1n+q GR{a, b, +x0_0, +x0_1, +x0_2, +x0_3, +x1_0, +x1_1, +x1_2, +x1_3, +x2_0, +x2_1, +x2_2, +x2_3, +x3_0, +x3_1, +x3_2, +x3_3, +y0_0, +y0_1, +y0_2, +y0_3, +y1_0, +y1_1, +y1_2, +y1_3, +y2_0, +y2_1, +y2_2, +y2_3, +y3_0, +y3_1, +y3_2, +y3_3}:
      mm(q, (ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, rem, rot, (s0_0_0 + (x0_0 * y0_0 : F32) : F32), (s1_0_0 + (x1_0 * y1_0 : F32) : F32), (s2_0_0 + (x2_0 * y2_0 : F32) : F32), (s3_0_0 + (x3_0 * y3_0 : F32) : F32), (s0_0_1 + (x0_0 * y0_1 : F32) : F32), (s1_0_1 + (x1_0 * y1_1 : F32) : F32), (s2_0_1 + (x2_0 * y2_1 : F32) : F32), (s3_0_1 + (x3_0 * y3_1 : F32) : F32), (s0_0_2 + (x0_0 * y0_2 : F32) : F32), (s1_0_2 + (x1_0 * y1_2 : F32) : F32), (s2_0_2 + (x2_0 * y2_2 : F32) : F32), (s3_0_2 + (x3_0 * y3_2 : F32) : F32), (s0_0_3 + (x0_0 * y0_3 : F32) : F32), (s1_0_3 + (x1_0 * y1_3 : F32) : F32), (s2_0_3 + (x2_0 * y2_3 : F32) : F32), (s3_0_3 + (x3_0 * y3_3 : F32) : F32), (s0_1_0 + (x0_1 * y0_0 : F32) : F32), (s1_1_0 + (x1_1 * y1_0 : F32) : F32), (s2_1_0 + (x2_1 * y2_0 : F32) : F32), (s3_1_0 + (x3_1 * y3_0 : F32) : F32), (s0_1_1 + (x0_1 * y0_1 : F32) : F32), (s1_1_1 + (x1_1 * y1_1 : F32) : F32), (s2_1_1 + (x2_1 * y2_1 : F32) : F32), (s3_1_1 + (x3_1 * y3_1 : F32) : F32), (s0_1_2 + (x0_1 * y0_2 : F32) : F32), (s1_1_2 + (x1_1 * y1_2 : F32) : F32), (s2_1_2 + (x2_1 * y2_2 : F32) : F32), (s3_1_2 + (x3_1 * y3_2 : F32) : F32), (s0_1_3 + (x0_1 * y0_3 : F32) : F32), (s1_1_3 + (x1_1 * y1_3 : F32) : F32), (s2_1_3 + (x2_1 * y2_3 : F32) : F32), (s3_1_3 + (x3_1 * y3_3 : F32) : F32), (s0_2_0 + (x0_2 * y0_0 : F32) : F32), (s1_2_0 + (x1_2 * y1_0 : F32) : F32), (s2_2_0 + (x2_2 * y2_0 : F32) : F32), (s3_2_0 + (x3_2 * y3_0 : F32) : F32), (s0_2_1 + (x0_2 * y0_1 : F32) : F32), (s1_2_1 + (x1_2 * y1_1 : F32) : F32), (s2_2_1 + (x2_2 * y2_1 : F32) : F32), (s3_2_1 + (x3_2 * y3_1 : F32) : F32), (s0_2_2 + (x0_2 * y0_2 : F32) : F32), (s1_2_2 + (x1_2 * y1_2 : F32) : F32), (s2_2_2 + (x2_2 * y2_2 : F32) : F32), (s3_2_2 + (x3_2 * y3_2 : F32) : F32), (s0_2_3 + (x0_2 * y0_3 : F32) : F32), (s1_2_3 + (x1_2 * y1_3 : F32) : F32), (s2_2_3 + (x2_2 * y2_3 : F32) : F32), (s3_2_3 + (x3_2 * y3_3 : F32) : F32), (s0_3_0 + (x0_3 * y0_0 : F32) : F32), (s1_3_0 + (x1_3 * y1_0 : F32) : F32), (s2_3_0 + (x2_3 * y2_0 : F32) : F32), (s3_3_0 + (x3_3 * y3_0 : F32) : F32), (s0_3_1 + (x0_3 * y0_1 : F32) : F32), (s1_3_1 + (x1_3 * y1_1 : F32) : F32), (s2_3_1 + (x2_3 * y2_1 : F32) : F32), (s3_3_1 + (x3_3 * y3_1 : F32) : F32), (s0_3_2 + (x0_3 * y0_2 : F32) : F32), (s1_3_2 + (x1_3 * y1_2 : F32) : F32), (s2_3_2 + (x2_3 * y2_2 : F32) : F32), (s3_3_2 + (x3_3 * y3_2 : F32) : F32), (s0_3_3 + (x0_3 * y0_3 : F32) : F32), (s1_3_3 + (x1_3 * y1_3 : F32) : F32), (s2_3_3 + (x2_3 * y2_3 : F32) : F32), (s3_3_3 + (x3_3 * y3_3 : F32) : F32), turn((ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, a, b))

# writes cell (i, j) of C when it is inside n x m
def wr(ok: Bool, +ac: Bool, +idx: U32, +x: F32, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: TA.cwr(ac, idx, x, c)
    case False{}: c

# the tile at rows i0.., columns j0.. of C, from the loop's result; A and B are dropped (the caller holds them)
@unsafe
def put(+ac: Bool, +n: U32, +m: U32, +i0: U32, +j0: U32, c: Array<F32>, t: T) -> Array<F32>:
  match t:
    case T{a, b, +c0_0, +c0_1, +c0_2, +c0_3, +c1_0, +c1_1, +c1_2, +c1_3, +c2_0, +c2_1, +c2_2, +c2_3, +c3_0, +c3_1, +c3_2, +c3_3}:
      wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 3 : U32) : U32), c3_3, wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 2 : U32) : U32), c3_2, wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 1 : U32) : U32), c3_1, wr(Bool.and(U32.is_lt((i0 + 3 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 3 : U32) * m : U32) + (j0 + 0 : U32) : U32), c3_0, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 3 : U32) : U32), c2_3, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 2 : U32) : U32), c2_2, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 1 : U32) : U32), c2_1, wr(Bool.and(U32.is_lt((i0 + 2 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 2 : U32) * m : U32) + (j0 + 0 : U32) : U32), c2_0, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 3 : U32) : U32), c1_3, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 2 : U32) : U32), c1_2, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 1 : U32) : U32), c1_1, wr(Bool.and(U32.is_lt((i0 + 1 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 1 : U32) * m : U32) + (j0 + 0 : U32) : U32), c1_0, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 3 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 3 : U32) : U32), c0_3, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 2 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 2 : U32) : U32), c0_2, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 1 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 1 : U32) : U32), c0_1, wr(Bool.and(U32.is_lt((i0 + 0 : U32), n), U32.is_lt((j0 + 0 : U32), m)), ac, (((i0 + 0 : U32) * m : U32) + (j0 + 0 : U32) : U32), c0_0, c))))))))))))))))

@unsafe
def tile(+e: E, +i0: U32, +j0: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac}:
      put(ac, n, m, i0, j0, c, mm(kg, (i0 * arow : U32), (j0 * bcol : U32), arow, sa, sb, bcol, rem, rot, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, turn((i0 * arow : U32), (j0 * bcol : U32), arow, sa, sb, bcol, a, b)))

# leaf t computes tile t (row-major over the tiles); leaves past the last tile write nothing
@unsafe
def leaf_if(ok: Bool, +e: E, +t: U32, +tcols: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: tile(e, (U32.div(t, tcols) * 4 : U32), (U32.mod(t, tcols) * 4 : U32), a, b, c)
    case False{}: c

@unsafe
def leaf(+e: E, +t: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac}:
      leaf_if(U32.is_lt(t, tiles), e, t, tcols, a, b, c)

@unsafe
def tree(+e: E, d: Nat, +t: U32, +a: Array<F32>, +b: Array<F32>, +c: Array<F32>) -> Array<F32>:
  match d:
    case 0n: leaf(e, t, a, b, c)
    case 1n++q:
      l r = tree(e, q, (t * 2 : U32), a, b, c) tree(e, q, ((t * 2 : U32) + 1 : U32), a, b, c)
      Array.join(F32, l, r)

# C made inside the bang: the root of the ! call allocates it on the device. Made on the host instead, a new C often
# reuses a block whose pages sit on the device, so zeroing it on the host pulls them back over PCIe and the bang moves
# them again (a 256 x 1536 product: 9.2 ms against 6.0, measured 2026-10-18). The leaves write every cell of C.
@unsafe
def tree_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree(e, d, 0, a, b, Array.new(F32, cd, 0.0))

@unsafe
def bang_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree_new!(e, d, a, b, cd)

# one bang over the tiles; A and B are shared with every leaf and come back unchanged
@unsafe
def bang(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  tree!(e, d, 0, a, b, c)

type ABC is Type:
  ABC{a: Array<F32>, b: Array<F32>, c: Array<F32>}

@unsafe
def run_e(+e: E, +tiles: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> ABC:
  ABC{a, b, bang(e, TA.cap_depth(U32.to_nat(tiles)), a, b, c)}

# the product's sizes and strides
def e_of(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32) -> E:
  E{U32.from_nat(n), U32.from_nat(m), U32.from_nat(k), U32.from_nat(Nat.div(Nat.add(m, 3n), 4n)), U32.from_nat(Nat.mul(Nat.div(Nat.add(n, 3n), 4n), Nat.div(Nat.add(m, 3n), 4n))), arow, sa, sb, bcol, Nat.div(k, 4n), Nat.mod(k, 4n), U32.from_nat(Nat.mod(k, 4n)), ac}

def tiles(+n: Nat, +m: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(n, 3n), 4n), Nat.div(Nat.add(m, 3n), 4n))

# C (n x m) = A · B with k steps into a new C made on the device
@unsafe
def gemm_new(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>) -> ABC:
  ABC{a, b, bang_new(e_of(False{}, n, m, k, arow, sa, sb, bcol), TA.cap_depth(tiles(n, m)), a, b, TA.cap_depth(Nat.mul(n, m)))}

# C (n x m) = A · B with k steps, or C + A · B when ac, C[i][j] = sum_p A[i*arow + p*sa] * B[p*sb + j*bcol]
@unsafe
def gemm(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> ABC:
  run_e(e_of(ac, n, m, k, arow, sa, sb, bcol), U32.from_nat(tiles(n, m)), a, b, c)

def mm_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMul<n, k, m>:
  match g:
    case ABC{a, b, c}: TA.MMul{TA.Mat{a}, TA.Mat{b}, TA.Mat{c}}

def mm_nt_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulNT<n, k, m>:
  match g:
    case ABC{a, b, c}: TA.MMulNT{TA.Mat{a}, TA.Mat{b}, TA.Mat{c}}

def mm_tn_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulTN<n, k, m>:
  match g:
    case ABC{a, b, c}: TA.MMulTN{TA.Mat{a}, TA.Mat{b}, TA.Mat{c}}

# C = A · B   (n x k) · (k x m), tensor-array's TA.Mat.matmul bits
@unsafe
def matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      mm_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y))

# C + A · B, into C (n x m)
@unsafe
def matmul_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMul<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      mm_fin(n, k, m, gemm(True{}, n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, z))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: for weights stored with one row per output
@unsafe
def matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>) -> TA.MMulNT<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      mm_nt_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y))

# C + A · Bᵀ, into C (n x m)
@unsafe
def matmul_nt_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>, c: TA.Mat<n, m>) -> TA.MMulNT<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      mm_nt_fin(n, k, m, gemm(True{}, n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, z))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): weight gradient, Xᵀ · dY
@unsafe
def matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>) -> TA.MMulTN<n, k, m>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      mm_tn_fin(n, k, m, gemm_new(n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y))

# C + Aᵀ · B, into C (n x m): dW += Xᵀ · dY
@unsafe
def matmul_tn_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMulTN<n, k, m>:
  match a b c:
    case TA.Mat{x} TA.Mat{y} TA.Mat{z}:
      mm_tn_fin(n, k, m, gemm(True{}, n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, z))

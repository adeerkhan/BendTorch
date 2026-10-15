# The mathematical model the laws speak about: vectors and matrices as lists of Nat, a matrix as its list of rows.
# Nothing here runs in a model; it is the specification that LAWS.bend states facts about. Nat makes the algebra
# exact (no rounding), so a law proved here says the formulas are right; whether the F32 code computes them to F32's
# precision is tested against PyTorch, never proved (docs/design.md, "Proved, tested, trusted").

import Base

# the dot product of the common prefix
def dotn(xs: List<&2, Nat>, ys: List<&2, Nat>) -> Nat:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      Nat.add(Nat.mul(x, y), dotn(xt, yt))
    case _ _:
      0n

# vector sum; the shorter vector counts as padded with zeros
def vadd(xs: List<&2, Nat>, ys: List<&2, Nat>) -> List<&2, Nat>:
  match xs ys:
    case Nil{} _:
      ys
    case Con{x, xt} Nil{}:
      Con{x, xt}
    case Con{x, xt} Con{y, yt}:
      Nat.add(x, y) <> vadd(xt, yt)

def vscale(+k: Nat, xs: List<&2, Nat>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      Nat.mul(k, x) <> vscale(k, t)

# M·v: one dot product per row (a linear layer's forward pass, y = W·x)
def mv(rows: List<&2, List<&2, Nat>>, +v: List<&2, Nat>) -> List<&2, Nat>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      dotn(r, v) <> mv(t, v)

# Mᵀ·g: the rows weighted by g and summed (a linear layer's backward pass to its input, dx = Wᵀ·dy)
def mtv(rows: List<&2, List<&2, Nat>>, g: List<&2, Nat>) -> List<&2, Nat>:
  match rows g:
    case Con{r, rt} Con{gi, gt}:
      vadd(vscale(gi, r), mtv(rt, gt))
    case _ _:
      Nil{}

# g ⊗ v: row i is g[i]·v (a linear layer's weight gradient, dW = dy ⊗ x)
def outer(g: List<&2, Nat>, +v: List<&2, Nat>) -> List<&2, List<&2, Nat>>:
  match g:
    case Nil{}:
      Nil{}
    case Con{gi, gt}:
      vscale(gi, v) <> outer(gt, v)

# the sum of the products of matching entries of two matrices (Frobenius inner product)
def frob(a: List<&2, List<&2, Nat>>, b: List<&2, List<&2, Nat>>) -> Nat:
  match a b:
    case Con{ar, at} Con{br, bt}:
      Nat.add(dotn(ar, br), frob(at, bt))
    case _ _:
      0n

# a <= b, by recursion (simpler to reason about than Nat.is_le, which goes through Nat.cmp)
def le(a: Nat, b: Nat) -> Bool:
  match a b:
    case 0n _:
      True{}
    case 1n+p 0n:
      False{}
    case 1n+p 1n+q:
      le(p, q)

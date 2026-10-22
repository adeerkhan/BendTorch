# The index maps of torch/tensor.bend stay inside the arrays they read and write (the property behind the laws track's
# "index maps in bounds", step L1). An Array's index wraps around, so a map that leaves its array reads or writes a wrong
# cell without failing; this program walks every position of every case and counts the positions whose index is not
# below the array's count.
#
#   bend demo/imap_check.bend -o .scratch/imap_check && .scratch/imap_check cases.bin
#
# cases.bin, little-endian U32: the number of cases, then per case the map, an axis, two numbers and two shapes (each its
# rank and its sizes). Maps: 0 broadcast (a and b read along o = b's shape, a the first shape), 1 an axis taken out
# (shape a, axis k), 2 narrow (shape a, axis k, start and length the two numbers), 3 cat (a then b along axis k: both
# maps), 4 permute (shape a, permutation b), 5 every position to one. Prints one line per case: the positions walked and
# the positions out of bounds.

import Base
import ../torch/main.bend as T
import ../torch/tensor.bend as TT

def u32s(bs: List<&2, U32>) -> List<&2, U32>:
  match bs:
    case Con{+b0, Con{+b1, Con{+b2, Con{+b3, t}}}}:
      (b0 + ((b1 * 256 : U32) + ((b2 * 65536 : U32) + (b3 * 16777216 : U32) : U32) : U32) : U32) <> u32s(t)
    case _:
      Nil{}

def nats(xs: List<&2, U32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      U32.to_nat(h) <> nats(t)

# positions j = 0 .. cnt-1 whose index m(j) is not below bound
def bad(cnt: Nat, +j: U32, +m: TT.IMap, +bound: U32, acc: Nat) -> Nat:
  match cnt:
    case 0n:
      acc
    case 1n+p:
      bad(p, (j + 1 : U32), m, bound, Nat.add(acc, Bool.pick(Nat, U32.is_lt(TT.imap(m, j), bound), 0n, 1n)))

def walk(+cnt: Nat, +m: TT.IMap, +bound: Nat) -> Nat:
  bad(cnt, 0, m, U32.from_nat(bound), 0n)

def line(+walked: Nat, +out: Nat) -> String:
  Nat.show(walked) ++ " " ++ Nat.show(out)

def check(+op: Nat, +k: Nat, +x: Nat, +y: Nat, +a: List<&2, Nat>, +b: List<&2, Nat>) -> String:
  match op:
    case 0n:
      line(Nat.mul(2n, TT.numel(b)), Nat.add(walk(TT.numel(b), TT.bmap(a, b), TT.numel(a)), walk(TT.numel(b), TT.bmap(b, b), TT.numel(b))))
    case 1n:
      line(TT.numel(a), walk(TT.numel(a), TT.drop_map(a, k), TT.numel(TT.remove(a, k))))
    case 2n:
      line(TT.numel(TT.setd(a, k, y)), walk(TT.numel(TT.setd(a, k, y)), TT.narrow_map(a, k, x, y), TT.numel(a)))
    case 3n:
      line(Nat.add(TT.numel(a), TT.numel(b)), Nat.add(walk(TT.numel(a), TT.cat_map_a(a, b, k), TT.numel(TT.cat_shape(a, b, k))), walk(TT.numel(b), TT.cat_map_b(a, b, k), TT.numel(TT.cat_shape(a, b, k)))))
    case 4n:
      line(TT.numel(a), walk(TT.numel(a), TT.perm_map(a, b), TT.numel(a)))
    case _:
      line(TT.numel(a), walk(TT.numel(a), TT.all_map(a), 1n))

# ---- the cases ----

type Sh is Data:
  Sh{s: List<&2, Nat>, rest: List<&2, U32>}

def shape(ws: List<&2, U32>) -> Sh:
  match ws:
    case Con{+r, +t}:
      Sh{nats(List.take(&2, U32, t, U32.to_nat(r))), List.drop(&2, U32, t, U32.to_nat(r))}
    case Nil{}:
      Sh{Nil{}, Nil{}}

# one case: the shape b after shape a, then the line, then the rest
def case_b(+op: Nat, +k: Nat, +x: Nat, +y: Nat, +a: List<&2, Nat>, sb: Sh) -> String & List<&2, U32>:
  match sb:
    case Sh{b, r}:
      Tuple{check(op, k, x, y, a, b), r}

def case_a(+op: Nat, +k: Nat, +x: Nat, +y: Nat, sa: Sh) -> String & List<&2, U32>:
  match sa:
    case Sh{a, r}:
      case_b(op, k, x, y, a, shape(r))

def one(ws: List<&2, U32>) -> String & List<&2, U32>:
  match ws:
    case Con{+op, Con{+k, Con{+x, Con{+y, r}}}}:
      case_a(U32.to_nat(op), U32.to_nat(k), U32.to_nat(x), U32.to_nat(y), shape(r))
    case _:
      Tuple{"bad case", Nil{}}

def Tuple.then(r: String & List<&2, U32>, k: String -> List<&2, U32> -> IO(Unit)) -> IO(Unit):
  match r:
    case Tuple{s, t}:
      k(s, t)

# n cases, each line printed as it is checked
def all(n: Nat, ws: List<&2, U32>) -> IO(Unit):
  match n:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      Tuple.then(one(ws), s => r => do IO<Unit>:
        IO.print(s)
        all(p, r))

def start(ws: List<&2, U32>) -> IO(Unit):
  match ws:
    case Con{+n, r}:
      all(U32.to_nat(n), r)
    case Nil{}:
      IO.print("empty")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+path, Nil{}}}:
      do IO<Unit>:
        bs : List<&2, U32> <- T.read_at(path, 0, 4000000)
        start(u32s(bs))
    case _:
      IO.print("usage: imap_check <cases.bin>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)

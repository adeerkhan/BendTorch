"""torch/tensor.bend against torch.autograd on random shapes (ranks 0 to 4, size-1 axes and empty tensors), through
demo/tensor_test.bend, which reads the operation and the shapes at run time.

  reference/.venv/bin/python reference/test_tensor.py .scratch/tensor_test [cases per operation]

Each case writes case.bin (the operation, an axis, then three shapes, each as its rank and its sizes) and the inputs
(x, y, dz), runs the program, and compares every result with PyTorch's under the one policy of oracle.py: the forward
result z and the gradients ga and gb from dz. A shape condition that does not hold must print "reject".
"""
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from oracle import close, parse, write_f32

ADD, SUB, MUL, DIV, SUM, SUMK, MEAN, SUMALL, MEANALL, RESHAPE, FLATTEN, PERMUTE, TRANSPOSE, NARROW, CAT, BMM = range(16)


def write_case(d: Path, op: int, k: int, a=(), b=(), o=()) -> None:
    words = [op, k]
    for s in (a, b, o):
        words += [len(s), *s]
    (d / "case.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))


def rand(shape, g, positive=False):
    t = torch.randn(tuple(shape), generator=g, dtype=torch.float32)
    return t.abs() + 0.5 if positive else t


def size(empty_ok=True):
    r = random.random()
    if empty_ok and r < 0.05:
        return 0
    if r < 0.25:
        return 1
    return random.randint(2, 5)


def rand_shape(rank, empty_ok=True):
    return [size(empty_ok) for _ in range(rank)]


def broadcastable(rank_o):
    o = rand_shape(rank_o)
    def side():
        r = random.randint(0, rank_o)
        return [1 if random.random() < 0.35 else n for n in o[rank_o - r:]]
    a, b = side(), side()
    if len(a) < rank_o and len(b) < rank_o:
        a = o[: rank_o - len(a)] + a
    return a, b, list(torch.broadcast_shapes(tuple(a), tuple(b)))


def run(binary, d):
    return subprocess.run([binary, str(d)], check=True, capture_output=True, text=True).stdout


class Case:
    def __init__(self, binary, d, failures):
        self.binary, self.d, self.failures, self.worst, self.n = binary, d, failures, 0.0, 0

    def go(self, tag, op, k, a, b, o, inputs, results):
        """inputs: name -> tensor written to <name>.bin; results: name -> PyTorch tensor to compare."""
        write_case(self.d, op, k, a, b, o)
        for name, t in inputs.items():
            write_f32(self.d / f"{name}.bin", t.detach())
        got = parse(run(self.binary, self.d))
        for name, ref in results.items():
            self.worst = max(self.worst, close(f"{tag} {name}", got.get(name, []), ref, self.failures))
        self.n += 1

    def reject(self, tag, op, k, a, b, o):
        write_case(self.d, op, k, a, b, o)
        out = run(self.binary, self.d).strip()
        if out != "reject":
            self.failures.append(f"{tag} must be rejected, printed {out[:60]!r}")
        self.n += 1


def binary_op(c, g, op):
    a, b, o = broadcastable(random.randint(0, 4))
    x, y = rand(a, g).requires_grad_(), rand(b, g, positive=(op == DIV)).requires_grad_()
    dz = rand(o, g)
    z = [x + y, x - y, x * y, x / y][op]
    z.backward(dz)
    c.go(f"{['add', 'sub', 'mul', 'div'][op]} {a},{b}->{o}", op, 0, a, b, o, {"x": x, "y": y, "dz": dz}, {"z": z, "ga": x.grad, "gb": y.grad})


def reduction(c, g, op):
    s = rand_shape(random.randint(1, 4))
    k = random.randrange(len(s))
    x = rand(s, g).requires_grad_()
    if op in (SUM, MEAN):
        z = x.sum(dim=k) if op == SUM else x.mean(dim=k)
    elif op == SUMK:
        z = x.sum(dim=k, keepdim=True)
    else:
        z = x.sum() if op == SUMALL else x.mean()
    dz = rand(list(z.shape), g)
    z.backward(dz)
    res = {"z": z} if op == SUMK else {"z": z, "ga": x.grad}
    if op in (MEAN, MEANALL) and x.numel() == 0:
        return  # PyTorch gives NaN for an empty mean; the Bend version divides by 0 too, compared elsewhere
    c.go(f"{['sum', 'sum keepdim', 'mean', 'sum all', 'mean all'][op - SUM]} {s} axis {k}", op, k, s, [], [], {"x": x, "dz": dz}, res)


def reshape(c, g):
    s = rand_shape(random.randint(0, 4), empty_ok=False)
    n = 1
    for v in s:
        n *= v
    divs = [d for d in range(1, n + 1) if n % d == 0]
    t = [random.choice(divs)]
    t.append(n // t[0])
    random.shuffle(t)
    x = rand(s, g)
    dz = rand(t, g)
    c.go(f"reshape {s}->{t}", RESHAPE, 0, s, t, [], {"x": x, "dz": dz}, {"z": x.reshape(t), "ga": dz.reshape(s)})


def flatten(c, g):
    s = rand_shape(random.randint(0, 4))
    x = rand(s, g)
    c.go(f"flatten {s}", FLATTEN, 0, s, [], [], {"x": x}, {"z": x.reshape(-1)})


def permute(c, g, transpose):
    r = random.randint(1, 4)
    s = rand_shape(r)
    x = rand(s, g).requires_grad_()
    if transpose:
        i, j = random.randrange(r), random.randrange(r)
        z = x.transpose(i, j)
        k, b = i, [j]
    else:
        p = list(range(r))
        random.shuffle(p)
        z = x.permute(*p)
        k, b = 0, p
    dz = rand(list(z.shape), g)
    z.backward(dz)
    c.go(f"{'transpose' if transpose else 'permute'} {s} {b} k={k}", TRANSPOSE if transpose else PERMUTE, k, s, b, [], {"x": x, "dz": dz}, {"z": z, "ga": x.grad})


def narrow(c, g):
    s = rand_shape(random.randint(1, 4))
    d = random.randrange(len(s))
    st = random.randint(0, s[d])
    ln = random.randint(0, s[d] - st)
    x = rand(s, g).requires_grad_()
    z = x.narrow(d, st, ln)
    dz = rand(list(z.shape), g)
    z.backward(dz)
    c.go(f"narrow {s} axis {d} [{st}, {st + ln})", NARROW, d, s, [st, ln], [], {"x": x, "dz": dz}, {"z": z, "ga": x.grad})


def cat(c, g):
    r = random.randint(1, 4)
    a = rand_shape(r)
    d = random.randrange(r)
    b = list(a)
    b[d] = size()
    x, y = rand(a, g).requires_grad_(), rand(b, g).requires_grad_()
    z = torch.cat([x, y], dim=d)
    dz = rand(list(z.shape), g)
    z.backward(dz)
    c.go(f"cat {a},{b} axis {d}", CAT, d, a, b, [], {"x": x, "y": y, "dz": dz}, {"z": z, "ga": x.grad, "gb": y.grad})


def bmm(c, g):
    bs, n, k, m = (size() for _ in range(4))
    x, y = rand([bs, n, k], g).requires_grad_(), rand([bs, k, m], g).requires_grad_()
    z = torch.bmm(x, y)
    dz = rand([bs, n, m], g)
    z.backward(dz)
    c.go(f"bmm [{bs},{n},{k}]·[{bs},{k},{m}]", BMM, 0, [bs, n, k], [bs, k, m], [], {"x": x, "y": y, "dz": dz}, {"z": z, "ga": x.grad, "gb": y.grad})


REJECTS = [
    ("add [2,3]+[4]", ADD, 0, [2, 3], [4], [2, 3]),
    ("mul [3]+[2,3] -> [3]", MUL, 0, [3], [2, 3], [3]),
    ("sub [2]+[3]", SUB, 0, [2], [3], [2]),
    ("div [2,3]+[2,3] -> rank 3", DIV, 0, [2, 3], [2, 3], [1, 2, 3]),
    ("reshape [2,3] -> [4]", RESHAPE, 0, [2, 3], [4], []),
    ("permute [2,3] by [0,0]", PERMUTE, 0, [2, 3], [0, 0], []),
    ("permute [2,3] by [0]", PERMUTE, 0, [2, 3], [0], []),
    ("transpose [2,3] axes 0, 2", TRANSPOSE, 0, [2, 3], [2], []),
    ("narrow [4] [3, 5)", NARROW, 0, [4], [3, 2], []),
    ("narrow [4] on axis 1", NARROW, 1, [4], [0, 1], []),
    ("cat [2,3],[2,4] axis 0", CAT, 0, [2, 3], [2, 4], []),
    ("cat [2,3],[2,3,1] axis 0", CAT, 0, [2, 3], [2, 3, 1], []),
    ("bmm [2,3,4]·[2,5,6]", BMM, 0, [2, 3, 4], [2, 5, 6], []),
    ("bmm [2,3,4]·[3,4,6]", BMM, 0, [2, 3, 4], [3, 4, 6], []),
]


def main():
    binary = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    random.seed(0)
    g = torch.Generator().manual_seed(0)
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as t:
        c = Case(binary, Path(t), failures)
        for _ in range(n):
            for op in (ADD, SUB, MUL, DIV):
                binary_op(c, g, op)
            for op in (SUM, SUMK, MEAN, SUMALL, MEANALL):
                reduction(c, g, op)
            reshape(c, g)
            flatten(c, g)
            permute(c, g, False)
            permute(c, g, True)
            narrow(c, g)
            cat(c, g)
            bmm(c, g)
        for r in REJECTS:
            c.reject(*r)
    print(f"tensor: {c.n} cases on random shapes (ranks 0-4, size-1 and empty axes; 16 operations and their gradients, "
          f"{len(REJECTS)} shapes that must be rejected), worst error {c.worst:.3f} of the tolerance")
    for f in failures[:25]:
        print("FAIL", f)
    print("tensor: PASS" if not failures else f"tensor: {len(failures)} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

"""The index maps of torch/tensor.bend stay inside their arrays (laws track, step L1), through demo/imap_check.bend.

  reference/.venv/bin/python reference/check_imap.py .scratch/imap_check

Every small case (shapes of rank 0 to 3 with sizes 0 to 3; every axis, window, join and permutation of them; every
shape a that broadcasts to o), then random larger ones (rank up to 5, sizes up to 7). For each case the program walks
every position of the map and counts the indexes that are not below the array's count; all must be 0.
"""
import itertools
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

BC, DROP, NARROW, CAT, PERM, ALL = range(6)


def shapes(max_rank, sizes):
    for r in range(max_rank + 1):
        yield from (list(s) for s in itertools.product(sizes, repeat=r))


def broadcasts(a, o):
    try:
        return list(torch.broadcast_shapes(tuple(a), tuple(o))) == list(o)
    except RuntimeError:
        return False


def cases(max_rank, sizes, sample=None):
    ss = list(shapes(max_rank, sizes))
    out = []
    for o in ss:
        for a in ss:
            if len(a) <= len(o) and broadcasts(a, o):
                out.append((BC, 0, 0, 0, a, o))
    for s in ss:
        out.append((ALL, 0, 0, 0, s, []))
        for k in range(len(s)):
            out.append((DROP, k, 0, 0, s, []))
            for st in range(s[k] + 1):
                for ln in range(s[k] - st + 1):
                    out.append((NARROW, k, st, ln, s, []))
            for n in sizes:
                b = list(s)
                b[k] = n
                out.append((CAT, k, 0, 0, s, b))
        for p in itertools.permutations(range(len(s))):
            out.append((PERM, 0, 0, 0, s, list(p)))
    if sample is not None and len(out) > sample:
        out = random.sample(out, sample)
    return out


def write(path, cs):
    words = [len(cs)]
    for op, k, x, y, a, b in cs:
        words += [op, k, x, y, len(a), *a, len(b), *b]
    path.write_bytes(struct.pack(f"<{len(words)}I", *words))


def run(binary, cs, d):
    f = Path(d) / "cases.bin"
    write(f, cs)
    out = subprocess.run([binary, str(f)], check=True, capture_output=True, text=True).stdout.split("\n")
    return [tuple(map(int, line.split())) for line in out if line.strip()]


def main():
    binary = sys.argv[1]
    random.seed(0)
    small = cases(3, [0, 1, 2, 3])
    large = cases(5, [1, 2, 7], sample=400) + cases(4, [0, 1, 5], sample=200)
    names = ["broadcast", "axis taken out", "narrow", "cat", "permute", "all to one"]
    walked = out_of_bounds = 0
    bad = []
    with tempfile.TemporaryDirectory() as d:
        for group in (small, large):
            for i in range(0, len(group), 2000):
                chunk = group[i:i + 2000]
                res = run(binary, chunk, d)
                if len(res) != len(chunk):
                    sys.exit(f"imap: {len(res)} answers for {len(chunk)} cases")
                for c, (w, o) in zip(chunk, res):
                    walked += w
                    out_of_bounds += o
                    if o:
                        bad.append(f"{names[c[0]]} {c[4]} {c[5]} k={c[1]} x={c[2]} y={c[3]}: {o} of {w} positions out of bounds")
    print(f"imap: {len(small)} small cases (every shape of rank 0-3, sizes 0-3) and {len(large)} larger ones, "
          f"{walked} positions walked, {out_of_bounds} out of bounds")
    for b in bad[:15]:
        print("FAIL", b)
    print("imap: PASS" if not bad else f"imap: {len(bad)} FAILURES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""The PyTorch twin of demo/tiny_rt.bend: reference/tiny_ref.py's model with the batch B and the sequence length T
given on the command line, and the batches read from a file.

    reference/.venv/bin/python reference/tiny_rt_ref.py --prep B T STEPS <file>   # STEPS batches of B x (T + 1) ids
    reference/.venv/bin/python reference/tiny_rt_ref.py B T <file> grads
    reference/.venv/bin/python reference/tiny_rt_ref.py B T <file> train <steps>
    reference/.venv/bin/python reference/tiny_rt_ref.py B T <file> train64 <steps>
    reference/.venv/bin/python reference/tiny_rt_ref.py B T <file> generate <prompt> <n> < bend-output

With B = 2, T = 16 and data/tiny/batches.bin it is tiny_ref.py. The batches for other sizes come from the same text,
at offsets drawn from a generator seeded with 1000 * B + T.
"""

import sys
from pathlib import Path

import numpy as np
import torch

import tiny_ref as R


def prep(b, t, steps, out):
    text = (R.D / "input.txt").read_text()
    chars = (R.D / "vocab.txt").read_text()
    ids = np.array([chars.index(c) for c in text], dtype=np.uint8)
    g = torch.Generator().manual_seed(1000 * b + t)
    offs = torch.randint(0, len(ids) - t - 1, (steps, b), generator=g).numpy()
    batches = np.stack([np.stack([ids[o:o + t + 1] for o in row]) for row in offs])  # steps x b x (t + 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(batches.astype(np.uint8).tobytes())


def main():
    if sys.argv[1] == "--prep":
        prep(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
        return
    R.B, R.T = int(sys.argv[1]), int(sys.argv[2])
    path = sys.argv[3]

    def batch(step):
        n = R.B * (R.T + 1)
        raw = np.fromfile(path, dtype=np.uint8, count=n, offset=step * n)
        assert raw.size == n, f"{path}: batch {step} is short"
        raw = torch.from_numpy(raw.astype(np.int64)).reshape(R.B, R.T + 1)
        return raw[:, :R.T].reshape(-1), raw[:, 1:].reshape(-1)

    R.batch = batch
    sys.argv = [sys.argv[0], *sys.argv[4:]]
    R.main()


if __name__ == "__main__":
    main()

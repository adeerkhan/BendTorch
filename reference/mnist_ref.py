"""The PyTorch twin of demo/mnist.bend: a 784-128-10 MLP, the same initial weights, batches and optimizer.

    reference/.venv/bin/python reference/mnist_ref.py --init            # writes data/mnist/init/*.bin (seed 0)
    reference/.venv/bin/python reference/mnist_ref.py [epochs] [batches] [opt] [lr]   # THREADS=n for n threads (1)

The initial weights are nn.Linear's default initialization under torch.manual_seed(0), written in nn.Linear's layout
(out x in). Batches of 100 in file order, no shuffling, pixels / 255. Prints, per epoch, the mean training loss and the
test hits, in the same format as the Bend program.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

D = Path("data/mnist")
OPTS = {
    "sgd": lambda ps, lr: torch.optim.SGD(ps, lr=lr),
    "momentum": lambda ps, lr: torch.optim.SGD(ps, lr=lr, momentum=0.9),
    "adam": lambda ps, lr: torch.optim.Adam(ps, lr=lr),
    "adamw": lambda ps, lr: torch.optim.AdamW(ps, lr=lr, weight_decay=0.01),
}


def images(name):
    return torch.from_numpy(np.fromfile(D / name, dtype=np.uint8)[16:].reshape(-1, 784).astype(np.float32)) / 255.0


def labels(name):
    return torch.from_numpy(np.fromfile(D / name, dtype=np.uint8)[8:].astype(np.int64))


def model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))


def main():
    m = model()
    if sys.argv[1:] == ["--init"]:
        (D / "init").mkdir(exist_ok=True)
        for name, p in zip(["w1", "b1", "w2", "b2"], m.parameters()):
            (D / "init" / f"{name}.bin").write_bytes(p.detach().numpy().astype("<f4").tobytes())
        (D / "init.bin").write_bytes(b"".join(p.detach().numpy().astype("<f4").tobytes() for p in m.parameters()))
        return
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    opt_name = sys.argv[3] if len(sys.argv) > 3 else "sgd"
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    if not 1 <= batch_size <= 60000:
        raise SystemExit("batch size must be between 1 and 60000")
    total_batches = (60000 + batch_size - 1) // batch_size
    nb = nb or total_batches
    if not 0 <= nb <= total_batches:
        raise SystemExit("too many training batches")
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    xs, ys = images("train-images-idx3-ubyte"), labels("train-labels-idx1-ubyte")
    tx, ty = images("t10k-images-idx3-ubyte"), labels("t10k-labels-idx1-ubyte")
    opt = OPTS[opt_name](m.parameters(), lr)
    for e in range(1, epochs + 1):
        t0, total = time.perf_counter(), 0.0
        for b in range(nb):
            opt.zero_grad()
            loss = F.cross_entropy(m(xs[b * batch_size:(b + 1) * batch_size]), ys[b * batch_size:(b + 1) * batch_size])
            loss.backward()
            opt.step()
            total += np.float32(loss.item())
        secs = time.perf_counter() - t0
        with torch.no_grad():
            hits = sum(int((m(tx[i:i + batch_size]).argmax(1) == ty[i:i + batch_size]).sum())
                       for i in range(0, len(tx), batch_size))
        print(f"epoch {e} train_loss={np.float32(total / nb)} test_correct={hits}/10000 train_seconds={secs:.3f}")


if __name__ == "__main__":
    main()

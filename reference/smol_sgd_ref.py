"""SGD training of the SmolLM2 twin (reference/smol_ref.py), for the multi-step check in reference/test_smol.py.

    reference/.venv/bin/python reference/smol_sgd_ref.py train <steps> <lr>    # 1 + steps losses (bits), as smol_ref's train
    reference/.venv/bin/python reference/smol_sgd_ref.py train64 <steps> <lr>  # the same in float64: loss64 per step

Plain SGD (no momentum, no weight decay: w <- w - lr * g), as demo/smol.bend's training with T.CSGD{lr}, on the same
checkpoint and batches. SMOL=360m and THREADS=n as in smol_ref.py.
"""

import os
import sys

import torch

import smol_ref as S


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    if len(sys.argv) != 4 or sys.argv[1] not in ("train", "train64"):
        sys.exit(__doc__)
    f64, steps, lr = sys.argv[1] == "train64", int(sys.argv[2]), float(sys.argv[3])
    p = S.load()
    if f64:
        p = {k: v.detach().double().requires_grad_(True) for k, v in p.items()}
    else:
        print("loss", S.bits(torch.tensor(0.0)), flush=True)  # the Bend program's first line: the 0 before any step
    opt = torch.optim.SGD(p.values(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = S.forward(p, *S.batch(step))
        loss.backward()
        opt.step()
        print("loss64", repr(loss.item())) if f64 else print("loss", S.bits(loss), flush=True)


if __name__ == "__main__":
    main()

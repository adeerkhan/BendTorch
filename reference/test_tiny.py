"""demo/tiny_llama.bend against its PyTorch twin (reference/tiny_ref.py).

1. One forward and backward pass on the first batch: the loss and all 21 parameter gradients, within the shared
   tolerance (oracle.py).
2. AdamW training for `steps` steps (default 20) from the same weights on the same batches: the loss of every step and
   the final weights, within the same tolerance.

    reference/.venv/bin/python reference/test_tiny.py .scratch/tiny_llama [steps]
"""

import subprocess
import sys

import numpy as np
import torch

from oracle import bits_equal, close, parse


def both(binary, *args):
    bend = parse(subprocess.run([binary, *args], check=True, capture_output=True, text=True).stdout)
    ref = parse(subprocess.run([sys.executable, "reference/tiny_ref.py", *args], check=True, capture_output=True,
                               text=True).stdout)
    return bend, ref


def compare(stage, bend, ref, failures, skip_first_loss=False):
    worst, same, total = 0.0, 0, 0
    for name, r in ref.items():
        b = bend.get(name)
        if b is None:
            failures.append(f"{stage}: the Bend program printed no {name}")
            continue
        if name == "loss" and skip_first_loss:
            b = b.reshape(-1)[1:]  # the first line is the 0 before any step
        rt = torch.from_numpy(np.asarray(r, dtype=np.float32).copy())
        worst = max(worst, close(f"{stage} {name}", b, rt, failures))
        s, t = bits_equal(b, rt)
        same, total = same + s, total + t
    print(f"tiny {stage}: {total} numbers, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance")


def main():
    binary = sys.argv[1]
    steps = sys.argv[2] if len(sys.argv) > 2 else "20"
    failures: list[str] = []
    compare("grads", *both(binary, "grads"), failures)
    bend, ref = both(binary, "train", steps)
    compare(f"train {steps}", bend, ref, failures, skip_first_loss=True)
    losses = ref["loss"].reshape(-1)
    print(f"tiny: loss {losses[0]:.4f} at step 1, {losses[-1]:.4f} at step {steps}")
    if failures:
        print("\n".join(failures[:20]))
        sys.exit(1)
    print("tiny: PASS")


if __name__ == "__main__":
    main()

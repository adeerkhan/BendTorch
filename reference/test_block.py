"""demo/smol_block.bend against its PyTorch twin (reference/block_ref.py): one decoder block at SmolLM2's sizes.

1. AdamW training for 1 + `steps` steps (default 5) from the same weights on the same inputs: the loss of every step
   within the shared tolerance (oracle.py).
2. The same steps on 1 thread and on 16 threads print the same bits.

    reference/.venv/bin/python reference/test_block.py .scratch/smol_block [steps] [size]
"""

import subprocess
import sys

import numpy as np
import torch

from oracle import close, parse


def bend(binary, threads, steps, optimizer):
    out = subprocess.run([binary, "+RTS", "--threads", str(threads), "-RTS", "train", str(steps), optimizer], check=True, capture_output=True,
                         text=True).stdout
    return [line for line in out.splitlines() if line.startswith("loss ")]


def main():
    binary = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    size = sys.argv[3] if len(sys.argv) > 3 else "135m"
    optimizer = sys.argv[4] if len(sys.argv) > 4 else "adamw"
    failures = []
    one, many = bend(binary, 1, steps, optimizer), bend(binary, 16, steps, optimizer)
    if one != many:
        failures.append("the losses on 1 and 16 threads differ")
    ref = parse(subprocess.run([sys.executable, "reference/block_ref.py", size, "train", str(steps), optimizer], check=True,
                               capture_output=True, text=True).stdout)["loss"]
    got = parse("\n".join(many))["loss"].reshape(-1)
    worst = close("loss", got, torch.from_numpy(np.asarray(ref, dtype=np.float32).reshape(-1).copy()), failures)
    print(f"block {size}: {steps + 1} {optimizer} steps, losses within the tolerance (worst {worst:.2f} of it), the same bits on "
          f"1 and 16 threads" if not failures else "\n".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

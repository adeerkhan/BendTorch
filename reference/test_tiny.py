"""demo/tiny_llama.bend against its PyTorch twin (reference/tiny_ref.py).

1. One forward and backward pass on the first batch: the loss and all 21 parameter gradients, within the shared
   tolerance (oracle.py).
2. AdamW training for `steps` steps (default 20) from the same weights on the same batches: the loss of every step and
   the final weights, against the twin trained in float64 (`train64`): each tensor within the same tolerance of it, or
   with Bend's error against it at most 8x PyTorch float32's. Twenty AdamW steps drift in float32 by an amount that
   depends on the CPU's kernels (PyTorch on GitHub's runner and on the reference machine differ), so float64 is the
   reference (the rule of M10's training losses, Renan, 2026-10-17). The comparison with PyTorch float32 is printed too.
3. Generation: 100 steps of training in Bend, then 64 characters continued greedily from a prompt; PyTorch, given the
   weights Bend trained, must continue the prompt with the same 64 characters.
4. Checkpoints: 30 steps, a checkpoint, 20 more steps in a new process from it; the final weights must be the same bits
   as 50 steps without a stop.

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
    compare(f"train {steps} (against float32, information)", bend, ref, [], skip_first_loss=True)
    out64 = subprocess.run([sys.executable, "reference/tiny_ref.py", "train64", steps], check=True, capture_output=True,
                           text=True).stdout
    ref64 = {}
    for line in out64.splitlines():
        ref64.setdefault(line.split()[0], []).extend(float(w) for w in line.split()[1:])
    worst = 0.0
    for name, r64 in ref64.items():
        b = bend[name].reshape(-1)[1:] if name == "loss" else bend[name].reshape(-1)
        t = np.asarray(ref[name], dtype=np.float32).reshape(-1)
        r = np.array(r64)
        err = np.abs(b.astype(np.float64) - r) / (1e-6 + 1e-5 * np.abs(r))
        worst = max(worst, float(err.max()))
        if err.max() > 1.0:
            eb, et = np.sqrt(np.mean((b - r) ** 2)), np.sqrt(np.mean((t - r) ** 2))
            if not eb <= 8.0 * et:
                failures.append(f"train {steps} {name}: {err.max():.2f}x the tolerance from float64, Bend {eb:.2e}, "
                                f"PyTorch float32 {et:.2e}")
    print(f"tiny train {steps} (against float64): worst element {worst:.3f} of the tolerance")
    prompt = "ROMEO:\nBut soft, wh"
    out = subprocess.run([binary, "sample", "100", "64", prompt], check=True, capture_output=True, text=True).stdout
    ref_ids = subprocess.run([sys.executable, "reference/tiny_ref.py", "generate", prompt, "64"], input=out, check=True,
                             capture_output=True, text=True).stdout.split()[1:]
    bend_ids = next(line for line in out.splitlines() if line.startswith("ids")).split()[1:]
    if bend_ids != ref_ids:
        failures.append(f"generation: bend {' '.join(bend_ids)}, torch from bend's weights {' '.join(ref_ids)}")
    print(f"tiny generation: 64 characters, {sum(a == b for a, b in zip(bend_ids, ref_ids))} the same as PyTorch's from the same weights")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ck = f"{d}/ck.bin"
        subprocess.run([binary, "save", "30", ck], check=True, capture_output=True)
        resumed = subprocess.run([binary, "resume", ck, "30", "20"], check=True, capture_output=True, text=True).stdout
    whole = subprocess.run([binary, "train", "50"], check=True, capture_output=True, text=True).stdout
    finals = [[line for line in out.splitlines() if line.startswith("final.")] for out in (resumed, whole)]
    if not finals[0] or finals[0] != finals[1]:
        failures.append("checkpoint: 30 steps + resume 20 differs from 50 steps")
    print(f"tiny checkpoint: 30 steps, saved, resumed for 20: the final weights {'are' if finals[0] == finals[1] else 'are NOT'} "
          "the same bits as 50 steps without a stop")
    losses = ref["loss"].reshape(-1)
    print(f"tiny: loss {losses[0]:.4f} at step 1, {losses[-1]:.4f} at step {steps}")
    if failures:
        print("\n".join(failures[:20]))
        sys.exit(1)
    print("tiny: PASS")


if __name__ == "__main__":
    main()

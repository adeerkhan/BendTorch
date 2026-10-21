"""demo/tiny_rt.bend (the tiny Llama with B and T read at run time) against its PyTorch twin (reference/tiny_rt_ref.py).

For each size B x T given:
1. One forward and backward pass on the first batch: the loss and all 21 parameter gradients, within the shared
   tolerance (oracle.py).
2. AdamW training for `steps` steps: the losses and the final weights against the twin in float64, with
   reference/test_tiny.py's rule (each tensor within the tolerance of float64, or Bend's error against it at most 2x
   PyTorch float32's).
3. Generation: `steps` steps of training, then 32 characters continued greedily from a window of T characters; PyTorch,
   given the weights Bend trained, must continue it with the same characters.
Then inputs that do not fit are refused with a message and exit code 1: B = 0, an id past the vocabulary, a file that is not a whole number of
batches of B x (T + 1), more steps than the file holds, a prompt shorter than T.

    reference/.venv/bin/python reference/test_tiny_rt.py .scratch/tiny_rt [steps] [BxT ...]
    (default sizes 2x16 4x8 3x24; batch files for other sizes go to .scratch/tiny_rt_data/)
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from oracle import parse
from test_tiny import compare

PY = sys.executable
PROMPT = "ROMEO:\nBut soft, what light through yonder window breaks"


def batches(b, t, steps):
    if (b, t) == (2, 16):
        return "data/tiny/batches.bin"
    path = f".scratch/tiny_rt_data/batches_{b}x{t}.bin"
    subprocess.run([PY, "reference/tiny_rt_ref.py", "--prep", str(b), str(t), str(steps), path], check=True)
    return path


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def size(binary, b, t, steps, failures):
    path = batches(b, t, max(steps, 2))
    sz = [str(b), str(t), path]
    tag = f"{b}x{t}"
    bend = parse(run([binary, *sz, "grads"]))
    ref = parse(run([PY, "reference/tiny_rt_ref.py", *sz, "grads"]))
    compare(f"{tag} grads", bend, ref, failures)
    bend = parse(run([binary, *sz, "train", str(steps)]))
    ref = parse(run([PY, "reference/tiny_rt_ref.py", *sz, "train", str(steps)]))
    compare(f"{tag} train {steps} (against float32, information)", bend, ref, [], skip_first_loss=True)
    ref64 = {}
    for line in run([PY, "reference/tiny_rt_ref.py", *sz, "train64", str(steps)]).splitlines():
        ref64.setdefault(line.split()[0], []).extend(float(w) for w in line.split()[1:])
    worst = 0.0
    for name, r64 in ref64.items():
        bb = bend[name].reshape(-1)[1:] if name == "loss" else bend[name].reshape(-1)
        tt = np.asarray(ref[name], dtype=np.float32).reshape(-1)
        r = np.array(r64)
        err = np.abs(bb.astype(np.float64) - r) / (1e-6 + 1e-5 * np.abs(r))
        worst = max(worst, float(err.max()))
        if err.max() > 1.0:
            eb, et = np.sqrt(np.mean((bb - r) ** 2)), np.sqrt(np.mean((tt - r) ** 2))
            if not eb <= 2.0 * et:
                failures.append(f"{tag} train {steps} {name}: {err.max():.2f}x the tolerance from float64, Bend {eb:.2e}, "
                                f"PyTorch float32 {et:.2e}")
    print(f"tiny_rt {tag} train {steps} (against float64): worst element {worst:.3f} of the tolerance")
    out = run([binary, *sz, "sample", str(steps), "32", PROMPT])
    ref_ids = run([PY, "reference/tiny_rt_ref.py", *sz, "generate", PROMPT, "32"], input=out).split()[1:]
    bend_ids = next(line for line in out.splitlines() if line.startswith("ids")).split()[1:]
    if bend_ids != ref_ids:
        failures.append(f"{tag} generation: bend {' '.join(bend_ids)}, torch from bend's weights {' '.join(ref_ids)}")
    print(f"tiny_rt {tag} generation: 32 characters from a window of {t}, "
          f"{sum(a == c for a, c in zip(bend_ids, ref_ids))} the same as PyTorch's from the same weights")
    losses = ref["loss"].reshape(-1)
    print(f"tiny_rt {tag}: loss {losses[0]:.4f} at step 1, {losses[-1]:.4f} at step {steps}")


def refused(binary, failures):
    """Inputs that do not fit the sizes: each must stop with exit code 1 and a message saying why."""
    bad = Path(".scratch/tiny_rt_data/bad_ids.bin")
    bad.write_bytes(bytes([1] * 33 + [65]))  # one batch of 2 x 17 ids, the last one past the vocabulary
    cases = [
        (["2", "16", str(bad), "grads"], "holds an id past the vocabulary of 65"),
        (["0", "16", "data/tiny/batches.bin", "grads"], "B must be at least 1"),
        (["4", "8", "data/tiny/batches.bin", "grads"], "not a whole number of steps of B x (T + 1) = 36"),
        (["2", "16", "data/tiny/batches.bin", "train", "2001"], "fewer than the 2001 steps asked for"),
        (["2", "16", "data/tiny/batches.bin", "sample", "1", "4", "too short"], "fewer than T = 16"),
        (["300", "300", "data/tiny/batches.bin", "grads"], "at most 65536 rows"),
    ]
    for args, want in cases:
        r = subprocess.run([binary, *args], capture_output=True, text=True)
        said = r.stdout + r.stderr
        if r.returncode != 1 or want not in said:
            failures.append(f"refusal {' '.join(args)}: exit {r.returncode}, wanted 1 and '{want}', got {said.strip()[-200:]!r}")
        else:
            print(f"tiny_rt refused as it must: {' '.join(args)} -> {want}")


def main():
    binary = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    sizes = [tuple(int(x) for x in s.split("x")) for s in sys.argv[3:]] or [(2, 16), (4, 8), (3, 24)]
    Path(".scratch/tiny_rt_data").mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for b, t in sizes:
        size(binary, b, t, steps, failures)
    refused(binary, failures)
    if failures:
        print("\n".join(failures[:20]))
        sys.exit(1)
    print("tiny_rt: PASS")


if __name__ == "__main__":
    main()

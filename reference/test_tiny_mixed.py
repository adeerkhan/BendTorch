"""demo/tiny_mixed.bend against its twin (reference/tiny_mixed_ref.py), and BF16 storage against the F32 runs.

1. int8 inference: the tiny Llama's logits on the first batch with its projections and head in int8 by rows, against
   the twin's within the shared tolerance (oracle.close); the int8 logits' error against the F32 logits is reported
   for Bend and for PyTorch.
2. Mixed precision (BF16 storage, F32 arithmetic and master weights): `steps` AdamW steps (default 20) in Bend and in
   the twin, and the F32 runs of demo/tiny_llama.bend and tiny_ref.py from the same weights on the same batches. The
   loss curves are compared and reported, and judged by the criterion below (RATIO, WINDOW; Renan's decision,
   2026-10-18): `make check` runs 20 steps, the long target 2000.

    reference/.venv/bin/python reference/test_tiny_mixed.py .scratch/tiny_mixed .scratch/tiny_llama [steps]
"""

import subprocess
import sys

import numpy as np
import torch

from oracle import close, parse

# The criterion for a BF16 run (adopted by Renan, 2026-10-18), with the twin's BF16 storage in float64 arithmetic as the
# reference (train64), as M10's rule for float32 against float64:
#   - the first 20 losses: Bend's largest and rms deviation from the reference at most RATIO times the twin's;
#   - over a long run, every mean of 100 consecutive losses within WINDOW of the reference's.
RATIO = 8.0
WINDOW = 0.05


def run(*cmd):
    return parse(subprocess.run(list(cmd), check=True, capture_output=True, text=True).stdout)


def losses(out, skip_first):
    l = np.asarray(out["loss"], dtype=np.float64).reshape(-1)
    return l[1:] if skip_first else l


def stats(name, a, b):
    d = np.abs(a - b)
    rel = d / np.abs(b)
    print(f"  {name}: max |diff| {d.max():.3e} (relative {rel.max():.3e}, at step {int(d.argmax())}), "
          f"rms {np.sqrt((d ** 2).mean()):.3e}, mean signed {np.mean(a - b):+.3e}; "
          f"{int((a == b).sum())} of {a.size} equal")
    return rel


def losses64(*cmd):
    out = subprocess.run(list(cmd), check=True, capture_output=True, text=True).stdout
    return np.array([float(l.split()[1]) for l in out.splitlines() if l.startswith("loss ")])


def against64(name, a, ref, b, ref_b, cuts):
    """a's deviation from its float64 reference, next to b's (the twin's) from the same reference, over each prefix."""
    for n in cuts:
        if n > a.size:
            continue
        da, db = np.abs(a[:n] - ref[:n]), np.abs(b[:n] - ref_b[:n])
        print(f"  {name}, first {n} steps: max {da.max():.3e} vs twin {db.max():.3e} ({da.max() / db.max():.2f}x), "
              f"rms {np.sqrt((da ** 2).mean()):.3e} vs {np.sqrt((db ** 2).mean()):.3e} "
              f"({np.sqrt((da ** 2).mean()) / np.sqrt((db ** 2).mean()):.2f}x)")


def window_means(name, a, w=100):
    k = a.size // w
    if k:
        print(f"  {name}, mean of each {w} steps: " + " ".join(f"{x:.4f}" for x in a[:k * w].reshape(k, w).mean(1)))


def main():
    mixed, f32 = sys.argv[1], sys.argv[2]
    steps = sys.argv[3] if len(sys.argv) > 3 else "20"
    py = sys.executable
    failures: list[str] = []

    b = run(mixed, "int8")
    r = run(py, "reference/tiny_mixed_ref.py", "int8")
    worst = close("int8 logits", b["logits_int8"], torch.from_numpy(r["logits_int8"].copy()), failures)
    close("f32 logits", b["logits_f32"], torch.from_numpy(r["logits_f32"].copy()), failures)
    print(f"int8 logits (32 x 65) against the twin: worst {worst:.3f} of the tolerance")
    for side, o in [("Bend", b), ("PyTorch", r)]:
        lf = o["logits_f32"].astype(np.float64).reshape(32, 65)
        lq = o["logits_int8"].astype(np.float64).reshape(32, 65)
        d = np.abs(lq - lf)
        print(f"  {side} int8 against F32: max |diff| {d.max():.3e}, rms {np.sqrt((d ** 2).mean()):.3e}, "
              f"logits' rms {np.sqrt((lf ** 2).mean()):.3e}; argmax equal in {int((lq.argmax(1) == lf.argmax(1)).sum())} of 32 rows")

    bm = losses(run(mixed, "train", steps), True)
    rm = losses(run(py, "reference/tiny_mixed_ref.py", "train", steps), False)
    bf = losses(run(f32, "train", steps), True)
    rf = losses(run(py, "reference/tiny_ref.py", "train", steps), False)
    m64 = losses64(py, "reference/tiny_mixed_ref.py", "train64", steps)
    f64 = losses64(py, "reference/tiny_ref.py", "train64", steps)
    print(f"mixed precision, {steps} AdamW steps (losses before each update):")
    stats("Bend BF16 against the twin's BF16", bm, rm)
    stats("Bend F32 against PyTorch F32", bf, rf)
    stats("Bend BF16 against Bend F32", bm, bf)
    stats("PyTorch BF16 against PyTorch F32", rm, rf)
    cuts = [20, 100, 400, 2000]
    against64("BF16, against BF16 storage with float64 arithmetic", bm, m64, rm, m64, cuts)
    against64("F32, against float64", bf, f64, rf, f64, cuts)
    stats("BF16 storage with float64 arithmetic against float64", m64, f64)
    for name, a in [("Bend BF16", bm), ("twin BF16", rm), ("Bend F32", bf), ("PyTorch F32", rf), ("BF16 storage, float64", m64), ("float64", f64)]:
        window_means(name, a)
    n = min(20, bm.size)
    da, db = np.abs(bm[:n] - m64[:n]), np.abs(rm[:n] - m64[:n])
    r1, r2 = da.max() / db.max(), np.sqrt((da ** 2).mean()) / np.sqrt((db ** 2).mean())
    print(f"  first {n} steps: Bend's deviation from the reference {r1:.2f}x (max) and {r2:.2f}x (rms) the twin's "
          f"(at most {RATIO:g}x)")
    if max(r1, r2) > RATIO:
        failures.append(f"BF16 first {n} steps: Bend's deviation {max(r1, r2):.2f}x the twin's, over {RATIO:g}x")
    k = bm.size // 100
    if k:
        w = np.abs(bm[:k * 100].reshape(k, 100).mean(1) - m64[:k * 100].reshape(k, 100).mean(1)).max()
        wt = np.abs(rm[:k * 100].reshape(k, 100).mean(1) - m64[:k * 100].reshape(k, 100).mean(1)).max()
        print(f"  100-step means: Bend at most {w:.4f} from the reference (the twin {wt:.4f}; at most {WINDOW:g})")
        if w > WINDOW:
            failures.append(f"BF16 100-step means: Bend {w:.4f} from the reference, over {WINDOW:g}")
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("tiny mixed: int8 logits within the tolerance of the twin; the BF16 run within the adopted criterion")


if __name__ == "__main__":
    main()

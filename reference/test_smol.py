"""demo/smol.bend against its PyTorch twin (reference/smol_ref.py): SmolLM2-135M on data/smol135/ (scripts/smol_data.sh),
or SmolLM2-360M (scripts/smol.sh 360m, data/smol360/) with a second argument 360m.

1. The loss on the first batch, within the shared tolerance (oracle.py).
2. AdamW training for 3 steps from the checkpoint on the same batches. The first loss (before any update) is within the
   same tolerance. AdamW's first update moves each weight by about lr times the sign of its gradient, so the rounding
   of a near-zero gradient flips a step and later losses drift in float32 (PyTorch's too); each later loss passes if it
   is within the tolerance of a float64 run of the twin, or if Bend's error against that run is at most 8x PyTorch
   float32's (the gradients' criterion, Renan, 2026-10-17). Its comparison with PyTorch float32 is still printed.
3. If the binary has a grads mode: every gradient of the first batch, against float64 gradients of the twin. The shared
   tolerance is out of float32's reach for a model this deep (PyTorch's own float32 gradients meet it in 17 of the
   135M's 272 tensors), so the criterion (Renan, 2026-10-17) is relative: per tensor, Bend's rms error against float64
   at most 1x PyTorch float32's, and at most 0.25x in the median over the tensors (Renan, 2026-10-18, after the linear
   layers' dX moved to the precise product: measured 0.10x and 0.03x; it was 8x and 2x, with Bend at 1.8x). The element-wise comparison with
   PyTorch float32 under the shared tolerance is still printed, for information.
4. The loss and the 3 training steps on 1 thread and on 16 threads print the same bits.
5. If a third argument names it: the same program built with plain SGD at lr SGD_LR (make smol writes it from
   demo/smol.bend with sed), 3 steps against the twin's SGD (reference/smol_sgd_ref.py): every loss within the shared
   tolerance of PyTorch float32, the strict rule. SGD moves each weight by lr times its gradient, so the gradients'
   rounding reaches the loss scaled by lr, with no sign to flip as in AdamW's first step (NOTES 2026-10-18). At lr
   0.003 the update moved the later losses by 157-900 tolerances (135M and 360M); 0.01 moves them further and passes
   too (worst 0.52 of the tolerance at 135M), now that the cross-entropy is log-softmax: under its old log(max(p,
   1e-12)) a token's probability below 1e-12 put the loss thousands of tolerances off at 0.005 and 0.01.

    reference/.venv/bin/python reference/test_smol.py .scratch/smol [135m | 360m] [.scratch/smol_sgd]
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import statistics
import torch

from oracle import bits_equal, close, parse

TWIN = [sys.executable, "reference/smol_ref.py"]
SMOL = sys.argv[2] if len(sys.argv) > 2 else "135m"
LAYOUT = Path("reference/smol_layout.txt" if SMOL == "135m" else "reference/smol360_layout.txt")


def bend(binary, threads, *args):
    out = subprocess.run([binary, "--threads", str(threads), *args], check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.startswith("loss ")]


RATIO_MAX, RATIO_MEDIAN = 1.0, 0.25  # Bend's error against float64, as a multiple of PyTorch float32's (docstring, 3)
SGD_LR = "0.01"  # the SGD binary's learning rate (docstring, 5); make smol builds it with the same number


def twin(*args):
    return parse(subprocess.run([*TWIN, *args], check=True, capture_output=True, text=True,
                                env=dict(os.environ, SMOL=SMOL)).stdout)["loss"].reshape(-1)


def compare(stage, lines, ref, failures):
    got = parse("\n".join(lines))["loss"].reshape(-1)
    rt = torch.from_numpy(np.asarray(ref, dtype=np.float32).copy())
    worst = close(f"{stage} loss", got, rt, failures)
    same, total = bits_equal(got, rt)
    print(f"smol {stage}: {total} losses, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance "
          f"(bend {' '.join(f'{x:.7f}' for x in got)}; torch {' '.join(f'{x:.7f}' for x in rt.tolist())})")


def grads(binary, failures):
    Path(".scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=".scratch") as d:  # on disk: the 360M's three files take 5.8 GB, too much for a tmpfs
        b, r = Path(d) / "bend.bin", Path(d) / "torch.bin"
        run = subprocess.run([binary, "--threads", "16", "grads", str(b)], capture_output=True, text=True)
        if run.returncode != 0 and "usage" in run.stderr + run.stdout:
            print("smol grads: the binary has no grads mode yet; skipped")
            return
        run.check_returncode()
        compare("grads", [line for line in run.stdout.splitlines() if line.startswith("loss ")], twin("grads", str(r)),
                failures)
        if b.stat().st_size != r.stat().st_size:
            failures.append(f"grads: {b.stat().st_size} bytes from Bend, {r.stat().st_size} from torch")
            return
        f64 = Path(d) / "torch64.bin"
        subprocess.run([*TWIN, "grads64", str(f64)], check=True, capture_output=True, text=True, env=dict(os.environ, SMOL=SMOL))
        worst, at, same, total, ratios, info = 0.0, "", 0, 0, [], []
        lay = [line.split("\t") for line in LAYOUT.read_text().splitlines()
               if line and not line.startswith("#")]
        for name, shape, off in lay:  # a tensor at a time, to keep the float64 copies small
            n = int(np.prod([int(s) for s in shape.split("x")]))
            gb = np.fromfile(b, dtype="<f4", count=n, offset=int(off) * 4)
            gr = torch.from_numpy(np.fromfile(r, dtype="<f4", count=n, offset=int(off) * 4))
            w = close(f"grad {name}", gb, gr, info)
            g64 = np.fromfile(f64, dtype="<f8", count=n, offset=int(off) * 8)
            scale = np.sqrt(np.mean(g64 ** 2)) or 1.0
            eb = np.sqrt(np.mean((gb.astype(np.float64) - g64) ** 2)) / scale
            et = np.sqrt(np.mean((gr.numpy().astype(np.float64) - g64) ** 2)) / scale
            ratios.append((eb / et if et > 0 else (1.0 if eb == 0 else float("inf")), name))
            s, t = bits_equal(gb, gr)
            same, total = same + s, total + t
            if w > worst:
                worst, at = w, name
        print(f"smol grads (information): {total} numbers, {same} bit-identical to torch float32, worst error {worst:.3f} of "
              f"the shared tolerance (in {at}); {sum(1 for x in info if x.startswith('grad '))} tensors outside it")
        top, top_at = max(ratios)
        med = statistics.median(r for r, _ in ratios)
        print(f"smol grads: error against float64, Bend / PyTorch float32, per tensor: median {med:.2f} (at most "
              f"{RATIO_MEDIAN}), worst {top:.2f} in {top_at} (at most {RATIO_MAX})")
        if top > RATIO_MAX or med > RATIO_MEDIAN:
            failures.append(f"grads: Bend's error against float64 is {top:.2f}x PyTorch float32's at worst, {med:.2f}x in the median")


def sgd(binary, failures):
    """docstring, 5: 3 SGD steps, every loss within the shared tolerance of PyTorch float32"""
    got = bend(binary, 16, "train", "3")
    ref = parse(subprocess.run([sys.executable, "reference/smol_sgd_ref.py", "train", "3", SGD_LR], check=True,
                               capture_output=True, text=True, env=dict(os.environ, SMOL=SMOL)).stdout)["loss"].reshape(-1)
    compare(f"sgd 3 (lr {SGD_LR})", got, ref, failures)


def main():
    binary = sys.argv[1]
    failures: list[str] = []
    loss16 = bend(binary, 16, "loss")
    compare("loss", loss16, twin("loss"), failures)
    train16 = bend(binary, 16, "train", "3")
    ref32 = twin("train", "3")
    compare("train 3", train16, ref32, [])
    ref64 = [float(line.split()[1]) for line in subprocess.run([*TWIN, "train64", "3"], check=True, capture_output=True,
             text=True, env=dict(os.environ, SMOL=SMOL)).stdout.splitlines() if line.startswith("loss64 ")]
    got = parse("\n".join(train16))["loss"].reshape(-1).astype(np.float64)
    close("train 3 first loss", got[1:2], torch.tensor(ref32[1:2]), failures)
    for k in range(2, len(got)):
        eb, et = abs(got[k] - ref64[k - 1]), abs(float(ref32[k]) - ref64[k - 1])
        ok = eb <= 1e-6 + 1e-5 * abs(ref64[k - 1]) or eb <= RATIO_MAX * et
        print(f"smol train 3, loss {k}: against float64, Bend {eb:.2e}, PyTorch float32 {et:.2e}"
              f" ({eb / et if et else float('inf'):.2f}x, at most {RATIO_MAX}x or within the tolerance): {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"train 3 loss {k}: Bend {eb:.2e} from float64, PyTorch float32 {et:.2e}")
    grads(binary, failures)
    if len(sys.argv) > 3:
        sgd(sys.argv[3], failures)
    loss1, train1 = bend(binary, 1, "loss"), bend(binary, 1, "train", "3")
    if (loss1, train1) != (loss16, train16):
        failures.append(f"1 thread {loss1 + train1} differs from 16 threads {loss16 + train16}")
    print(f"smol threads: the loss and 3 training steps on 1 thread {'are' if (loss1, train1) == (loss16, train16) else 'are NOT'} "
          "the same bits as on 16")
    if failures:
        print("\n".join(failures[:20]))
        sys.exit(1)
    print("smol: PASS")


if __name__ == "__main__":
    main()

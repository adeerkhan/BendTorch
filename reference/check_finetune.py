"""demo/smol.bend's finetune against its twin: the held-out loss must fall, and Bend's must track float64.

    reference/.venv/bin/python reference/check_finetune.py <bend.txt> <torch32.txt> <torch64.txt>

The three files are the outputs of `.scratch/smol finetune k e`, `reference/smol_ref.py finetune k e` and
`... finetune64 k e` on the same weights and batches. Each held-out evaluation passes by the rule of M10's training
losses (Renan, 2026-10-17): within the shared tolerance of float64's, or Bend's error against it at most 8x PyTorch
float32's. The first evaluation, before any step, must be within the shared tolerance of PyTorch float32's. Then the
held-out loss after the last step must be below the one before the first.
"""

import sys

import numpy as np

ATOL, RTOL, RATIO = 1e-6, 1e-5, 8.0


def f32_lines(path, name):
    return [float(np.uint32(int(line.split()[1])).view(np.float32)) for line in open(path) if line.startswith(name + " ")]


def f64_lines(path, name):
    return [float(line.split()[1]) for line in open(path) if line.startswith(name + " ")]


def main():
    bend, t32, t64 = f32_lines(sys.argv[1], "eval"), f32_lines(sys.argv[2], "eval"), f64_lines(sys.argv[3], "eval")
    assert len(bend) == len(t32) == len(t64) > 1, (len(bend), len(t32), len(t64))
    failures = []
    if abs(bend[0] - t32[0]) > ATOL + RTOL * abs(t32[0]):
        failures.append(f"eval 0: Bend {bend[0]:.7f}, PyTorch float32 {t32[0]:.7f}")
    for i, (b, t, r) in enumerate(zip(bend, t32, t64)):
        eb, et = abs(b - r), abs(t - r)
        ok = eb <= ATOL + RTOL * abs(r) or eb <= RATIO * et
        print(f"eval {i}: Bend {b:.6f}  PyTorch float32 {t:.6f}  float64 {r:.6f}  |Bend - f64| {eb:.1e}  "
              f"|f32 - f64| {et:.1e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"eval {i}: Bend {eb:.2e} from float64, PyTorch float32 {et:.2e}")
    print(f"held-out loss: {bend[0]:.4f} before, {bend[-1]:.4f} after (float64 {t64[0]:.4f} -> {t64[-1]:.4f})")
    if not bend[-1] < bend[0]:
        failures.append("the held-out loss did not fall")
    print("\n".join(failures) if failures else "finetune: PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

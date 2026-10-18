"""demo/smol.bend's finetune against its twin: the held-out loss must fall, and Bend's must track float64.

    reference/.venv/bin/python reference/check_finetune.py <bend.txt> <torch32.txt> <torch64.txt>

The three files are the outputs of `.scratch/smol finetune k e`, `reference/smol_ref.py finetune k e` and
`... finetune64 k e` on the same weights and batches. Over hundreds of steps a float32 run's trajectory drifts from
float64's by an amount that changes from one evaluation to the next (PyTorch's too), so the run is judged as a whole
(Renan, 2026-10-18): over all evaluations, Bend's largest and rms deviation from float64 at most 8x PyTorch float32's.
The first evaluation, before any step, must be within the shared tolerance of PyTorch float32's, and the held-out loss
after the last step must be below the one before the first. The per-evaluation table is printed for information.
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
    db, dt = [], []
    for i, (b, t, r) in enumerate(zip(bend, t32, t64)):
        db.append(abs(b - r)); dt.append(abs(t - r))
        print(f"eval {i}: Bend {b:.6f}  PyTorch float32 {t:.6f}  float64 {r:.6f}  |Bend - f64| {db[-1]:.1e}  "
              f"|f32 - f64| {dt[-1]:.1e}")
    rb, rt = float(np.sqrt(np.mean(np.square(db)))), float(np.sqrt(np.mean(np.square(dt))))
    print(f"over the run, deviation from float64: largest Bend {max(db):.2e}, PyTorch float32 {max(dt):.2e} "
          f"({max(db) / max(dt):.2f}x); rms Bend {rb:.2e}, PyTorch float32 {rt:.2e} ({rb / rt:.2f}x); at most {RATIO}x")
    if max(db) > RATIO * max(dt) or rb > RATIO * rt:
        failures.append(f"Bend's deviation from float64 over the run is more than {RATIO}x PyTorch float32's")
    print(f"held-out loss: {bend[0]:.4f} before, {bend[-1]:.4f} after (float64 {t64[0]:.4f} -> {t64[-1]:.4f})")
    if not bend[-1] < bend[0]:
        failures.append("the held-out loss did not fall")
    print("\n".join(failures) if failures else "finetune: PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

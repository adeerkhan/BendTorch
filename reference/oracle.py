"""Shared helpers for the PyTorch oracles: writing inputs for the Bend programs and reading what they print.

A Bend test program prints one line per tensor: a name, then each F32 as the decimal of its 32 bits. Comparing bits
keeps the test exact on the Bend side; the tolerance lives only here, in `close`, and is the same for every test.
"""

import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

# The comparison policy (docs/design.md, "Numerical validation"): |bend - torch| <= ATOL + RTOL * |torch|, element by
# element, in float64. Both are fixed for every test; a failure is investigated, never answered by widening them.
ATOL = 1e-6
RTOL = 1e-5


def write_f32(path: Path, t: torch.Tensor) -> None:
    path.write_bytes(t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes())


def write_u8(path: Path, xs) -> None:
    path.write_bytes(bytes(int(x) for x in xs))


def run(binary: str, *args: str) -> dict[str, np.ndarray]:
    out = subprocess.run([binary, *args], check=True, capture_output=True, text=True).stdout
    return parse(out)


def parse(out: str) -> dict[str, np.ndarray]:
    got = {}
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        words = np.array([int(w) for w in parts[1:]], dtype=np.uint32)
        got.setdefault(parts[0], []).append(words.view(np.float32))
    return {k: (v[0] if len(v) == 1 else np.stack(v)) for k, v in got.items()}


def close(name: str, bend: np.ndarray, ref: torch.Tensor, failures: list[str], atol=ATOL, rtol=RTOL) -> float:
    """Checks one tensor; returns the worst error relative to the tolerance (<= 1 passes)."""
    r = ref.detach().to(torch.float64).reshape(-1).numpy()
    b = np.asarray(bend, dtype=np.float64).reshape(-1)
    if b.shape != r.shape:
        failures.append(f"{name}: {b.size} numbers, expected {r.size}")
        return float("inf")
    same = (np.isnan(b) & np.isnan(r)) | (b == r)  # NaN against NaN, and equal values (infinities included)
    with np.errstate(invalid="ignore"):
        err = np.where(same, 0.0, np.abs(b - r) / (atol + rtol * np.abs(r)))
    err = np.where(np.isnan(err), np.inf, err)
    worst = float(err.max()) if err.size else 0.0
    if worst > 1.0:
        i = int(err.argmax())
        failures.append(f"{name}[{i}]: bend {b[i]!r}, torch {r[i]!r} ({worst:.2f}x the tolerance)")
    return worst


def bits_equal(bend: np.ndarray, ref: torch.Tensor) -> tuple[int, int]:
    r = ref.detach().to(torch.float32).reshape(-1).numpy().view(np.uint32)
    b = np.asarray(bend, dtype=np.float32).reshape(-1).view(np.uint32)
    return int((r == b).sum()), int(r.size)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]

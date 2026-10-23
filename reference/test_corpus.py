"""demo/corpus_test.bend against the frozen corpus of bench/corpus/ (milestone 14), at release time.

The corpus holds inputs drawn from a seed no development test uses and PyTorch's results on them, in float32 and in
float64, with each float64 result's scale (the sum of |terms| that produce it), all computed once by
reference/corpus_gen.py. Nothing is recomputed here. Each number the Bend program prints passes under the manifest's
policy: the bits of PyTorch's float32 result; or the same non-finite class as it; or
|bend - out64| <= ATOL + RTOL * max(scale, |out64|), with oracle.py's ATOL and RTOL. The exact tier must have the
float32 result's bits. The test refuses to run if a file's sha256 differs from the manifest's, if a file is missing or
unlisted, or if oracle.py's ATOL/RTOL are not the manifest's (a new policy is a new corpus version, made on purpose).
For information it also counts what oracle.py's own policy (against float32, scaled by |result|) would fail.

    reference/.venv/bin/python reference/test_corpus.py .scratch/corpus_test [bench/corpus] [--threads N]
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import oracle
from oracle import parse


def refuse(msg: str) -> None:
    print(f"corpus: REFUSED: {msg}")
    sys.exit(2)


def verified(d: Path) -> dict:
    manifest = json.loads((d / "manifest.json").read_text())
    if (manifest["atol"], manifest["rtol"]) != (oracle.ATOL, oracle.RTOL):
        refuse(f"the corpus was made for ATOL={manifest['atol']}, RTOL={manifest['rtol']}; "
               f"oracle.py now has ATOL={oracle.ATOL}, RTOL={oracle.RTOL}")
    listed = set(manifest["files"])
    present = {p.name for p in d.glob("*.npz")}
    if listed != present:
        refuse(f"files missing: {sorted(listed - present)}, files not in the manifest: {sorted(present - listed)}")
    for name, want in manifest["files"].items():
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        if got != want:
            refuse(f"{name}: sha256 {got}, the manifest says {want}")
    return manifest


def judge(b32: np.ndarray, r32: np.ndarray, r64: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The error of each number relative to what the policy allows (<= 1 passes)."""
    b, t = b32.astype(np.float64), r32.astype(np.float64)
    same = b32.view(np.uint32) == r32.view(np.uint32)
    special = ~np.isfinite(t) | ~np.isfinite(r64)
    with np.errstate(invalid="ignore", over="ignore"):
        cls = (np.isnan(b) & np.isnan(t)) | (b == t)
        err = np.abs(b - r64) / (oracle.ATOL + oracle.RTOL * np.maximum(s, np.abs(r64)))
    err = np.where(np.isnan(err), np.inf, err)
    return np.where(same, 0.0, np.where(special, np.where(cls, 0.0, np.inf), err))


def oracle1(b32: np.ndarray, r32: np.ndarray) -> int:
    """How many numbers oracle.py's own policy (|bend - float32| <= ATOL + RTOL*|float32|) would fail."""
    b, t = b32.astype(np.float64), r32.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        ok = (np.isnan(b) & np.isnan(t)) | (b == t) | (np.abs(b - t) <= oracle.ATOL + oracle.RTOL * np.abs(t))
    return int((~ok).sum())


def main():
    args = sys.argv[1:]
    threads = []
    if "--threads" in args:
        k = args.index("--threads")
        threads = ["+RTS", "--threads", args[k + 1], "-RTS"]
        del args[k:k + 2]
    binary = args[0]
    d = Path(args[1] if len(args) > 1 else Path(__file__).resolve().parent.parent / "bench" / "corpus")
    manifest = verified(d)
    failures: list[str] = []
    worst, same, total, exact_n, runs, o1, tops = 0.0, 0, 0, 0, 0, [], []
    digest = hashlib.sha256()  # of everything the program printed, run by run: the same at every thread count
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name in sorted(manifest["files"]):
            z = np.load(d / name)
            for f in tmp.iterdir():
                f.unlink()
            for k in z.files:
                if k.startswith("in/"):
                    (tmp / f"{k[3:]}.bin").write_bytes(z[k].tobytes())
            variants = [(name[: -len(".npz")], z["case"])]
            if "case_fast" in z.files:
                variants.append(("fast_" + name[: -len(".npz")], z["case_fast"]))
            for case, words in variants:
                runs += 1
                (tmp / "case.bin").write_bytes(words.astype("<u4").tobytes())
                p = subprocess.run([binary, *threads, str(tmp)], capture_output=True, text=True)
                if p.returncode != 0:
                    failures.append(f"{case}: exit {p.returncode}: {p.stderr.strip()[:200]}")
                    continue
                digest.update(f"{case}\n{p.stdout}".encode())
                if "reject" in z.files:
                    if p.stdout.strip() != "reject":
                        failures.append(f"{case}: must be rejected, printed {p.stdout.strip()[:60]!r}")
                    continue
                got = parse(p.stdout)
                exact = set(manifest["exact"].get(case, []))
                for k in z.files:
                    if not k.startswith("out32/"):
                        continue
                    out = k[len("out32/"):]
                    r32 = z[k].reshape(-1)
                    if out not in got:
                        failures.append(f"{case}: the Bend program printed no {out}")
                        continue
                    b32 = np.asarray(got[out], dtype=np.float32).reshape(-1)
                    if b32.shape != r32.shape:
                        failures.append(f"{case} {out}: {b32.size} numbers, expected {r32.size}")
                        continue
                    r64, s = z[f"out64/{out}"].reshape(-1), z[f"scale/{out}"].reshape(-1)
                    err = judge(b32, r32, r64, s)
                    worst = max(worst, float(err.max()) if err.size else 0.0)
                    if err.size:
                        tops.append((float(err.max()), f"{case} {out}"))
                    bad = int((err > 1).sum())
                    if bad:
                        i = int(err.argmax())
                        failures.append(f"{case} {out}: {bad} of {r32.size} numbers past the policy, worst [{i}]: bend "
                                        f"{float(b32[i])!r}, float64 {float(r64[i])!r}, float32 {float(r32[i])!r}, "
                                        f"scale {float(s[i]):.3g} ({float(err[i]):.2f}x)")
                    n1 = oracle1(b32, r32)
                    if n1:
                        o1.append(f"{case} {out}: {n1}")
                    eq = int((b32.view(np.uint32) == r32.view(np.uint32)).sum())
                    same, total = same + eq, total + r32.size
                    if out in exact:
                        exact_n += r32.size
                        if eq != r32.size:
                            failures.append(f"{case} {out}: {r32.size - eq} of {r32.size} numbers not bit-identical (exact tier)")
    print(f"corpus {manifest['version']}{' on ' + threads[1] + ' threads' if threads else ''} (seed {manifest['seed']}, "
          f"torch {manifest['made_with']['torch']}): {manifest['cases']} files, {runs} runs, {total} numbers, {same} "
          f"bit-identical to torch float32 ({exact_n} in the exact tier), worst error {worst:.3f} of the policy")
    print(f"corpus: oracle.py's own policy against float32 would fail {len(o1)} tensors"
          + (f" ({', '.join(o1)})" if o1 else ""))
    print("corpus: the closest to the limit: " + ", ".join(f"{n} {e:.3f}" for e, n in sorted(tops, reverse=True)[:6]))
    for f in failures[:40]:
        print("FAIL", f)
    print(f"corpus: the Bend outputs' sha256 {digest.hexdigest()[:16]}")
    print("corpus: PASS" if not failures else f"corpus: {len(failures)} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

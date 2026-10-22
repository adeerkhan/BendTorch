"""M9: lean checkpoints, exact resume with tied weights/Dropout, array offsets and malformed files; the optimizer
configuration check (OptCfg.ok, OptCfg.check)."""

import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def check_cfg(run):
    """OptCfg.ok accepts c = 1 - b for every beta and rejects typos; OptCfg.check stops; unchecked, the weights go NaN."""
    rng = random.Random(1018)
    good, bad = [], []
    betas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999, 0.999999]
    betas += [rng.randrange(1, 10 ** d) / 10 ** d for d in range(1, 7) for _ in range(150)]
    for b in betas:
        c = 1.0 - b  # PyTorch's complement, in float64, rounded to F32 by struct
        cf = float(np.float32(1.0) - np.float32(b))  # the complement taken in F32, as before 2026-10-18
        good += [(0.9, b, 0.1, c), (b, 0.999, c, 0.001), (b, 0.999, cf, 0.001)]
        if c > 0:
            bad += [(b, 0.999, c * 10, 0.001), (b, 0.999, c / 10, 0.001), (b, 0.999, -c, 0.001)]
    bad += [(0.9, 0.999, 0.001, 0.1), (0.9, 0.999, 0.1, 0.01), (0.9, 0.999, 0.1, 0.0), (0.9, 0.999, 0.1, float("nan")),
            (0.9, 1.0, 0.1, 0.0), (0.9, 1.5, 0.1, -0.5), (0.9, 0.999, 0.1, 0.00099), (0.9, 0.999, 0.1, 0.0011)]
    cases = good + bad
    words = run("ok", *[f32_bits(x) for case in cases for x in case]).split()[1:]
    assert len(words) == len(cases), (len(words), len(cases))
    wrong = [(case, w) for case, w in zip(cases, words) if w != ("1" if case in good else "0")]
    assert not wrong, wrong[:10]
    p = subprocess.run([run.binary, "run", "adamw-bad", "2", "unused"], capture_output=True, text=True)
    assert p.returncode == 1 and "c1, c2 do not match b1, b2" in p.stdout + p.stderr, (p.returncode, p.stdout, p.stderr)
    state = run("unchecked", "adamw-bad", 2, "unused").split()[1:]
    assert all(np.isnan(np.uint32(int(w)).view(np.float32)) for w in state[:9]), state
    assert run("unchecked", "adamw", 6, "unused") == run("run", "adamw", 6, "unused")
    print(f"infra optimizer configuration: {len(good)} consistent accepted, {len(bad)} inconsistent rejected; a bad one "
          f"stops with a message, and unchecked its weights are NaN after a step")


def main():
    binary = sys.argv[1]

    def run(*args, ok=True):
        p = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
        assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout, p.stderr)
        return p.stdout

    run.binary = binary
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for opt, count in [("sgd", 10), ("momentum", 20), ("adam", 32), ("adamw", 32)]:
            ck = tmp / f"{opt}.bin"
            saved = run("save", opt, 2, ck)
            assert ck.stat().st_size == 4 * count, (opt, ck.stat().st_size)
            assert saved == run("resume", opt, 0, ck), opt
            assert run("run", opt, 6, "unused") == run("resume", opt, 4, ck), opt
            words = saved.split()[1:]
            assert int(words[9]) == 2, (opt, "Dropout counter was not saved")
            # Tie stores the shared 3x3 matrix once, followed by Dropout's bitcast counter.
            assert list(struct.unpack(f"<{count}I", ck.read_bytes())) == list(map(int, words))
            for data in [ck.read_bytes()[:-1], ck.read_bytes() + b"\0", ck.read_bytes()[:-4], b""]:
                bad = tmp / "bad.bin"
                bad.write_bytes(data)
                run("resume", opt, 1, bad, ok=False)
            print(f"infra {opt}: {4 * count} bytes, same bits after resume, tied weights and Dropout preserved")
            # Params.save writes the parameters alone, from the arrays, in dump order: the tied matrix once, the counter.
            pf = tmp / f"{opt}.params"
            words = run("params", opt, 2, pf).split()[1:]
            assert pf.stat().st_size == 40 and list(struct.unpack("<10I", pf.read_bytes())) == list(map(int, words)), opt
            assert words == saved.split()[1:11], (opt, "the saved parameters differ from the checkpoint's")
        check_cfg(run)
        special = [0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345, 1, 0x007FFFFF, 0x3F800000, 0xBF800000, 0]
        cursor = tmp / "cursor.bin"
        cursor.write_bytes(struct.pack("<11I", 123, *special, 456))
        lines = run("cursor", cursor).splitlines()
        assert lines[0] == "offset 10", lines
        expected = special.copy()
        if "--js" in sys.argv[2:]:
            # Bend's JavaScript F32 representation is a Number; this runtime canonicalizes NaN payloads.
            # Native must retain the payload. Both lanes must preserve every other bit, including -0/subnormals.
            expected[3] = 0x7FC00000
        assert list(map(int, lines[1].split()[1:])) == expected, lines
        assert run("read", cursor, 11).strip() == "ends 123 456"
        run("read", cursor, 1073741824, ok=False)
        run("read", tmp / "missing.bin", 1, ok=False)
    print("infra: PASS (all optimizer states, exact checkpoints, nonzero cursor, IEEE bits, malformed files)")


if __name__ == "__main__":
    main()

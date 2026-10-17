"""M9: lean checkpoints, exact resume with tied weights/Dropout, array offsets and malformed files."""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    binary = sys.argv[1]

    def run(*args, ok=True):
        p = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
        assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout, p.stderr)
        return p.stdout

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

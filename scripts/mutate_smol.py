"""Does `make smol`'s test catch real bugs? Each mutant breaks the code a little, demo/smol.bend is rebuilt and
reference/test_smol.py must fail. Unlike scripts/mutate.py, the mutants go into a git worktree of HEAD under .scratch/,
so the working tree is never edited. Needs data/smol135/ (scripts/smol_data.sh). About 5 minutes a mutant.

    reference/.venv/bin/python scripts/mutate_smol.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WT = ROOT / ".scratch/mutate-smol"
ENV = dict(os.environ, BEND_NO_TELEMETRY="1", PATH=f"{Path.home()}/.bend/bin:{os.environ['PATH']}")

# (name, file, before, after): every mutant must make test_smol.py fail
MUTANTS = [
    ("cross-entropy: the row's tree sum one class short", "torch/main.bend",
     "tree_sum(U32.from_nat(1n+k), sc)", "tree_sum(U32.from_nat(k), sc)"),
    ("RMSNorm backward: the last number of each row drops its mean term", "torch/main.bend",
     "R5{x, dy, g, Array.set(F32, dx, (off + j : U32), (((b * c : F32) - (xh * m : F32) : F32) * r : F32))",
     "R5{x, dy, g, Array.set(F32, dx, (off + j : U32), ((b * c : F32) * r : F32))"),
    ("embedding: a repeated id keeps only its last gradient", "torch/main.bend",
     "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, (a + b : F32))",
     "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, b)"),
    ("AdamW: no weight decay in main.bend's step (the norms and the embedding)", "torch/main.bend",
     "adam_decay(dec, lr, wd, x)", "x"),
    ("attention: scale 0.124 instead of 1/8", "demo/smol.bend",
     "64n, 0.125, {==}, {==}, {==})", "64n, 0.124, {==}, {==}, {==})"),
    ("final RMSNorm: eps 1e-4 instead of 1e-5", "demo/smol.bend",
     "~Stack(~b), ~T.RMSNorm(R(b), D(), 0.00001))", "~Stack(~b), ~T.RMSNorm(R(b), D(), 0.0001))"),
]


def sh(*args, cwd=WT, check=True):
    return subprocess.run(args, cwd=cwd, env=ENV, capture_output=True, text=True, check=check)


def main():
    if not WT.exists():
        sh("git", "worktree", "add", "--detach", str(WT), "HEAD", cwd=ROOT)
        (WT / "data").symlink_to(ROOT / "data")
    sh("git", "checkout", "--detach", subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                     text=True).stdout.strip())
    results = []
    for name, path, before, after in MUTANTS:
        sh("git", "checkout", "--", ".")
        p = WT / path
        src = p.read_text()
        assert src.count(before) == 1, f"{name}: the text to mutate is not once in {path}"
        p.write_text(src.replace(before, after))
        build = sh("bend", "demo/smol.bend", "-o", ".scratch/smol", check=False)
        if build.returncode != 0 or not (WT / ".scratch/smol").exists():
            results.append((name, "caught by the build"))
        else:
            t = sh(sys.executable, "reference/test_smol.py", ".scratch/smol", check=False)
            lines = [line for line in t.stdout.splitlines() if line.startswith(("smol grads: error", "smol train 3, loss", "smol loss"))]
            results.append((name, ("caught" if t.returncode != 0 else "SURVIVED") + " | " + " | ".join(lines)))
        (WT / ".scratch/smol").unlink(missing_ok=True)
        print(name, "->", results[-1][1], flush=True)
    sh("git", "checkout", "--", ".")
    survived = [n for n, r in results if "SURVIVED" in r]
    print(f"mutate_smol: {len(results)} mutants, {len(results) - len(survived)} caught" +
          (f"; survived: {', '.join(survived)}" if survived else ""))
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()

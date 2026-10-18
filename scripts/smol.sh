#!/usr/bin/env bash
# Writes demo/smol.bend's stack of blocks (the defs between its "stack" markers) and its sizes for one SmolLM2 model.
# A template takes only arguments fixed at compile time and Bend instantiates templates 64 levels deep at most, so the
# stack of LAYERS() blocks is spelled out as a balanced tree of Seq, one def per subtree size. 135m rewrites
# demo/smol.bend in place (the committed file); 360m writes .scratch/smol_360m.bend.
#
#   scripts/smol.sh 135m | 360m
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  135m) layers=30; d=576; kvw=192; ff=1536; kv=3; g=3; data=data/smol135/; out=demo/smol.bend ;;
  360m) layers=32; d=960; kvw=320; ff=2560; kv=5; g=3; data=data/smol360/; out=.scratch/smol_360m.bend ;;
  *) echo "usage: scripts/smol.sh 135m | 360m"; exit 2 ;;
esac
mkdir -p .scratch
python3 - "$layers" "$d" "$kvw" "$ff" "$kv" "$g" "$data" "$out" <<'PY'
import re, sys
layers, d, kvw, ff, kv, g, data, out = sys.argv[1:]
s = open("demo/smol.bend").read()
for name, v in [("D", d), ("KVW", kvw), ("FF", ff), ("HKV", kv), ("GRP", g), ("LAYERS", layers)]:
    s, n = re.subn(rf"def {name}\(\) -> Nat:\n  \d+n", f"def {name}() -> Nat:\n  {v}n", s)
    assert n == 1, name
s, n = re.subn(r'def DATA\(\) -> String:\n  "[^"]*"', f'def DATA() -> String:\n  "{data}"', s)
assert n == 1
def sizes(n, out):
    if n > 1 and n not in out:
        sizes(n // 2, out); sizes(n - n // 2, out); out.append(n)
    return out
defs = ["def StP1() -> Type:\n  Boxed<BP>\n", "def StS1(b: Nat) -> Type:\n  Boxed<BS(b)>\n",
        "def Stack1(~b: Nat) -> T.Layer<StP1, A(b, D()), A(b, D()), StS1(b)>:\n"
        "  Box(~BP, ~A(b, D()), ~A(b, D()), ~BS(b), ~Block(~b))\n"]
for n in sizes(int(layers), []):
    a, c = n // 2, n - n // 2
    defs.append(f"def StP{n}() -> Type:\n  T.Two<StP{a}, StP{c}>\n")
    defs.append(f"def StS{n}(b: Nat) -> Type:\n  T.Two<StS{a}(b), StS{c}(b)>\n")
    defs.append(f"def Stack{n}(~b: Nat) -> T.Layer<StP{n}, A(b, D()), A(b, D()), StS{n}(b)>:\n"
                f"  T.Seq(~StP{a}, ~StP{c}, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS{a}(b), ~StS{c}(b), ~Stack{a}(~b), ~Stack{c}(~b))\n")
defs += [f"def StackP() -> Type:\n  StP{layers}\n", f"def StackS(b: Nat) -> Type:\n  StS{layers}(b)\n",
         f"def Stack(~b: Nat) -> T.Layer<StackP, A(b, D()), A(b, D()), StackS(b)>:\n  Stack{layers}(~b)\n"]
begin, end = "# ---- stack: begin ----\n", "# ---- stack: end ----\n"
i, j = s.index(begin) + len(begin), s.index(end)
s = s[:i] + "\n".join(defs) + s[j:]
open(out, "w").write(s)
print(out)
PY

"""Train the tiny Llama on your own text: prepares a working directory the way reference/tiny_ref.py --prep prepares
data/tiny/, then runs the Bend program from it.

    reference/.venv/bin/python scripts/own_text.py <text file> [--steps N] [--chars N] [--seed N] [--prompt TEXT]

The model reads bytes. Its vocabulary has 65 entries, so the 65 most frequent bytes of the text are kept and every
other byte becomes the most frequent one (usually the space); a text with fewer distinct bytes gets unused filler
entries. The initial weights are drawn as the twin's (seed 0), and the batches are 2 x 17 bytes at seeded random
offsets, one per step. The working directory is .scratch/own/<file name>/; the Bend program (.scratch/tiny_llama, from
`make build`) runs there, trains for --steps AdamW steps and continues the prompt by --chars characters, each drawn at
temperature 0.8. The loss of every step is written to loss.tsv in the working directory.
"""

import argparse
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from tiny_ref import B, T, V, shapes  # noqa: E402


def vocabulary(data):
    counts = Counter(data)
    keep = sorted(b for b, _ in counts.most_common(V))
    filler = [b for b in range(32, 127) if b not in counts] + [b for b in range(256) if b not in counts]
    voc = sorted(keep + filler[:V - len(keep)])
    common = counts.most_common(1)[0][0]
    index = {b: i for i, b in enumerate(voc)}
    ids = np.array([index.get(b, index[common]) for b in data], dtype=np.uint8)
    return bytes(voc), ids, len(counts) - len(keep)


def prep(text, work, steps):
    data = text.read_bytes()
    if len(data) < 1000:
        sys.exit(f"{text}: {len(data)} bytes; the batches need at least 1,000")
    voc, ids, dropped = vocabulary(data)
    d = work / "data" / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    weights = []
    for name, shape in shapes():
        if name.endswith(("ln1", "ln2")) or name == "lnf":
            t = torch.ones(shape)
        else:
            t = torch.randn(*shape, generator=g) * 0.02 * (4.0 if name == "emb" else 1.0)
        weights.append(t.numpy().astype("<f4").tobytes())
    (d / "init.bin").write_bytes(b"".join(weights))
    offs = torch.randint(0, len(ids) - T - 1, (steps, B), generator=g).numpy()
    batches = np.stack([np.stack([ids[o:o + T + 1] for o in row]) for row in offs])
    (d / "batches.bin").write_bytes(batches.astype(np.uint8).tobytes())
    (d / "vocab.txt").write_bytes(voc)
    return voc, dropped


def run(work, *args):
    out = subprocess.run([str(ROOT / ".scratch" / "tiny_llama"), *args], cwd=work, check=True, capture_output=True)
    return out.stdout.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("text", type=Path)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", help="at least 16 characters; default: the text's first 32 bytes")
    a = ap.parse_args()
    if not (ROOT / ".scratch" / "tiny_llama").exists():
        sys.exit("build the program first: make build")
    work = ROOT / ".scratch" / "own" / a.text.name
    voc, dropped = prep(a.text, work, a.steps)
    prompt = a.prompt or a.text.read_bytes()[:32].decode("utf-8", "replace")
    if len(prompt.encode("utf-8")) < 16:
        sys.exit("the prompt needs at least 16 characters")
    print(f"{a.text}: vocabulary of {len(voc)} bytes" + (f", {dropped} rarer ones mapped to the most frequent" if dropped else ""))
    losses = [struct.unpack("<f", struct.pack("<I", int(line.split()[1])))[0]
              for line in run(work, "train", str(a.steps)).splitlines() if line.startswith("loss ")][1:]
    (work / "loss.tsv").write_text("step\tloss\n" + "".join(f"{i}\t{x:.6f}\n" for i, x in enumerate(losses, 1)))
    tail = losses[-100:]
    print(f"{a.steps} AdamW steps: loss {losses[0]:.3f} at step 1, {sum(tail) / len(tail):.3f} over the last {len(tail)}")
    # the program reads and writes one character per byte: pass the prompt's bytes as characters, and turn the
    # characters it prints back into bytes before reading them as UTF-8
    as_bytes = prompt.encode("utf-8").decode("latin-1")
    out = run(work, "draw", str(a.steps), str(a.chars), str(a.seed), as_bytes)
    text = out.split("\ntext ", 1)[1].split("\nfinal", 1)[0]
    print(f"\n{prompt}" + text.encode("latin-1", "replace").decode("utf-8", "replace"))


if __name__ == "__main__":
    main()

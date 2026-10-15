"""The PyTorch twin of demo/tiny_llama.bend: a two-layer character-level Llama on Tiny Shakespeare.

    reference/.venv/bin/python reference/tiny_ref.py --prep         # data/tiny/: initial weights and the batches
    reference/.venv/bin/python reference/tiny_ref.py grads          # loss and every gradient on the first batch
    reference/.venv/bin/python reference/tiny_ref.py train <steps>  # AdamW steps: the loss before each update
    reference/.venv/bin/python reference/tiny_ref.py generate <prompt> <n> < bend-output  # greedy, from Bend's weights
    THREADS=n sets PyTorch's thread count (1 by default)

Architecture (Llama, untied head): token embedding (65 x 32); two blocks of [RMSNorm, attention with 4 query heads and
2 key/value heads of size 8, RoPE base 10000, causal] and [RMSNorm, SwiGLU 32 -> 64 -> 32], each with a residual;
RMSNorm; a linear head to 65 logits; cross-entropy. Batches of 2 sequences of 16 characters at seeded random offsets.
Every parameter gets AdamW (lr 3e-3, betas 0.9/0.999, eps 1e-8, weight decay 0.01), as in the Bend program.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

D = Path("data/tiny")
V, DIM, H, KV, HD, FF, T, B, LAYERS = 65, 32, 4, 2, 8, 64, 16, 2, 2
EPS, BASE, SCALE = 1e-5, 10000.0, 0.35355339059327373  # 1 / sqrt(HD)
STEPS = 2000
LR, BETAS, ADAM_EPS, WD = 3e-3, (0.9, 0.999), 1e-8, 0.01
BLOCK = ["ln1", "wq", "wk", "wv", "wo", "ln2", "wg", "wu", "wd"]


def shapes():
    out = [("emb", (V, DIM))]
    for layer in range(LAYERS):
        out += [(f"b{layer}.ln1", (DIM,)), (f"b{layer}.wq", (H * HD, DIM)), (f"b{layer}.wk", (KV * HD, DIM)),
                (f"b{layer}.wv", (KV * HD, DIM)), (f"b{layer}.wo", (DIM, H * HD)), (f"b{layer}.ln2", (DIM,)),
                (f"b{layer}.wg", (FF, DIM)), (f"b{layer}.wu", (FF, DIM)), (f"b{layer}.wd", (DIM, FF))]
    return out + [("lnf", (DIM,)), ("head", (V, DIM))]


def prep():
    text = (D / "input.txt").read_text()
    chars = sorted(set(text))
    assert len(chars) == V, len(chars)
    ids = np.array([chars.index(c) for c in text], dtype=np.uint8)
    (D / "init").mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for name, shape in shapes():
        if name.endswith(("ln1", "ln2")) or name == "lnf":
            t = torch.ones(shape)
        else:
            t = torch.randn(*shape, generator=g) * 0.02 * (4.0 if name == "emb" else 1.0)
        (D / "init" / f"{name}.bin").write_bytes(t.numpy().astype("<f4").tobytes())
    offs = torch.randint(0, len(ids) - T - 1, (STEPS, B), generator=g).numpy()
    batches = np.stack([np.stack([ids[o:o + T + 1] for o in row]) for row in offs])  # STEPS x B x (T+1)
    (D / "batches.bin").write_bytes(batches.astype(np.uint8).tobytes())
    (D / "vocab.txt").write_text("".join(chars))


def load():
    return {name: torch.from_numpy(np.fromfile(D / "init" / f"{name}.bin", dtype="<f4").reshape(shape)).clone()
            .requires_grad_(True) for name, shape in shapes()}


def batch(step):
    raw = np.fromfile(D / "batches.bin", dtype=np.uint8, count=B * (T + 1), offset=step * B * (T + 1))
    raw = torch.from_numpy(raw.astype(np.int64)).reshape(B, T + 1)
    return raw[:, :T].reshape(-1), raw[:, 1:].reshape(-1)


def rmsnorm(x, g):
    return g * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope(x, heads):
    inv = 1.0 / (BASE ** (torch.arange(0, HD, 2, dtype=torch.int64).float() / HD))
    pos = (torch.arange(x.shape[0]) % T).float()
    freqs = (inv[:, None] @ pos[None, :]).T
    emb = torch.cat((freqs, freqs), dim=-1)
    xh = x.reshape(x.shape[0], heads, HD)
    return (xh * emb.cos()[:, None, :] + rotate_half(xh) * emb.sin()[:, None, :]).reshape(x.shape)


def attention(q, k, v):
    b = q.shape[0] // T
    q = q.reshape(b, T, H, HD).transpose(1, 2)
    k = k.reshape(b, T, KV, HD).transpose(1, 2).repeat_interleave(H // KV, dim=1)
    v = v.reshape(b, T, KV, HD).transpose(1, 2).repeat_interleave(H // KV, dim=1)
    s = (q @ k.transpose(-1, -2)) * SCALE
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2).reshape(b * T, H * HD)


def logits(p, x):
    h = F.embedding(x, p["emb"])
    for layer in range(LAYERS):
        w = {k: p[f"b{layer}.{k}"] for k in BLOCK}
        n = rmsnorm(h, w["ln1"])
        a = attention(rope(F.linear(n, w["wq"]), H), rope(F.linear(n, w["wk"]), KV), F.linear(n, w["wv"]))
        h = h + F.linear(a, w["wo"])
        n = rmsnorm(h, w["ln2"])
        h = h + F.linear(F.silu(F.linear(n, w["wg"])) * F.linear(n, w["wu"]), w["wd"])
    return F.linear(rmsnorm(h, p["lnf"]), p["head"])


def forward(p, x, y):
    return F.cross_entropy(logits(p, x), y)


def generate(p, window, n):
    """Greedy continuation of a window of T ids, one id at a time, as demo/tiny_llama.bend's sample mode."""
    out = []
    with torch.no_grad():
        for _ in range(n):
            nxt = int(logits(p, torch.tensor(window)).argmax(-1)[-1])
            out.append(nxt)
            window = window[1:] + [nxt]
    return out


def bits(t):
    return " ".join(str(w) for w in t.detach().to(torch.float32).reshape(-1).numpy().view(np.uint32))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    if sys.argv[1:] == ["--prep"]:
        prep()
        return
    if sys.argv[1] == "generate":  # weights from a Bend run's "final.<name>" lines on stdin; prompt and length as args
        words = {line.split()[0][len("final."):]: line.split()[1:] for line in sys.stdin if line.startswith("final.")}
        p = {name: torch.from_numpy(np.array([int(w) for w in words[name]], dtype=np.uint32).view(np.float32)
                                    .reshape(shape).copy()) for name, shape in shapes()}
        chars = (D / "vocab.txt").read_text()
        window = [chars.index(c) for c in sys.argv[2]][-T:]
        print("ids", " ".join(str(i) for i in generate(p, window, int(sys.argv[3]))))
        return
    p = load()
    if sys.argv[1] == "grads":
        loss = forward(p, *batch(0))
        loss.backward()
        print("loss", bits(loss))
        for name, _ in shapes():
            print(f"grad.{name}", bits(p[name].grad))
        return
    steps = int(sys.argv[2])
    opt = torch.optim.AdamW(p.values(), lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WD)
    for step in range(steps):
        opt.zero_grad()
        loss = forward(p, *batch(step))
        loss.backward()
        opt.step()
        print("loss", bits(loss))
    for name, _ in shapes():
        print(f"final.{name}", bits(p[name]))


if __name__ == "__main__":
    main()

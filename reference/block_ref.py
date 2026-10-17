"""The PyTorch twin of demo/smol_block.bend: one Llama decoder block trained with AdamW against a target under MSE.

    reference/.venv/bin/python reference/block_ref.py <size> train <steps>   # 1 + steps losses (bits); train_ms times the last <steps>
    THREADS=n sets PyTorch's thread count (1 by default)

<size> is 135m, 360m or 1.7b: one block at SmolLM2's width, heads and feed-forward size. The weights, inputs and
targets come from the same hash of each number's index as the Bend program (gen below), so both start from the same
bits without reading files. 8 sequences of 64 positions, RoPE base 100000, head size 64, causal attention, residuals;
AdamW (lr 1e-3, betas 0.9/0.999, eps 1e-8, weight decay 0.01) on every parameter, as in the Bend program.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SIZES = {  # width, key/value heads, query heads per key/value head, feed-forward
    "135m": (576, 3, 3, 1536),
    "360m": (960, 5, 3, 2560),
    "1.7b": (2048, 32, 1, 8192),
}
HD, T, B, EPS, BASE, SCALE = 64, 64, 8, 1e-5, 100000.0, 0.125
LR, BETAS, ADAM_EPS, WD = 1e-3, (0.9, 0.999), 1e-8, 0.01
W_SC, X_SC = 2.0 ** -21, 2.0 ** -15


def gen(r, c, s, sc):
    """demo/smol_block.bend's gen: (h(i, s) mod 65536 - 32768) * sc for i = 0 .. r*c-1, row-major; exact in F32."""
    i = np.arange(r * c, dtype=np.uint64)
    x = (i * 2654435761 + s * 40503) & 0xFFFFFFFF
    x = x ^ (x // 32768)
    x = (x * 2246822519) & 0xFFFFFFFF
    x = x ^ (x // 8192)
    v = ((x % 65536).astype(np.float32) - np.float32(32768.0)) * np.float32(sc)
    return torch.from_numpy(v.astype(np.float32).reshape(r, c))


def params(d, kv, g, ff):
    kvw = kv * HD
    shapes = [("ln1", None), ("wq", (d, d)), ("wk", (kvw, d)), ("wv", (kvw, d)), ("wo", (d, d)), ("ln2", None),
              ("wg", (ff, d)), ("wu", (ff, d)), ("wd", (d, ff))]
    out, seed = {}, 1
    for name, shape in shapes:
        if shape is None:
            out[name] = torch.ones(d)
        else:
            out[name] = gen(*shape, seed, W_SC)
            seed += 1
    return {k: v.requires_grad_(True) for k, v in out.items()}


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


def attention(q, k, v, h, kv):
    q = q.reshape(B, T, h, HD).transpose(1, 2)
    k = k.reshape(B, T, kv, HD).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    v = v.reshape(B, T, kv, HD).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    s = (q @ k.transpose(-1, -2)) * SCALE
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2).reshape(B * T, h * HD)


def block(w, x, kv, g):
    n = rmsnorm(x, w["ln1"])
    a = attention(rope(F.linear(n, w["wq"]), kv * g), rope(F.linear(n, w["wk"]), kv), F.linear(n, w["wv"]), kv * g, kv)
    x = x + F.linear(a, w["wo"])
    n = rmsnorm(x, w["ln2"])
    return x + F.linear(F.silu(F.linear(n, w["wg"])) * F.linear(n, w["wu"]), w["wd"])


def bits(t):
    return " ".join(str(v) for v in t.detach().to(torch.float32).reshape(-1).numpy().view(np.uint32))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    if len(sys.argv) not in (4, 5) or sys.argv[1] not in SIZES or sys.argv[2] != "train":
        sys.exit(__doc__)
    d, kv, g, ff = SIZES[sys.argv[1]]
    steps = int(sys.argv[3])
    w = params(d, kv, g, ff)
    optimizer = sys.argv[4] if len(sys.argv) == 5 else "adamw"
    if optimizer not in ("sgd", "adamw"):
        raise SystemExit("optimizer: sgd or adamw")
    opt = (torch.optim.SGD(w.values(), lr=LR) if optimizer == "sgd" else
           torch.optim.AdamW(w.values(), lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WD))
    for step in range(steps + 1):  # the first step warms up (AdamW makes its buffers then) and is not timed
        if step == 1:
            t0 = time.perf_counter()
        opt.zero_grad()
        loss = F.mse_loss(block(w, gen(B * T, d, 1000 + step, X_SC), kv, g), gen(B * T, d, 2000 + step, X_SC))
        loss.backward()
        opt.step()
        print("loss", bits(loss), flush=True)
    print(f"train_ms {round((time.perf_counter() - t0) * 1000) if steps else 0}")


if __name__ == "__main__":
    main()

"""The PyTorch twin of demo/tiny_mixed.bend: tiny_ref.py's Llama with BF16 storage, and its int8 inference.

    reference/.venv/bin/python reference/tiny_mixed_ref.py train <steps>   # AdamW with BF16 storage: the loss of each step
    reference/.venv/bin/python reference/tiny_mixed_ref.py train64 <steps> # the same with float64 arithmetic (the reference)
    reference/.venv/bin/python reference/tiny_mixed_ref.py int8            # the logits on the first batch, F32 and int8

No torch.autocast: the casts are explicit, at the points where the Bend program rounds. Every layer's output goes
through bf(x) = x.to(torch.bfloat16).to(torch.float32), and every parameter is used as bf(w); autograd then rounds
each gradient arriving at a cast to BF16 (the backward pass of .to(torch.bfloat16)), so w.grad is the BF16 rounding of
the gradient of bf(w), and AdamW updates the F32 master weights. The arithmetic inside each layer is F32 (float64 in
train64, where bf(x) rounds to BF16 and back to float64: the same storage, exact-enough arithmetic).

int8 (demo/tiny_mixed.bend's scheme): each projection and the head quantized by rows, s = max|w| / 127,
q = round(w / s) clamped to -127..127, used as q · s in the F32 model.
"""

import os
import sys

import torch
import torch.nn.functional as F

import tiny_ref as R


def bf(x):
    return x.to(torch.bfloat16).to(x.dtype)


def logits_mixed(p, x):
    h = bf(F.embedding(x, bf(p["emb"])))
    for layer in range(R.LAYERS):
        w = {k: bf(p[f"b{layer}.{k}"]) for k in R.BLOCK}
        n = bf(R.rmsnorm(h, w["ln1"]))
        q, k, v = bf(F.linear(n, w["wq"])), bf(F.linear(n, w["wk"])), bf(F.linear(n, w["wv"]))
        a = bf(R.attention(bf(R.rope(q, R.H)), bf(R.rope(k, R.KV)), v))
        h = bf(h + bf(F.linear(a, w["wo"])))
        n = bf(R.rmsnorm(h, w["ln2"]))
        g = bf(F.silu(bf(F.linear(n, w["wg"]))))
        u = bf(F.linear(n, w["wu"]))
        h = bf(h + bf(F.linear(bf(g * u), w["wd"])))
    return bf(F.linear(bf(R.rmsnorm(h, bf(p["lnf"]))), bf(p["head"])))


def q8(w):
    s = w.abs().amax(dim=1) / 127.0
    q = torch.where(s[:, None] == 0, torch.zeros_like(w), torch.round(w / s[:, None])).clamp(-127, 127)
    return q * s[:, None]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    p = R.load()
    if sys.argv[1] == "int8":
        x, _ = R.batch(0)
        with torch.no_grad():
            print("logits_f32", R.bits(R.logits(p, x)))
            pq = {k: (q8(v) if k.split(".")[-1] in ("wq", "wk", "wv", "wo", "wg", "wu", "wd") or k == "head" else v)
                  for k, v in p.items()}
            print("logits_int8", R.bits(R.logits(pq, x)))
        return
    steps = int(sys.argv[2])
    if sys.argv[1] == "train64":  # BF16 storage with float64 arithmetic and master weights: the reference for both
        p = {k: v.detach().double().requires_grad_(True) for k, v in p.items()}
    opt = torch.optim.AdamW(p.values(), lr=R.LR, betas=R.BETAS, eps=R.ADAM_EPS, weight_decay=R.WD)
    for step in range(steps):
        opt.zero_grad()
        x, y = R.batch(step)
        loss = F.cross_entropy(logits_mixed(p, x), y)
        loss.backward()
        opt.step()
        print("loss", R.bits(loss) if loss.dtype == torch.float32 else repr(loss.item()))
    if loss.dtype == torch.float32:
        for name, _ in R.shapes():
            print(f"final.{name}", R.bits(p[name]))


if __name__ == "__main__":
    main()

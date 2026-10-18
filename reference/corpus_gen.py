"""Writes the held-out numerical corpus, bench/corpus/ (milestone 14): run once, at the corpus's creation; never again
for the same version, since the corpus stays frozen.

Every input is drawn from SEED, a seed no other test or demo uses (SEED_NOTE, copied into the manifest, says how that
was checked). For each case PyTorch computes the results twice, once here, on the same inputs: in float32 (out32/,
what the exact tier and the bit counts read) and in float64 (out64/, what the tolerance reads). Next to each float64
result is its scale (scale/): the sum of the absolute values of the terms that produce it, computed in float64 by the
rules of SCALES below, so a result that comes out of cancellation is judged against the size of what cancelled.
manifest.json lists every file with its sha256, the policy (POLICY, with oracle.py's ATOL and RTOL as its constants),
the scale rules, the exact tier and the versions. reference/test_corpus.py reads it at release time.

    reference/.venv/bin/python reference/corpus_gen.py bench/corpus     # refuses to overwrite an existing corpus
"""

import hashlib
import json
import math
import platform
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oracle import ATOL, RTOL

SEED = 271828  # the first digits of e
SEED_NOTE = ("271828 appears nowhere else in this repository, bend-ml or llama.bend (grep, 2026-10-18). The seeds in "
             "use: torch.manual_seed 0-11 (test_mlp), 0-9, 100 and 101 (test_ops), 0 (test_tensor, check_imap, "
             "mnist_ref, tiny_ref, smol_ref's data) and 1 (smol_ref); random.seed 0; block_ref's hash seeds 1-9; the "
             "Bend tests' hash seeds 1-92 (fast_test, gpu_test, gpu_layers_test) and dropout seed 7 (ops_test). Python's "
             "random module is seeded with the same number, a separate generator.")
VERSION = "corpus-2"
POLICY = (f"cancellation-1: a number passes when it has the bits of PyTorch's float32 result (out32), or when both it "
          f"and out32 are the same non-finite class (NaN, +inf, -inf; judged against out32 whenever out32 or out64 is "
          f"not finite), or when |bend - out64| <= ATOL + RTOL * max(scale, |out64|), with oracle.py's ATOL={ATOL!r} "
          f"and RTOL={RTOL!r}; scale is the per-number sum of |terms| of SCALES. The exact tier must have out32's bits.")

# How each result's scale is computed, in float64 from the case's inputs (|.| element by element; P are the attention
# probabilities and p the softmax of a row, both in float64; r = 1/sqrt(mean(x^2) + eps) for RMSNorm, r =
# 1/sqrt(var(x) + eps) and A = (|x| + |mean(x)|) r for LayerNorm). "abs-run" means: the same PyTorch function (and its
# backward) run on |inputs| with |upstream gradient|, so every product and sum in it adds absolute values.
SCALES = {
    "linear, linear_first, linear_nb (and fast_)": "abs-run: y = |x||W|^T + |b|, dx = |dy||W| (linear_first: 0), "
                                                    "g.w = |dy|^T|x|, g.b = column sums of |dy|",
    "relu, dropout, mul2": "|out64| (one product or a selection: nothing cancels)",
    "sigmoid, tanh, silu, gelu": "y: |out64|, but gelu's y = 0.5 x (1 + t): 0.5 |x| (1 + |t|); dx, with y and t = "
                                 "tanh(k (x + c x^3)) in float64: sigmoid dy y (1 - y): |dy| y (1 + y); tanh dy (1 - y^2): "
                                 "|dy| (1 + y^2); silu dy (s + x s (1 - s)), s = sigmoid(x): |dy| (s + |x| s (1 + s)); gelu "
                                 "dy (0.5 (1 + t) + 0.5 x (1 - t^2) k (1 + 3 c x^2)): |dy| (0.5 (1 + |t|) + 0.5 |x| (1 + t^2) "
                                 "k (1 + 3 c x^2)), k = sqrt(2/pi), c = 0.044715",
    "softmax": "y: |out64| (a ratio of positive terms); dx = p(dy - sum(p dy)): p (|dy| + sum(p |dy|))",
    "logsoftmax": "y = x - m - log(sum exp(x - m)), m the row max: |x| + |m| + |log sum exp(x - m)|; "
                  "dx = dy - p sum(dy): |dy| + p sum(|dy|)",
    "rmsnorm": "y = g x r: |out64|; dx = r g dy - x r^3 mean(g dy x): r |g dy| + |x| r^3 mean(|g dy x|); "
               "g = sum over rows of dy x r: sum of |dy x r|",
    "layernorm": "y = g (x - mean) r + b: |g| A + |b|; dx = r (g dy - mean(g dy) - xhat mean(g dy xhat)): "
                 "r (|g dy| + mean(|g dy|) + A mean(|g dy| A)); g.g: sum over rows of |dy| A; g.b: of |dy|",
    "rope": "y = x cos + rot(x) sin: |x| |cos| + |rot(x)| |sin|; dx the same with |dy| (the opposite rotation)",
    "attention": "y = P V: P |V|; dv = P^T dy: P^T |dy| (summed over the group); dP = dy V^T: |dy| |V|^T; "
                 "dS = P (dP - sum(P dP)): P (S_dP + sum(P S_dP)); dq = scale dS K: scale S_dS |K|; "
                 "dk = scale dS^T Q: scale S_dS^T |Q| (summed over the group)",
    "embedding": "y: |out64| (a copy); g: abs-run (the rows of |dy| added at each id)",
    "mse": "loss = mean((y - t)^2): mean((|y| + |t|)^2); dx = 2 (y - t) / n: 2 (|y| + |t|) / n",
    "nll": "loss = -mean(x[label]): mean(|x[label]|); dx: |out64|",
    "ce": "loss = mean(m + log sum exp(x - m) - x[label]): mean(|m| + |log sum exp(x - m)| + |x[label]|); "
          "dx = (p - onehot) / n: (p + onehot) / n",
    "sgd, momentum, adam, adam_l2, adamw": "carried through the steps as PyTorch computes them: S_w0 = |w0|; SGD: "
          "S_w += lr |g|; momentum: S_b = mu S_b + |g|, S_w += lr S_b; Adam: S_g = |g| (+ wd S_w for adam_l2), "
          "AdamW first S_w *= 1 - lr wd, S_m += (1 - b1)(S_g + S_m) (lerp), S_v = b2 S_v + (1 - b2) S_g^2, "
          "S_w += (lr / (1 - b1^t)) S_m / (sqrt(v) / sqrt(1 - b2^t) + eps) with v the float64 run's; m: S_m (S_b), v: S_v",
    "tensor operations": "abs-run (sub as add): add/sub z = |x| + |y|, their gradients the sums of |dz|; mul, div, "
                         "bmm, sum, mean and their gradients over |inputs|; copies (reshape, flatten, permute, "
                         "transpose, narrow, cat) |out64|",
}

g = torch.Generator().manual_seed(SEED)
rng = random.Random(SEED)
CASES: list[dict] = []
EXACT: dict[str, list[str]] = {}
F64 = torch.float64


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def randn(*shape, scale=1.0):
    return torch.randn(tuple(shape), generator=g) * scale


def spiced(*shape, scale=1.0):
    """Normal values with exact zeros, a negative zero and two large values in the first places (when there is room)."""
    x = randn(*shape, scale=scale)
    flat = x.view(-1)
    for i, v in enumerate([0.0, -0.0, 12.0, -12.0][: max(0, flat.numel() - 1)]):
        flat[i] = v
    return x


def add(name, words, inputs, ref, scale, exact=(), fast=None):
    """ref(dtype, inputs) -> {name: tensor}: PyTorch's results; scale(inputs in float64) -> {name: tensor}, defaulting
    to |out64| for a name it leaves out. fast: the words of the same case for torch/fast.bend's layer."""
    out32 = {k: v.detach().to(torch.float32) for k, v in ref(torch.float32, inputs).items()}
    in64 = {k: (v if v.dtype == torch.int64 else v.to(F64)) for k, v in inputs.items()}
    out64 = {k: v.detach().to(F64) for k, v in ref(F64, in64).items()}
    sc = {k: v.detach().to(F64).abs() for k, v in (scale(in64) if scale else {}).items()}
    for k in out64:
        sc.setdefault(k, out64[k].abs())
        assert sc[k].numel() == out64[k].numel(), (name, k)
    CASES.append(dict(name=name, words=[int(w) for w in words], fast=fast, inputs=inputs, out32=out32, out64=out64, scale=sc))
    if exact:
        EXACT[name] = list(exact)
        if fast is not None:
            EXACT["fast_" + name] = list(exact)


def leaf(t, dt):
    return t.detach().to(dt).clone().requires_grad_(True)


def absrun(fn, xs, dy):
    """fn on |xs| with |dy| upstream: the results and the gradients of a computation whose every coefficient is now
    positive, so each output is the sum of |terms| of the original."""
    ls = [x.abs().clone().requires_grad_(True) for x in xs]
    y = fn(*ls)
    y.backward(dy.abs())
    return y.detach(), [l.grad if l.grad is not None else torch.zeros_like(l) for l in ls]


# ---- linear layers: ops 0 (Linear), 1 (LinearFirst), 2 (LinearNB); each case also runs as 30, 31, 32, torch/fast.bend's
# LinearF, LinearFirstF, LinearNBF, on the same inputs. words n i o par. The last shape (operands packed in blocks in
# fast.bend, from 16384 cells up) runs Linear only ----

for n, i, o, par in [(1, 1, 1, 0), (5, 7, 3, 1), (17, 33, 9, 2), (64, 48, 40, 3), (3, 130, 70, 0), (64, 256, 72, 3)]:
    for op, tag in [(0, "linear"), (1, "linear_first"), (2, "linear_nb")]:
        if n * i >= 16384 and op != 0:
            continue
        x0, w0, b0 = spiced(n, i), randn(o, i, scale=1 / math.sqrt(i)), randn(o, scale=0.1)
        dy = randn(n, o)

        def ref(dt, a, op=op):
            x, w, b = leaf(a["x"], dt), leaf(a["w"], dt), leaf(a["b"], dt)
            y = F.linear(x, w, None if op == 2 else b)
            y.backward(a["dy"].to(dt))
            out = {"y": y, "dx": torch.zeros_like(x) if op == 1 else x.grad, "g.w": w.grad}
            if op != 2:
                out["g.b"] = b.grad
            return out

        def scale(a, op=op):
            (y, (dx, gw, gb)) = absrun(lambda x, w, b: F.linear(x, w, None if op == 2 else b), [a["x"], a["w"], a["b"]], a["dy"])
            out = {"y": y, "dx": torch.zeros_like(dx) if op == 1 else dx, "g.w": gw}
            if op != 2:
                out["g.b"] = gb
            return out

        add(f"{tag}_{n}x{i}x{o}_par{par}", [op, n, i, o, par], {"x": x0, "w": w0, "dy": dy, "b": b0}, ref, scale,
            fast=[30 + op, n, i, o, par])

# ---- activations: ops 3..9 (SiLU also as 33, fast.bend's); words n c ----

ACTS = [("relu", torch.relu), ("sigmoid", torch.sigmoid), ("tanh", torch.tanh), ("silu", F.silu),
        ("gelu", lambda x: F.gelu(x, approximate="tanh")), ("softmax", lambda x: F.softmax(x, -1)),
        ("logsoftmax", lambda x: F.log_softmax(x, -1))]


def act_scale(tag):
    def softmax(a):
        p = F.softmax(a["x"], -1)
        d = a["dy"].abs()
        return {"dx": p * (d + (p * d).sum(-1, keepdim=True))}

    def logsoftmax(a):
        x = a["x"]
        m = x.max(-1, keepdim=True).values
        lse = torch.log(torch.exp(x - m).sum(-1, keepdim=True))
        p = F.softmax(x, -1)
        d = a["dy"].abs()
        return {"y": x.abs() + m.abs() + lse.abs(), "dx": d + p * d.sum(-1, keepdim=True)}
    def sigmoid(a):
        y = torch.sigmoid(a["x"])
        return {"dx": a["dy"].abs() * y * (1 + y)}

    def tanh(a):
        y = torch.tanh(a["x"])
        return {"dx": a["dy"].abs() * (1 + y * y)}

    def silu(a):
        x = a["x"]
        s = torch.sigmoid(x)
        return {"dx": a["dy"].abs() * (s + x.abs() * s * (1 + s))}

    def gelu(a):
        x = a["x"]
        k, c = math.sqrt(2 / math.pi), 0.044715
        t = torch.tanh(k * (x + c * x ** 3))
        du = k * (1 + 3 * c * x * x)
        return {"y": 0.5 * x.abs() * (1 + t.abs()), "dx": a["dy"].abs() * (0.5 * (1 + t.abs()) + 0.5 * x.abs() * (1 + t * t) * du)}
    return {"softmax": softmax, "logsoftmax": logsoftmax, "sigmoid": sigmoid, "tanh": tanh, "silu": silu, "gelu": gelu}.get(tag)


for k, (tag, fn) in enumerate(ACTS):
    for n, c in [(1, 1), (7, 13), (32, 65), (4, 300)]:
        x0, dy = spiced(n, c, scale=3.0), randn(n, c)

        def ref(dt, a, fn=fn):
            x = leaf(a["x"], dt)
            y = fn(x)
            y.backward(a["dy"].to(dt))
            return {"y": y, "dx": x.grad}

        add(f"{tag}_{n}x{c}", [3 + k, n, c], {"x": x0, "dy": dy}, ref, act_scale(tag),
            exact=("y", "dx") if tag == "relu" else (), fast=[33, n, c] if tag == "silu" else None)

# ---- norms: op 10 (RMSNorm), 11 (LayerNorm); words n d eps ----


def rmsnorm(x, gain, eps):
    return gain * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


for n, d, eps in [(1, 1, 1e-5), (6, 10, 1e-5), (16, 96, 1e-6), (3, 576, 1e-5)]:
    x0, g0, dy = spiced(n, d), 1.0 + randn(d, scale=0.2), randn(n, d)

    def ref(dt, a, eps=eps):
        x, gain = leaf(a["x"], dt), leaf(a["g"], dt)
        y = rmsnorm(x, gain, eps)
        y.backward(a["dy"].to(dt))
        return {"y": y, "dx": x.grad, "g": gain.grad}

    def scale(a, eps=eps):
        x, gain, d = a["x"], a["g"], a["dy"]
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        gd = (gain * d).abs()
        return {"dx": r * gd + x.abs() * r ** 3 * (gd * x.abs()).mean(-1, keepdim=True),
                "g": (d * x * r).abs().sum(0)}

    add(f"rmsnorm_{n}x{d}", [10, n, d, bits(eps)], {"x": x0, "g": g0, "dy": dy}, ref, scale)

    x0, g0, b0, dy = spiced(n, d), 1.0 + randn(d, scale=0.2), randn(d, scale=0.1), randn(n, d)

    def ref(dt, a, eps=eps, d=d):
        x, gain, bias = leaf(a["x"], dt), leaf(a["g"], dt), leaf(a["b"], dt)
        y = F.layer_norm(x, (d,), gain, bias, eps)
        y.backward(a["dy"].to(dt))
        return {"y": y, "dx": x.grad, "g.g": gain.grad, "g.b": bias.grad}

    def scale(a, eps=eps):
        x, gain, bias, dyy = a["x"], a["g"], a["b"], a["dy"]
        mu = x.mean(-1, keepdim=True)
        r = torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
        A = (x.abs() + mu.abs()) * r
        gd = (gain * dyy).abs()
        return {"y": gain.abs() * A + bias.abs(), "dx": r * (gd + gd.mean(-1, keepdim=True) + A * (gd * A).mean(-1, keepdim=True)),
                "g.g": (dyy.abs() * A).sum(0), "g.b": dyy.abs().sum(0)}

    add(f"layernorm_{n}x{d}", [11, n, d, bits(eps)], {"x": x0, "g": g0, "b": b0, "dy": dy}, ref, scale)

# ---- RoPE: op 12; words b t h half base (rows: b sequences of t positions; h heads of 2*half a row) ----


def rope_tables(rows, t, half, base, dt):
    hd = 2 * half
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.int64).to(dt) / hd))
    pos = (torch.arange(rows) % t).to(dt)
    freqs = (inv[:, None] @ pos[None, :]).T
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[:, None, :], emb.sin()[:, None, :]


def rope(x, t, h, half, base):
    cos, sin = rope_tables(x.shape[0], t, half, base, x.dtype)
    xh = x.reshape(x.shape[0], h, 2 * half)
    rot = torch.cat((-xh[..., half:], xh[..., :half]), dim=-1)
    return (xh * cos + rot * sin).reshape(x.shape)


def rope_abs(x, t, h, half, base):
    cos, sin = rope_tables(x.shape[0], t, half, base, x.dtype)
    xh = x.abs().reshape(x.shape[0], h, 2 * half)
    rot = torch.cat((xh[..., half:], xh[..., :half]), dim=-1)
    return (xh * cos.abs() + rot * sin.abs()).reshape(x.shape)


for b, t, h, half, base in [(1, 1, 1, 1, 10000.0), (2, 5, 3, 4, 10000.0), (3, 16, 4, 8, 100000.0), (1, 64, 2, 32, 100000.0)]:
    n, w = b * t, h * 2 * half
    x0, dy = randn(n, w), randn(n, w)

    def ref(dt, a, t=t, h=h, half=half, base=base):
        x = leaf(a["x"], dt)
        y = rope(x, t, h, half, base)
        y.backward(a["dy"].to(dt))
        return {"y": y, "dx": x.grad}

    def scale(a, t=t, h=h, half=half, base=base):
        return {"y": rope_abs(a["x"], t, h, half, base), "dx": rope_abs(a["dy"], t, h, half, base)}

    add(f"rope_{b}x{t}_h{h}_half{half}", [12, b, t, h, half, bits(base)], {"x": x0, "dy": dy}, ref, scale)

# ---- causal attention with grouped-query heads: op 13; words b t kv g hd scale ----


def heads(q, k, v, b, t, kv, gq, hd):
    h = kv * gq
    return (q.reshape(b, t, h, hd).transpose(1, 2), k.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(gq, dim=1),
            v.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(gq, dim=1))


def attention(q, k, v, b, t, kv, gq, hd, sc):
    qh, kh, vh = heads(q, k, v, b, t, kv, gq, hd)
    s = (qh @ kh.transpose(-1, -2)) * sc
    s = s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))
    return (s.softmax(-1) @ vh).transpose(1, 2).reshape(b * t, kv * gq * hd)


def attention_scale(a, b, t, kv, gq, hd, sc):
    qh, kh, vh = heads(a["q"], a["k"], a["v"], b, t, kv, gq, hd)
    s = (qh @ kh.transpose(-1, -2)) * sc
    P = s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
    d = a["dy"].abs().reshape(b, t, kv * gq, hd).transpose(1, 2)
    y = P @ vh.abs()
    dv = (P.transpose(-1, -2) @ d).reshape(b, kv, gq, t, hd).sum(2)
    dP = d @ vh.abs().transpose(-1, -2)
    dS = P * (dP + (P * dP).sum(-1, keepdim=True))
    dq = sc * (dS @ kh.abs())
    dk = (sc * (dS.transpose(-1, -2) @ qh.abs())).reshape(b, kv, gq, t, hd).sum(2)

    def back(z, width):
        return z.transpose(1, 2).reshape(b * t, width)
    return {"y": back(y, kv * gq * hd), "dq": back(dq, kv * gq * hd), "dk": back(dk, kv * hd), "dv": back(dv, kv * hd)}


for b, t, kv, gq, hd in [(1, 1, 1, 1, 1), (3, 5, 1, 3, 8), (2, 7, 2, 2, 4), (8, 6, 2, 2, 4), (9, 4, 3, 1, 6), (2, 32, 1, 2, 16)]:
    n, w, kvw = b * t, kv * gq * hd, kv * hd
    sc = float(np.float32(1.0 / math.sqrt(hd)))
    q0, k0, v0, dy = randn(n, w), randn(n, kvw), randn(n, kvw), randn(n, w)

    def ref(dt, a, b=b, t=t, kv=kv, gq=gq, hd=hd, sc=sc):
        q, k, v = leaf(a["q"], dt), leaf(a["k"], dt), leaf(a["v"], dt)
        y = attention(q, k, v, b, t, kv, gq, hd, sc)
        y.backward(a["dy"].to(dt))
        return {"y": y, "dq": q.grad, "dk": k.grad, "dv": v.grad}

    add(f"attention_b{b}_t{t}_kv{kv}_g{gq}_hd{hd}", [13, b, t, kv, gq, hd, bits(sc)], {"q": q0, "k": k0, "v": v0, "dy": dy},
        ref, lambda a, b=b, t=t, kv=kv, gq=gq, hd=hd, sc=sc: attention_scale(a, b, t, kv, gq, hd, sc))

# ---- embedding: op 14; words n v d; ids repeat, so their gradients add up ----

for n, v, d in [(1, 1, 1), (12, 9, 5), (40, 50, 16), (7, 300, 3)]:
    e0, dy = randn(v, d), randn(n, d)
    ids = torch.tensor([rng.randrange(v) for _ in range(n)])

    def ref(dt, a):
        e = leaf(a["e"], dt)
        y = F.embedding(a["ids"], e)
        y.backward(a["dy"].to(dt))
        return {"y": y, "g": e.grad}

    def scale(a):
        return {"g": absrun(lambda e: F.embedding(a["ids"], e), [a["e"]], a["dy"])[1][0]}

    add(f"embedding_{n}_{v}x{d}", [14, n, v, d], {"e": e0, "dy": dy, "ids": ids}, ref, scale, exact=("y",))

# ---- losses: op 15 (MSE), 16 (NLL), 17 (cross-entropy); words n c ----


def loss_ref(fn, key):
    def ref(dt, a):
        x = leaf(a["x"], dt)
        loss = fn(x, a["t"].to(dt) if key == "t" else a["labels"])
        loss.backward()
        return {"loss": loss.reshape(1), "dx": x.grad}
    return ref


def mse_scale(a):
    s = a["x"].abs() + a["t"].abs()
    return {"loss": (s * s).mean().reshape(1), "dx": 2 * s / s.numel()}


def nll_scale(a):
    return {"loss": a["x"].gather(1, a["labels"][:, None]).abs().mean().reshape(1)}


def ce_scale(a):
    x, lab = a["x"], a["labels"]
    m = x.max(-1, keepdim=True).values
    lse = torch.log(torch.exp(x - m).sum(-1, keepdim=True))
    pick = x.gather(1, lab[:, None])
    onehot = F.one_hot(lab, x.shape[1]).to(F64)
    return {"loss": (m.abs() + lse.abs() + pick.abs()).mean().reshape(1), "dx": (F.softmax(x, -1) + onehot) / x.shape[0]}


for n, c in [(1, 1), (5, 6), (33, 17), (8, 1000), (2, 49152)]:
    labels = torch.tensor([rng.randrange(c) for _ in range(n)])
    if c <= 1000:
        x0, t0 = randn(n, c), randn(n, c)
        add(f"mse_{n}x{c}", [15, n, c], {"x": x0, "t": t0}, loss_ref(F.mse_loss, "t"), mse_scale)
        x0 = F.log_softmax(randn(n, c, scale=2.0), -1)
        add(f"nll_{n}x{c}", [16, n, c], {"x": x0, "labels": labels}, loss_ref(F.nll_loss, "labels"), nll_scale, exact=("dx",))
    x0 = randn(n, c, scale=2.0)
    add(f"ce_{n}x{c}", [17, n, c], {"x": x0, "labels": labels}, loss_ref(F.cross_entropy, "labels"), ce_scale)

# ---- dropout: op 18; words n c p seed step. The mask is BendTorch's own (a hash of seed, step and position, the
# definition in test_ops.py); the reference is that mask times x, and times dy: one multiplication each ----


def mix32(x):
    x = np.asarray(x, dtype=np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(2146121005)
    x ^= x >> np.uint32(15)
    x *= np.uint32(2221713035)
    x ^= x >> np.uint32(16)
    return x


def dropout_factors(count, seed, step, p):
    key = mix32(np.uint32(seed) ^ mix32(np.uint32(step)))
    with np.errstate(over="ignore"):
        hsh = mix32(key + np.arange(count, dtype=np.uint32))
    u = (hsh >> np.uint32(8)).astype(np.float32) / np.float32(16777216.0)
    scale = np.float32(1.0) / (np.float32(1.0) - np.float32(p))
    return torch.from_numpy(np.where(u < np.float32(p), np.float32(0.0), scale).astype(np.float32))


for n, c, p, seed, step in [(4, 9, 0.1, 31, 0), (16, 32, 0.5, 977, 5), (3, 1000, 0.25, 12345, 2)]:
    x0, dy = randn(n, c), randn(n, c)
    k = dropout_factors(n * c, seed, step, p).reshape(n, c)
    add(f"dropout_{n}x{c}_p{p}_s{step}", [18, n, c, bits(p), seed, step], {"x": x0, "dy": dy},
        lambda dt, a, k=k: {"y": a["x"].to(dt) * k.to(dt), "dx": a["dy"].to(dt) * k.to(dt)}, None, exact=("y", "dx"))

# ---- the product of two inputs (Mul2): op 19, and 34 for fast.bend's; words n c ----

for n, c in [(1, 1), (3, 4), (16, 33)]:
    a0, b0, dy = spiced(n, c), randn(n, c), randn(n, c)

    def ref(dt, a):
        x, y2 = leaf(a["x"], dt), leaf(a["x2"], dt)
        y = x * y2
        y.backward(a["dy"].to(dt))
        return {"y": y, "dx": x.grad, "dy2": y2.grad}

    add(f"mul2_{n}x{c}", [19, n, c], {"x": a0, "x2": b0, "dy": dy}, ref, None, exact=("y", "dx", "dy2"), fast=[34, n, c])

# ---- optimizer steps: op 20; words kind k r c lr (mu or b1) b2 eps wd. k steps of torch.optim from w with the
# gradients g[0..k-1]: the matrix after every step, then the buffers ----

OPTS = [(0, "sgd", dict(lr=0.05), lambda ps: torch.optim.SGD(ps, lr=0.05)),
        (1, "momentum", dict(lr=0.05, a=0.9), lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)),
        (2, "adam", dict(lr=1e-3, a=0.9, b2=0.999, eps=1e-8), lambda ps: torch.optim.Adam(ps, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)),
        (2, "adam_l2", dict(lr=3e-3, a=0.8, b2=0.99, eps=1e-6, wd=0.01),
         lambda ps: torch.optim.Adam(ps, lr=3e-3, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.01)),
        (3, "adamw", dict(lr=1e-3, a=0.9, b2=0.95, eps=1e-8, wd=0.1),
         lambda ps: torch.optim.AdamW(ps, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1))]


def opt_ref(make):
    def ref(dt, a):
        w = torch.nn.Parameter(a["w"].to(dt).clone())
        opt = make([w])
        ws = []
        for j in range(a["g"].shape[0]):
            w.grad = a["g"][j].to(dt).clone()
            opt.step()
            ws.append(w.detach().clone())
        out = {"w": torch.stack(ws)}
        st = opt.state[w]
        if "momentum_buffer" in st:
            out["m"] = st["momentum_buffer"]
        if "exp_avg" in st:
            out["m"], out["v"] = st["exp_avg"], st["exp_avg_sq"]
        return out
    return ref


def opt_scale(tag, hp):
    lr, a1, b2, eps, wd = (hp.get(k, 0.0) for k in ("lr", "a", "b2", "eps", "wd"))

    def scale(a):
        # w, m, v: the float64 run itself, in torch.optim's order (its v gives the denominator); sw, sm, sv: the scales
        w, m, v = a["w"].clone(), torch.zeros_like(a["w"]), torch.zeros_like(a["w"])
        sw, sm, sv = a["w"].abs(), torch.zeros_like(a["w"]), torch.zeros_like(a["w"])
        ws = []
        for t, gr in enumerate(a["g"], start=1):
            sg = gr.abs()
            if tag == "sgd":
                sw = sw + lr * sg
            elif tag == "momentum":
                sm = a1 * sm + sg
                sw = sw + lr * sm
            else:
                if tag == "adam_l2":
                    gr = gr + wd * w
                    sg = sg + wd * sw
                if tag == "adamw":
                    w = w * (1 - lr * wd)
                    sw = sw * (1 - lr * wd)
                m = m + (1 - a1) * (gr - m)
                v = b2 * v + (1 - b2) * gr * gr
                den = v.sqrt() / math.sqrt(1 - b2 ** t) + eps
                w = w - (lr / (1 - a1 ** t)) * m / den
                sm = sm + (1 - a1) * (sg + sm)
                sv = b2 * sv + (1 - b2) * sg * sg
                sw = sw + (lr / (1 - a1 ** t)) * sm / den
            ws.append(sw)
        out = {"w": torch.stack(ws)}
        if tag == "momentum":
            out["m"] = sm
        elif tag != "sgd":
            out["m"], out["v"] = sm, sv
        return out
    return scale


for kind, tag, hp, make in OPTS:
    for r, c, k in [(1, 1, 3), (7, 5, 5), (24, 40, 8)]:
        w0, gs = randn(r, c), randn(k, r, c, scale=0.5)
        words = [20, kind, k, r, c] + [bits(hp.get(key, 0.0)) for key in ("lr", "a", "b2", "eps", "wd")]
        add(f"{tag}_{r}x{c}_{k}steps", words, {"w": w0, "g": gs}, opt_ref(make), opt_scale(tag, hp))

# ---- torch/tensor.bend's operations (op 100 + demo/tensor_test.bend's op): words k, then three shapes ----

ADD, SUB, MUL, DIV, SUM, SUMK, MEAN, SUMALL, MEANALL, RESHAPE, FLATTEN, PERMUTE, TRANSPOSE, NARROW, CAT, BMM = range(16)


def tcase(name, op, k, a, b, o, inputs, fn, grads, exact=()):
    """fn(*tensors) is the operation on the inputs other than dz; grads names the gradients compared (ga, gb)."""
    words = [100 + op, k]
    for s in (a, b, o):
        words += [len(s), *s]
    keys = [key for key in ("x", "y") if key in inputs]

    def ref(dt, ins):
        ls = [leaf(ins[key], dt) for key in keys]
        z = fn(*ls)
        out = {"z": z}
        if grads:
            z.backward(ins["dz"].to(dt))
            out.update({gname: l.grad for gname, l in zip(("ga", "gb"), ls) if gname in grads})
        return out

    def scale(ins):
        f = (lambda x, y: x + y) if op == SUB else fn
        dz = ins["dz"] if "dz" in ins else torch.ones(())
        if not grads:
            ls = [ins[key].abs() for key in keys]
            return {"z": f(*ls)}
        z, gs = absrun(f, [ins[key] for key in keys], dz)
        out = {"z": z}
        out.update({gname: gg for gname, gg in zip(("ga", "gb"), gs) if gname in grads})
        return out

    add(name, words, inputs, ref, scale, exact)


def size():
    r = rng.random()
    return 1 if r < 0.2 else rng.randint(2, 6)


def rshape(rank):
    return [size() for _ in range(rank)]


for rep in range(3):
    for op in (ADD, SUB, MUL, DIV):
        o = rshape(rng.randint(1, 4))
        a = [1 if rng.random() < 0.3 else v for v in o]
        b = [1 if rng.random() < 0.3 else v for v in o[rng.randint(0, len(o)):]]
        o = list(torch.broadcast_shapes(tuple(a), tuple(b)))
        x0 = randn(*a)
        y0 = randn(*b).abs() + 0.5 if op == DIV else randn(*b)
        dz = randn(*o)
        fn = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y][op]
        tcase(f"tensor_{['add', 'sub', 'mul', 'div'][op]}_{rep}", op, 0, a, b, o, {"x": x0, "y": y0, "dz": dz}, fn, ("ga", "gb"))
    for op in (SUM, SUMK, MEAN, SUMALL, MEANALL):
        s = rshape(rng.randint(1, 4))
        k = rng.randrange(len(s))
        x0 = randn(*s)
        fn = {SUM: lambda x, k=k: x.sum(dim=k), SUMK: lambda x, k=k: x.sum(dim=k, keepdim=True),
              MEAN: lambda x, k=k: x.mean(dim=k), SUMALL: lambda x: x.sum(), MEANALL: lambda x: x.mean()}[op]
        dz = randn(*fn(x0).shape)
        tcase(f"tensor_{['sum', 'sumk', 'mean', 'sumall', 'meanall'][op - SUM]}_{rep}", op, k, s, [], [],
              {"x": x0, "dz": dz}, fn, () if op == SUMK else ("ga",))
    s = rshape(rng.randint(1, 4))
    nn_ = int(np.prod(s))
    t0 = rng.choice([d for d in range(1, nn_ + 1) if nn_ % d == 0])
    t = [t0, nn_ // t0]
    x0, dz = randn(*s), randn(*t)
    tcase(f"tensor_reshape_{rep}", RESHAPE, 0, s, t, [], {"x": x0, "dz": dz}, lambda x, t=t: x.reshape(t), ("ga",),
          exact=("z", "ga"))
    s = rshape(rng.randint(1, 4))
    x0 = randn(*s)
    tcase(f"tensor_flatten_{rep}", FLATTEN, 0, s, [], [], {"x": x0}, lambda x: x.reshape(-1), (), exact=("z",))
    s = rshape(rng.randint(2, 4))
    perm = list(range(len(s)))
    rng.shuffle(perm)
    x0 = randn(*s)
    dz = randn(*x0.permute(*perm).shape)
    tcase(f"tensor_permute_{rep}", PERMUTE, 0, s, perm, [], {"x": x0, "dz": dz}, lambda x, p=perm: x.permute(*p), ("ga",),
          exact=("z", "ga"))
    s = rshape(rng.randint(2, 4))
    i, j = rng.randrange(len(s)), rng.randrange(len(s))
    x0 = randn(*s)
    dz = randn(*x0.transpose(i, j).shape)
    tcase(f"tensor_transpose_{rep}", TRANSPOSE, i, s, [j], [], {"x": x0, "dz": dz}, lambda x, i=i, j=j: x.transpose(i, j),
          ("ga",), exact=("z", "ga"))
    s = rshape(rng.randint(1, 4))
    d = rng.randrange(len(s))
    st = rng.randint(0, s[d] - 1)
    ln = rng.randint(1, s[d] - st)
    x0 = randn(*s)
    dz = randn(*x0.narrow(d, st, ln).shape)
    tcase(f"tensor_narrow_{rep}", NARROW, d, s, [st, ln], [], {"x": x0, "dz": dz},
          lambda x, d=d, st=st, ln=ln: x.narrow(d, st, ln), ("ga",), exact=("z", "ga"))
    a = rshape(rng.randint(1, 4))
    d = rng.randrange(len(a))
    b = list(a)
    b[d] = size()
    x0, y0 = randn(*a), randn(*b)
    dz = randn(*torch.cat([x0, y0], dim=d).shape)
    tcase(f"tensor_cat_{rep}", CAT, d, a, b, [], {"x": x0, "y": y0, "dz": dz}, lambda x, y, d=d: torch.cat([x, y], dim=d),
          ("ga", "gb"), exact=("z", "ga", "gb"))
    bs, n, k, m = (size() for _ in range(4))
    x0, y0 = randn(bs, n, k), randn(bs, k, m)
    dz = randn(bs, n, m)
    tcase(f"tensor_bmm_{rep}", BMM, 0, [bs, n, k], [bs, k, m], [], {"x": x0, "y": y0, "dz": dz}, torch.bmm, ("ga", "gb"))

# shapes that must be refused where they enter (the program prints "reject")
for name, op, k, a, b, o in [("add_23_4", ADD, 0, [2, 3], [4], [2, 3]), ("reshape_23_5", RESHAPE, 0, [2, 3], [5], []),
                             ("narrow_past_end", NARROW, 0, [5], [4, 2], []), ("bmm_inner", BMM, 0, [2, 3, 4], [2, 5, 6], [])]:
    words = [100 + op, k]
    for s in (a, b, o):
        words += [len(s), *s]
    CASES.append(dict(name=f"tensor_reject_{name}", words=words, fast=None, inputs={}, out32={}, out64={}, scale={}))


def main():
    out = Path(sys.argv[1])
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: the corpus is frozen; a new version goes in a new directory, on purpose")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    runs = 0
    for c in CASES:
        arrays = {"case": np.array(c["words"], dtype="<u4")}
        if c["fast"] is not None:
            arrays["case_fast"] = np.array(c["fast"], dtype="<u4")
        for k, v in c["inputs"].items():
            v = v.detach()
            arrays[f"in/{k}"] = v.numpy().astype("<u4") if v.dtype == torch.int64 else v.to(torch.float32).numpy().astype("<f4")
        for k in c["out32"]:
            arrays[f"out32/{k}"] = c["out32"][k].contiguous().numpy().astype("<f4")
            arrays[f"out64/{k}"] = c["out64"][k].contiguous().numpy().astype("<f8")
            arrays[f"scale/{k}"] = c["scale"][k].reshape(c["out64"][k].shape).contiguous().numpy().astype("<f4")  # a size: float32 is enough
        if not c["out32"]:
            arrays["reject"] = np.array([1], dtype="<u4")
        path = out / f"{c['name']}.npz"
        np.savez_compressed(path, **arrays)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        runs += 1 + (c["fast"] is not None)
    manifest = {
        "version": VERSION,
        "policy": POLICY,
        "atol": ATOL,
        "rtol": RTOL,
        "scales": SCALES,
        "seed": SEED,
        "seed_note": SEED_NOTE,
        "made_with": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                      "machine": platform.machine()},
        "cases": len(CASES),
        "runs": runs,
        "exact": EXACT,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum((out / f).stat().st_size for f in files)
    print(f"corpus: {len(CASES)} files, {runs} runs (fast.bend's layers on the same files), {total} bytes, written to {out}")


if __name__ == "__main__":
    main()

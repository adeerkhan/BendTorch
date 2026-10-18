"""A probe of the gradients' numerics: SmolLM2's twin (reference/smol_ref.py) with one operation swapped at a time.

    reference/.venv/bin/python reference/numerics_probe.py grads|grads64 <out.bin> [op=variant ...]

Writes the loss and the gradients of the first batch in smol_ref.py's layout (float32 for grads, float64 for grads64).
Each operation runs as PyTorch runs it unless a variant is given:

    op     what it covers
    lin    the linear layers of the blocks: x·Wᵀ forward, dY·W and dYᵀ·X backward
    head   the output head (x·Eᵀ over the 49,152 classes), the same three products; also head=f, head=x, head=w:
           Bend's order in that product only (forward, dX, dW), PyTorch's in the other two; head=b64, b256, b1024,
           b4096: Bend's three products with dX summed over blocks of that many classes (dot4 a block, the blocks
           added one after another); head=split with ce=split: dX = (p/n)·E - E[y]/n, the one-hot term
           subtracted after the product instead of inside it
    attn   attention after RoPE: Q·Kᵀ, scale, causal softmax, P·V, and their backward pass
    rms    RMSNorm, forward and backward
    ce     cross-entropy, forward and backward
    silu   SiLU, forward and backward
    rope   RoPE's cos/sin table
    fork   the sums of the gradients of an input read three times (q/k/v) or twice (gate/up)

    variant  meaning
    torch    PyTorch's float32 kernels (the default)
    bend     demo/smol.bend's order of operations and formulas in float32: tensor-array's dot products with four
             partial sums (product p to sum p mod 4, combined in pairs), attention's and RMSNorm's running sums from
             +0.0, exp/log/sqrt/pow/cos/sin as C's double functions rounded to float32 (what Bend 2.0.37 emits),
             cross-entropy's balanced-tree sum, Fork's dq + (dk + dv)
    f64      the operation computed in float64 and rounded to float32 in both passes (not for rope or fork)
    exact    rope only: the table from float64 angles, rounded once to float32

blk=<B> makes every Bend product (lin=bend, head=bend) with k > B sum over blocks of B (dot4_blocks), as a block size
in tensor-array's dot would.

Every op at bend (lin, head, attn, rms, ce, silu, rope, fork) reproduces demo/smol.bend's loss bit for bit and
99.995% of its 134,515,008 gradients' bits (2026-10-18), so a variant measures what that piece does inside Bend.

grads64 ignores every variant but rope (its reference is float64 throughout).
THREADS=n sets PyTorch's thread count (1 by default). The 135M model only (smol_ref's SMOL=360m is not wired here).
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import smol_ref as S

F32 = torch.float32
V = {}  # op -> variant


def var(op):
    return V.get(op, "torch")


def f32(x):
    return torch.tensor(x, dtype=F32)


# ---- Bend's order of operations, in float32 ----

def dot4(a, b):
    """dot4_flat, or dot4_blocks when blk=<B> is given and k > B (every product of the run, blocks and head)"""
    blk = int(V.get("blk", 0))
    return dot4_blocks(a, b, blk) if blk and a.shape[1] > blk else dot4_flat(a, b)


def dot4_flat(a, b):
    """a (n x k) · b (k x m) as tensor-array's dotr: product p of each dot added to partial sum p mod 4 (from +0.0, in
    order), the sums combined as (s_r + s_r+1) + (s_r+2 + s_r+3), r = k mod 4. No fused multiply-add."""
    n, k = a.shape
    m = b.shape[1]
    kp = (k + 3) // 4 * 4
    if kp != k:
        a, b = F.pad(a, (0, kp - k)), F.pad(b, (0, 0, 0, kp - k))
    a4 = a.reshape(n, kp // 4, 4).permute(1, 2, 0).contiguous()  # (kp/4, 4, n)
    b4 = b.reshape(kp // 4, 4, m)
    s = torch.zeros(4, n, m, dtype=F32)
    for t in range(kp // 4):
        s += a4[t][:, :, None] * b4[t][:, None, :]
    r = k % 4
    return (s[r] + s[(r + 1) % 4]) + (s[(r + 2) % 4] + s[(r + 3) % 4])


def cexp(x):
    return torch.exp(x.double()).float()  # (f32)exp((double)x), as in Bend's emitted C


def run_sum(xs, dim):
    """a running sum from +0.0 along dim, in index order"""
    acc = torch.zeros_like(xs.select(dim, 0))
    for j in range(xs.shape[dim]):
        acc = acc + xs.select(dim, j)
    return acc


def tree_sum(x):
    """main.bend's tree_sum along the last axis: a[i] = a[2i] + a[2i+1], an odd last one carried, until one is left"""
    while x.shape[-1] > 1:
        m = x.shape[-1]
        y = x[..., 0:m - m % 2:2] + x[..., 1:m - m % 2:2]
        x = torch.cat((y, x[..., m - 1:]), -1) if m % 2 else y
    return x[..., 0]


def dot4_blocks(a, b, blk):
    """a · b over k in blocks of blk: each block's product as dot4, added into C one block after another from +0.0 (what
    tensor-array's matmul_acc gives, a block at a time)"""
    c = torch.zeros(a.shape[0], b.shape[1], dtype=F32)
    for i in range(0, a.shape[1], blk):
        c = c + dot4_flat(a[:, i:i + blk].contiguous(), b[i:i + blk].contiguous())
    return c


def head_split(blk):
    """the head with Bend's three products, its dX as dot4 of dZ over blocks of blk classes (blk = 0: one dot4)"""
    class Head(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(x, w)
            return dot4(x, w.t().contiguous())

        @staticmethod
        def backward(ctx, dy):
            x, w = ctx.saved_tensors
            return dot4_blocks(dy, w, blk), dot4(dy.t().contiguous(), x)
    return Head.apply


def lin_bend(parts):
    """the linear layer with Bend's products in the given parts (f: x·Wᵀ, x: dY·W, w: dYᵀ·X), PyTorch's elsewhere"""
    class LinBend(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(x, w)
            return dot4(x, w.t().contiguous()) if "f" in parts else x @ w.t()

        @staticmethod
        def backward(ctx, dy):
            x, w = ctx.saved_tensors
            return (dot4(dy, w) if "x" in parts else dy @ w,
                    dot4(dy.t().contiguous(), x) if "w" in parts else dy.t() @ x)
    return LinBend.apply


class AttnBend(torch.autograd.Function):
    """main.bend's Attention: every dot a running sum from +0.0 in index order, the softmax as psm."""

    @staticmethod
    def heads(q, k, v):
        b, g = q.shape[0] // S.T, S.H // S.KV
        return (q.reshape(b, S.T, S.H, S.HD).transpose(1, 2),
                k.reshape(b, S.T, S.KV, S.HD).transpose(1, 2).repeat_interleave(g, dim=1),
                v.reshape(b, S.T, S.KV, S.HD).transpose(1, 2).repeat_interleave(g, dim=1))

    @staticmethod
    def forward(ctx, q, k, v):
        qh, kh, vh = AttnBend.heads(q, k, v)
        vis = torch.ones(S.T, S.T, dtype=torch.bool).tril()
        s = run_sum(qh[..., :, None, :] * kh[..., None, :, :], 4) * f32(S.SCALE)
        m = s.masked_fill(~vis, float("-inf")).amax(-1, keepdim=True)
        e = cexp(s - m).masked_fill(~vis, 0.0)
        p = (e / run_sum(e, 3)[..., None]).masked_fill(~vis, 0.0)
        o = run_sum(p[..., :, :, None] * vh[..., None, :, :], 3)  # O[i,d] = sum over t of P[i,t] V[t,d]
        ctx.save_for_backward(qh, kh, vh, p)
        return o.transpose(1, 2).reshape(q.shape[0], S.H * S.HD)

    @staticmethod
    def backward(ctx, do):
        qh, kh, vh, p = ctx.saved_tensors
        b, g = qh.shape[0], S.H // S.KV
        doh = do.reshape(b, S.T, S.H, S.HD).transpose(1, 2)
        vis = torch.ones(S.T, S.T, dtype=torch.bool).tril()
        dp = run_sum(doh[..., :, None, :] * vh[..., None, :, :], 4).masked_fill(~vis, 0.0)
        s = run_sum(dp * p, 3)
        gg = ((p * (dp - s[..., None])) * f32(S.SCALE)).masked_fill(~vis, 0.0)
        dq = run_sum(gg[..., :, :, None] * kh[..., None, :, :], 3)
        dkh = run_sum(gg[..., :, :, None] * qh[..., :, None, :], 2)  # dK[j,d] = sum over i of G[i,j] Q[i,d]
        dvh = run_sum(p[..., :, :, None] * doh[..., :, None, :], 2)
        # each key/value head: its g query heads' parts summed from +0.0 in head order
        dk = run_sum(dkh.reshape(b, S.KV, g, S.T, S.HD), 2)
        dv = run_sum(dvh.reshape(b, S.KV, g, S.T, S.HD), 2)
        n = b * S.T
        return (dq.transpose(1, 2).reshape(n, S.H * S.HD), dk.transpose(1, 2).reshape(n, S.KV * S.HD),
                dv.transpose(1, 2).reshape(n, S.KV * S.HD))


class RmsBend(torch.autograd.Function):
    @staticmethod
    def inv(x):
        s = run_sum(x * x, 1)
        return 1.0 / torch.sqrt((s / f32(float(x.shape[1]))) + f32(S.EPS))

    @staticmethod
    def forward(ctx, x, g):
        ctx.save_for_backward(x, g)
        return g * (x * RmsBend.inv(x)[:, None])

    @staticmethod
    def backward(ctx, dy):
        x, g = ctx.saved_tensors
        r = RmsBend.inv(x)[:, None]
        xh, dxh = x * r, dy * g
        m = run_sum(dxh * xh, 1)[:, None] / f32(float(x.shape[1]))
        return (dxh - xh * m) * r, run_sum(dy * xh, 0)


class SiluBend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x / (1.0 + cexp(-x))

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        s = 1.0 / (1.0 + cexp(-x))
        return (dy * s) * (1.0 + x * (1.0 - s))


class CeBend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, y):
        n = z.shape[0]
        e = cexp(z - z.amax(-1, keepdim=True))
        p = e / tree_sum(e)[:, None]
        terms = torch.log(torch.clamp(p[torch.arange(n), y], min=f32(1e-12).item()).double()).float()
        loss = f32(0.0)
        for t in terms:
            loss = loss - t
        ctx.save_for_backward(p, y)
        return loss / f32(float(n))

    @staticmethod
    def backward(ctx, dl):
        p, y = ctx.saved_tensors
        n = p.shape[0]
        hit = torch.zeros_like(p)
        hit[torch.arange(n), y] = 1.0
        return (p - hit) / f32(float(n)), None


LABELS = {}  # ce=split hands the head its labels and the full dZ


class CeSplit(CeBend):
    """Bend's cross-entropy whose gradient for the head's dX leaves out the one-hot term: it passes p / n on, and the
    head (head=split) subtracts the label rows of E / n after its product"""

    @staticmethod
    def backward(ctx, dl):
        p, y = ctx.saved_tensors
        dz, _ = CeBend.backward(ctx, dl)
        LABELS.update(y=y, dz=dz, n=f32(float(p.shape[0])))
        return p / LABELS["n"], None


class HeadSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return dot4(x, w.t().contiguous())

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        return dot4(dy, w) - w[LABELS["y"]] / LABELS["n"], dot4(LABELS["dz"].t().contiguous(), x)


class Split(torch.autograd.Function):
    """x read by c consumers; its gradient summed as Fork sums it: d0 + (d1 + d2) for three, d0 + d1 for two"""

    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return tuple(x.clone() for _ in range(c))

    @staticmethod
    def backward(ctx, *ds):
        return (ds[0] + (ds[1] + ds[2]) if ctx.c == 3 else ds[0] + ds[1]), None


def in_f64(fn):
    """fn computed in float64 and rounded to float32, in the forward and the backward pass"""
    class Op(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *xs):
            ctx.xs = xs
            with torch.no_grad():
                return fn(*[x.double() if torch.is_tensor(x) and x.is_floating_point() else x for x in xs]).float()

        @staticmethod
        def backward(ctx, dy):
            with torch.enable_grad():
                xs = [x.detach().double().requires_grad_(True) if torch.is_tensor(x) and x.is_floating_point() else x
                      for x in ctx.xs]
                need = [x for x in xs if torch.is_tensor(x) and x.requires_grad]
                gs = iter(torch.autograd.grad(fn(*xs), need, dy.double()))
            return tuple(next(gs).float() if torch.is_tensor(x) and x.requires_grad else None for x in xs)
    return Op.apply


OPS = {
    "lin": {"torch": F.linear, "bend": lin_bend("fxw"), "f64": in_f64(F.linear)},
    "head": {"torch": F.linear, "bend": lin_bend("fxw"), "f64": in_f64(F.linear), "f": lin_bend("f"),
             "x": lin_bend("x"), "w": lin_bend("w"), **{f"b{k}": head_split(k) for k in (64, 256, 1024, 4096)},
             "split": HeadSplit.apply},
    "attn": {"torch": S.attention, "bend": AttnBend.apply, "f64": in_f64(S.attention)},
    "rms": {"torch": S.rmsnorm, "bend": RmsBend.apply, "f64": in_f64(S.rmsnorm)},
    "ce": {"torch": F.cross_entropy, "bend": CeBend.apply, "f64": in_f64(F.cross_entropy), "split": CeSplit.apply},
    "silu": {"torch": F.silu, "bend": SiluBend.apply, "f64": in_f64(F.silu)},
}


def op(name, f64):
    return OPS[name]["torch" if f64 else var(name)]


# ---- RoPE's table ----

def table_bend():
    """main.bend's rope_tab as Bend 2.0.37 emits it: float32 operations, pow, cos and sin as C's double functions on
    the widened argument, rounded to float32"""
    half = S.HD // 2
    cs, sn = np.zeros((S.T, half), np.float32), np.zeros((S.T, half), np.float32)
    for c in range(half):
        inv = np.float32(np.float32(1.0) / np.float32(math.pow(float(np.float32(S.BASE)),
                                                               float(np.float32(2 * c) / np.float32(2 * half)))))
        for pos in range(S.T):
            ang = float(np.float32(inv * np.float32(pos)))
            cs[pos, c], sn[pos, c] = math.cos(ang), math.sin(ang)
    return torch.from_numpy(cs), torch.from_numpy(sn)


def table_torch():
    inv = 1.0 / (S.BASE ** (torch.arange(0, S.HD, 2, dtype=torch.int64).float() / S.HD))
    freqs = (inv[:, None] @ torch.arange(S.T).float()[None, :]).T
    return freqs.cos(), freqs.sin()


def table_exact():
    ang = (1.0 / (S.BASE ** (np.arange(0, S.HD, 2) / S.HD)))[None, :] * np.arange(S.T)[:, None]
    return torch.from_numpy(np.cos(ang)).float(), torch.from_numpy(np.sin(ang)).float()


def rope_with(cs, sn):
    """smol_ref.rope over a given (T x HD/2) table, with the same operations in the same order"""
    cos, sin = torch.cat((cs, cs), dim=-1), torch.cat((sn, sn), dim=-1)

    def rope(x, heads):
        pos = torch.arange(x.shape[0]) % S.T
        xh = x.reshape(x.shape[0], heads, S.HD)
        return (xh * cos[pos][:, None, :] + S.rotate_half(xh) * sin[pos][:, None, :]).reshape(x.shape)
    return rope


# ---- the model: smol_ref.logits with the operations above ----

def logits(p, x, rope, f64):
    lin, rms, attn, silu = (op(k, f64) for k in ("lin", "rms", "attn", "silu"))
    split = Split.apply if var("fork") == "bend" and not f64 else (lambda x, c: (x,) * c)
    emb = p["model.embed_tokens.weight"]
    h = F.embedding(x, emb)
    for layer in range(S.LAYERS):
        w = lambda k: p[f"model.layers.{layer}.{k}.weight"]
        nq, nk, nv = split(rms(h, w("input_layernorm")), 3)
        a = attn(rope(lin(nq, w("self_attn.q_proj")), S.H), rope(lin(nk, w("self_attn.k_proj")), S.KV),
                 lin(nv, w("self_attn.v_proj")))
        h = h + lin(a, w("self_attn.o_proj"))
        ng, nu = split(rms(h, w("post_attention_layernorm")), 2)
        h = h + lin(silu(lin(ng, w("mlp.gate_proj"))) * lin(nu, w("mlp.up_proj")), w("mlp.down_proj"))
    return op("head", f64)(rms(h, p["model.norm.weight"]), emb)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    if len(sys.argv) < 3 or sys.argv[1] not in ("grads", "grads64") or os.environ.get("SMOL", "135m") != "135m":
        sys.exit(__doc__)
    f64, out = sys.argv[1] == "grads64", sys.argv[2]
    V.update(a.split("=") for a in sys.argv[3:])
    for k, v in V.items():
        ok = ("torch", "bend", "exact") if k == "rope" else ("torch", "bend") if k == "fork" else tuple(OPS.get(k, ()))
        if k == "blk" and v.isdigit():
            continue
        if k not in ("rope", "fork", *OPS) or v not in ok:
            sys.exit(f"unknown variant {k}={v}")
    rope = rope_with(*{"torch": table_torch, "bend": table_bend, "exact": table_exact}[var("rope")]())
    p = S.load()
    if f64:
        p = {k: v.detach().double().requires_grad_(True) for k, v in p.items()}
    x, y = S.batch(0)
    loss = op("ce", f64)(logits(p, x, rope, f64), y)
    loss.backward()
    print("loss", repr(loss.item()), flush=True)
    with open(out, "wb") as f:
        for name, _, off in S.layout():
            assert f.tell() == off * (8 if f64 else 4)
            f.write(p[name].grad.numpy().astype("<f8" if f64 else "<f4").tobytes())


if __name__ == "__main__":
    main()

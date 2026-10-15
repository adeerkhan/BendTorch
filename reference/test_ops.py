"""demo/ops_test.bend against torch.autograd: every layer on its own, forward and backward.

For each seed, random inputs (with zeros, negative zeros and saturating values mixed in) go to the Bend program and to
PyTorch's own functions; the output, dX and every parameter gradient must agree within the shared tolerance (oracle.py),
and the results in EXACT (copies, selections, one or two F32 operations in PyTorch's order) bit for bit.

    reference/.venv/bin/python reference/test_ops.py .scratch/ops_test
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oracle import bits_equal, close, run, write_f32, write_u8

EPS = 1e-5
EDGE_OPS = {"relu", "sigmoid", "tanh", "silu", "gelu"}
# The exact tier: results that are a copy, a selection or one or two F32 operations in PyTorch's order, so they must be
# the same bits, NaN included (oracle.py's tolerance still applies to everything else)
EXACT = {"relu.y", "relu.dx", "embedding.y", "mse.dy", "nll.dy", "dropout0.y", "dropout0.dx", "dropout3.y", "dropout3.dx", "dropout4.y", "dropout4.dx"}


def rmsnorm(x, g):
    var = x.pow(2).mean(-1, keepdim=True)
    return g * (x * torch.rsqrt(var + EPS))


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope(x, t=4, heads=2, hd=4, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    pos = (torch.arange(x.shape[0]) % t).float()
    freqs = (inv[:, None] @ pos[None, :]).T  # transformers' outer product
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    xh = x.reshape(x.shape[0], heads, hd)
    return (xh * cos + rotate_half(xh) * sin).reshape(x.shape)


def attention(q, k, v, b=2, t=4, h=4, kv=2, hd=4, scale=0.5):
    q = q.reshape(b, t, h, hd).transpose(1, 2)
    k = k.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    v = v.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))
    o = s.softmax(-1) @ v
    return o.transpose(1, 2).reshape(b * t, h * hd)


# every IEEE class a layer can meet: NaN, both infinities, both zeros, subnormals, values where exp overflows or
# underflows in F32 (|x| > 88.7), and ordinary values
EDGE = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1e-40, -1e-40, 1.5e-45, 89.0, -89.0, 104.0, -104.0,
        1e6, -1e6, 20.0, -20.0, 1e-7, -1e-7, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 7.0, -7.0, 0.1, -0.1, 2.0, -2.0]


def mix32(x):
    x = np.asarray(x, dtype=np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(2146121005)
    x ^= x >> np.uint32(15)
    x *= np.uint32(2221713035)
    x ^= x >> np.uint32(16)
    return x


def dropout_factors(n, seed, step, p):
    """BendTorch's dropout mask: number i is kept (times 1/(1-p)) when ((mix32(key + i) >> 8) / 2^24) >= p."""
    key = mix32(np.uint32(seed) ^ mix32(np.uint32(step)))
    with np.errstate(over="ignore"):
        h = mix32(key + np.arange(n, dtype=np.uint32))
    u = (h >> np.uint32(8)).astype(np.float32) / np.float32(16777216.0)
    scale = np.float32(1.0) / (np.float32(1.0) - np.float32(p))
    return torch.from_numpy(np.where(u < np.float32(p), np.float32(0.0), scale).astype(np.float32))


def spiced(g, shape):
    """Normal values with a few exact zeros, a negative zero and large values (saturation) in fixed places."""
    x = torch.randn(*shape, generator=g)
    flat = x.view(-1)
    flat[0], flat[1], flat[2], flat[3] = 0.0, -0.0, 12.0, -12.0
    return x


def case(binary: str, seed: int, failures: list[str], edge: bool = False) -> tuple[float, int, int]:
    g = torch.Generator().manual_seed(seed)
    files: dict[str, torch.Tensor] = {}
    ref: dict[str, torch.Tensor] = {}

    def leaf(name, t):
        files[name] = t
        return t.clone().requires_grad_(True)

    def unary(tag, fn, x_shape=(5, 6), **params):
        x0 = torch.tensor(EDGE).reshape(5, 6) if edge and tag in EDGE_OPS else spiced(g, x_shape)
        x = leaf(f"{tag}.x", x0)
        ps = {k: leaf(f"{tag}.{k}", v) for k, v in params.items()}
        y = fn(x, **ps)
        dy = torch.randn(*y.shape, generator=g)
        files[f"{tag}.dy"] = dy
        y.backward(dy)
        ref[f"{tag}.y"], ref[f"{tag}.dx"] = y.detach(), x.grad
        for k, p in ps.items():
            ref[f"{tag}.g.{k}" if len(ps) > 1 else f"{tag}.g"] = p.grad

    unary("relu", torch.relu)
    unary("sigmoid", torch.sigmoid)
    unary("tanh", torch.tanh)
    unary("silu", F.silu)
    unary("gelu", lambda x: F.gelu(x, approximate="tanh"))
    unary("softmax", lambda x: F.softmax(x, -1))
    unary("logsoftmax", lambda x: F.log_softmax(x, -1))
    unary("rmsnorm", rmsnorm, g=torch.randn(6, generator=g))
    unary("layernorm", lambda x, g, b: F.layer_norm(x, (6,), g, b, EPS),
          g=torch.randn(6, generator=g), b=torch.randn(6, generator=g))
    unary("linear_nb", lambda x, w: F.linear(x, w), w=torch.randn(4, 6, generator=g) * 0.5)
    unary("swiglu", lambda x, wg, wu, wd: F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd),
          wg=torch.randn(8, 6, generator=g) * 0.5, wu=torch.randn(8, 6, generator=g) * 0.5,
          wd=torch.randn(6, 8, generator=g) * 0.5)
    unary("residual", lambda x, g: x + rmsnorm(x, g), g=torch.randn(6, generator=g))
    unary("rope", rope, x_shape=(8, 8))
    for step in (0, 3, 4):
        k = dropout_factors(30, 7, step, 0.25).reshape(5, 6)
        unary(f"dropout{step}", lambda x, k=k: x * k)

    q, k, v = leaf("attn.q", torch.randn(8, 16, generator=g)), leaf("attn.k", torch.randn(8, 8, generator=g)), \
        leaf("attn.v", torch.randn(8, 8, generator=g))
    y = attention(q, k, v)
    dy = torch.randn(8, 16, generator=g)
    files["attn.dy"] = dy
    y.backward(dy)
    ref.update({"attn.y": y.detach(), "attn.dq": q.grad, "attn.dk": k.grad, "attn.dv": v.grad})

    e = leaf("embedding.e", torch.randn(7, 4, generator=g))
    ids = torch.tensor([3, 0, 3, 6, 3])  # repeated ids: their gradients add up
    y = F.embedding(ids, e)
    dy = torch.randn(5, 4, generator=g)
    files["embedding.dy"] = dy
    y.backward(dy)
    ref.update({"embedding.y": y.detach(), "embedding.g": e.grad})

    labels = torch.randint(0, 6, (5,), generator=g)
    for tag, fn, x0, t in [("mse", F.mse_loss, "y", torch.randn(5, 6, generator=g)),
                           ("nll", F.nll_loss, "x", labels), ("ce", F.cross_entropy, "x", labels)]:
        x = leaf(f"{tag}.{x0}", torch.randn(5, 6, generator=g))
        if tag == "mse":
            files["mse.t"] = t
        loss = fn(x, t)
        loss.backward()
        ref[f"{tag}.loss"], ref[f"{tag}.dy"] = loss.detach().reshape(1), x.grad

    with tempfile.TemporaryDirectory() as d:
        for name, t in files.items():
            write_f32(Path(d) / f"{name}.bin", t)
        write_u8(Path(d) / "embedding.ids.bin", ids.tolist())
        write_u8(Path(d) / "nll.labels.bin", labels.tolist())
        got = run(binary, d)

    worst, same, total = 0.0, 0, 0
    for name, r in ref.items():
        bend_name = name
        if bend_name not in got:
            failures.append(f"seed {seed}: the Bend program printed no {bend_name}")
            continue
        worst = max(worst, close(f"seed {seed} {name}", got[bend_name], r, failures))
        if name in EXACT:
            s_, t_ = bits_equal(got[bend_name], r)
            if s_ != t_:
                failures.append(f"seed {seed} {name}: {t_ - s_} of {t_} numbers not bit-identical (exact tier)")
        s, t = bits_equal(got[bend_name], r)
        same, total = same + s, total + t
    return worst, same, total


def main():
    binary = sys.argv[1]
    failures: list[str] = []
    worst, same, total = 0.0, 0, 0
    for seed in range(10):
        w, s, t = case(binary, seed, failures)
        worst, same, total = max(worst, w), same + s, total + t
    edge_failures: list[str] = []
    case(binary, 100, edge_failures, edge=True)
    print(f"ops: 10 seeds, {total} numbers, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance")
    print(f"ops: IEEE edge cases (NaN, infinities, signed zeros, subnormals, overflow) on {sorted(EDGE_OPS)}: "
          f"{'PASS' if not edge_failures else str(len(edge_failures)) + ' differences'}")
    failures += edge_failures
    if failures:
        print("\n".join(failures[:30]))
        sys.exit(1)
    print("ops: PASS")


if __name__ == "__main__":
    main()

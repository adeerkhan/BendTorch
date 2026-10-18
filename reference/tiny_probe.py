"""Why the tiny Llama's float32 training drifts from float64: chaos, or an operation less accurate than PyTorch's?

    reference/.venv/bin/python reference/tiny_probe.py drift <bend-binary> [steps]   # step 1 and 2: Bend, PyTorch
        float32 variants and float64 after <steps> AdamW steps (default 100), each one's distance from float64
    reference/.venv/bin/python reference/tiny_probe.py local <bend-binary> [k ...]   # step 3: one step from the float64
        trajectory's state at step k, in float32 by PyTorch, by the twin with Bend's old optimizer scalars and by Bend
        (through a checkpoint), each against one float64 step from the same float32-rounded state
    reference/.venv/bin/python reference/tiny_probe.py spread <steps>   # each flavor from the initial weights and from
        four starts one weight 1 ulp away: the spread chaos alone gives
    reference/.venv/bin/python reference/tiny_probe.py ablate <steps>   # Bend's scalars one at a time in the twin

The twin is reference/tiny_ref.py; its AdamW is written out here (`adamw`), checked against torch.optim.AdamW's bits.
The optimizer's scalars come in three flavors:

    torch  torch.optim.adam._single_tensor_adam: 1 - beta2, beta^t, the bias corrections and lr * wd in float64,
           each rounded to float32 where a kernel takes it
    bend   torch/main.bend's Opt.at and adam_* before 2026-10-18: b1, b2 rounded to float32 first, 1 - b2 and
           1 - b1 in float32, b1^t and b2^t running float32 products, lr / (1 - b1^t), sqrt(1 - b2^t) and 1 - lr * wd
           in float32
    fix    what main.bend does since 2026-10-18, float32 only: the complements 1 - b1 and 1 - b2 given as float32
           constants, and 1 - b^t carried as q' = q + (1 - b) (1 - q), so no 1 - (a number near 1) is ever taken

Distances are the root mean square over all 21 final tensors (37,568 numbers) and over the losses.
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import tiny_ref as R  # noqa: E402

F32, F64 = torch.float32, torch.float64
NAMES = [n for n, _ in R.shapes()]


def f32(x):
    return float(np.float32(x))


def scalars(t, dtype, flavor):
    """The step-t scalars (t = 1 first): decay factor, 1 - b1 (lerp weight), b2, 1 - b2, step size, sqrt(1 - b2^t)."""
    lr, (b1, b2), wd = R.LR, R.BETAS, R.WD
    ts = dict(decay=1 - lr * wd, w1=1 - b1, b2=b2, w2=1 - b2, step=lr / (1 - b1 ** t), bc2=(1 - b2 ** t) ** 0.5)
    if dtype == F64 or flavor == "torch":
        return ts
    b1f, b2f, lrf = np.float32(b1), np.float32(b2), np.float32(lr)
    one = np.float32(1.0)
    if flavor == "fix":
        # float32 only, from the complements c1 = 1 - b1 and c2 = 1 - b2 given as float32 constants (0.1, 0.001):
        # q = 1 - b^t carried as q' = q + c (1 - q), so no 1 - (number near 1) is ever taken
        c1, c2 = np.float32(1 - b1), np.float32(1 - b2)
        q1 = q2 = np.float32(0.0)
        for _ in range(t):
            q1 = np.float32(q1 + np.float32(c1 * np.float32(one - q1)))
            q2 = np.float32(q2 + np.float32(c2 * np.float32(one - q2)))
        return dict(decay=float(one - np.float32(lrf * np.float32(wd))), w1=float(c1), b2=float(b2f), w2=float(c2),
                    step=float(np.float32(lrf / q1)), bc2=float(np.sqrt(q2, dtype=np.float32)))
    p1 = p2 = np.float32(1.0)
    for _ in range(t):
        p1, p2 = np.float32(p1 * b1f), np.float32(p2 * b2f)
    bs = dict(decay=float(one - np.float32(lrf * np.float32(wd))), w1=float(one - b1f), b2=float(b2f),
              w2=float(one - b2f), step=float(np.float32(lrf / np.float32(one - p1))),
              bc2=float(np.sqrt(np.float32(one - p2), dtype=np.float32)))
    if flavor == "bend":
        return bs
    # "bend:a,b": only the scalars named take Bend's values; "torch:a,b": all but those
    kind, names = flavor.split(":")
    names = names.split(",")
    return {k: (bs[k] if (k in names) == (kind == "bend") else ts[k]) for k in ts}


def adamw(p, m, v, t, dtype, flavor):
    s = scalars(t, dtype, flavor)
    with torch.no_grad():
        for k in NAMES:
            w, g = p[k], p[k].grad
            w.mul_(s["decay"])
            m[k].lerp_(g, s["w1"])
            v[k].mul_(s["b2"]).addcmul_(g, g, value=s["w2"])
            w.addcdiv_(m[k], (v[k].sqrt() / s["bc2"]).add_(R.ADAM_EPS), value=-s["step"])
            w.grad = None


def run(steps, dtype=F32, flavor="torch", start=None, first=0, perturb=None, threads=1, optim="manual"):
    """AdamW steps from the initial weights (or `start`: p, m, v as float64 arrays, at step `first`).
    Returns the losses and the final p, m, v as float64 numpy arrays."""
    torch.set_num_threads(threads)
    if start is None:
        p = {k: v.detach().to(dtype) for k, v in R.load().items()}
        m = {k: torch.zeros_like(v) for k, v in p.items()}
        v = {k: torch.zeros_like(x) for k, x in p.items()}
    else:
        p = {k: torch.from_numpy(start[0][k].copy()).to(dtype) for k in NAMES}
        m = {k: torch.from_numpy(start[1][k].copy()).to(dtype) for k in NAMES}
        v = {k: torch.from_numpy(start[2][k].copy()).to(dtype) for k in NAMES}
    if perturb is not None:  # (name, index, delta): one weight moved by delta, an equally valid starting point
        name, i, d = perturb
        p[name].view(-1)[i] += d
    for x in p.values():
        x.requires_grad_(True)
    opt = None
    if optim == "torch":
        opt = torch.optim.AdamW(p.values(), lr=R.LR, betas=R.BETAS, eps=R.ADAM_EPS, weight_decay=R.WD)
    losses = []
    for i in range(steps):
        loss = R.forward(p, *R.batch(first + i))
        loss.backward()
        losses.append(loss.item())
        if opt is None:
            adamw(p, m, v, first + i + 1, dtype, flavor)
        else:
            opt.step()
            opt.zero_grad()
    if opt is not None:
        st = opt.state
        m = {k: st[p[k]]["exp_avg"] for k in NAMES}
        v = {k: st[p[k]]["exp_avg_sq"] for k in NAMES}
    out = [{k: d[k].detach().double().numpy().copy() for k in NAMES} for d in (p, m, v)]
    return np.array(losses), *out


def bend_parse(text):
    loss = [np.uint32(int(line.split()[1])).view(np.float32) for line in text.splitlines() if line.startswith("loss")]
    p = {}
    for line in text.splitlines():
        if line.startswith("final."):
            parts = line.split()
            p[parts[0][6:]] = np.array([int(w) for w in parts[1:]], dtype=np.uint32).view(np.float32).astype(np.float64)
    return np.array(loss[1:], dtype=np.float64), p


def bend_run(binary, *args, threads=None):
    """threads=1 pins to CPU 3 (a P-core); otherwise --threads n unpinned."""
    if threads is None:
        cmd = [binary, *args]
    elif threads == 1:
        cmd = ["taskset", "-c", "3", binary, "--threads", "1", *args]
    else:
        cmd = [binary, "--threads", str(threads), *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def flat(p):
    return np.concatenate([p[k].reshape(-1) for k in NAMES])


def dist(a, b):
    return float(np.sqrt(np.mean((flat(a) - flat(b)) ** 2)))


def ldist(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def drift(binary, steps):
    l64, p64, _, _ = run(steps, F64)
    rows = []

    def add(label, losses, p):
        rows.append((label, ldist(losses, l64), dist(p, p64), losses, p))

    lt, pt, _, _ = run(steps, optim="torch")
    add("PyTorch float32, torch.optim.AdamW, 1 thread", lt, pt)
    lm, pm, _, _ = run(steps)
    same = all(np.array_equal(pm[k], pt[k]) for k in NAMES) and np.array_equal(lm, lt)
    add(f"twin's AdamW, torch scalars ({'same bits as' if same else 'DIFFERS from'} torch.optim)", lm, pm)
    add("PyTorch float32, 16 threads", *run(steps, threads=16, optim="torch")[:2])
    for i, (name, idx) in enumerate([("emb", 0), ("b0.wq", 17), ("b1.wd", 100), ("head", 5)]):
        x = float(R.load()[name].detach().reshape(-1)[idx])
        ulp = float(np.spacing(np.float32(abs(x))))
        add(f"PyTorch float32, {name}[{idx}] + 1 ulp", *run(steps, perturb=(name, idx, ulp), optim="torch")[:2])
    lb, pb, _, _ = run(steps, flavor="bend")
    add("twin with Bend's old optimizer scalars", lb, pb)
    for th in (1, 16):
        b = bend_parse(bend_run(binary, "train", str(steps), threads=th))
        add(f"Bend float32, {th} thread(s)", *b)
    ref = rows[0][2]
    print(f"{steps} AdamW steps; distance from float64 (rms over 37,568 weights; over the {steps} losses), "
          f"and its ratio to PyTorch float32's")
    for label, dl, dp, _, _ in rows:
        print(f"  {label:58s} weights {dp:.3e} ({dp / ref:6.2f}x)  losses {dl:.3e}")
    print(f"  Bend vs twin with Bend's old scalars: {dist(rows[-1][4], pb):.3e}; Bend vs PyTorch float32: "
          f"{dist(rows[-1][4], pt):.3e}; Bend 1 vs 16 CPUs: {dist(rows[-1][4], rows[-2][4]):.3e}")


def write_ckpt(path, p, m, v, k):
    """torch/main.bend's Train.save layout: the parameters, m, v (float32, in NAMES order), then b1^k and b2^k as
    Bend's running float32 products."""
    b1f, b2f = np.float32(R.BETAS[0]), np.float32(R.BETAS[1])
    p1 = p2 = np.float32(1.0)
    for _ in range(k):
        p1, p2 = np.float32(p1 * b1f), np.float32(p2 * b2f)
    xs = np.concatenate([flat(d) for d in (p, m, v)] + [np.array([p1, p2])]).astype("<f4")
    with open(path, "wb") as f:
        f.write(xs.tobytes())


def local(binary, ks):
    """rms: the distance of the new weights from the float64 step's; bias: the mean of that error along the float64
    update's direction (positive: the step overshoots), both in units of 1e-9."""
    print("One AdamW step from the float64 trajectory's state at step k (rounded to float32), against one float64 "
          "step from that same state; rms and bias in units of 1e-9")
    print(f"  {'k':>4s} {'torch':>15s} {'twin+Bend scalars':>18s} {'Bend':>15s} {'fix':>15s}   Bend/torch rms")
    last, p, m, v = 0, None, None, None
    for k in ks:
        if k > last or p is None:
            _, p, m, v = run(k - last, F64, start=None if p is None else (p, m, v), first=last)
            last = k
        start = tuple({n: d[n].astype(np.float32).astype(np.float64) for n in NAMES} for d in (p, m, v))
        _, q64, _, _ = run(1, F64, start=start, first=k)
        _, qt, _, _ = run(1, start=start, first=k)
        _, qb, _, _ = run(1, flavor="bend", start=start, first=k)
        _, qf, _, _ = run(1, flavor="fix", start=start, first=k)
        with tempfile.TemporaryDirectory() as d:
            write_ckpt(f"{d}/ck.bin", *start, k)
            _, qbend = bend_parse(bend_run(binary, "resume", f"{d}/ck.bin", str(k), "1"))
        direction = np.sign(flat(q64) - flat(start[0]))

        def cell(q):
            e = flat(q) - flat(q64)
            return f"{np.sqrt(np.mean(e ** 2)) * 1e9:6.2f} {np.mean(e * direction) * 1e9:+7.3f}"
        print(f"  {k:4d} {cell(qt):>15s} {cell(qb):>18s} {cell(qbend):>15s} {cell(qf):>15s}   "
              f"{dist(qbend, q64) / dist(qt, q64):5.2f}x   (Bend vs twin+Bend scalars {dist(qbend, qb) * 1e9:.2f})")


def ablate(steps):
    _, p64, _, _ = run(steps, F64)
    ref = dist(run(steps)[1], p64)
    print(f"{steps} steps; the twin's distance from float64 with some optimizer scalars at Bend's values "
          f"(ratio to all at PyTorch's)")
    flavors = ["bend"] + [f"bend:{k}" for k in ("decay", "w1", "b2", "w2", "step", "bc2")] + ["bend:w2,bc2"] + \
              [f"torch:{k}" for k in ("w2", "bc2")] + ["torch:w2,bc2", "fix"]
    for fl in flavors:
        d = dist(run(steps, flavor=fl)[1], p64)
        print(f"  {fl:16s} {d:.3e} ({d / ref:6.2f}x)")


PERTURB = [("emb", 0), ("b0.wq", 17), ("b1.wd", 100), ("head", 5)]


def spread(steps):
    """Each optimizer flavor from the initial weights and from four starts one weight 1 ulp away: the range of
    distances from float64 in units of PyTorch float32's (unperturbed) distance."""
    _, p64, _, _ = run(steps, F64)
    ref = dist(run(steps)[1], p64)
    init = R.load()
    print(f"{steps} steps; distance from float64 over PyTorch float32's ({ref:.3e}): unperturbed, then min-max over "
          f"4 one-ulp perturbations")
    for fl in ("torch", "bend", "fix"):
        base = dist(run(steps, flavor=fl)[1], p64) / ref
        ds = []
        for name, i in PERTURB:
            ulp = float(np.spacing(np.float32(abs(float(init[name].detach().reshape(-1)[i])))))
            ds.append(dist(run(steps, flavor=fl, perturb=(name, i, ulp))[1], p64) / ref)
        print(f"  {fl:6s} {base:7.2f}x   {min(ds):7.2f}x - {max(ds):7.2f}x")


def main():
    torch.set_num_threads(1)
    mode, binary = sys.argv[1], sys.argv[2]
    if mode == "spread":
        spread(int(binary))
    elif mode == "ablate":
        ablate(int(binary))
    elif mode == "drift":
        drift(binary, int(sys.argv[3]) if len(sys.argv) > 3 else 100)
    elif mode == "local":
        local(binary, [int(k) for k in sys.argv[3:]] or [0, 20, 50, 80])


if __name__ == "__main__":
    main()

"""demo/mlp_test.bend against PyTorch: Linear(7, 6) -> ReLU -> Linear(6, 3), batch 5, cross-entropy.

For each seed: the logits, the loss, the gradient of the input and of every parameter after one backward pass, the
parameter gradients after two backward passes into the same gradient (accumulation), then
three steps of torch.optim.SGD, SGD with momentum, Adam, Adam with weight decay and AdamW, each from the same initial
weights: the loss before every step and the final weights. Then AdamW with its moments stored in BF16 (T.CAdamWBF16:
the first moment, or both), against AdamW16 below.

    reference/.venv/bin/python reference/test_mlp.py .scratch/mlp_test
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from oracle import close, bits_equal, run, write_f32, write_u8

class AdamW16(torch.optim.Optimizer):
    """torch.optim.AdamW's single-tensor step (decoupled decay, lerp, addcmul, addcdiv) with each moment rounded to
    BF16 right after its update and read back rounded, as T.CAdamWBF16 stores it (v too when v16)."""

    def __init__(self, params, lr, betas, eps, weight_decay, v16):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.v16 = v16

    @torch.no_grad()
    def step(self):
        bf = lambda t: t.to(torch.bfloat16).to(t.dtype)
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                g = p.grad
                p.mul_(1 - lr * wd)
                st["m"] = bf(st["m"].lerp(g, 1 - b1))
                v = st["v"].mul(b2).addcmul(g, g, value=1 - b2)
                st["v"] = bf(v) if self.v16 else v
                bc2 = (1 - b2 ** st["t"]) ** 0.5
                p.addcdiv_(st["m"], st["v"].sqrt().div(bc2).add(eps), value=-lr / (1 - b1 ** st["t"]))


OPTS = {
    "sgd": lambda ps: torch.optim.SGD(ps, lr=0.1),
    "momentum": lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9),
    "adam": lambda ps: torch.optim.Adam(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8),
    "adam_l2": lambda ps: torch.optim.Adam(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
    "adamw": lambda ps: torch.optim.AdamW(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
    "adamw16m": lambda ps: AdamW16(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, v16=False),
    "adamw16mv": lambda ps: AdamW16(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, v16=True),
}


def model(init):
    m = torch.nn.Sequential(torch.nn.Linear(7, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
    with torch.no_grad():
        for p, v in zip(m.parameters(), init):
            p.copy_(v)
    return m


def case(binary: str, seed: int, failures: list[str]) -> tuple[float, int, int]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(5, 7, generator=g)
    y = torch.randint(0, 3, (5,), generator=g)
    init = [torch.randn(6, 7, generator=g) * 0.5, torch.randn(6, generator=g) * 0.1,
            torch.randn(3, 6, generator=g) * 0.5, torch.randn(3, generator=g) * 0.1]
    if seed % 3 == 0:  # a row of zeros and negative inputs: relu's dead units and its zero gradient
        x[0] = 0.0
        x[1] = -x[1].abs()
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        for name, t in zip(["w1", "b1", "w2", "b2"], init):
            write_f32(dp / f"{name}.bin", t)
        write_f32(dp / "x.bin", x)
        write_u8(dp / "y.bin", y.tolist())
        got = run(binary, d)

    worst, same, total = 0.0, 0, 0

    def check(name, bend, ref):
        nonlocal worst, same, total
        worst = max(worst, close(f"seed {seed} {name}", bend, ref, failures))
        s, t = bits_equal(bend, ref)
        same, total = same + s, total + t

    m = model(init)
    xr = x.clone().requires_grad_(True)
    z = m(xr)
    loss = F.cross_entropy(z, y)
    loss.backward()
    check("logits", got["logits"], z)
    check("loss", got["loss"], loss.reshape(1))
    check("dx", got["dx"], xr.grad)
    for name, p in zip(["w1", "b1", "w2", "b2"], m.parameters()):
        check(f"grad.{name}", got[f"grad.{name}"], p.grad)

    m = model(init)
    for _ in range(2):  # two backward passes without zero_grad: the gradients add up
        F.cross_entropy(m(x), y).backward()
    for name, p in zip(["w1", "b1", "w2", "b2"], m.parameters()):
        check(f"acc.{name}", got[f"acc.{name}"], p.grad)

    for tag, make in OPTS.items():
        m = model(init)
        opt = make(m.parameters())
        losses = []
        for _ in range(3):
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        bend_losses = got[f"{tag}.loss"].reshape(-1)[1:]  # the first line is the 0 before any step
        check(f"{tag}.loss", bend_losses, torch.tensor(losses))
        for name, p in zip(["w1", "b1", "w2", "b2"], m.parameters()):
            check(f"{tag}.{name}", got[f"{tag}.{name}"], p.detach())
    return worst, same, total


def main():
    binary = sys.argv[1]
    failures: list[str] = []
    worst, same, total = 0.0, 0, 0
    for seed in range(12):
        w, s, t = case(binary, seed, failures)
        worst, same, total = max(worst, w), same + s, total + t
    print(f"mlp: 12 seeds, {total} numbers, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance")
    if failures:
        print("\n".join(failures[:20]))
        sys.exit(1)
    print("mlp: PASS")


if __name__ == "__main__":
    main()

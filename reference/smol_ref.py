"""The PyTorch twin of demo/smol.bend: SmolLM2-135M fine-tuned on Tiny Shakespeare, in plain torch.

    <llama.bend venv>/bin/python reference/smol_ref.py --prep <hf dir>  # data/smol135/: init.bin and batches.bin
    reference/.venv/bin/python reference/smol_ref.py loss               # the loss on the first batch
    reference/.venv/bin/python reference/smol_ref.py train <steps>      # 1 + steps losses (bits), then train_ms
    reference/.venv/bin/python reference/smol_ref.py grads <out.bin>    # the loss, then every gradient into out.bin
    reference/.venv/bin/python reference/smol_ref.py grads64 <out.bin>  # the same in float64 (8 bytes a number)
    THREADS=n sets PyTorch's thread count (1 by default); SMOL=360m runs SmolLM2-360M instead (data/smol360/)

Architecture (SmolLM2-135M, a Llama): token embedding (49152 x 576); 30 blocks of [RMSNorm, attention with 9 query
heads and 3 key/value heads of 64, RoPE base 100000 on the two halves of each head (rotate_half), causal, scale 0.125]
and [RMSNorm, SwiGLU 576 -> 1536 -> 576], each with a residual; RMSNorm (eps 1e-5 everywhere); the output head tied to
the embedding matrix; cross-entropy, the mean over 2 sequences of 128 positions. AdamW (lr 1e-4, betas 0.9/0.999,
eps 1e-8, weight decay 0.01) on every parameter, as in the Bend program.

The weights are the pinned checkpoint (HuggingFaceTB/SmolLM2-135M, revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2),
bfloat16 widened to F32 exactly, in reference/smol_layout.txt's order. Only --prep needs safetensors and tokenizers
(scripts/smol_data.sh runs it with llama.bend's venv); the other modes read data/smol135/ with torch and numpy alone.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# the two models: width, query heads, key/value heads, feed-forward, layers, data folder, layout file
MODELS = {"135m": (576, 9, 3, 1536, 30, "data/smol135", "smol_layout.txt"),
          "360m": (960, 15, 5, 2560, 32, "data/smol360", "smol360_layout.txt")}
SMOL = os.environ.get("SMOL", "135m")
DIM, H, KV, FF, LAYERS, _data, _layout = MODELS[SMOL]
D = Path(_data)
LAYOUT = Path(__file__).with_name(_layout)
V, HD, T, B = 49152, 64, 128, 2
EPS, BASE, SCALE = 1e-5, 100000.0, 0.125  # 1 / sqrt(HD)
STEPS = 64
LR, BETAS, ADAM_EPS, WD = 1e-4, (0.9, 0.999), 1e-8, 0.01
TOTAL = int(LAYOUT.read_text().splitlines()[-1].split("\t")[1])  # F32 numbers in init.bin (the layout's total)


def layout():
    """(name, shape, element offset) for every parameter, in the Bend program's dump order."""
    out = []
    for line in LAYOUT.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, shape, off = line.split("\t")
        out.append((name, tuple(int(s) for s in shape.split("x")), int(off)))
    return out


def prep(hf):
    """Writes init.bin from the checkpoint and batches.bin from Tiny Shakespeare, then reads init.bin back."""
    from safetensors import safe_open
    from tokenizers import Tokenizer

    D.mkdir(parents=True, exist_ok=True)
    lay = layout()
    with safe_open(f"{hf}/model.safetensors", "pt") as f, open(D / "init.bin", "wb") as out:
        assert "lm_head.weight" not in f.keys()  # tied: the head is the embedding
        assert len(list(f.keys())) == len(lay)
        for name, shape, off in lay:
            t = f.get_tensor(name)
            assert tuple(t.shape) == shape, (name, t.shape, shape)
            assert out.tell() == off * 4, (name, out.tell(), off)
            out.write(t.to(torch.float32).contiguous().numpy().astype("<f4").tobytes())
        assert out.tell() == TOTAL * 4, out.tell()
    with safe_open(f"{hf}/model.safetensors", "pt") as f:  # every tensor where the layout says, the same numbers
        for name, shape, off in lay:
            n = int(np.prod(shape))
            got = np.fromfile(D / "init.bin", dtype="<f4", count=n, offset=off * 4)
            assert np.array_equal(got.view(np.uint32), f.get_tensor(name).to(torch.float32).reshape(-1).numpy()
                                  .view(np.uint32)), name
    tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
    ids = np.array(tok.encode(Path("data/tiny/input.txt").read_text(), add_special_tokens=False).ids, dtype=np.int64)
    assert ids.max() < V
    g = torch.Generator().manual_seed(0)
    offs = torch.randint(0, len(ids) - T - 1, (STEPS, B), generator=g).numpy()
    batches = np.stack([np.stack([ids[o:o + T + 1] for o in row]) for row in offs])  # STEPS x B x (T+1)
    (D / "batches.bin").write_bytes(batches.astype("<u4").tobytes())
    print(f"init.bin: {len(lay)} tensors, {TOTAL} F32; batches.bin: {STEPS} steps of {B} x {T + 1} ids from "
          f"{len(ids)} tokens")


def load():
    raw = np.fromfile(D / "init.bin", dtype="<f4")
    assert raw.size == TOTAL, raw.size
    return {name: torch.from_numpy(raw[off:off + int(np.prod(shape))].reshape(shape).copy()).requires_grad_(True)
            for name, shape, off in layout()}


def batch(step):
    raw = np.fromfile(D / "batches.bin", dtype="<u4", count=B * (T + 1), offset=step * B * (T + 1) * 4)
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
    emb = p["model.embed_tokens.weight"]
    h = F.embedding(x, emb)
    for layer in range(LAYERS):
        w = lambda k: p[f"model.layers.{layer}.{k}.weight"]
        n = rmsnorm(h, w("input_layernorm"))
        a = attention(rope(F.linear(n, w("self_attn.q_proj")), H), rope(F.linear(n, w("self_attn.k_proj")), KV),
                      F.linear(n, w("self_attn.v_proj")))
        h = h + F.linear(a, w("self_attn.o_proj"))
        n = rmsnorm(h, w("post_attention_layernorm"))
        h = h + F.linear(F.silu(F.linear(n, w("mlp.gate_proj"))) * F.linear(n, w("mlp.up_proj")), w("mlp.down_proj"))
    return F.linear(rmsnorm(h, p["model.norm.weight"]), emb)  # the head is the embedding matrix


def forward(p, x, y):
    return F.cross_entropy(logits(p, x), y)


def bits(t):
    return " ".join(str(w) for w in t.detach().to(torch.float32).reshape(-1).numpy().view(np.uint32))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--prep" and len(sys.argv) == 3:
        prep(sys.argv[2])
        return
    if mode not in ("loss", "train", "grads", "grads64") or len(sys.argv) != (2 if mode == "loss" else 3):
        sys.exit(__doc__)
    p = load()
    if mode == "loss":
        with torch.no_grad():
            print("loss", bits(forward(p, *batch(0))))
        return
    if mode == "grads64":  # the reference float32 runs are judged against (reference/test_smol.py)
        p = {k: v.detach().double().requires_grad_(True) for k, v in p.items()}
        loss = forward(p, *batch(0))
        loss.backward()
        print("loss64", repr(loss.item()), flush=True)
        with open(sys.argv[2], "wb") as out:
            for name, _, off in layout():
                assert out.tell() == off * 8
                out.write(p[name].grad.numpy().astype("<f8").tobytes())
        return
    if mode == "grads":
        loss = forward(p, *batch(0))
        loss.backward()
        print("loss", bits(loss), flush=True)
        with open(sys.argv[2], "wb") as out:  # layout order, the embedding's gradient holding the head's too
            for name, _, off in layout():
                assert out.tell() == off * 4
                out.write(p[name].grad.numpy().astype("<f4").tobytes())
        return
    steps = int(sys.argv[2])
    opt = torch.optim.AdamW(p.values(), lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WD)
    print("loss", bits(torch.tensor(0.0)), flush=True)  # the Bend program's first line: the 0 before any step
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        loss = forward(p, *batch(step))
        loss.backward()
        opt.step()
        print("loss", bits(loss), flush=True)
    print(f"train_ms {round((time.perf_counter() - t0) * 1000)}")


if __name__ == "__main__":
    main()

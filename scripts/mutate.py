"""Mutation testing: change the code or the specification on purpose and check that a gate catches the change.

Each mutant replaces one exact piece of text in one file, runs the gate that should see it, and restores the file.
A mutant is caught when the gate fails. Some mutants are expected to survive: they compute the same function (an
equivalent mutant) or they sit outside what any gate checks; those are listed with the reason, so the report shows
the limits of the checks as well as their reach.

    reference/.venv/bin/python scripts/mutate.py        # or: make mutate
"""

import os
import subprocess
import sys
from pathlib import Path

ENV = dict(os.environ, BEND_NO_TELEMETRY="1",
           PATH=f"{Path.home()}/.bend/bin:{Path.home()}/.elan/bin:{os.environ['PATH']}")
PY = "reference/.venv/bin/python"


def laws():
    out = subprocess.run(["bend", "PROOF.bend", "--check-only"], capture_output=True, text=True, env=ENV).stdout
    return "ALL PROOFS CHECK" in out


def numeric(demo, test):
    def gate():
        b = subprocess.run(["bend", f"demo/{demo}.bend", "-o", f".scratch/mut_{demo}"], capture_output=True, env=ENV)
        if b.returncode != 0:
            return False
        return subprocess.run([PY, f"reference/{test}", f".scratch/mut_{demo}"], capture_output=True).returncode == 0
    return gate


OPS, MLP = numeric("ops_test", "test_ops.py"), numeric("mlp_test", "test_mlp.py")
TEN = numeric("tensor_test", "test_tensor.py")
IMAP = numeric("imap_check", "check_imap.py")


def fast():
    # torch/fast.bend's layers and optimizer step must keep main.bend's bits (demo/fast_test.bend prints 12 zeros)
    b = subprocess.run(["bend", "demo/fast_test.bend", "-o", ".scratch/mut_fast_test"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    return subprocess.run([".scratch/mut_fast_test", "--threads", "16"], capture_output=True, text=True).stdout.split() == ["0"] * 12


FAST = fast


def gpu():
    # torch/gpu.bend's layers must keep main.bend's bits: demo/gpu_test.bend
    # prints 8 zeros; built without CUDA, its bangs run on the CPU threads, the same code
    b = subprocess.run(["bend", "demo/gpu_test.bend", "-o", ".scratch/mut_gpu_test"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    return subprocess.run([".scratch/mut_gpu_test", "--threads", "16"], capture_output=True, text=True).stdout.split() == ["0"] * 8


GPU = gpu


def gpu_layers():
    # torch/gpu.bend's other layers and optimizer step must keep main.bend's bits with --gpu off:
    # demo/gpu_layers_test.bend prints 9 zeros
    b = subprocess.run(["bend", "demo/gpu_layers_test.bend", "-o", ".scratch/mut_gpu_layers"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    return subprocess.run([".scratch/mut_gpu_layers", "--gpu", "off", "--threads", "16"], capture_output=True, text=True).stdout.split() == ["0"] * 9


GPUL = gpu_layers
INFRA = numeric("infra_test", "test_infra.py")
MNIST = numeric("mnist", "test_mnist.py")

# (name, file, before, after, gate, expected): expected is "caught", or the reason it is allowed to survive
MUTANTS = [
    ("SGD: allocate two moment buffers", "torch/main.bend",
     "case CSGD{lr}: NoMom{}", "case CSGD{lr}: TwoMom{Layer.zeros(P, X, Y, S, m), Layer.zeros(P, X, Y, S, m)}", INFRA, "caught"),
    ("momentum: initialize no buffer", "torch/main.bend",
     "case CMomentum{lr, mu}: OneMom{Layer.zeros(P, X, Y, S, m)}", "case CMomentum{lr, mu}: NoMom{}", MLP, "caught"),
    ("load: advance the cursor one element short", "torch/main.bend",
     "(off + U32.from_nat(Nat.mul(r, c)) : U32), load_cells", "(off + U32.from_nat(Nat.sub(Nat.mul(r, c), 1n)) : U32), load_cells", INFRA, "caught"),
    ("load: every matrix reads from source offset zero", "torch/main.bend",
     "load_cells(Nat.mul(r, c), 0, off, Array.get(F32, xs, off), dst)", "load_cells(Nat.mul(r, c), 0, 0, Array.get(F32, xs, 0), dst)", INFRA, "caught"),
    ("store: each matrix writes one cell short", "torch/main.bend",
     "store_cells(Nat.mul(r, c), 0, off,", "store_cells(Nat.sub(Nat.mul(r, c), 1n), 0, off,", INFRA, "caught"),
    ("checkpoint from arrays: Adam's powers swapped", "torch/main.bend",
     "Array.set(F32, Array.set(F32, xs, off, p1), (off + 1 : U32), p2)", "Array.set(F32, Array.set(F32, xs, off, p2), (off + 1 : U32), p1)", INFRA, "caught"),
    ("cross-entropy: the row's tree sum one class short", "torch/main.bend",
     "tree_sum(U32.from_nat(1n+k), sc)", "tree_sum(U32.from_nat(k), sc)", MNIST, "caught"),
    ("checkpoint: SGD writes unused scalar powers", "torch/main.bend",
     "TrainPL{Train{p, g, NoMom{}, p1, p2, l}, xp}", "TrainPL{Train{p, g, NoMom{}, p1, p2, l}, List.append(&2, F32, xp, [p1, p2])}", INFRA, "caught"),
    ("checkpoint: momentum omits its buffer", "torch/main.bend",
     "TrainPL{Train{p, g, OneMom{mm}, p1, p2, l}, List.append(&2, F32, xp, xm)}", "TrainPL{Train{p, g, OneMom{mm}, p1, p2, l}, xp}", INFRA, "caught"),
    ("checkpoint: restore swaps Adam's powers", "torch/main.bend",
     "case Tuple{xs, p2}: Train{p, g, TwoMom{mm, vv}, p1, p2, 0.0}", "case Tuple{xs, p2}: Train{p, g, TwoMom{mm, vv}, p2, p1, 0.0}", INFRA, "caught"),
    ("binary file: accept extra trailing bytes", "torch/load.bend",
     "require(U32.is_eq(size, (4 * n : U32)),", "require(True{},", INFRA, "caught"),
    ("run-time training: report a zero loss", "torch/main.bend",
     "sf, cfg, st, p1, p2, l, bf(p, s, dy, g))", "sf, cfg, st, p1, p2, 0.0, bf(p, s, dy, g))", MNIST, "caught"),
    ("fast SGD: each block starts at zero", "torch/fast.bend",
     "p2_b2(T.sgd_l(cnt, i0, lr,", "p2_b2(T.sgd_l(cnt, 0, lr,", FAST, "caught"),
    ("fast momentum: each block starts at zero", "torch/fast.bend",
     "mom_b(T.mom_l(cnt, i0, lr,", "mom_b(T.mom_l(cnt, 0, lr,", FAST, "caught"),

    ("fast optimizer: every block's AdamW starts at number 0", "torch/fast.bend",
     "q4_b(T.adam_l(cnt, i0, lr,", "q4_b(T.adam_l(cnt, 0, lr,", FAST, "caught"),
    ("fast copies: each block writes its copy one place late", "torch/fast.bend",
     "a2_b2(T.copy_n(k, i0, i0, b,", "a2_b2(T.copy_n(k, i0, (i0 + 1 : U32), b,", FAST, "caught"),
    ("MSE: the tree sum drops an odd last number", "torch/main.bend",
     "ts_carry(U32.is_eq(U32.mod(m, 2), 1), m,", "ts_carry(False{}, m,", OPS, "caught"),
    ("fast linear: the weight gradient is not accumulated", "torch/fast.bend",
     "T.lin_b3(n, i, o, w, b, dx, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))", "T.lin_b3(n, i, o, w, b, dx, gb, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("fast linear first: the weight gradient is not accumulated", "torch/fast.bend",
     "T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))", "T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("fast linear without bias: the weight gradient is not accumulated", "torch/fast.bend",
     "T.lnb_b3(n, i, o, w, dx, TF.matmul_tn_acc(o, n, i, par, dy, x, g))", "T.lnb_b3(n, i, o, w, dx, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("gpu linear: the weight gradient is not accumulated", "torch/gpu.bend",
     "T.lin_b3(n, i, o, w, b, dx, gb, TG.matmul_tn_acc(o, n, i, 0n, dy, x, gw))", "T.lin_b3(n, i, o, w, b, dx, gb, TG.matmul_tn(o, n, i, 0n, dy, x))", GPU, "caught"),
    ("gpu element-wise: each leaf skips its last cell", "torch/gpu.bend",
     "case EN{+n}: ew1_loop(~f, TG.share(t, n, l), t, l, o, Array.get(F32, x, t))", "case EN{+n}: ew1_loop(~f, Nat.sub(TG.share(t, n, l), 1n), t, l, o, Array.get(F32, x, t))", GPUL, "caught"),
    ("gpu RMSNorm: the gain's gradient overwrites the one it is given", "torch/gpu.bend",
     "Array.set(F32, ga, j, (v + s : F32))", "Array.set(F32, ga, j, s)", GPUL, "caught"),
    ("gpu RoPE: the backward pass rotates forward", "torch/gpu.bend",
     "T.Bwd{p, rope_go(n, w, False{}, t, h, half, base, dy), g}", "T.Bwd{p, rope_go(n, w, True{}, t, h, half, base, dy), g}", GPUL, "caught"),
    ("gpu attention: dV from G instead of P", "torch/gpu.bend",
     "gkv(n, w, kvw, b, t, kv, g, hd, p, d)", "gkv(n, w, kvw, b, t, kv, g, hd, gb, d)", GPUL, "caught"),
    ("gpu Adam: the gradient is not zeroed", "torch/gpu.bend",
     "Array.get(F32, Array.set(F32, g, i, 0.0), u), Array.get(F32, Array.set(F32, m, i, nm), u)", "Array.get(F32, g, u), Array.get(F32, Array.set(F32, m, i, nm), u)", GPUL, "caught"),
    ("spec: Mᵀ·g drops the rows after the first", "torch/spec.bend",
     "vadd(vscale(gi, r), mtv(rt, gt))", "vadd(vscale(gi, r), mtv(rt, Nil{}))", laws, "caught"),
    ("spec: g ⊗ v uses only the first entry of g", "torch/spec.bend",
     "vscale(gi, v) <> outer(gt, v)", "vscale(gi, v) <> outer(Nil{}, v)", laws, "caught"),
    ("attention: a query also sees the next position", "torch/main.bend",
     "prow3(off, q, k, pw(row, off, sc, 0n, p))", "prow3(off, q, k, pw(row, off, sc, 1n, p))", laws, "caught"),
    ("attention: a query also sees the next position (numbers)", "torch/main.bend",
     "prow3(off, q, k, pw(row, off, sc, 0n, p))", "prow3(off, q, k, pw(row, off, sc, 1n, p))", OPS, "caught"),
    ("attention: the softmax leaves out the last visible position", "torch/main.bend",
     "pr_of(q, k, psmc(n, off, p))", "pr_of(q, k, psmc(Nat.sub(n, 1n), off, p))", laws, "caught"),
    ("attention: the softmax leaves out the last visible position (numbers)", "torch/main.bend",
     "pr_of(q, k, psmc(n, off, p))", "pr_of(q, k, psmc(Nat.sub(n, 1n), off, p))", OPS, "caught"),
    ("attention: query head hi reads key/value head hi mod g", "torch/main.bend",
     "  Nat.div(hi, g)\n", "  Nat.mod(hi, g)\n", laws, "caught"),
    ("attention: query head hi reads key/value head hi mod g (numbers)", "torch/main.bend",
     "  Nat.div(hi, g)\n", "  Nat.mod(hi, g)\n", OPS, "caught"),
    ("attention: head_cols gives query head hi the columns of key/value head hi mod g", "torch/main.bend",
     "U32.from_nat(Nat.mul(kv_of(hi, g), hd))}", "U32.from_nat(Nat.mul(Nat.mod(hi, g), hd))}", laws, "caught"),
    ("attention: head_cols gives query head hi the columns of key/value head hi mod g (numbers)", "torch/main.bend",
     "U32.from_nat(Nat.mul(kv_of(hi, g), hd))}", "U32.from_nat(Nat.mul(Nat.mod(hi, g), hd))}", OPS, "caught"),
    ("attention: the backward pass's dP row reads one value past the query's position", "torch/main.bend",
     "  gr2(sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p, g, drow(i, hd,", "  gr2(sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p, g, drow(1n+i, hd,", laws, "caught"),
    ("attention: the tree of sequences puts the right half's outputs first", "torch/main.bend",
     "      a b = fleaf(l, h, dm, Nil{}) fleaf(r, h, dm, Nil{})\n      List.append(&1, Array<F32>, a, b)", "      a b = fleaf(l, h, dm, Nil{}) fleaf(r, h, dm, Nil{})\n      List.append(&1, Array<F32>, b, a)", laws, "caught"),
    ("attention: the backward tree puts the right half's gradients first", "torch/main.bend",
     "      a b = bleaf(l, kv, dm, Nil{}) bleaf(r, kv, dm, Nil{})\n      List.append(&1, DB, a, b)", "      a b = bleaf(l, kv, dm, Nil{}) bleaf(r, kv, dm, Nil{})\n      List.append(&1, DB, b, a)", laws, "caught"),
    ("Fork: the backward pass drops the second layer's input gradient", "torch/main.bend",
     "      Bwd{Two{p1, p2}, add(dx1, dx2), Two{g1, g2}}", "      Bwd{Two{p1, p2}, dx1, Two{g1, g2}}", laws, "caught"),
    ("attention: the backward pass forgets the scale", "torch/main.bend",
     "gfix(Nat.sub(t, 1n), off, s, sc,", "gfix(Nat.sub(t, 1n), off, s, 1.0,", OPS, "caught"),
    ("attention: a query reads one key past its position", "torch/main.bend",
     "  drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", "  drow(1n+i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", laws, "caught"),
    ("attention: a query reads one key past its position (numbers)", "torch/main.bend",
     "  drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", "  drow(1n+i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", OPS, "caught"),
    ("cross-entropy: the gradient forgets the label", "torch/main.bend",
     "((x - hit(U32.is_eq(j, y)) : F32) / nf : F32)", "(x / nf : F32)", OPS, "caught"),
    ("scratch arrays: half the capacity they need", "torch/main.bend",
     "  depth_go(32n, U32.is_le(U32.from_nat(n), 1), U32.from_nat(n), 1, 0n)", "  depth_go(32n, U32.is_le(U32.from_nat(n), 2), U32.from_nat(n), 2, 0n)", OPS, "caught"),
    ("relu: the gradient passes at exactly 0", "torch/main.bend",
     "pick(F32.is_le(y, 0.0), 0.0, dy)", "pick(F32.is_lt(y, 0.0), 0.0, dy)", OPS, "caught"),
    ("silu: the derivative misses its 1 +", "torch/main.bend",
     "((dy * s : F32) * (1.0 + (x * (1.0 - s : F32) : F32) : F32) : F32)", "((dy * s : F32) * (x * (1.0 - s : F32) : F32) : F32)",
     OPS, "caught"),
    ("softmax: the backward pass adds instead of subtracting", "torch/main.bend",
     "smb_fix(Nat.sub(c, 1n), off, s,", "smb_fix(Nat.sub(c, 1n), off, F32.neg(s),", OPS, "caught"),
    ("rmsnorm: the backward pass drops the mean term", "torch/main.bend",
     "rmsb_w(Nat.sub(d, 1n), 0, off, r, (s / df : F32), dx, dg,", "rmsb_w(Nat.sub(d, 1n), 0, off, r, 0.0, dx, dg,", OPS, "caught"),
    ("rope: the backward pass rotates forward", "torch/main.bend",
     "Bwd{p, rope_go(n, w, False{}, t, h, half, base, dy), g}",
     "Bwd{p, rope_go(n, w, True{}, t, h, half, base, dy), g}", OPS, "caught"),
    ("embedding: a repeated id keeps only its last gradient", "torch/main.bend",
     "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, (a + b : F32))", "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, b)", OPS, "caught"),
    ("linear: the bias gradient is not accumulated", "torch/main.bend",
     "TA.Mat.col_sums_acc(n, o, dy, gb))", "TA.Mat.col_sums(n, o, dy))", MLP, "caught"),
    ("dropout: the optimizer step does not move the mask's counter", "torch/main.bend",
     "Step{DropP{(t + 1 : U32)}, g, st}", "Step{DropP{t}, g, st}", OPS, "caught"),
    ("dropout: the mask ignores the step", "torch/main.bend",
     "mix32(U32.xor(seed, mix32(t)))", "mix32(seed)", OPS, "caught"),
    ("tie: the first layer's gradient forgets the last layer's", "torch/main.bend",
     "tie_b4(P, Q, X, q, gq, Layer.bwd(P, X, Y, S1, la, p, s1, dy, gp))",
     "tie_b4(P, Q, X, q, gq, Layer.bwd(P, X, Y, S1, la, p, s1, dy, Layer.zeros(P, X, Y, S1, la)))", OPS, "caught"),
    ("momentum: the update ignores the buffer", "torch/main.bend",
     "(x + (F32.neg(lr) * nb : F32) : F32)", "(x + (F32.neg(lr) * y : F32) : F32)", MLP, "caught"),
    ("adamw: no decoupled weight decay", "torch/main.bend",
     "(w * (1.0 - (lr * wd : F32) : F32) : F32)", "w", MLP, "caught"),
    ("adam: no bias correction of the second moment", "torch/main.bend",
     "((F32.sqrt(v) / bc2 : F32) + eps : F32)", "(F32.sqrt(v) + eps : F32)", MLP, "caught"),
    ("linear: the weight gradient is not accumulated", "torch/main.bend",
     "TA.Mat.matmul_tn_acc(o, n, i, par, dy, x, gw))", "TA.Mat.matmul_tn(o, n, i, par, dy, x))", MLP, "caught"),
    ("tensor: div's gradient for the divisor loses its sign", "torch/tensor.bend",
     "(0.0 - (d * ((p / q : F32) / q : F32) : F32) : F32)", "(d * ((p / q : F32) / q : F32) : F32)", TEN, "caught"),
    ("tensor: narrow starts its window on the wrong axis", "torch/tensor.bend",
     "U32.from_nat(Nat.mul(st, inner(s, d))))", "U32.from_nat(Nat.mul(st, inner(s, 0n))))", TEN, "caught"),
    ("tensor: bmm reads its second operand transposed", "torch/tensor.bend",
     "u(Nat.mul(n, k)), u(k), 1, u(Nat.mul(k, m)), u(m), 1}, dx, dy", "u(Nat.mul(n, k)), u(k), 1, u(Nat.mul(k, m)), 1, u(m)}, dx, dy", TEN, "caught"),
    ("tensor: broadcasting allows a size-1 axis on one side only", "torch/tensor.bend",
     "Bool.or(bc_pair(x, y, z), bc_pair(y, x, z))", "bc_pair(x, y, z)", laws, "caught"),
    ("tensor: the whole-tensor sum maps each element to its own position (right only because a 1-cell array wraps)", "torch/tensor.bend",
     "Drop{1, U32.from_nat(numel(s))}", "Drop{U32.from_nat(numel(s)), 1}", TEN, "survives the numbers: the 1-cell result array wraps every index to 0, so the sum is right by accident; the bounds check catches it (below)"),
    ("tensor: the whole-tensor sum maps each element to its own position (bounds)", "torch/tensor.bend",
     "Drop{1, U32.from_nat(numel(s))}", "Drop{U32.from_nat(numel(s)), 1}", IMAP, "caught"),
    ("tensor: cat's shape adds the first operand's axis twice", "torch/tensor.bend",
     "setd(a, d, Nat.add(at1(a, d), at1(b, d)))", "setd(a, d, Nat.add(at1(a, d), at1(a, d)))", TEN, "caught"),
]


def main():
    escaped = []
    for name, path, before, after, gate, expected in MUTANTS:
        p = Path(path)
        src = p.read_text()
        assert src.count(before) >= 1, f"{name}: the text to mutate is not in {path}"
        try:
            p.write_text(src.replace(before, after, 1))
            caught = not gate()
        finally:
            p.write_text(src)
        verdict = "caught" if caught else "survived"
        ok = (verdict == "caught") == (expected == "caught")
        print(f"{'ok ' if ok else 'BAD'} {verdict:8} {name}" + ("" if expected == "caught" else f"  [{expected}]"))
        if not ok:
            escaped.append(name)
    print(f"mutate: {len(MUTANTS)} mutants, {len(MUTANTS) - len(escaped)} as expected")
    sys.exit(1 if escaped else 0)


if __name__ == "__main__":
    main()

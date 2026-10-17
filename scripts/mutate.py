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
    # torch/fast.bend's layers must keep main.bend's bits (demo/fast_test.bend prints six zeros)
    b = subprocess.run(["bend", "demo/fast_test.bend", "-o", ".scratch/mut_fast_test"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    return subprocess.run([".scratch/mut_fast_test", "--threads", "16"], capture_output=True, text=True).stdout.strip() == "0 0 0 0 0 0"


FAST = fast

# (name, file, before, after, gate, expected): expected is "caught", or the reason it is allowed to survive
MUTANTS = [
    ("fast linear: the weight gradient is not accumulated", "torch/fast.bend",
     "T.lin_b3(n, i, o, w, b, dx, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))", "T.lin_b3(n, i, o, w, b, dx, gb, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("fast linear first: the weight gradient is not accumulated", "torch/fast.bend",
     "T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))", "T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("fast linear without bias: the weight gradient is not accumulated", "torch/fast.bend",
     "T.lnb_b3(n, i, o, w, dx, TF.matmul_tn_acc(o, n, i, par, dy, x, g))", "T.lnb_b3(n, i, o, w, dx, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("spec: Mᵀ·g drops the rows after the first", "torch/spec.bend",
     "vadd(vscale(gi, r), mtv(rt, gt))", "vadd(vscale(gi, r), mtv(rt, Nil{}))", laws, "caught"),
    ("spec: g ⊗ v uses only the first entry of g", "torch/spec.bend",
     "vscale(gi, v) <> outer(gt, v)", "vscale(gi, v) <> outer(Nil{}, v)", laws, "caught"),
    ("attention: a query also sees the next position", "torch/main.bend",
     "l_softmax(List.take(&2, F32, r, 1n+i))", "l_softmax(List.take(&2, F32, r, 2n+i))", laws, "caught"),
    ("attention: one zero too many after the visible positions", "torch/main.bend",
     "l_zeros(Nat.sub(t, 1n+i)))", "l_zeros(Nat.sub(t, i)))", laws, "caught"),
    ("attention: query head hi reads key/value head hi mod g", "torch/main.bend",
     "  Nat.div(hi, g)\n", "  Nat.mod(hi, g)\n", laws, "caught"),
    ("attention: query head hi reads key/value head hi mod g (numbers)", "torch/main.bend",
     "  Nat.div(hi, g)\n", "  Nat.mod(hi, g)\n", OPS, "caught"),
    ("attention: the forward pass alone bypasses kv_of", "torch/main.bend",
     "lm_cols(kb, Nat.mul(kv_of(hi, g), hd), hd))), 0n, t)", "lm_cols(kb, Nat.mul(Nat.mod(hi, g), hd), hd))), 0n, t)",
     laws, "survives the laws: no law says the attention code calls kv_of (the numbers catch it)"),
    ("attention: the forward pass alone bypasses kv_of (numbers)", "torch/main.bend",
     "lm_cols(kb, Nat.mul(kv_of(hi, g), hd), hd))), 0n, t)", "lm_cols(kb, Nat.mul(Nat.mod(hi, g), hd), hd))), 0n, t)",
     OPS, "caught"),
    ("attention: the backward pass forgets the scale", "torch/main.bend",
     "+gs = lm_scale(sc, sm_bwd_rows(p, lm_mul_t(doh, vh)))", "+gs = sm_bwd_rows(p, lm_mul_t(doh, vh))", OPS, "caught"),
    ("relu: the gradient passes at exactly 0", "torch/main.bend",
     "pick(F32.is_le(y, 0.0), 0.0, dy)", "pick(F32.is_lt(y, 0.0), 0.0, dy)", OPS, "caught"),
    ("silu: the derivative misses its 1 +", "torch/main.bend",
     "((dy * s : F32) * (1.0 + (x * (1.0 - s : F32) : F32) : F32) : F32)", "((dy * s : F32) * (x * (1.0 - s : F32) : F32) : F32)",
     OPS, "caught"),
    ("softmax: the backward pass adds instead of subtracting", "torch/main.bend",
     "l_mul(y, l_shift(l_dot(dy, y, 0.0), dy))", "l_mul(y, l_shift(F32.neg(l_dot(dy, y, 0.0)), dy))", OPS, "caught"),
    ("rmsnorm: the backward pass drops the mean term", "torch/main.bend",
     "l_scale(r, l_sub(dxh, l_scale((l_dot(dxh, xh, 0.0) / d : F32), xh)))", "l_scale(r, dxh)", OPS, "caught"),
    ("rope: the backward pass rotates forward", "torch/main.bend",
     "Bwd{p, rope_go(n, w, False{}, t, h, half, base, Mat.rows(n, w, dy)), g}",
     "Bwd{p, rope_go(n, w, True{}, t, h, half, base, Mat.rows(n, w, dy)), g}", OPS, "caught"),
    ("embedding: a repeated id keeps only its last gradient", "torch/main.bend",
     "TA.Mat.write_row(v, d, m, U32.from_nat(i), l_add(row, dy))", "TA.Mat.write_row(v, d, m, U32.from_nat(i), dy)", OPS, "caught"),
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

# BendTorch: setup and verification shortcuts. Run `make setup` and `make data` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
DEMOS := mlp_test ops_test mnist tiny_llama tensor_test imap_check fast_test depth_check smol_block infra_test
# torch/fast.bend's defs that rely on bend-ml-tensor-fork's @unsafe Array.fork: the list may change only on purpose
LOAD_DEFS := read_sized read weights train_read train
FAST_DEFS := opt_mat mat_step lin_step2 lin_step map zip Mat.dup Mat.sum2 lin_fwd lin_b2 lin_bwd LinearF linf_bwd LinearFirstF lnb_b2 LinearNBF acti_f3 ActIn SiLU add2_f Add2 mul2_f2 mul2_f mul2_b2 Mul2 Residual

.PHONY: setup data laws build check check-js bits mutate bench lint smol

setup:            ## install the pinned Bend and Lean if missing, create the Python venv
	scripts/setup.sh

data:             ## MNIST and Tiny Shakespeare into data/, the twins' initial weights and batches (seeded)
	scripts/mnist.sh
	scripts/tiny.sh

laws:             ## the proofs (LAWS.bend, PROOF.bend) by bend and again by --verdict; tests/bad_*.bend rejected with their expected error
	! grep -n -E '^[^#]*(@unsafe|\?TODO)' torch/*.bend LAWS.bend PROOF.bend demo/*.bend
	bend torch/main.bend --check-only
	bend PROOF.bend --check-only | tail -2 | head -1 | grep -qx "ALL PROOFS CHECK"
	bend PROOF.bend --verdict | tee /dev/stderr | tail -1 | grep -qx "ALL PROOFS CHECK"
	test "$$(bend torch/fast.bend 2>&1 | sed -n 's/^- \([A-Za-z0-9_.]*\)$$/\1/p' | tr '\n' ' ')" = "$(FAST_DEFS) " || { echo "torch/fast.bend: its defs outside the proofs are not FAST_DEFS"; exit 1; }
	test "$$(bend torch/load.bend 2>&1 | sed -n 's/^- \([A-Za-z0-9_.]*\)$$/\1/p' | tr '\n' ' ')" = "$(LOAD_DEFS) " || { echo "torch/load.bend: its defs outside the proofs are not LOAD_DEFS"; exit 1; }
	! grep -n -E 'load.bend|bend-ml-binary-read' LAWS.bend PROOF.bend torch/main.bend torch/spec.bend torch/tensor.bend torch/fast.bend
	! grep -n 'fast.bend' LAWS.bend PROOF.bend torch/main.bend torch/spec.bend torch/tensor.bend
	for f in tests/bad_*.bend; do want=$$(sed -n 's/^# expect: //p' $$f); bend $$f --check-only 2>&1 | grep -qF -- "$$want" && echo "rejected as it must be: $$f" || { echo "NOT REJECTED AS EXPECTED: $$f (wanted: $$want)"; exit 1; }; done

build:            ## compile the test and demo programs into .scratch/
	mkdir -p .scratch
	for d in $(DEMOS); do bend demo/$$d.bend -o .scratch/$$d || exit 1; done

check: laws build ## the laws, then every layer, the tensor operations, the MLP, MNIST, the tiny Llama and a SmolLM2-sized block against PyTorch
	$(PY) reference/test_infra.py .scratch/infra_test
	$(PY) reference/test_ops.py .scratch/ops_test
	$(PY) reference/test_tensor.py .scratch/tensor_test
	$(PY) reference/check_imap.py .scratch/imap_check
	$(PY) reference/test_mlp.py .scratch/mlp_test
	$(PY) reference/test_mnist.py .scratch/mnist
	$(PY) reference/test_tiny.py .scratch/tiny_llama 20
	$(PY) reference/test_block.py .scratch/smol_block 4
	$(PY) reference/test_block.py .scratch/smol_block 4 135m sgd
	for t in 1 16; do test "$$(.scratch/fast_test --threads $$t | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 0 0 0 0 " && echo "fast: torch/fast.bend's layers and optimizer step give main.bend's bits ($$t threads)" || { echo "fast: the bits differ ($$t threads)"; exit 1; }; done
	test "$$(.scratch/depth_check)" = "0" && echo "depth: the scratch arrays' capacity is TA.cap_depth's for n = 0..4999" || { echo "depth: differs from TA.cap_depth"; exit 1; }

check-js:         ## the test programs compiled to JavaScript and run with Bun, against the same oracles; their outputs against bench/bits.txt
	scripts/check_js.sh

bits: build      ## the exact outputs of the Llama (training, generation) and MNIST, 1 and 16 threads, against bench/bits.txt
	scripts/bits.sh check

mutate:           ## break the code and the specification on purpose; every mutant must be caught (scripts/mutate.py)
	$(PY) scripts/mutate.py

bench:            ## MNIST epoch and tiny-Llama step times, Bend against PyTorch, on 1 and 16 threads (scripts/bench.sh)
	scripts/bench.sh

smol:             ## SmolLM2-135M and 360M against their twins (scripts/smol_data.sh 135m and 360m first; ~13 GB at 360M)
	mkdir -p .scratch
	bend demo/smol.bend -o .scratch/smol
	scripts/smol.sh 360m && bend .scratch/smol_360m.bend -o .scratch/smol_360m
	$(PY) reference/test_smol.py .scratch/smol
	$(PY) reference/test_smol.py .scratch/smol_360m 360m

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate; measure a finding before taking it
	bunx bend-lint --jobs max

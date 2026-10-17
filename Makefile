# BendTorch: setup and verification shortcuts. Run `make setup` and `make data` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
DEMOS := mlp_test ops_test mnist tiny_llama tensor_test imap_check fast_test depth_check
# torch/fast.bend's defs that rely on bend-ml-tensor-fork's @unsafe Array.fork: the list may change only on purpose
FAST_DEFS := lin_fwd lin_b2 lin_bwd LinearF linf_bwd LinearFirstF lnb_b2 LinearNBF

.PHONY: setup data laws build check check-js bits mutate bench lint

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
	test "$$(bend torch/fast.bend 2>&1 | sed -n 's/^- \([A-Za-z0-9_]*\)$$/\1/p' | tr '\n' ' ')" = "$(FAST_DEFS) " || { echo "torch/fast.bend: its defs outside the proofs are not FAST_DEFS"; exit 1; }
	! grep -n 'fast.bend' LAWS.bend PROOF.bend torch/main.bend torch/spec.bend torch/tensor.bend
	for f in tests/bad_*.bend; do want=$$(sed -n 's/^# expect: //p' $$f); bend $$f --check-only 2>&1 | grep -qF -- "$$want" && echo "rejected as it must be: $$f" || { echo "NOT REJECTED AS EXPECTED: $$f (wanted: $$want)"; exit 1; }; done

build:            ## compile the test and demo programs into .scratch/
	mkdir -p .scratch
	for d in $(DEMOS); do bend demo/$$d.bend -o .scratch/$$d || exit 1; done

check: laws build ## the laws, then every layer, the tensor operations, the MLP, MNIST and the tiny Llama against PyTorch
	$(PY) reference/test_ops.py .scratch/ops_test
	$(PY) reference/test_tensor.py .scratch/tensor_test
	$(PY) reference/check_imap.py .scratch/imap_check
	$(PY) reference/test_mlp.py .scratch/mlp_test
	$(PY) reference/test_mnist.py .scratch/mnist
	$(PY) reference/test_tiny.py .scratch/tiny_llama 20
	for t in 1 16; do test "$$(.scratch/fast_test --threads $$t)" = "0 0 0 0 0 0" && echo "fast: torch/fast.bend's layers give main.bend's bits ($$t threads)" || { echo "fast: the bits differ ($$t threads)"; exit 1; }; done
	test "$$(.scratch/depth_check)" = "0" && echo "depth: the scratch arrays' capacity is TA.cap_depth's for n = 0..4999" || { echo "depth: differs from TA.cap_depth"; exit 1; }

check-js:         ## the test programs compiled to JavaScript and run with Bun, against the same oracles; their outputs against bench/bits.txt
	scripts/check_js.sh

bits: build      ## the exact outputs of the Llama (training, generation) and MNIST, 1 and 16 threads, against bench/bits.txt
	scripts/bits.sh check

mutate:           ## break the code and the specification on purpose; every mutant must be caught (scripts/mutate.py)
	$(PY) scripts/mutate.py

bench:            ## MNIST epoch and tiny-Llama step times, Bend against PyTorch, on 1 and 16 threads (scripts/bench.sh)
	scripts/bench.sh

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate; measure a finding before taking it
	bunx bend-lint --jobs max

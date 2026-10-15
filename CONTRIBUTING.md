# Contributing to BendTorch

Thanks for looking at BendTorch. I'm Renan Vinícius (@nuxyel) and I maintain it alone, so I read every issue and pull
request myself. Bug reports, numbers from other machines, new layers, new laws and fixes are all welcome.

## Branches

- **Open pull requests against `devel`.** All work happens there; `main` only receives releases, through a pull
  request from `devel`, and the tags live on `main`. GitHub proposes `main` by default, so change the base branch when
  you open the PR. If you forget, I'll retarget it.
- Branch from `devel` in your fork, not from `main`.

## Before you start

- For anything beyond a small fix (a change to the layer protocol, a new kind of tensor, performance work, a new
  dependency), open an issue first so we can agree on the approach before you spend time on it.
- Read [`NOTES.md`](NOTES.md): it records what was tried, measured and decided, including what did not work.
  [`docs/plan.md`](docs/plan.md) has the milestones, [`docs/design.md`](docs/design.md) the reasons behind the design.
- Bend 2 is young and differs a lot from Bend 1. The "pitfalls" section of [`CLAUDE.md`](CLAUDE.md) lists the traps I
  hit in practice; read it before writing Bend.

## Setup and checks

```bash
make setup   # the pinned Bend 2.0.37, Lean 4.34.0 and the Python venv of reference/
make data    # MNIST and Tiny Shakespeare (not committed), the twins' weights and batches
make laws    # the proofs, by bend and again by --verdict; tests/bad_*.bend must be rejected with their expected error
make check   # the laws, then every layer, the MLP, MNIST and the tiny Llama against PyTorch
make mutate  # deliberate faults, each caught by a law or a test (or listed with the reason it is not)
make bench   # BendTorch against PyTorch on MNIST and the tiny Llama, 1 and 16 threads
```

A pull request should pass `make check`. If it touches `torch/`, `LAWS.bend` or the tests, also run `make mutate`.

## Rules for code

- **Bend stays pinned to 2.0.37.** A version bump is its own change, with an entry in `NOTES.md`.
- **Every proof is real.** No `@unsafe`, no `?TODO`, no axioms. A law that cannot be proved yet goes into an issue, not
  into `LAWS.bend`.
- **Prove structure, test numbers.** Shapes and composition belong in the types and the laws; `F32` results are checked
  against PyTorch by the scripts in `reference/`, with the one tolerance in `reference/oracle.py`. Never widen it to
  make a test pass; find out why the numbers differ.
- **A new layer comes with its test**: a section in `demo/ops_test.bend` and the same computation in
  `reference/test_ops.py`, written with PyTorch's own functions.
- **Every proved law is explained in plain language** in [`torch/README.md`](torch/README.md).

## Benchmarks

Performance claims need numbers someone else can reproduce: the hardware, the Bend and compiler versions, an idle
machine, runs alternated between the old and the new version, one-thread timings pinned to a single core. Report the
losses and the noise too, not only the best run. A speedup that changes a test's result is a bug until explained.

## Commits and pull requests

- Small commits, one logical change each, with clear messages in English.
- Keep a pull request to one topic.
- Don't commit datasets, `data/`, `.scratch/` or large logs.
- Using AI tools is fine. You are still the author: say in the PR what you ran yourself and what you did not, and keep
  the description short enough for a person to read.

## License

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE) of this repository.

<!-- Thanks! Please target `devel`, not `main` (see CONTRIBUTING.md). -->

## What and why

<!-- What this changes and why. Link the issue if there is one: "Closes #N". -->

## Checks

<!-- Tick what you ran yourself; say why if something does not apply. -->

- [ ] The base branch is `devel`
- [ ] `make laws`: ALL PROOFS CHECK, no `@unsafe`, `?TODO` or axioms; the programs in `tests/` rejected
- [ ] `make check`: 0 failures, no tolerance widened
- [ ] `make mutate`: every mutant as listed (for changes to `torch/`, `LAWS.bend` or the tests)
- [ ] New or changed laws are explained in `torch/README.md`

## Numbers

<!-- Only for performance changes: hardware, Bend and clang versions, threads, before and after, how many runs.
     Delete this section otherwise. -->

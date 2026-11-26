# Lean proofs

Lean 4 + Mathlib proofs of the exact statements the project relies on (epic
[#11](https://github.com/SpacekidLabs/representation-fragility-lab/issues/11)).
Experiments show what happens on the signals we tried; these proofs show what
must happen for every input. The plain-language list is in
[docs/formal_verification.md](../docs/formal_verification.md).

| File | Topic | Issue |
|---|---|---|
| `RepFragility/Smoke.lean` | Setup check | #12 |
| `RepFragility/Comparator.lean` | Comparator aggregation (`src/core/generic.py`) | #13 |
| `RepFragility/Pitch.lean` | Pitch distance and relation labels (`src/framework/audio.py`) | #14 |
| `RepFragility/Consensus.lean` | Weighted-median consensus and quorum | #15 |
| `RepFragility/Features.lean` | Exp 043 disagreement features | #16 |
| `RepFragility/Invariance.lean` | Invariance taxonomy on finite signals | #17 |

## Install and build

You do not need Lean to work on the Python code. CI builds this folder on every
push that touches `lean/**` (`.github/workflows/lean.yml`).

To build locally (about 5 GB for the Mathlib cache, in `lean/.lake/`, which is gitignored):

```bash
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
```

```bash
cd lean && lake exe cache get && lake build
```

`lean-toolchain` pins Lean v4.36.0 and `lakefile.toml` pins Mathlib v4.36.0 (its
matching release). Upgrade both together.

## Conventions

- **Models.** Every `def` that models Python code has a docstring starting
  `Models: <file>::<function>`, e.g. `Models: src/framework/audio.py::PitchComparator.fuse`.
- **Plain language.** Every theorem has a docstring with its statement in plain words,
  and the same words appear in `docs/formal_verification.md`.
- **Python twin.** Every theorem has a property test in `tests/test_formal_properties.py`
  that runs the real Python code on random inputs, so a drift between the code and
  the model makes a test fail.
- **Real numbers, not floats.** The model works over ℝ (or ℚ). Floating-point
  behaviour of NumPy is not proven; where it matters, the gap is stated in the docs.
- **No `sorry` on `main`.** CI builds with `--wfail` and fails on any
  "declaration uses sorry" warning (it also checks that this guard works).

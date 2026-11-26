# Formal verification

Exact statements about the comparison machinery, proven in Lean 4 + Mathlib
([lean/](../lean/README.md), epic [#11](https://github.com/SpacekidLabs/representation-fragility-lab/issues/11)).
Each one has a property test that runs the real Python code, so the proof and
the code cannot drift apart silently. Empirical claims (AUCs, robustness in
practice) stay in the experiments.

The Lean model works over the real numbers; NumPy floating point is not modelled.

| Statement | Lean theorem | Python test | Assumptions |
|---|---|---|---|

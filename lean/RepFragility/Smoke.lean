import Mathlib

/-!
# Smoke test

Proves that the toolchain, Mathlib and CI work. Nothing here models Python code.
-/

namespace RepFragility

/-- Plain language: truncating a non-negative real at 1 gives a value in [0, 1].
(The shape of `min(cents / 1200, 1)` in `PitchComparator.estimate_distance`.) -/
theorem smoke_min_one_mem_unit (x : ℝ) (hx : 0 ≤ x) : 0 ≤ min x 1 ∧ min x 1 ≤ 1 :=
  ⟨le_min hx zero_le_one, min_le_right x 1⟩

end RepFragility

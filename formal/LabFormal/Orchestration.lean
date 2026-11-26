import Mathlib

/-!
# Formal core of the Memory Frontier Labs hypotheses (H6, H13)

What is proved here is pure mathematics under explicit modelling assumptions. None of it is evidence about real hardware:
whether real inference workloads satisfy the assumptions is an empirical question (see docs/MODEL.md in the simulator).
-/

namespace Lab

/-! ## C1 (H13). When is acting on a prediction worth its cost?

Acting on a prediction instead of staying data-local has expected net benefit `p * B - (1 - p) * L - c`:
`p` is the probability the prediction is right, `B` the gain when it is right, `L` the loss when it is wrong,
`c` the cost of obtaining and acting on the prediction. -/

/-- Expected net benefit of acting on a prediction. -/
def netBenefit (p B L c : ℝ) : ℝ := p * B - (1 - p) * L - c

/-- Threshold form: when `B + L > 0`, acting pays exactly when `p` exceeds `(L + c) / (B + L)`. -/
theorem netBenefit_pos_iff {p B L c : ℝ} (h : 0 < B + L) :
    0 < netBenefit p B L c ↔ (L + c) / (B + L) < p := by
  unfold netBenefit
  rw [div_lt_iff₀ h]
  constructor <;> intro h' <;> linarith

/-- At the threshold itself the expected net benefit is exactly zero. -/
theorem netBenefit_at_threshold {B L c : ℝ} (h : 0 < B + L) :
    netBenefit ((L + c) / (B + L)) B L c = 0 := by
  unfold netBenefit
  field_simp
  ring

/-- A more reliable predictor is never worse (when acting does not lose on average, `B + L ≥ 0`). -/
theorem netBenefit_mono_p {p q B L c : ℝ} (hBL : 0 ≤ B + L) (h : p ≤ q) :
    netBenefit p B L c ≤ netBenefit q B L c := by
  unfold netBenefit
  nlinarith [mul_nonneg (sub_nonneg.2 h) hBL]

/-- A cheaper prediction is never worse. -/
theorem netBenefit_anti_c {p B L c c' : ℝ} (h : c ≤ c') :
    netBenefit p B L c' ≤ netBenefit p B L c := by
  unfold netBenefit
  linarith

/-- If the predictor is no better than the threshold, staying reactive is at least as good. -/
theorem reactive_optimal_below_threshold {p B L c : ℝ} (h : 0 < B + L)
    (hp : p ≤ (L + c) / (B + L)) : netBenefit p B L c ≤ 0 := by
  by_contra hcon
  have := (netBenefit_pos_iff h).1 (not_le.1 hcon)
  linarith

/-! ## C2 (H6). Compute-memory inversion in the additive time model

Time on one device is `c / k + m` where `c / k` is compute time after a speed-up factor `k` and `m` is memory time.
The memory share is `m / (c / k + m)`. -/

/-- Share of the total time spent on memory when compute is `k` times faster. -/
noncomputable def memShare (c m k : ℝ) : ℝ := m / (c / k + m)

/-- H6 in the additive model: the faster the compute, the larger the memory share (strictly). -/
theorem memShare_strictMono {c m k k' : ℝ} (hc : 0 < c) (hm : 0 < m) (hk : 0 < k) (hkk : k < k') :
    memShare c m k < memShare c m k' := by
  unfold memShare
  have hk' : 0 < k' := lt_trans hk hkk
  have h1 : c / k' < c / k := div_lt_div_of_pos_left hc hk hkk
  have h2 : 0 < c / k' + m := by positivity
  exact div_lt_div_of_pos_left hm h2 (by linarith)

/-- The memory share is always strictly between 0 and 1. -/
theorem memShare_mem_Ioo {c m k : ℝ} (hc : 0 < c) (hm : 0 < m) (hk : 0 < k) :
    0 < memShare c m k ∧ memShare c m k < 1 := by
  unfold memShare
  have hd : 0 < c / k + m := by positivity
  refine ⟨div_pos hm hd, ?_⟩
  rw [div_lt_one hd]
  have : 0 < c / k := by positivity
  linarith

/-- Roofline form: compute time `F / P` is at most memory time `M / B` iff the arithmetic intensity `F / M`
is at most the ridge point `P / B`. -/
theorem memoryBound_iff {F M P B : ℝ} (hM : 0 < M) (hP : 0 < P) (hB : 0 < B) :
    F / P ≤ M / B ↔ F / M ≤ P / B := by
  rw [div_le_div_iff₀ hP hB, div_le_div_iff₀ hM hB]
  constructor <;> intro h <;> linarith [mul_comm M P]

/-! ## C3. Regret against the minimum-cost placement

For a finite set `s` of placements and a cost `c`, regret of a placement is its cost minus the minimum cost. -/

section Regret

variable {α : Type*} {s : Finset α} {c : α → ℝ}

/-- Regret of placement `a`: cost minus the minimum cost over `s`. -/
noncomputable def regret (s : Finset α) (hs : s.Nonempty) (c : α → ℝ) (a : α) : ℝ :=
  c a - s.inf' hs c

/-- Regret is never negative. -/
theorem regret_nonneg (hs : s.Nonempty) {a : α} (ha : a ∈ s) : 0 ≤ regret s hs c a := by
  unfold regret
  have := Finset.inf'_le c ha
  linarith

/-- Regret is zero exactly for the optimal placements. -/
theorem regret_eq_zero_iff (hs : s.Nonempty) {a : α} (ha : a ∈ s) :
    regret s hs c a = 0 ↔ ∀ b ∈ s, c a ≤ c b := by
  unfold regret
  constructor
  · intro h b hb
    have := Finset.inf'_le c hb
    linarith
  · intro h
    have h1 : c a ≤ s.inf' hs c := (Finset.le_inf'_iff hs c).2 h
    have h2 := Finset.inf'_le c ha
    linarith

/-- The oracle is a lower bound: no placement costs less than the minimum. -/
theorem oracle_lower_bound (hs : s.Nonempty) {a : α} (ha : a ∈ s) : s.inf' hs c ≤ c a :=
  Finset.inf'_le c ha

/-- Total regret (with a non-negative controller overhead and prediction cost) is at least the overhead. -/
theorem totalRegret_ge_overhead (hs : s.Nonempty) {a : α} (ha : a ∈ s) {o : ℝ} (ho : 0 ≤ o) :
    o ≤ (c a + o) - s.inf' hs c := by
  have := regret_nonneg (c := c) hs ha
  unfold regret at this
  linarith

end Regret

/-! ## C4 (H13, KV prefetch). Break-even accuracy when the loss and the cost scale with the stall

If the stall saved by a correct prefetch is `T`, a wrong prefetch costs `u * T` (link utilisation `u`) and the predictor costs `γ * T`,
the break-even accuracy is `(u + γ) / (1 + u)`. -/

/-- Prefetching pays off exactly when the accuracy exceeds `(u + γ) / (1 + u)`. -/
theorem prefetch_threshold_iff {p T u γ : ℝ} (hT : 0 < T) (hu : 0 ≤ u) :
    0 < netBenefit p T (u * T) (γ * T) ↔ (u + γ) / (1 + u) < p := by
  have hBL : 0 < T + u * T := by nlinarith
  rw [netBenefit_pos_iff hBL]
  have h : (u * T + γ * T) / (T + u * T) = (u + γ) / (1 + u) := by
    rw [div_eq_div_iff hBL.ne' (by linarith : (1 + u) ≠ 0)]
    ring
  rw [h]

/-- The threshold never decreases as the link gets busier (as long as the predictor costs at most the stall it hides). -/
theorem prefetch_threshold_mono_u {u u' γ : ℝ} (hu : 0 ≤ u) (huu : u ≤ u') (hγ : γ ≤ 1) :
    (u + γ) / (1 + u) ≤ (u' + γ) / (1 + u') := by
  have hu' : 0 ≤ u' := le_trans hu huu
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_nonneg (sub_nonneg.2 hγ) (sub_nonneg.2 huu)]

/-- A predictor that is right more than half the time pays off whenever `u + 2γ < 1`. -/
theorem prefetch_threshold_lt_half {u γ : ℝ} (hu : 0 ≤ u) (h : u + 2 * γ < 1) :
    (u + γ) / (1 + u) < 1 / 2 := by
  rw [div_lt_iff₀ (by linarith)]
  linarith

/-- The threshold is below 1 (some predictor can pay off) iff the predictor costs less than the stall it hides. -/
theorem prefetch_threshold_lt_one_iff {u γ : ℝ} (hu : 0 ≤ u) :
    (u + γ) / (1 + u) < 1 ↔ γ < 1 := by
  rw [div_lt_one (by linarith)]
  constructor <;> intro h <;> linarith

/-! ## C5. No balance point when the KV cache dominates

One decode step with batch `B`: compute time `B * f`, memory time `w + B * k` (`w` = weight read time, read once; `f`, `k` = compute time and KV read
time per sequence). With context `T`: `f = a + c * T` (weight matmuls plus attention) and `k = s * T`. -/

/-- If the KV read time per sequence is at least the compute time per sequence, no batch size makes the step compute-bound. -/
theorem no_balance_point {w f k B : ℝ} (hw : 0 < w) (hfk : f ≤ k) (hB : 0 ≤ B) : B * f < w + B * k := by
  nlinarith

/-- Otherwise the step is compute-bound exactly for batches larger than `w / (f - k)`. -/
theorem compute_bound_iff {w f k B : ℝ} (hfk : k < f) : w + B * k < B * f ↔ w / (f - k) < B := by
  rw [div_lt_iff₀ (by linarith)]
  constructor <;> intro h <;> nlinarith

/-- The critical context: KV reads dominate compute per sequence exactly when the context is at least `a / (s - c)` (needs `c < s`). -/
theorem kv_dominates_iff {a c s T : ℝ} (hs : c < s) : a + c * T ≤ s * T ↔ a / (s - c) ≤ T := by
  rw [div_le_iff₀ (by linarith)]
  constructor <;> intro h <;> nlinarith

/-- A larger KV read cost per token (for example a cache with more bits per element) lowers the critical context, so shrinking the KV raises it. -/
theorem critical_context_anti {a c s s' : ℝ} (ha : 0 ≤ a) (hs : c < s) (hss : s ≤ s') :
    a / (s' - c) ≤ a / (s - c) := by
  exact div_le_div_of_nonneg_left ha (by linarith) (by linarith)

/-! ## C6. Stability of a shared link with wasted prefetches

Own demand fetches every `n` transfer times, background utilisation `u`, prefetch accuracy `p`: the offered load of an always-prefetch policy is
`u + (2 - p) / n` (one transfer per epoch, plus one wasted transfer with probability `1 - p`). -/

/-- The always-prefetch policy keeps the link below saturation exactly when `p > 2 - n * (1 - u)`. -/
theorem stable_iff {u p n : ℝ} (hn : 0 < n) : u + (2 - p) / n < 1 ↔ 2 - n * (1 - u) < p := by
  constructor
  · intro h
    have h' : (2 - p) / n < 1 - u := by linarith
    have := (div_lt_iff₀ hn).1 h'
    nlinarith
  · intro h
    have h' : 2 - p < (1 - u) * n := by nlinarith
    have := (div_lt_iff₀ hn).2 h'
    linarith

/-- If reactive fetching alone already saturates the link (`n * (1 - u) ≤ 1`), no predictor of accuracy at most 1 keeps it stable. -/
theorem no_stable_predictor {u n : ℝ} (hn : 0 < n) (h : n * (1 - u) ≤ 1) : ∀ p : ℝ, p ≤ 1 → ¬ (u + (2 - p) / n < 1) := by
  intro p hp hlt
  have h1 : 1 / n ≤ (2 - p) / n := div_le_div_of_nonneg_right (by linarith) hn.le
  have h2 : 1 - u ≤ 1 / n := by
    rw [le_div_iff₀ hn]
    linarith
  linarith

end Lab

#print axioms Lab.netBenefit_pos_iff
#print axioms Lab.netBenefit_at_threshold
#print axioms Lab.netBenefit_mono_p
#print axioms Lab.reactive_optimal_below_threshold
#print axioms Lab.memShare_strictMono
#print axioms Lab.memShare_mem_Ioo
#print axioms Lab.memoryBound_iff
#print axioms Lab.regret_nonneg
#print axioms Lab.regret_eq_zero_iff
#print axioms Lab.totalRegret_ge_overhead
#print axioms Lab.prefetch_threshold_iff
#print axioms Lab.prefetch_threshold_mono_u
#print axioms Lab.prefetch_threshold_lt_half
#print axioms Lab.prefetch_threshold_lt_one_iff
#print axioms Lab.no_balance_point
#print axioms Lab.compute_bound_iff
#print axioms Lab.kv_dominates_iff
#print axioms Lab.critical_context_anti
#print axioms Lab.stable_iff
#print axioms Lab.no_stable_predictor

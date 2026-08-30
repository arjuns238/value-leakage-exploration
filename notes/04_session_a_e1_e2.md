# Session A results — E1 (preference vector) + E2 sampling (2026-08-25)

*Pod stopped at session end. Status: E1 COMPLETE ✅, E2 sampled (judging pending).*

## E1 — the instrument exists and works

**Pipeline** (Gilg's, adapted): 125,000 pairwise task choices (10k tasks, temp 0.7, truncation-at-32 speedup — choices committed in first tokens, completions unused) + 12,500 eval choices (1k disjoint tasks). Early-stopped after active-learning iteration 1 (asri's call); Thurstonian fits converged; utilities sane (harmful/manipulative μ=−10, math/creative/coding μ=+10). Refusal/unparsed 2.1% with regex-only choice parsing.

**Probe**: ridge on task_mean residual activations (10,991 tasks × layers {25,32,39,46,53}), standardized, alpha-swept. **Held-out r = 0.792 (L25), pairwise acc 0.834**; all layers 0.776–0.792. Gilg full-pipeline reference: 0.867 — early-stop cost ≈0.08, accepted deliberately.

**Fig-3 steering sanity (THE gate) — PASSED**: differential steering (±c·mean_norm on the two tasks' token spans, L25, mean_norm 37,620 ≈ their 35,708):

| c | −0.06 | −0.03 | 0 | +0.03 | +0.06 |
|---|---|---|---|---|---|
| P(chose steered-up task) | 0.00 | 0.00 | 0.67 | 1.00 | 1.00 |

n=192/cell, coherent completions at extremes (spot-checked). Saturation even at ±0.03 — our probe steers at least as hard as theirs. Note the pitfall for future analysis: steering targets **pair roles** (task_a/b), so presentation-slot analyses flatten to 0.5 by design — always join through `ordering`/`choice_original`.

**Deviations from Gilg**: iteration-1-only utilities; regex-only choice parse (no LLM judge); max_new_tokens 32 with truncation-allowed patch; 20s→600s client timeout patch; length-sorted extraction batches. All patched in our copies of their repo, commented `LOCAL PATCH (vd)`.

## E2 — sampled, awaiting judging
10,000 stated scores (100% parse; 500 activities spread 3–95; food/books/scenic top, adult venues/hunting/pseudoscience bottom), 10,000 "pick randomly" responses, 3,000 matched genuine-preference controls. All on laptop (`artifacts/choosing/`). Next: subagent choice-judging (isolated work dirs + mechanical cross-checks), then selection-rate vs stated-score correlation (the leakage number), then E3/E4 design.

## Artifacts
- Laptop: `artifacts/gilg_preferences/results/probes/vd_main/` (probe weights + manifest), `artifacts/gilg_runs/` (steering checkpoint, pairs, norms, measurement backups), `artifacts/thurstonian_0d87ba86.csv` (train utilities), `artifacts/choosing/` (E2 samples), measurement checkpoints (train+eval).
- Pod volume only (asri's call): activations npz (6.7GB, `gilg_preferences/activations/gemma-3-27b_it/pref_main/`).
- Pod notebook: `vd_experiments/02_session_a.ipynb`.

## Session-B queue (needs GPU)
E3 (probe-score gap predicts covert picks — needs activations for pick prompts), E4 (steer during covert picks), then E5/E6. Prereq before GPU: E2 judging + correlation (offline), and E3/E4 harness design (prefill + hooks on the pick prompts).

## Ops lessons (see also memory)
Gilg client timeout 20s → mass silent drops against vLLM batching; completions truncatable at 32 tokens for choice-only measurement (17× speedup); extraction was CPU-bound per batch (~7.5s constant), length-sort helps the tail; `Message` is a TypedDict (`.get`, not `getattr`); origins must be enum member names in measurements.yaml; steering checkpoint analysis must use pair roles.

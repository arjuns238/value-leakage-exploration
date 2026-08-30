# Roadmap 03 — Finding and mediating the value direction (agreed with asri, 2026-08-25)

## Goal
Find a linear direction (or low-rank subspace) carrying Gemma-3-27B's valuation of options/answers; test whether it **reads** (predicts) and **writes** (steers/mediates) **value leakage** — preference influence that survives an instruction to suppress it (Betley) — using Gilg et al.'s preference vector as the candidate instrument.

## What exp02 (Donation Bet, notes/02) established
Value→behavior chain decomposes: (1) valuation (present, verbalized in 77% of rollouts) → (2) intent ("adjust for the good cause", overt) → (3) execution (broken: conditional inversion ~18%, numeric-control failure ~21% in the above-good arm). Bias metric = motivation × competence; Gemma fails competence, so Donation Bet can't host the mediation test. Words and numbers are dissociated — which makes output-language readouts (stated intent) a first-class dependent variable.

## Key facts from Gilg's paper constraining our design
- Steering the preference vector at L23 (|c| ≤ 0.06 of resid norm, task-token span) fully controls **overt** pairwise choice (their Fig. 3). Sufficiency: proven. We do not re-run this beyond a sanity check.
- **Their Fig. 41: rank-1 ablation of the direction leaves overt choice unchanged** (0.98–0.99 agreement) at every layer incl. L23 — not necessary; computation routes around rank-1 deletion. They explicitly leave rank-k subspace ablation open. → our necessity tests use rank-k projection and activation patching, never rank-1 deletion.
- Probe: ridge on residual stream, end-of-turn token, L32 readout (r ≈ 0.867 held-out); persona finding = same axis, different loadings (extension E8).

## Experiments (priority order)

**E1 — Instrument: re-derive preference vector + Fig 3 sanity.** Full-ish Gilg pipeline on our checkpoint: ~10k-task pairwise choices (adaptive), Thurstonian utilities, ridge probe (L32 readout), steering S-curve at L23 on ~100 pairs + random-direction control. Gate: monotone large swing. ~3–5 GPU h.

**E2 — Find real leakage: Choosing Activities, expanded.** ~500 activities (LLM-generated in Betley's style + their 100), stated 0–100 scores (5 paraphrases × repeats), "pick randomly" pairwise picks ~10k variations (≥30 decisive appearances/activity), choice+randomness judging via subagents (isolated work dirs + mechanical cross-checks). Deliverable: selection-rate vs stated-score r with n≈500. Also: 500 activities is enough to train an activity-specific utility probe if E1's task probe doesn't transfer. ~2–3 GPU h.

**E3 — Read: probe predicts covert picks.** Per "pick randomly" rollout, probe-score gap between the two options' tokens → does it predict the pick? Also: is the gap suppressed vs the overt context (locates whether the "be random" instruction gates the preference computation or only the narration)?

**E4 — Write: steer during covert picks.** Gilg's steering (±v, L23) applied inside Betley's "pick totally randomly" prompt, same pairs/coefficients that flip overt choice. Outcomes tracked separately: pick distribution, and the response's randomness claim (disclosure judge). Three informative worlds: (i) picks flip & still claims random → same axis mediates covert leakage, disclosure blind; (ii) picks don't move (though overt flips) → covert pathway ≠ overt pathway; (iii) picks flip & narration changes → behavior and disclosure coupled.

**E5 — Necessity, properly: rank-k projection sweep + activation patching.** Project out top-k probe subspace (sweep k) during covert picks → leakage-death curve; patch option-token activations between liked/disliked activities → does the "random" pick follow the patch? (Fig. 41 predicts rank-1 does nothing; don't headline it.)

**E6 — Value-sway under ambiguous prompts (asri's addition).** Steering sweep c ∈ [−0.06, +0.06] while answering Donation Bet prompts. Dependent variables are **output-language measures**, not just the number: stated-intent rate (baseline 77%), stated aim direction, bet-mention rate, magnitude of the common downward shift, and the estimate. Tests which chain link (valuation/intent/execution) the vector writes to; Gemma's word–number dissociation makes "values sway in words but not numbers" (or the reverse) detectable.

**E7 — Prompted-CoT Donation Bet.** "Think step by step in <thinking>, then answer." If bias becomes nonzero → leakage is gated by serial reasoning, and we gain a second mediation arena + probe readout at plan-tokens vs answer-tokens. Also fixes/exposes the two competence failures separately (plan direction visible in CoT).

**E8 (extension) — Personas.** Choosing Activities under 2–3 persona system prompts: does leakage flip with persona while the *same* vector predicts/mediates it (Gilg's shared-axis claim extended to covert leakage)? Contingent on E1–E4.

## Controls & smaller items
- Equal-charity donation-bet variant (anchoring vs value-drive for the common shift); shifted thresholds (20th/80th pct — restores metric sensitivity + tests anchor-tracking); explicit-direction wording ("good cause needs your estimate ABOVE X") — competence control.
- Judge QA standard: subagents with isolated work dirs, mechanical cross-check on every extraction round.

## Budget & sequencing
Pod session A (~1 long day, ~$15–20): E1 + E2 sampling + E7 sampling (+controls sampling). Offline: judging + analysis. Pod session B: E3–E6 (need hooks: HF/TransformerLens or nnsight, not vLLM — activations & steering). E5/E6 sized after E4 results.

## Risks
Task-probe may not transfer to activities (fallback: activity-trained probe from E2 data); Choosing Activities could be null for Gemma (then Gemma may genuinely not leak — steering-only story remains); downscaled-vs-full probe quality (E1 run at full scale to remove this); vLLM (sampling) vs hooked-HF (interventions) checkpoint parity must be verified (same weights, greedy-match spot check).

## Session-A operational notes (prepped 2026-08-25, before pod start)
- **Gilg repo adapted** (`gilg_preferences/`): `configs/vd/{measure_train_10k,measure_eval_3k,extract_pref,probe_main,steer_fig3}.yaml`; parser patched to `RegexOnly` (no API keys — monitor refusal rate; their `vllm` backend is hardcoded to http://localhost:8000/v1, model name `gemma-3-27b`).
- **vLLM must serve BOTH names**: `--served-model-name gemma-3-27b-it gemma-3-27b` (first for our harness entry — cache-hash pinned; second for Gilg's registry).
- **E2 sampler**: `value_leakage/local_tools/run_choosing_activities.py` (stages score/pick/prefer, resumable, matched pairing; dry-run tested). Activities file: 100 originals + 400 generated (pending) → combined YAML.
- **E7**: `main_experiment_cot` registered in the harness (accurate wording + `_COT_SUFFIX`, own baseline).
- **Pod order**: (1) dry-run Gilg configs; (2) vLLM up → E1-Step1 train+eval measurements (~2-4h, temp 0.7 short-ish completions), E2 stages, E7 smoke+full sampling — all HTTP against vLLM, can interleave; (3) judge rounds offline (subagents, isolated dirs, cross-checks); (4) STOP vLLM → E1-Step2 activation extraction (HF, batched, no generation) → pull npz (~1GB) + run dirs local; (5) Step 3 probes can run on pod CPU or locally; (6) Fig-3 steering (HF hooks) after probes; (7) pull everything, stop pod.
- Gilg extraction needs `measured_task_ids.json` (union of train+eval task ids) written to /workspace/gilg_runs/ after Step 1; steering needs `pairs_benign.json` built from utilities + per-layer `mean_norm` from `src/steering/calibration.py`.
- Install on pod: `uv pip install -e /workspace/gilg_preferences` into vd-venv **without extras**; verify `import vllm` still works afterwards (dependency-clobber check), and `python -c "import torch; torch.cuda.is_available()"`.

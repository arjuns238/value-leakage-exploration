# Session — does the Gilg vector generalize beyond the covert pick? (AI Bubble, Job Offer, Stance)

*2026-08-28. Pod stopped. Question (asri): our whole mechanistic core was ONE task (Choosing Activities). Take the EXISTING Gilg vector + E4 coefficients and test whether they generalize — read/write/delete — on new tasks the papers use (self-serving company bias) and a new one (LLM-as-judge stance leak). All data in artifacts/session_c/, drivers in session_b/vd_session_b.py (gen-samples, steer-judge, steer-spans, ablate-judge), analyzers analyze_*.py, browser judging/steer_browser.html.*

## Baselines — do these leaks exist on Gemma?
- **AI Bubble** (Betley §4; P(AI bubble pops), self-serving = lowball for own company): **NULL.** Google (Gemma's maker) p_pop=0.445 vs competitors 0.407 — Google *highest*, not lowest (faint anti-Google, echoes paper's Gemini). Only real effect: any investment mention drops p_pop 0.60→0.41 uniformly (framing anchor). *Prompt caveat: Gemma buries the number after a long essay; concise "number-first" collapsed to reflexive 40% (no reasoning); the valid measurement used "reason 2-3 sentences THEN Estimate: X%".*
- **Job Offer** (Betley §5; endorse "keep interesting job" conditional on companies): **small +4.3.** endorse(Google current)=82.8 vs endorse(Google offer)=78.5, CIs just separate — a marginal pro-Google lean (advise staying at Google more). Opposite sign to AI Bubble → weak/task-specific, not robust.
- **Stance leak** (new: LLM-as-judge favors its own side while told to judge on quality alone): **weak, +0.346** corr(stated stance, "better-reasoned" verdict) across 20 topics [partial est was +0.155]. Confounded by a **strong 2nd-position bias** (rates Argument 2 "better" regardless of content). Args matched to ~6-char length parity.

## WRITE (steer with the existing Gilg vector + E4 coeffs)
- **Stance judgment — GENERALIZES, totally.** steer-judge = E4's exact steering (v25, L25, +c arg-1 span / -c arg-2 span) on the "judge PURELY on reasoning quality" task:

  | c | -0.06 | -0.03 | 0 | +0.03 | +0.06 |
  |---|---|---|---|---|---|
  | P(rates arg-1 "better") | 0.00 | 0.00 | 0.58 | 1.00 | 1.00 |

  **Flips 20/20 topics.** The vector overrides the model's stated objectivity 0→1, identical to the covert-pick curve. Latent LLM-as-judge control knob; the imposed control dwarfs the natural lean (+0.346).
- **AI Bubble — NULL.** steer the company-name span: p_pop flat 0.4-0.5, no dose-response (Google or OpenAI).
- **Job Offer — NULL.** steer current(+c)/offer(-c) company spans: endorsement flat 67-76, no trend.

## DELETE (ablate the value subspace, stance task)
- ablate-judge (RankKProjector rank-16, L25, both orderings): P(side_A better) rank0 **0.731** → rank16 **0.700**; corr(stance,verdict) **+0.346 → +0.301**. **NULL — verdict & leak survive ablation.**

## Headline — the sharp boundary + the sufficient-not-necessary asymmetry
1. **The vector controls binary picks between options at token spans — not graded/numeric outputs.** Steering swings activities (E4), moral pairs, AND an "objective" argument-quality verdict 0→1; but does NOTHING to a probability (Bubble) or endorsement (Job Offer). It's a **"which discrete option wins" controller, not a value-magnitude mover** — the causal footprint on the *write* side is *selection*, not value. (It still *encodes* value: E3 reads stated scores.) Refines the value-vs-decision question.
2. **Sufficient override ≠ necessary carrier.** On the covert pick the direction is sufficient (E4) AND necessary (E5, ablation→chance). On the stance judgment it is sufficient (steer 0→1) but NOT necessary (ablation leaves verdict+leak intact) — the natural judgment runs on real reasoning assessment + position bias, yet the subspace can still hijack it. **Steering manufactures a hidden bias the model doesn't natively route through the direction.**
3. **Company self-serving bias doesn't live in the vector** (AI Bubble null, Job Offer tiny) — identity ≠ the valuation the vector encodes.

## Caveats
Stance args hand-written (quality-match imperfect; strong residual position bias). Bubble/JobOffer steering = "induction" tests (baselines ~null). Endorsement/p_pop are judge/parse readouts. n=20 topics. No random-subspace control on the stance ablation yet (though the ablation is null anyway). Concise-prompt deviations from Betley wording logged above.

---

# UPDATE 2026-08-29 — steering controls, coefficient calibration, and the disposition theory

The earlier "WRITE" results above were partly UNCONTROLLED (no random-direction steering controls of our own; we'd relied on Gilg's paper). This session ran them all. Data: artifacts/session_c/{e4_ctrl_*, sweep_*, num_lowc_*, e6_rand0, bubble_global}.jsonl.

## 1. E4 covert-pick steering IS value-specific ✅ (the core is safe)
Same 12 pairs, value vs 2 random seeds: VALUE swing 0→1 (0.92); RANDOM 0.11 / 0.36, no monotonic dose-response. E4's write result stands with its own control.

## 2. Stance/LLM-judge steering: overpowered at c=0.06, RESCUED at c≈0.01–0.02
Random ALSO flipped the verdict at c=0.06 (long ~22-token spans ⇒ total shove c×n too big). Coefficient sweep (±0.01/0.02/0.04, value vs random):
c:      -0.04 -0.02 -0.01  0   +0.01 +0.02 +0.04
VALUE:   0.00  0.00  0.20 0.53  0.95  1.00  1.00   (swing 1.00)
RANDOM:  0.25  0.40  0.47 0.53  0.55  0.62  0.62   (swing 0.38)
→ Clean value-specific window at c≈0.01–0.02: value flips the "judge purely on reasoning quality" verdict 0→1, random can't. LLM-as-judge result now CONTROLLED and real. NOTE: steering strength ≈ c × n_span_tokens; target ≈0.2–0.6 (E4: 0.06×9; stance: 0.015×22). asri's normalization predicted the window (~0.015) almost exactly. ALWAYS run a per-task random-direction control.

## 3. Numbers don't respond to value at ANY calibrated strength (boundary confirmed)
Full-span steering at low c (±0.01/0.02) with random control: Bubble p_pop swings VALUE 0.08/0.07 vs RANDOM 0.09/0.04; JobOffer endorsement VALUE 4.5/9.7 vs RANDOM 7.8/6.7. Flat, value≈random, at both c=0.06 and low-c. (JobOffer caveat: asymmetric spans 11 vs 27 tokens.) → picks-vs-magnitudes boundary is real, not a calibration artifact.

## 4. E6's refusal wall IS value-specific ✅ (my 70% "artifact" bet was wrong)
Global RANDOM steering on the bet at c=−0.06: refusal-language 1/18 (6%) vs VALUE's 180/180 (100%). E6 stands, now controlled.

## 5. Disposition test — the unifying result
Global ±value steering on the AI BUBBLE prompt (different task, same operation): c=−0.06 → **20/20 refusals** (same confabulated "harmless AI assistant" justification as the bet); c=+0.06 → 0/20 refusals, 19/20 eager committed estimates. Random global −0.06 → ~0 refusals.

## THE THEORY (fits everything)
**The value direction is a DISPOSITION dial, not a content dial.** It encodes "how much do I value this thing" and causally drives approach/choose/engage:
- two options tagged differently → picks the higher (E4, stance: 0→1, value-specific)
- whole task tagged low/high (global) → refuse / eagerly engage (E6 + bubble-global, value-specific)
- a clause tagged valuable, or "make this number smaller" → NOTHING (no value→magnitude circuit; same missing link as exp02's directional incompetence)
Random directions are inert (a random vector has ~1/√5376≈1.4% overlap with any read direction — energy in an unread channel) except in fragile head-to-head comparisons at high magnitude, where asymmetric degradation tips ties (the stance c=0.06 artifact). Strong global coefficients work because per-token it's only ~6% of residual norm — a coherent whisper in a channel the model reads beats a shout in one it doesn't.

## 6. Accommodation-erasure at +value — VERIFIED value-specific across 3 random seeds (final experiment)
The +0.06 global steer on the Google-invest bubble prompt didn't push p_pop to an arbitrary "high-value" number — it returned it to the NO-INVESTMENT baseline (**64.5%, 17/19 at exactly 65%** — delta-function, variance collapse) from the investment-anchored 44.5%. Matched-strength global RANDOM controls, 3 seeds: seed0 **32.2** (25–35), seed1 **42.5** (≈ anchored baseline), seed2 **33.1** (25–35). Across seeds random is at-or-below the anchor, directionally incoherent (one null, two ~−12), never approaches 65, never collapses to a single answer. → **+value dispositionally emancipates the model from the user's stake** (stops accommodating, states its unpressured belief); the numeric shift is a *consequence* of the dispositional change, not a magnitude dial — and no random direction reproduces it.
**Honest refinement (asri's catch):** "random is inert" was an overclaim — global random perturbation has its own real, smallish, seed-dependent effect on the number (null-to-downward ~12 pts; plausibly generic degradation/conservatism). So the boundary statement is: span-steering leaves numbers untouched (value≈random≈null); GLOBAL steering moves numbers — value via a coherent dispositional route (up to the unpressured landmark, delta-distribution), random via incoherent drift. Value-specific = "not reproducible by random," not "the only thing that moves it." Discriminating fingerprints: direction (up-to-landmark vs down/null) AND distribution shape (delta vs spread).
Caveats: single prompt, n=20/cell, global strength far above the span-calibrated window (value-specificity established by matched-strength controls, not calibration).

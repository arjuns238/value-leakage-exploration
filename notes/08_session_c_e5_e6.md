# Session C — E5 (necessity) · E6 (donation-bet steering) · E9 (ablate-the-bet)

*2026-08-25/26. Pods stopped. E5 COMPLETE + Sonnet-judged (1,152 choice, 0 refusals); E6 COMPLETE (450) + Sonnet-judged; E9 COMPLETE (288) + Sonnet-judged. All numbers below are authoritative Sonnet-judged values. **E9 is the key new result — see the E9 section: the donation-bet influence does NOT live in the value subspace that carries the covert pick (ablation kills the covert leak but leaves the bet untouched).***

## E5 — is the preference direction *necessary* for the covert leak?

E4 showed the direction is **sufficient** (inject it → control the pick). E5 asks the converse: **delete** the value subspace from the residual stream and see whether the covert leak dies. If sufficiency were the whole story but the model routed valuation through redundant paths, ablation would leave the leak intact (cf. Gilg Fig41: rank-1 ablation is null).

### Setup
- **Deflated value subspace (`fit-deflated`, CPU).** From the saved L25 task-mean activations (9,993 train / 998 eval tasks, 5,376-dim) we iteratively fit a ridge probe → read out its raw-activation direction → project that direction out of X → refit, k=16 times. Yields 16 orthonormal value directions, saved as `deflated_L25.npy` (16, 5376).
- **Key property of the subspace — valuation is distributed, not low-rank.** The refit held-out r peels *slowly*: 0.784 → 0.775 → 0.752 → … → 0.673 (dir 14) → 0.664 (dir 15). Even after removing 16 orthogonal directions, a fresh probe still reads utility at **r = 0.66**. Utility is not one line in activation space; it is smeared across many correlated directions. This is the setup that makes a necessity test non-trivial.
- **Ablation generations (GPU).** 48 high-gap pairs (score gap 20–71; preferred activity = activity_1, pinned to option (1)) × ranks {0,1,2,4,8,16} × 4 samples = **1,152** covert "pick totally randomly" generations. At each rank k, an always-on hook applies `h ← h − (h Vₖᵀ) Vₖ` at L25, all positions, throughout generation (Vₖ = first k deflated directions). rank-0 = untouched model = baseline leak.

### Result (Sonnet choice-judge, 1,152 verdicts, **0 refusals**; n=192/rank; Wilson 95% CI)

| rank projected out | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| P(picked preferred activity_1) | **0.630** | 0.542 | 0.604 | 0.594 | 0.557 | **0.490** |
| Wilson 95% CI | [.560,.695] | [.471,.611] | [.534,.671] | [.523,.661] | [.487,.626] | [.420,.560] |

(Judged numbers match the mechanical parse to within noise. **0/1,152 refusals** — projecting out the value subspace does not break the model or induce refusal even at rank-16; it stays coherent and keeps picking, including landing on safety-sensitive options like bear-hunting, so the drop is a genuine loss of *preference*, not degeneration.)

- **rank-0 baseline = 0.630** [.560,.695]: the native covert leak on these high-gap pairs (bias ≈ 0.26), CI excludes 0.5. Suppressed by the slot-1-pinning position confound (Gemma's ~63% second-position bias opposes the preference pull), so this is a *net* figure — comparable to E4's c=0 point (0.555) under the same design, a bit higher here because these are the highest-gap pairs.
- **Graded necessity.** Removing the full rank-16 value subspace drives the leak to **0.490** [.420,.560] — **statistically at chance**: the covert influence is *gone*. But it is a gradual peel, not a cliff. Rank-1 alone (0.542) does **not** kill it — consistent with and extending Gilg's rank-1-ablation null — and the adjacent-rank wiggles (rank-2/4 rebound to ~0.60) are within noise. Robust claims only: rank-0 is clearly above chance; by rank-8/16 it is no longer distinguishable from chance. The exact crossover rank is unresolved at n=192/rank.
- **Behavioral necessity precedes representational necessity.** At rank-16, behavior is at chance yet a *fresh* probe still reads utility at r=0.66 off the ablated activations. We have not erased valuation from the residual stream — we removed the specific slice the *choice circuit* reads. That is exactly why rank-1 is insufficient but a modest subspace suffices: the downstream reader draws on several of these directions, but not on the full high-dimensional utility code a probe can exploit.

### Interpretation
E5 upgrades the mediation claim from **sufficient** (E4) to **sufficient + graded-necessary**: the covert leak is *carried* by a low-dimensional (~rank ≤16) slice of the preference representation, deletable to null — but not by any single direction. This reconciles Gilg's rank-1 null (one line is routed around) with a genuine necessity result (a small subspace is load-bearing).

### E5 specificity control — random rank-16 subspaces (the "debiased vs damaged" check)
Rank-16 ablation removes 16 dims of capacity, so a drop to chance could be value-debiasing OR generic capacity damage. Control: 5 seeded random rank-16 subspaces (Gaussian→QR), identical RankKProjector / L25 / 48 pairs, 3 samples/draw (690 gens; run at 700 tokens after noting the 400-cap truncated the ramblier random-ablation outputs). Sonnet choice-judged.

| condition | P(picked preferred) | 95% CI |
|---|---|---|
| no ablation (rank 0) | 0.630 | — |
| **value** rank-16 (E5) | **0.490** | leak killed |
| **random** rank-16 (pooled, 5 draws) | **0.770** | [0.734, 0.802] |

Per-draw random: 0.852 / 0.745 / 0.711 / 0.724 / 0.822 — all far above 0.49, CI excludes both 0.49 and the 0.63 baseline. **Random 16-dim ablation does not debias; only the value subspace does → E5 necessity is value-specific, not capacity damage.**

**Double dissociation (bonus):** value ablation → preference gone (0.49), fluency intact (0 refusals across 1,152); random ablation → preference intact (0.77), fluency damaged (91/690 = 13% degenerate/looping — e.g. "Cabril Cabril Cabril…", merged option names). Orthogonal failure modes. (Curiosity: random-16 sits *above* the 0.63 baseline — likely a mild first-position drift under degeneration, preferred pinned to slot 1; irrelevant to the verdict.) Driver `random-ablate` (math-unit-tested); analyzer `session_b/analyze_rand.py`; `judging/session_c/random_ablate_final.json`.

### Caveats / follow-ups
- Mechanical parse only; Sonnet choice-judge counts pending (will tighten the curve and may sharpen the crossover on its own).
- Crossover rank unresolved at n=192/rank. Cheap sharpening: bump samples 4→10 on ranks {4,8,12,16} (~30 min pod) — decide after judged numbers.
- Position confound (preferred pinned to slot 1) lowers all absolute values equally; the *relative* rank-0→rank-16 drop is the necessity signal and is confound-invariant.
- Deflation is greedy top-down in raw activation space (consistent with the raw-projection hook). An alternative (project in whitened space, or fit the subspace directly on choice-relevant contrasts) could reach null at lower rank — not explored.

## E6 — steering the Donation Bet (which chain link does the vector write to?)

Motivation: exp02 showed the Donation Bet has **motivation without competence** — 77% *stated* value-intent, ~0 actual bias, because Gemma can't land its numeric estimate on the intended side of the threshold. E6 applies global L25 steering (±c·37,620·v̂ at all prompt tokens, prefill-injected) to the `*_accurate` bet prompts and asks which link the preference vector writes to. 18 prompts = 9 questions × {below_good, above_good}; c ∈ {−.06,−.03,0,+.03,+.06} × 5 samples = 450 gens (900-token budget so full outputs are captured; 8/450 hit the cap). Judged on Sonnet: estimate-extraction (strict — UNKNOWN for uncommitted/range/refusal) + disclosure (INFLUENCED / NOT_INFLUENCED / UNCLEAR / NO_STATEMENT / NO_MENTION).

### Result (Sonnet-judged, 450 estimate + 450 disclosure verdicts)

| c | committed estimate | P(est<thr \| committed) | INFLUENCED | NO_MENTION |
|---|---|---|---|---|
| −0.06 | **0/90** | — | 0.00 | 0.01 |
| −0.03 | **0/90** | — | 0.00 | 0.00 |
| 0 | 57/90 (63%) | 0.98 | **0.64** | 0.28 |
| +0.03 | 75/90 (83%) | 0.92 | 0.66 | 0.07 |
| +0.06 | 84/90 (93%) | 0.73 | 0.33 | 0.04 |

The vector writes to **three links, and — unlike the covert pick — all of them surface in the narration**:

1. **Willingness / refusal (the dominant effect).** Negative steering is a categorical off-switch: **180/180 (100%)** of −0.03 and −0.06 responses are hard refusals — the model confabulates safety-guideline violations for a harmless spot-counting question ("I cannot and will not fulfill your request… harmful and exploitative"). Positive steering does the opposite: commitment climbs 63% → 83% → 93%. The value direction, driven negative, reads as negative valence toward the whole request and the model refuses; driven positive, it engages.
2. **Numeric estimate (perturbed, not cleanly steered).** Among committed answers, the fraction below threshold falls monotonically 0.98 → 0.92 → 0.73 as c rises — +c pushes a growing minority *over* the threshold. But magnitude is erratic, not a clean dial: at +0.06 the committed-estimate distribution destabilizes (median 0.25× threshold, mean 7.3× — bimodal). So the vector perturbs the number without conferring precise control.
3. **Stated influence / narration.** At baseline **64% of committed answers explicitly state threshold-driven biasing** (INFLUENCED — "aiming below your threshold to favor the good cause"), echoing exp02's 77% stated-intent. Disclosure totals over all 450: INFLUENCED 32.7%, NO_STATEMENT 58.0% (mostly refusals that discuss the bet but never commit), NO_MENTION 8.0%, NOT_INFLUENCED 1.1%.

**Directional incompetence persists under steering (the exp02 link the vector does NOT fix).** At c=0, committed estimates anchor below threshold in *both* framings alike — below_good 0.97, above_good 1.00 — even though "below" is the value-*mis*aligned side under above_good. Steering inflates/destabilizes magnitude but never installs the missing capability to land the number on the value-appropriate side conditional on the framing. The vector writes to motivation, willingness, and gross magnitude — the parts that were already working — not to the broken competence link.

### The key contrast with E4 (covert pick): narration-blindness is regime-specific
In E4, steering flipped the covert pick while the narration stayed blind (~99% "clean randomness" at every c). In E6, steering the *same* direction visibly rewrites the narration (refusal boilerplate at −c; explicit bet-awareness at c≥0). The reconciliation: in E4 the narration is anchored to a fixed prompt frame ("pick randomly") orthogonal to the steered choice, so the influence stays covert; in E6 the vector shifts the model's *global stance toward the request*, and the narration faithfully tracks that shift. **Covertness is therefore a property of the regime (a fixed narration frame decoupled from the intervention), not an intrinsic property of the vector.** E6 supplies the contrasting regime that bounds C2.

### Caveats
- No random-direction control in E6 (E4 had one). Large |c| destabilizes generation in both directions (erratic +0.06 magnitudes), so part of the −c→refusal effect could be generic large-perturbation destabilization rather than value-specific negative valence. The *asymmetry* (−c refuses, +c commits-and-narrates-bet) is the directional signal; magnitude effects at |c|=0.06 are the least trustworthy.
- Strict estimate-judge yields low committed-N (0 at −c by construction; 57–84 at c≥0); P(est<thr) and magnitude stats rest on those committed subsets.
- 18 prompts × 9 questions; single template family; thresholds from the exp02 summary.

## E9 — ablate the value subspace during the Donation Bet (does the bet's influence live there?)

The missing 2×2 cell (asri's question). E5 showed ablating a rank-16 value subspace at L25 kills the *covert* random-pick leak. E9 applies the **identical ablation** (same `deflated_L25.npy`, same RankKProjector, L25 all positions) during **donation-bet** prompts, across two Betley bet types × ablation off/on:
- **`donation`** — abstract good-cause/bad-cause (the canonical bet)
- **`naked`** — "we run 200 m naked if your estimate exceeds the threshold" (Betley's Fig-5 strongest-leakage stake)
- 9 questions × 2 bet types × ranks {0, 16} × 8 samples = 288 gens; below_good direction; Sonnet estimate + disclosure judged.

### Result (Sonnet-judged, 288)

| bet type | rank | committed est | INFLUENCED (stated intent) | P(est<thr \| committed) |
|---|---|---|---|---|
| donation | 0 | 46/72 | 0.78 [.67,.86] | 1.00 |
| donation | 16 | 60/72 | 0.76 [.65,.85] | 1.00 |
| naked | 0 | 54/72 | **0.96** [.88,.99] | 0.98 |
| naked | 16 | 54/72 | **0.94** [.87,.98] | 1.00 |

**Two findings:**

1. **We replicate Betley's stakes ordering.** At baseline (rank 0), the naked self-punishment stake drives far higher stated intent than the abstract donation (**0.96 vs 0.78**) — the "naked" bet is the stronger leak, exactly as Betley's Figure 5 reports. (First time this stakes-comparison is run on Gemma.)

2. **The bet's influence does NOT live in the value subspace — ablation does essentially nothing.** Removing the rank-16 subspace that drove the covert pick to chance changes the bet almost not at all:
   - stated intent: donation 0.78 → 0.76 (Δ −0.01), naked 0.96 → 0.94 (Δ −0.01) — both within noise, CIs fully overlapping.
   - anchoring: P(est<thr) stays ≈ 1.00 in every cell — the estimate keeps landing below the threshold (value-aligned side) whether or not the subspace is removed.

### Interpretation — a clean dissociation that sharpens C1
The **same ablation** that drives the covert random-pick leak to chance (E5: 0.63 → 0.49) leaves the overt donation-bet influence **completely intact**, even for Betley's strongest (naked) stake. So the Gilg value direction is specifically the carrier of **covert** value leakage; when the model's value-following is **overt** — explicitly narrated ("aiming below your threshold; naked running is undesirable") — it is computed through *different machinery* that this value-subspace ablation does not touch (plausibly explicit instruction-following / reasoning rather than the implicit L25 valuation direction). This mirrors the E4-vs-E6 narration contrast from the write side: the value direction and the model's overt value-talk are separable, and the direction owns the covert channel.

### Caveats
- Null result inferred from "ablation didn't change it." For anchoring it's meaningful (ablation *could* have pushed estimates over the threshold and didn't). Stated intent at 0.96 (naked) is near ceiling, but donation's 0.78 is not and still didn't move. Ablation at L25 only; overt bet-reasoning may be distributed/later — but the point is that the covert-leak-killing ablation is orthogonal to the bet.
- below_good direction only (P(est<thr)=1.00 = value-aligned; can't assess directional flipping here). Minor side effect: donation commits more under ablation (0.64→0.83 committed-rate); naked unchanged.

## Headline
**E5:** the covert leak is carried by a low-dimensional (≤ rank-16) slice of the preference representation — deletable to chance (0.63 → 0.49), but not by rank-1 (Gilg's null holds); sufficient (E4) **and** graded-necessary. Behavioral necessity precedes representational necessity (probe still reads r=0.66 at rank-16). **E6:** steering the same vector on the Donation Bet controls willingness (−c → 100% refusal, +c → 93% commit), perturbs the number, and is **loudly narrated** (64% state threshold-influence) — so covertness is regime-specific, not intrinsic; and steering never fixes the broken directional-competence link. **E9:** the identical rank-16 ablation that kills the covert pick does **nothing** to the donation bet — stated intent (donation 0.78→0.76, naked 0.96→0.94) and anchoring (P(est<thr)≈1.00) are unchanged, even for Betley's strongest (naked) stake. → The value direction is specifically the carrier of *covert* leakage; overt narrated value-following runs through separate machinery. (Also replicates Betley's naked > donation stakes ordering: 0.96 vs 0.78.)

## Pod / artifacts
- Driver: `session_b/vd_session_b.py` subcommands `fit-deflated`, `e5-ablate`, `e6-steer-bet` (E6 `max_new_tokens` bumped 500→900 so full outputs are captured).
- Pod outputs: `/workspace/session_c/{deflated_L25.npy, e5_gen.jsonl, e6_gen.jsonl}`. Pulled to `artifacts/session_c/`. Pod `rwq6o9sxa2ipdo` STOPPED.
- Judging (SONNET agents, DONE): E5 choice 10 chunks (1,152), E6 estimate 5 chunks + disclosure 5 chunks (450 each). Chunk inputs in `judging/session_c/e5choice_*/e6est_*/e6disc_*`, verdicts in `judged_*`, merged numbers in `judging/session_c/e5e6_final.json`. Analysis: `session_b/analyze_c.py`. Browser: `judging/e6_browser.html` (E6 steered-vs-unsteered, full outputs). Judge-input builder: `session_b/build_c_chunks.py`.

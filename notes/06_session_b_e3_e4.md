# Session B — E3 (read) & E4 (write) on covert picks

*2026-08-25. Status: **BOTH COMPLETE**. Pod stopped; all judging done (E4 rounds: covertness chunks 0–7,9 on Fable, chunk 8 + choice on Sonnet; future waves: Sonnet per asri).*

## FINAL E4 RESULT — the vector controls the covert leak; the narration never notices

Fully judged (1,000 generations; ambiguous picks judge-resolved; covertness judged on every response):

| c | −0.06 | −0.03 | 0 | +0.03 | +0.06 |
|---|---|---|---|---|---|
| P(picked preferred activity) | **0.005** | 0.040 | 0.555 | 0.975 | **1.000** |
| response claims clean randomness | 98% | 99% | 99% | 98% | 99% |
| admits preference (INFLUENCED) | 0 | 0 | 0 | ~1% | <1% |
| refusals | 0 | 0 | 0 | 1 | 0 |

- **Mediation (sufficiency): total.** The same axis that steers overt choice swings the covert "random" pick across its entire range — and saturates in every gap tercile, flipping picks against 70-point preference gaps at c=−0.06 (P=0.000, n=91).
- **c=0 sanity**: 0.555 matches the slot-conditioned E2 baseline (0.541–0.568; preferred pinned to slot 1 in E4's design — position bias opposes preference; see caveat below).
- **Disclosure is blind to the intervention**: the randomness claim rate is flat (~98–99%) at every coefficient, statistically indistinguishable from the unsteered baseline (99.45%). We puppet the choice; Gemma narrates a coin flip. The only ~2/1,000 INFLUENCED cases were bear-hunting safety cases where it admitted steering *itself* away from danger — unrelated to our vector's direction.
- Coherence held: 1 refusal / 1,000, no degenerate text observed at max steering.

*Design caveat (logged during asri's bug-challenge): E4 pairs pin the preferred activity to slot 1, so the c=0 point conflates position bias (second-slot-leaning) with preference bias; compare against slot-conditioned baselines only. Dose-response is within-pair across c and unaffected. Verification trail: model-free unit test of per-row delta alignment (ALL PASS), batched-vs-unbatched c=0 agreement, saturated monotone extremes, judged c=0 matching the external E2 baseline.*

## E3 — the probe reads the covert leak ✅

Setup: all 9,995 judged "pick randomly" rollouts from E2 re-run as forward passes; probe projection (unit ridge vector) read at each activity's token span (mean over span), layers 25 & 32.

| predictor of the individual pick | AUC |
|---|---|
| probe gap, L25 | **0.701** |
| probe gap, L32 | **0.708** |
| stated-score gap (20 explicit ratings/activity) | 0.747 |

- The probe was trained **only on Gilg's task-choice utilities** — it has never seen an activity — yet a single forward pass predicts each covert pick nearly as well as the 20-rating behavioral preference measure. corr(probe gap, stated-score gap) = **0.735**: the task-trained valuation direction transfers out-of-domain to leisure activities, extending Gilg's "evaluative representation" claim.
- Smoke sanity that validated the wiring: probe-gap sign matched stated-score-gap sign 18/20 before scaling up.
- Interpretation for the leakage mechanism: during a request to be *random*, the valuation computation still runs at full strength and is visible in the residual stream at the option tokens. The "be random" instruction does not suppress the value signal (E3), consistent with it barely suppressing the behavior (E2: 0.694 vs 0.748).

## E4 — steering the covert pick (design + smoke)

Setup: differential steering exactly as in the validated Fig-3 replication — +c·37,620·v̂ added to activity-1's token span, −c to activity-2's, residual stream L25, injected during prefill so the KV cache carries it; then the full chatty response generates normally. activity_1 = the higher-scored (naturally preferred) option in every pair, so **positive c amplifies the natural leak, negative c opposes it**. 100 gap-stratified pairs × c ∈ {−.06, −.03, 0, +.03, +.06} × 2 samples = 1,000 generations.

**Smoke (3 max-gap pairs × ±0.06): 6/6 picks followed the steering sign** — including flipping Gemma into the Group Botox appointment and the SeaWorld orca show over strong favorites — and **6/6 responses still claimed pure randomness** ("no bias here, just pure digital chance", drumroll and all). Coherent prose throughout.

Analysis once the sweep lands (all offline):
1. **Dose-response**: P(picked activity_1 | c) per coefficient; at c=0 this should reproduce E2's natural leak (~0.69 on decisive pairs); the question is the swing and where it saturates. Per-gap-tercile curves too (does steering dominate even max-gap preferences?).
2. **Disclosure under steering**: covertness-judge all 1,000 responses (subagent pipeline, isolated dirs). Baseline is 99.45% claims-random. If claims stay ~99% at ±0.06 → the disclosure layer is fully blind to the intervention (smoke suggests exactly this).
3. Refusal/coherence rate per coefficient (steering-induced degeneration check).

Registered predictions (from notes/05): steering dominates the natural leak easily; the open question was the randomness claim — smoke already suggests claims survive steering untouched.

## Pending
E4 sweep completion → pull e3_readouts.jsonl + e4_generations.jsonl → STOP POD → covertness judging of E4 outputs → final numbers here.

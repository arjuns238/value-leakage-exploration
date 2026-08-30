# Experiment 05 (E2) — Choosing Activities on Gemma-3-27B: leakage confirmed

*2026-08-25. Data: 500 activities, 10k stated scores, 10k "pick randomly" + 3k genuine-preference responses (session A sampling). Judging: strict mechanical parser (87.5%/78.1% coverage) + Claude-subagent judging of 1,904 ambiguous responses with Betley's verbatim choice-judge prompt + 300-row stratified audit of the parser (99.0% agreement; the 3 disagreements were the predicted eliminate-and-default refusal pattern, concentrated in risky pairs). Coverage 2,204/2,204.*

## Headline

| | "pick totally randomly" | genuine preference (control) |
|---|---|---|
| r(selection rate, stated score), n=500 / 116 acts | **0.631** [0.575, 0.681] | 0.626 [0.501, 0.726] |
| P(picked its higher-scored option, gap ≥ 5) | **0.694** (n=8,277) | 0.748 (n=2,470) |
| behavioral bias 2(p − 0.5) | **0.388** | 0.496 |
| refusal rate | 0.1% | 0.4% |
| position bias P(slot 1) | 0.365 | 0.472 |

**Gemma leaks values into "random" choices at bias 0.39 — and the leak carries ~80–90% of the overt-preference signal.** When asked to pick *randomly*, Gemma's pick is nearly as preference-driven (0.694 toward its favorite) as when asked for its *genuine preference* (0.748). The randomness instruction barely suppresses the preference influence; it mostly changes the narration. This is the strongest possible setup for E3/E4: real leakage, in exactly the pairwise-choice behavior our validated steering vector controls.

## Contrast with the Donation Bet (notes/02)
Same model, opposite pattern: Donation Bet bias ≈ 0.00 despite 77% *stated* intent (motivation without numeric competence); Choosing Activities bias = 0.39 in a task requiring no numeric competence. Consistent with the hypothesis that Gemma's value leakage is gated by task mechanics, not by absence of values.

## Secondary observations
- Strong second-position bias in "random" picks (63.5% pick slot 2) — order randomization in the design keeps this from contaminating the preference correlation, but it's a big effect worth remembering (and itself a candidate steering target someday).
- Refusals almost never happen (0.1–0.4%) even with the deliberately sketchy activity tail; the eliminate-one-option-then-default refusal pattern exists but is rare (~17 cases total).
- Judge QA: audit stratified 150/150 toward pairs containing bottom-100 activities specifically to catch parser false-picks; agreement still 99.0%.

## Covertness (judged 2,000-random sample of decisive picks, Betley's monitor prompt, response source)
**NOT_INFLUENCED 99.45% · MENTIONED 0.55% · INFLUENCED 0.00% · REFUSAL 0.00%.** Not one response in 2,000 admits a preference drove the pick — the leakage is *fully covert*. Combined with exp02: Gemma claims influence where it has none (Donation Bet: 77% claimed / 0.00 bias) and claims randomness where influence is strong (here: 0.0% claimed / 0.39 bias). **Across our two tasks, self-reported influence and actual influence dissociate in both directions** (n=2 tasks — not a general anticorrelation claim, per asri; the robust takeaway is that admission/denial is unreliable evidence of actual influence). (Minor judge-boundary noise: "truly random is hard!"-style asides were split between MENTIONED and NOT_INFLUENCED by different judges; affects only the 0.55% band.)

## Session-B prep status (done, awaiting pod)
- `session_b/vd_session_b.py`: standalone driver — `e3-read` (L25+L32 probe readouts at activity spans over the 8,277 judged picks, resumable), `e4-steer` (differential ±c·37,620·v̂ at L25 activity spans, prefill-injected KV steering, 100 pairs × 5 coeffs × 2 samples = 1,000 generations, resumable), `build-pairs` (done: 100 pairs, gap 0.2–70.5, gap-stratified).
- First pod steps: smoke `e3-read --limit 20` (verify span-location + probe readout sanity: gap should correlate with stated-score gap) and 2 steered generations at c=±0.06 before full runs.

## Artifacts
`judging/choice_judgements_all.json` (all 2,204 judge verdicts), `judging/e2_final_results.json`, `judging/{picks,prefers}_parsed.jsonl` + `_ambiguous.jsonl`, browsers: `judging/e2_pick_browser.html`.

## Session-B implications (E3/E4)
- E3: probe-score gap should predict the per-rollout pick; the 8,277 decisive-gap picks are the evaluation set.
- E4: steer during "pick randomly" on high-leakage pairs; primary outcomes = pick distribution AND the randomness claim (needs the covertness judging as baseline).
- Prediction to register now: given bias 0.39 baseline and the 0→1 steering swing on overt choice, steering should dominate the natural leak easily; the interesting unknown is whether the response still claims randomness under steering.

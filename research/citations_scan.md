# Citation scan: arXiv:2607.14345 (Betley et al., "Value Leakage") and arXiv:2605.13339 (Gilg et al., "Probing Persona-Dependent Preferences")

Scan date: 2026-08-25. Method: Semantic Scholar citations API, Google Scholar author profile (Jan Betley), arXiv HTML full-text search for citing sentences, LessWrong/AlignmentForum search, general web search. ~15 queries.

## Summary

- **arXiv:2607.14345** (Betley, Treutlein, Dubiński, Mayne, Gałązka, Warncke, Sztyber-Betley, Evans — "Value Leakage: An LLM's Answers Are Silently Shaped by Its Own Values", submitted 2026-07-15, now v4) has **2 confirmed citing papers** (Semantic Scholar citationCount = 2, corroborated on Jan Betley's Google Scholar profile).
- **arXiv:2605.13339** (Gilg, Beckmann, Paleka, Butlin — "Probing Persona-Dependent Preferences in Language Models", submitted 2026-05-13, v2 2026-05-18) has **1 confirmed citing paper**.
- No paper found (via Semantic Scholar, arXiv full-text search, or general web search) cites *both* papers together or explicitly connects covert value leakage to the Gilg preference-direction/probe as a mechanistic carrier. **Our combination (probe predicts covert "random" picks, steering controls them, narration stays invariant) appears to be a novel connection** not yet made in the literature as of this scan.
- No follow-up/v-next paper from either author group (Truthful AI / Owain Evans; Oscar Gilg / Patrick Butlin / Daniel Paleka) was found that connects values/preferences to internals or to covert-bias steering beyond what's already in their respective original papers. The Gilg et al. paper itself already reports Assistant-trained-probe steering transferring across personas and (per one secondary summary) affecting refusal/ethical-flagging behavior — this is the paper's own content, not a new follow-up.
- The Gilg et al. work has a companion self-authored LessWrong post (MATS 9.0 research report), not a third-party citation.

## Items found

### Citing arXiv:2607.14345 (Value Leakage)

1. **"Chain-of-Thought Monitoring Can Be Unreliable in Implicit-Influence Settings"**
   - arXiv: https://arxiv.org/abs/2608.04735 (submitted ~2026-08-08)
   - One-sentence summary: Introduces a benchmark comparing CoT-monitor detection rates between "explicit-influence" settings (model told to hide something) and "implicit-influence" settings (behavior shifted by contextual features with no instruction to hide), finding CoT monitorability drops sharply (by 41-46 pts, to as low as 5% with realistic system-prompt additions) in the implicit case even though the behavioral influence is preserved.
   - Citation context (Related Work): "Most recently, Betley et al. (2026) show that models' answers are covertly biased by their own values, with the influence arising from the model itself rather than any inserted cue." — cited as the canonical example of implicit-influence bias.
   - Overlap with our contribution: **High topical overlap, no mechanistic overlap.** This paper studies covert influence via CoT-monitor detection (behavioral/interpretive), not via activation probes/steering vectors. It does not touch preference directions or mechanistic carriers. Directly relevant as a "sibling" implicit-bias paper but does not anticipate our mechanistic result.

2. **"Jagged Judges: Epistemic Stability Under Silence, Pressure, and Persistence"**
   - arXiv: https://arxiv.org/abs/2608.12645
   - One-sentence summary: Introduces the "Wiggle Framework" stress-testing LLM-judge epistemic stability (mechanical consistency, single-turn conviction, multi-turn persistence) across 9 models and 14 judging tasks, finding judges flip verdicts 25-91% of the time under pushback, mostly toward errors.
   - Citation context: "Recent work further shows a model's outputs can be silently shaped by its own priors and surrounding context, with the model's stated reasoning failing to disclose the influence while being systematically swayed (Betley et al. 2026)." — cited in passing as evidence that models' stated reasoning can be non-disclosing/unreliable.
   - Overlap with our contribution: **Low.** Judge-robustness paper, no probes/steering, no preference-direction content. Cites Value Leakage only as generic evidence of non-disclosure.

### Citing arXiv:2605.13339 (Probing Persona-Dependent Preferences)

3. **"Readable, Faithful, Used: Three Dissociable Properties of Demographic Identity in a Language Model"**
   - arXiv: https://arxiv.org/abs/2608.18768 (author: Fathin Difa Robbani)
   - One-sentence summary: Uses representational similarity analysis (Mistral-7B vs. Pew survey data, 169 demographic cells) to show that demographic identity in an LLM can be "readable" (probeable) and "faithful" (matches real survey differences) while being minimally "causally used" in the model's actual predictions — i.e., readability, fidelity, and causal use are dissociable.
   - Citation context (Related Work, "Localizing social attributes" subsection): "Closest to our dissociation result, Gilg et al. find preference machinery largely shared across personas—probes and steering vectors transfer between opposed personas—converging from the steering side with our finding that what a location encodes and what the model answers with are separable properties."
   - Overlap with our contribution: **Moderate/conceptual.** This paper's readable-vs-causally-used dissociation is thematically adjacent to our narration-invariance-under-steering finding (i.e., the "story" a model tells vs. what actually drives its behavior can decouple) but is about demographic identity, not covert value/preference leakage, and does not touch the Betley "random choice" paradigm or AUC-style prediction of covert picks. Worth citing as related dissociation-of-representation-and-behavior evidence.

## Not found / ruled out

- No LessWrong/AlignmentForum post (other than the authors' own companion posts) discussing a mechanistic/probe-based account of value leakage. One LessWrong commenter (Jasmine Brazilek) on the original Value Leakage post explicitly called for exactly this kind of mechanistic evidence ("something other than CoT carrying the covertness claim... models compute the difference internally") — Johannes Treutlein (co-author) replied defending the behavioral evidence but did not point to any internals-based follow-up. This is a good citable "gap identified by the community" pointer for our paper's motivation.
- No alphaxiv page content beyond the standard arXiv mirror was found with additional discussion/citations.
- No v-next/follow-up paper from Gilg/Beckmann/Paleka/Butlin or from the Truthful AI / Betley / Evans group connecting the two papers.
- Semantic Scholar API rate-limited before a second citationCount check on 2605.13339 could be repeated, but the citations-list endpoint (fetched earlier, listing exactly the "Readable, Faithful, Used" paper) is consistent with count = 1.

## Author/venue notes

- Value Leakage (2607.14345): authors Jan Betley, Johannes Treutlein, Jan Dubiński, Harry Mayne, Karol Gałązka, Niels Warncke, Anna Sztyber-Betley, Owain Evans (Truthful AI). v1 2026-07-15 → v4 2026-08-14 (3 revisions since posting — worth re-checking for added related-work/mechanistic discussion in v4 vs v1). Companion GitHub repos: TruthfulAI-research/value_leakage and value_leakage_data. Companion site: valueleakage.net. LessWrong/AlignmentForum post: https://www.lesswrong.com/posts/hbMw4Yqw6RnFaExDy/value-leakage-an-llm-s-answers-are-silently-shaped-by-its-1
- Probing Persona-Dependent Preferences (2605.13339): authors Oscar Gilg, Pierre Beckmann, Daniel Paleka, Patrick Butlin. v1 2026-05-13, v2 2026-05-18. Code: https://github.com/oscar-gilg/Preferences (also mirrored at oscar-gilg/probing-persona-preferences). Companion LessWrong post (self-authored MATS 9.0 report): https://www.lesswrong.com/posts/pxC2RAeoBrvK8ivMf/models-have-linear-representations-of-what-tasks-they-like-1

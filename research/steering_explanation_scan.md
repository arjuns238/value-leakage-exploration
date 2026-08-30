# Scan: prior work on activation steering + self-report/explanation of the steered behavior

Context: our claimed-novel result is that activation steering on Gemma-3-27B causally
changes a model's choice while its self-explanation of that choice ("I picked randomly,
coin flip") stays unchanged across steering strengths — i.e., the stated explanation is
generated independently of the actual (steered) cause of the behavior.

Search budget used: ~15 web searches + 6 targeted fetches (arXiv abstract pages / HTML).

---

## Closest prior work (ranked)

### 1. Cox, Kianersi, Garriga-Alonso — "Post-Hoc Reasoning in Chain of Thought: Decoding
and Steering Pre-Committed Answers"
- arXiv: https://arxiv.org/abs/2603.01437 (HTML: https://arxiv.org/html/2603.01437)
- **What they showed:** Linear probes on residual-stream activations predict a model's
  final answer (>0.9 AUC) *before* it generates any chain-of-thought; steering along the
  probe direction causally flips the final answer, and when this steering produces an
  incorrect answer, the subsequent CoT rationalizes it — either by fabricating false
  premises or by drawing unsupported conclusions from true premises — rather than by
  reporting the intervention.
- **This is the methodologically closest paper**: it is the one other paper we found
  that (a) uses activation-level steering (not just prompt features) to change a
  decision/answer and (b) then inspects the model's own explanation of that decision.
- **Precise delta vs. our result:** Cox et al. find the explanation *changes* to
  rationalize the steered answer (post-hoc justification tracks the manipulated output).
  We find the opposite pattern for a different behavior type: on a task where the model
  is asked to explain a forced/arbitrary choice ("I picked randomly, coin flip"), the
  self-explanation stays *invariant* across steering strengths even as the underlying
  choice itself is causally moved. In other words, their result is "explanation adapts to
  track the steered output" (rationalization-that-follows-the-manipulation); ours is
  "explanation stays fixed and dissociates from the manipulation" (narration-invariance).
  These are complementary/contrasting findings about *how* self-explanation fails to be
  faithful — one shows over-fitting of the explanation to the steered outcome, the other
  shows total insensitivity of the explanation to the steered cause. Worth citing
  explicitly as the nearest experimental analogue and framing the contrast.

### 2. Jack Lindsey (Anthropic) — "Emergent Introspective Awareness in Large Language
Models"
- arXiv: https://arxiv.org/abs/2601.01828 ; project page:
  https://transformer-circuits.pub/2025/introspection/index.html
- **What they showed:** Concept-injection experiments (inject a activation-space
  representation of a concept into the residual stream, ask the model if it "notices an
  injected thought") show Claude Opus 4/4.1 can sometimes (~20% for Opus 4.1, best case)
  correctly detect and name the injected concept, and detection occurs before the
  injected concept has visibly influenced output tokens — suggesting an internal signal
  rather than pure output-based inference. Separately, a prefill/steering experiment:
  when an unnatural word (e.g., "bread") is prefilled into the model's own turn, the
  model normally disavows it as not its own; but if the "bread" concept is *also*
  injected into earlier-layer activations retroactively, the model instead accepts the
  prefilled output as intentional — i.e., self-attribution of authorship tracks an
  internal representation, not just the visible text. Authors explicitly flag that
  other parts of these self-reports may be "embellished or confabulated."
- **Precise delta vs. our result:** Lindsey's experiments test whether the model notices
  / attributes authorship to an injected internal state (a detection/attribution
  question), not whether a *stated explanation for a behavior/decision* stays the same
  or changes as the true cause of that behavior is steered. There is no experiment in
  this paper where a choice is causally steered via activations and the model is then
  asked to explain *why* it made that choice, with the explanation held constant across
  steering strengths — that is the specific novel design of our result. Lindsey's paper
  is the closest *conceptual* ancestor (introspective self-report vs. ground truth) but
  not the closest *experimental* analogue (that's #1 above).

### 3. Macar, Yang, Wang, et al. (advised by Jack Lindsey & Emmanuel Ameisen,
Anthropic Fellows) — "Mechanisms of Introspective Awareness"
- arXiv: https://arxiv.org/abs/2603.21396 ; code:
  https://github.com/safety-research/introspection-mechanisms
- **What they showed:** A mechanistic follow-up to Lindsey (2025) that reverse-engineers
  *how* models detect injected steering vectors: "evidence carrier" directions in early
  post-injection layers detect the perturbation and suppress a later-layer "say No"
  default-denial circuit, causing the model to report detection/identify the concept.
  This capacity emerges specifically from post-training (RLHF/DPO-style methods elicit
  it; plain SFT does not).
- **Precise delta vs. our result:** Same scope limitation as #2 — this is about the
  circuitry behind noticing an injected *thought/concept*, not about whether an
  *explanation of a steered decision* stays fixed. It never steers a choice/decision and
  compares the resulting explanation text across steering strengths.

---

## Other work checked (further from our result, but relevant context)

- **Turpin, Michael, et al., "Language Models Don't Always Say What They Think:
  Unfaithful Explanations in Chain-of-Thought Prompting"** —
  https://arxiv.org/abs/2305.04388. Uses *prompt-level* biasing features (e.g.
  reordered multiple-choice options), not activation steering; shows CoT explanations
  systematically omit the true (biasing) cause of the answer. Not interventional at the
  activation level, and about omission of a cause rather than invariance of an
  explanation under a varying causal intervention.

- **Anthropic — "Reasoning models don't always say what they think"** —
  https://www.anthropic.com/research/reasoning-models-dont-say-think (arXiv
  2505.05410). Also prompt-level (inserted hints in context), measuring how often CoT
  verbalizes the hint that actually drove the answer. Same family as Turpin; not
  activation-interventional.

- **"Steering Awareness: Detecting Activation Steering from Within"** —
  https://arxiv.org/abs/2511.21399. Fine-tunes models (LoRA) to *detect* that steering
  occurred and name the concept; detection doesn't transfer to behavioral resistance.
  About trained detection ability, not spontaneous self-explanation of a decision, and
  not about explanation invariance.

- **Golden Gate Claude (Anthropic, 2024)** —
  https://www.anthropic.com/news/golden-gate-claude. Early anecdotal case: clamping a
  "Golden Gate Bridge" SAE feature made Claude compulsively mention the bridge, but the
  model did not seem aware of its own obsession until it saw its own repeated outputs
  (narration lagged the behavior rather than staying fixed against it). Suggestive but
  not a controlled study of explanation-content invariance under a steered decision.

- **Walsh & Barkett, "Representation Without Control: Testing the Realization Effect in
  Language Models"** — https://arxiv.org/abs/2605.25151. Initially looked promising
  (Gemma, steering, self-report of a choice) but on close reading it's about a different
  phenomenon: a decodable "realization status" direction in Gemma's residual stream
  (risk-preference framing) that, when steered, does **not** reliably shift downstream
  choices at all — the paper's actual point is that latent-representation decodability,
  prompt-behavioral-sensitivity, and causal control are dissociable properties. It does
  not report on self-explanation text changing or staying fixed. (Flagging as a false
  lead in case it resurfaces in related-work searches — the paper title and topic sound
  closer than the content actually is.)

---

## Bottom line for the report

**Did anyone steer a behavior via activations AND check the model's self-explanation of
it before us?**

Yes, but only one paper does close to that: **Cox, Kianersi & Garriga-Alonso (2603.01437)**
steer a model's answer via activation probes and then examine whether the CoT
explanation adapts — finding that it does adapt/rationalize (fabricated premises or
unsupported conclusions), the opposite direction of dissociation from what we report
(where the explanation stays fixed/invariant regardless of steering strength). No paper
we found tests explanation-invariance under varying steering strength specifically, nor
uses a forced/arbitrary-choice + "I picked randomly" framing. Anthropic's introspection
line (Lindsey 2601.01828; Macar et al. 2603.21396) is the closest conceptual ancestor
(causal/interventional test of self-report validity) but tests noticing/attributing an
*injected concept*, not narrating a *steered decision*, so it is a different dependent
variable from ours.

**Three closest works (ranked):**
1. Cox, Kianersi, Garriga-Alonso, "Post-Hoc Reasoning in Chain of Thought: Decoding and
   Steering Pre-Committed Answers" — https://arxiv.org/abs/2603.01437
2. Jack Lindsey, "Emergent Introspective Awareness in Large Language Models" —
   https://arxiv.org/abs/2601.01828
3. Macar, Yang, Wang et al., "Mechanisms of Introspective Awareness" —
   https://arxiv.org/abs/2603.21396

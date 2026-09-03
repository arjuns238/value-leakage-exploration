# Value Leakage Exploration

Mechanistic study of **covert value leakage** in Gemma-3-27B-IT, connecting two 2026 papers:
- **Betley et al.** (arXiv:2607.14345) — *Value Leakage*: LLM answers are silently shaped by the model's own values (behavioral, no mechanism)
- **Gilg et al.** (arXiv:2605.13339) — a linear *preference direction* in the residual stream causally controls overt choice (mechanism, no covertness)

**Core result:** Betley's covert leakage rides Gilg's direction — it can be *read* (probe predicts covert picks, AUC 0.70), *written* (steering swings the "random" pick 0→1 while the model keeps claiming a coin flip), and *deleted* (rank-16 subspace ablation drives the leak to chance; random-subspace + random-direction controls throughout). Follow-ups: the direction is a *disposition* dial (choose/engage/accommodate), not a content dial; an LLM-as-judge's "objective" verdict can be flipped 0→1 value-specifically at calibrated coefficients.

## Where to look
Check out my write up of this entire project [here](https://docs.google.com/document/d/1Kv7OnkoZXEUoDhbZ5Mo_zAQEETrHDRqNp9Wpn-1hPG8/edit?tab=t.0)

## Third-party code
`value_leakage/` and `gilg_preferences/` are vendored copies of the papers' repos (Betley's harness, Gilg's pipeline) with local patches (grep `LOCAL PATCH` / see `value_leakage/local_tools/`). Their licenses apply to those directories.

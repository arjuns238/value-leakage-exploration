# Value Leakage Exploration

Mechanistic study of **covert value leakage** in Gemma-3-27B-IT, connecting two 2026 papers:
- **Betley et al.** (arXiv:2607.14345) — *Value Leakage*: LLM answers are silently shaped by the model's own values (behavioral, no mechanism)
- **Gilg et al.** (arXiv:2605.13339) — a linear *preference direction* in the residual stream causally controls overt choice (mechanism, no covertness)

**Core result:** Betley's covert leakage rides Gilg's direction — it can be *read* (probe predicts covert picks, AUC 0.70), *written* (steering swings the "random" pick 0→1 while the model keeps claiming a coin flip), and *deleted* (rank-16 subspace ablation drives the leak to chance; random-subspace + random-direction controls throughout). Follow-ups: the direction is a *disposition* dial (choose/engage/accommodate), not a content dial; an LLM-as-judge's "objective" verdict can be flipped 0→1 value-specifically at calibrated coefficients.

## Where to look
- `HANDOFF.md` — project state, pod conventions, full experiment map
- `notes/07_writeup.md` — paper draft
- `notes/01–09` — per-session experiment logs & results
- `session_b/vd_session_b.py` — the standalone experiment driver (steering / ablation / sampling)
- `judging/*.html` — result browsers; `judging/session_c/` judged outputs
- `artifacts/` — pulled experiment data (activations excluded for size)

## Third-party code
`value_leakage/` and `gilg_preferences/` are vendored copies of the papers' repos (Betley's harness, Gilg's pipeline) with local patches (grep `LOCAL PATCH` / see `value_leakage/local_tools/`). Their licenses apply to those directories.

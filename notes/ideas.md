# Ideas parking lot

- **SFT a model to believe its chain of thought should be bad — then what happens?** (asri, 2026-08-24)
  Possible framing: fine-tune Gemma on data where the CoT is deliberately low-quality/misleading while final answers stay correct (or vice versa). Questions it could probe: does the disclosure/covertness behaviour dissociate from the bias behaviour? Does the preference vector's readout during CoT change? Connects to the Value Leakage finding that Claude denies bias in-CoT while Qwen admits it — is "what the CoT says" trainable independently of "what the answer does"?
  Status: parked until replication is done.

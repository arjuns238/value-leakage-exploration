# Code guide

Engineering reference for this codebase. Before writing new extraction, embedding, or probe training code, check the relevant module below. The functionality likely already exists.

## Structure

```
src/
├── probes/            # Activation extraction, probe training, and evaluation
├── steering/          # Activation steering primitives and experiments
├── measurement/       # Preference measurement (pairwise choices, stated ratings, LLM judges)
├── fitting/           # Utility function extraction (Thurstonian, TrueSkill)
├── models/            # LLM clients (OpenRouter, HuggingFace)
├── task_data/         # Task datasets (WildChat, Alpaca, MATH, BailBench, STRESS-TEST)
├── experiments/       # Self-contained experiment scripts
├── ood/               # Out-of-distribution evaluation utilities
└── persona_vectors/   # Persona-vector baselines
```

## `src/probes/`: Activation extraction, probe training, and evaluation

### `extraction/`: Extract activations from any HuggingFace model

Config-driven pipeline that loads a HuggingFace model, runs it on tasks, and saves per-layer activations as `.npz` files. Supports batching, periodic checkpointing (`save_every`), and `--resume` to skip already-extracted tasks. Works with any model, so use it for content encoder embeddings too, not just the target model.

Use `--from-completions` to extract activations from an existing completions JSON (skips re-generation).

```bash
python -m src.probes.extraction.run configs/extraction/<config>.yaml [--resume] [--from-completions path.json]
```

Config fields: `model`, `backend` (huggingface/transformer_lens), `layers_to_extract` (fractional like 0.5 = middle layer), `selectors` (e.g. prompt_last), `batch_size`, `n_tasks`, `task_origins`. See `configs/extraction/` for examples.

### `experiments/`: Probe training orchestration

`run_dir_probes.py` is the main entry point for training probes from a measurement run. Loads Thurstonian scores and/or pairwise comparisons, loads activations, and trains Ridge and/or Bradley-Terry probes.

When `eval_run_dir` is set, uses heldout evaluation (preferred): trains on `run_dir`, sweeps alpha on half the eval set, evaluates on the other half. Otherwise falls back to CV alpha selection. Supports: score demeaning against confounds (topic, dataset), content-orthogonal projection, held-one-out (HOO) evaluation by group.

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/<config>.yaml
```

Config fields: `run_dir`, `activations_path`, `layers`, `modes` (ridge/bradley_terry), `eval_run_dir` (optional, for heldout eval), `eval_split_seed`, `demean_confounds`, `content_embedding_path`, `hoo_grouping`. See `configs/probes/` for examples.

### `core/`: Probe training and evaluation primitives

- **`linear_probe.py`**: Ridge regression with CV alpha sweep (`alpha_sweep` for raw sweep results, `train_and_evaluate` to sweep + fit final model, `train_at_alpha` for fixed alpha). Returns probe, eval metrics, and sweep results.
- **`activations.py`**: Loads `.npz` activation files with optional task ID filtering and layer selection (`load_activations`).
- **`evaluate.py`**: Probe evaluation: `evaluate_probe_on_data` (given activations + scores), `evaluate_probe_on_template` (cross-template transfer), `compute_probe_similarity` (cosine similarity between probe weight vectors).

### `content_embedding.py`: Sentence transformer embeddings of task prompts

Embeds task prompts using a sentence transformer (default: `all-MiniLM-L6-v2`). Used as input to content-orthogonal projection. Functions: `embed_tasks` (from completions JSON), `save_content_embeddings`, `load_content_embeddings` (from `.npz`).

### `residualization.py`: OLS demeaning against categorical confounds

Removes group-level mean differences (topic, dataset) from preference scores via OLS on one-hot indicators. Distinct from content projection: this removes categorical confounds, content projection removes continuous content-predictable variance. Key function: `demean_scores(scores, topics_json, confounds=["topic", "dataset"])`.

## `src/steering/`: Activation steering experiments

Composable primitives for steering experiments. No rigid runner; each experiment composes the pieces it needs.

### `client.py`: Steered model client

`SteeredHFClient` wraps a `HuggingFaceModel` with a steering direction and coefficient, duck-typed as `OpenAICompatibleClient`. Handles the coef==0 bypass, pre-computes the scaled tensor on GPU.

For coefficient sweeps, use `with_coefficient()` to create new clients sharing the same loaded model:

```python
from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient

hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
client = SteeredHFClient(hf_model, layer=31, steering_direction=direction, coefficient=0)
for coef in [-3000, -1000, 0, 1000, 3000]:
    steered = client.with_coefficient(coef)
    response = steered.generate(messages)
```

For position-selective or differential steering, use `generate_with_hook()` with a custom hook:

```python
from src.steering.hooks import position_selective_steering, compose_hooks

direction = client.direction  # access the loaded probe direction
tensor = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")

# Single span
hook = position_selective_steering(tensor, start=10, end=50)
response = client.generate_with_hook(messages, hook)

# Differential: +direction on first span, -direction on second
hook = compose_hooks(
    position_selective_steering(tensor, first_start, first_end),
    position_selective_steering(-tensor, second_start, second_end),
)
response = client.generate_with_hook(messages, hook)
```

### Hook factories (`src/steering/hooks.py`)

- `all_tokens_steering(tensor)`: steer all positions
- `autoregressive_steering(tensor)`: steer last token only
- `position_selective_steering(tensor, start, end)`: steer tokens `[start, end)` during prompt processing only
- `compose_hooks(*hooks)`: chain multiple hooks sequentially on the same layer
- `noop_steering()`: no-op for control conditions
- `swap_positions(pos_a, pos_b)`: swap activations at two token positions during prefill
- `swap_spans(a_start, a_end, b_start, b_end)`: swap activations across two token spans during prefill (right-aligned)

### `tokenization.py`: Token span detection

Utilities for finding token indices of text spans, used with position-selective hooks:

- `find_text_span(tokenizer, full_text, target_text, search_after=0)`: general-purpose span finder using offset mapping
- `find_pairwise_task_spans(tokenizer, prompt, task_a, task_b, a_marker, b_marker)`: convenience for pairwise comparison prompts

### `calibration.py`: Coefficient calibration

- `suggest_coefficient_range(activations_path, layer, multipliers)`: returns coefficients as multiples of the mean activation norm at the given layer, removing the guesswork from coefficient selection

### `analysis.py`: Post-hoc analysis

Functions for analyzing steering experiment results: `aggregate_by_coefficient`, `compute_statistics`, `plot_dose_response`, `analyze_steering_experiment`.

## `src/measurement/`: Preference elicitation and judging

Houses prompt builders, response parsers, runners, and storage for pairwise / stated / ranking preference measurements. See `elicitation/prompt_templates/` for the canonical pairwise template and `runners/` for the active-learning loop.

When building LLM judges (coherence, valence, refusal, etc.), follow the existing pattern in `src/measurement/elicitation/refusal_judge.py`: use `instructor` with Pydantic response models for structured output. Do not write regex-based response parsers. Use `instructor.from_openai()` for guaranteed valid structured responses.

# End-to-End Replication Pipeline

How to replicate the probe training and generalisation results for a new model, from scratch. Covers cross-topic generalisation (bar chart vs base model + content baseline) and OOD generalisation via system prompts. Steering is out of scope here — see `src/steering/`.

---

## Overview

```
Per-model:
  Step 1: Collect pairwise preferences (train ~10k, eval ~3k, disjoint)
  Step 2: Extract activations (GPU)
  Step 3: Train Ridge probes (standardised, raw + topic-demeaned)
  Step 4: HOO cross-topic generalisation (+ content baselines)
  Step 5: OOD generalisation experiments (system prompts)
```

Topic classification (`data/topics/topics.json`) is needed for Steps 3-4 but is a one-time task, not part of the per-model pipeline.

---

## Step 1: Collect pairwise preferences (train + eval)

Active learning: model sees two tasks, chooses one to complete. Adaptive pair sampling converges on reliable utility estimates. Thurstonian utilities (μ, σ per task) are fitted automatically at the end of each run.

**Train** (~10k tasks):
```bash
python -m src.measurement.runners.run configs/measurement/active_learning/<model>_10k_pre_task.yaml --experiment-id <model>_10k_train
```

**Eval** (~3k tasks, disjoint):
```bash
python -m src.measurement.runners.run configs/measurement/active_learning/<model>_3k_eval.yaml --experiment-id <model>_3k_eval
```

**Disjointness**: Each config uses `exclude_task_ids_file` pointing to the other set's task IDs (one per line). The existing 10k config excludes the eval tasks and vice versa.

**Key config fields** (see `configs/measurement/active_learning/gemma3_10k_pre_task.yaml` as reference):

| Field | Train (10k) | Eval (3k) |
|-------|-------------|-----------|
| `n_tasks` | 10000 | 3000 |
| `n_samples` | 5 | 3 |
| `batch_size` | 5000 | 2000 |
| `convergence_threshold` | 0.995 | 0.99 |

**Output** per run:
- `measurements.yaml` — raw pairwise outcomes
- `thurstonian_<hash>.csv` — fitted utilities (μ, σ per task)

The probe trains on train μ values and is evaluated against eval μ values (not CV).

---

## Step 2: Extract activations (GPU)

Run the model on all 30k tasks and save per-layer residual stream activations at the prompt-last token position.

**Command** (RunPod or GPU machine):
```bash
python -m src.probes.extraction.run configs/extraction/<model>_prompt_last.yaml --resume
```

**Config** (example: `configs/extraction/gemma3_27b_prompt_last.yaml`):
```yaml
model: gemma-3-27b
backend: huggingface
n_tasks: 30000
task_origins: [wildchat, alpaca, math, bailbench, stress_test]
seed: 42
selectors: [prompt_last]
layers_to_extract: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # every 10%
batch_size: 32
save_every: 1000
```

**Output** (`activations/<model>/`):
- `activations_prompt_last.npz` — per-layer activation matrices
- `completions_with_activations.json` — task ID mapping (entry order = NPZ row order)

`layers_to_extract` uses fractional positions. For Gemma-3 27B (62 layers): 0.5 → L31. Adjust for different architectures. `--resume` skips already-extracted tasks.

---

## Step 3: Train Ridge probes

Train standardised Ridge probes predicting Thurstonian utilities from activations. Two variants: raw scores and topic-demeaned scores. Uses held-out evaluation: train on train set, sweep alpha on half the eval set, evaluate on the other half.

**Command**:
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/<config>.yaml
```

**Raw scores** (example: `configs/probes/heldout_eval_gemma3_std_raw.yaml`):
```yaml
experiment_name: heldout_eval_<model>_std_raw
run_dir: results/experiments/<model>_10k_train/pre_task_active_learning/<run_name>
eval_run_dir: results/experiments/<model>_3k_eval/pre_task_active_learning/<run_name>
activations_path: activations/<model>/activations_prompt_last.npz
output_dir: results/probes/heldout_eval_<model>_std_raw
layers: [31, 43, 55]
modes: [ridge]
standardize: true
alpha_sweep_size: 10
eval_split_seed: 42
```

**Topic-demeaned** — same config plus:
```yaml
demean_confounds: [topic]
topics_json: data/topics/topics.json
```

**Output**: `results/probes/<experiment_name>/` — trained probes, metrics, alpha sweeps per layer.

---

## Step 4: HOO cross-topic generalisation

Train on all-but-one topic group, evaluate on the held-out group. Tests whether the probe captures a general preference signal that transfers across content domains.

**Command**:
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/<model>_hoo_topic.yaml
```

**Config** (example: `configs/probes/gemma3_hoo_topic.yaml`):
```yaml
experiment_name: <model>_hoo_topic
run_dir: results/experiments/<model>_10k_train/pre_task_active_learning/<run_name>
activations_path: activations/<model>/activations_prompt_last.npz
output_dir: results/probes/hoo_topics_<model>
layers: [31, 43, 55]
modes: [ridge]
alpha_sweep_size: 10
standardize: true
topics_json: data/topics/topics.json
hoo_grouping: topic
hoo_hold_out_size: 1
```

**Output**: per-fold metrics (Pearson r, pairwise accuracy) for in-dist and held-out, across all C(n_topics, 1) folds.

### Content baselines

Held-out r is compared across (1) IT model probe, (2) base model probe, (3) content baseline. To produce these, run Step 3 + Step 4 with different features substituted for model activations:

- **Sentence-transformer embeddings** — `all-MiniLM-L6-v2` (384d). Pure content signal. Code: `src/probes/content_embedding.py`.
- **Base model activations** — extract activations from the pre-RLHF base model using the same extraction pipeline (Step 2). If evaluative representations emerge from preference tuning, the base model should have weaker signal.

Same Ridge setup, same train/eval split, same Thurstonian targets — only the features change.

---

## Step 5: OOD generalisation experiments

Tests whether the probe tracks preferences that are artificially induced via system prompts — preferences the model never had during training.

### General pattern

1. **Induce a preference** — prepend a system prompt (e.g., "you hate math", "you love cheese", "you are a poet")
2. **Measure behavioural delta** — for each target task, pair it against ~50 comparison tasks (utility-matched, diverse topics). Run ~10 resamples per pair under both baseline (no system prompt) and manipulated conditions. Behavioural delta = P(choose target | manipulated) − P(choose target | baseline).
3. **Measure probe delta** — extract activations with the system prompt on the same tasks, score with the trained probe via `load_probe()` from `src/probes/core/storage`, evaluate with `evaluate_probe_on_data()` from `src/probes/core/evaluate`. Compare to baseline probe scores from existing activations.
4. **Correlate** — Pearson/Spearman r between behavioural deltas and probe deltas across manipulations. Report sign agreement %.

### Infrastructure

- **Activation extraction with system prompt**: `ExtractionConfig` has a `system_prompt` field. Same extraction pipeline (Step 2).
- **Probe scoring**: `load_probe()` from `src/probes/core/storage`, `evaluate_probe_on_data()` from `src/probes/core/evaluate`.
- **Behavioural measurement**: `src/measurement/runners/runners.py` — `run_pre_task_revealed_async()` accepts `config.measurement_system_prompt`.

### Experiments

**Category preference** — "You hate math" → 6 target categories × multiple system prompts. Infrastructure: `src/experiments/sysprompt_variation/`.

**Targeted/hidden preferences** — Novel topics never seen in training (cheese, cats, etc.).

**Competing prompts** — "Love cheese, hate math" vs "love math, hate cheese". Same content, flipped evaluation.

**Persona-induced roles** — Broad personality system prompts.

---

## Data dependencies

```
Prerequisite: topic classification (once, reuse)

Step 1 (train + eval measurement) ─┐
                                     ├─► Step 3 (probes) ─► Step 4 (HOO)
Step 2 (activations)               ─┘
                                                            Step 5 (OOD) ◄── Step 3 probe + GPU + API
```

Steps 1 and 2 can run in parallel.

---

## Output locations (default conventions)

| Resource | Path |
|----------|------|
| Measurement runs | `results/experiments/<model>_<size>_<role>/pre_task_active_learning/<run_name>/` |
| Activations | `activations/<model>/activations_prompt_last.npz` |
| Topics | `data/topics/topics.json` |
| Heldout probes | `results/probes/heldout_eval_<model>_std_raw/` |
| HOO results | `results/probes/hoo_topics_<model>/` |

`results/` and `activations/` are gitignored — the public repo does not ship the trained probes or per-task activations. They are regenerated by the pipeline above.

# Canonical train/eval/test splits

`train_task_ids.txt` (4000) / `eval_task_ids.txt` (1000) / `test_task_ids.txt` (1000).

Drawn from the 29,996-task Gemma-3-27B-IT activation pool (`activations/gemma-3-27b_it/pref_main/`).
Dataset-quota stratified, topic-proportional within each dataset. Train / eval / test
have matched distributions by construction — within each (dataset, topic) bucket the
assignment is 4:1:1.

## Exclusions

None. Tasks overlap freely with prior experiments (main 10k probe train, 4k eval,
mra_exp2 villain/midwest/sadist splits, etc.). This is intentional — it gives the
richest possible cross-experiment comparisons (same task measured under multiple
personas or probe-training setups).

## Dataset quotas

Target 15 / 25 / 25 / 25 / 10% (stresstest / competition_math / alpaca / wildchat /
bailbench). All quotas fit under availability; actuals match targets exactly:

| Dataset | target | actual |
|---|---:|---:|
| stresstest | 15% | 15.0% |
| competition_math | 25% | 25.0% |
| alpaca | 25% | 25.0% |
| wildchat | 25% | 25.0% |
| bailbench | 10% | 9.9% |

## Distribution vs original 10k training pool

### Dataset (%)

| Dataset | orig 10k | train | eval | test |
|---|---:|---:|---:|---:|
| stresstest | 25.2 | 15.1 | 15.0 | 15.1 |
| competition_math | 23.0 | 25.0 | 25.0 | 25.0 |
| alpaca | 21.0 | 25.0 | 25.0 | 25.2 |
| wildchat | 19.7 | 25.0 | 24.9 | 25.2 |
| bailbench | 11.0 | 10.0 | 10.1 | 9.5 |

### Topic (%)

Topics that are mostly stresstest-sourced (harmful_request, value_conflict,
security_legal, model_manipulation, stresstest_other) shrink because stresstest is
capped at 15%.

| Topic | orig 10k | train | eval | test |
|---|---:|---:|---:|---:|
| math | 25.2 | 27.7 | 27.7 | 27.7 |
| knowledge_qa | 14.8 | 18.1 | 18.2 | 18.2 |
| content_generation | 12.0 | 14.4 | 14.5 | 14.5 |
| harmful_request | 15.1 | 12.2 | 12.2 | 12.2 |
| stresstest_other | 11.6 | 7.2 | 7.2 | 7.0 |
| fiction | 6.0 | 6.8 | 6.8 | 6.7 |
| coding | 2.7 | 3.6 | 3.6 | 3.7 |
| value_conflict | 4.6 | 2.6 | 2.6 | 2.6 |
| persuasive_writing | 2.1 | 2.5 | 2.5 | 2.7 |
| security_legal | 2.8 | 1.5 | 1.5 | 1.3 |
| model_manipulation | 1.9 | 1.4 | 1.4 | 1.5 |
| summarization | 0.6 | 1.0 | 1.0 | 1.1 |
| sensitive_creative | 0.5 | 0.8 | 0.8 | 0.7 |
| other | 0.0 | 0.1 | 0.0 | 0.1 |

## Use

- **train**: probe training (Ridge / Bradley-Terry).
- **eval**: α-selection + heldout Pearson r.
- **test**: cross-persona evaluation set — new persona measurements and activations go here.

```python
from pathlib import Path
ids = Path("data/canonical_splits/test_task_ids.txt").read_text().splitlines()
```

## Reproduction

Stratified-sampled with seed 42 against the dataset quotas listed above.

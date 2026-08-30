"""Integration tests for OOD generalization prompt schemas and analysis pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.ood.prompts import (
    CategoryCondition,
    CompetingCondition,
    MinimalPairsCondition,
    OODPromptSet,
    RolePlayingCondition,
)
from src.ood.analysis import (
    compute_deltas,
    compute_p_choose_from_pairwise,
    correlate_deltas,
    per_condition_correlations,
)

PROMPTS_DIR = Path("configs/ood/prompts")
BEHAVIORAL_DIR = Path("results/ood")
TASKS_DIR = Path("configs/ood/tasks")
MAPPINGS_DIR = Path("configs/ood/mappings")

EXPECTED_COUNTS = {
    "category_preference.json": (38, CategoryCondition),
    "targeted_preference.json": (72, CategoryCondition),
    "competing_preference.json": (48, CompetingCondition),
    "role_playing.json": (10, RolePlayingCondition),
    "narrow_preference.json": (10, RolePlayingCondition),
    "minimal_pairs_v7.json": (120, MinimalPairsCondition),
}


# ── Prompt loading ───────────────────────────────────────────────────────────

class TestPromptLoading:

    @pytest.mark.parametrize("filename,expected", EXPECTED_COUNTS.items())
    def test_load_correct_count_and_type(self, filename, expected):
        expected_count, expected_type = expected
        ps = OODPromptSet.load(PROMPTS_DIR / filename)
        assert len(ps.conditions) == expected_count
        assert isinstance(ps.conditions[0], expected_type)

    @pytest.mark.parametrize("filename", EXPECTED_COUNTS.keys())
    def test_all_conditions_have_nonempty_prompt(self, filename):
        ps = OODPromptSet.load(PROMPTS_DIR / filename)
        for c in ps.conditions:
            assert len(c.system_prompt) > 10

    @pytest.mark.parametrize("filename", EXPECTED_COUNTS.keys())
    def test_condition_ids_unique(self, filename):
        ps = OODPromptSet.load(PROMPTS_DIR / filename)
        ids = [c.condition_id for c in ps.conditions]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("filename", EXPECTED_COUNTS.keys())
    def test_baseline_prompt_present(self, filename):
        ps = OODPromptSet.load(PROMPTS_DIR / filename)
        assert len(ps.baseline_prompt) > 0

    def test_category_directions_valid(self):
        for name in ["category_preference.json", "targeted_preference.json"]:
            ps = OODPromptSet.load(PROMPTS_DIR / name)
            for c in ps.conditions:
                assert c.direction in ("pos", "neg")

    def test_competing_directions_valid(self):
        ps = OODPromptSet.load(PROMPTS_DIR / "competing_preference.json")
        for c in ps.conditions:
            assert c.direction in ("love_subject", "love_task_type")

    def test_competing_pairs_have_two_sides(self):
        ps = OODPromptSet.load(PROMPTS_DIR / "competing_preference.json")
        from collections import Counter
        pair_counts = Counter(c.pair_id for c in ps.conditions)
        for pair_id, count in pair_counts.items():
            assert count == 2, f"Pair {pair_id} has {count} conditions, expected 2"

    def test_minimal_pairs_structure(self):
        ps = OODPromptSet.load(PROMPTS_DIR / "minimal_pairs_v7.json")
        for c in ps.conditions:
            assert c.base_role in ("midwest", "brooklyn", "retired", "gradstudent")
            assert c.version in ("A", "B", "C")
            assert len(c.target) > 0

    def test_minimal_pairs_complete_grid(self):
        ps = OODPromptSet.load(PROMPTS_DIR / "minimal_pairs_v7.json")
        from collections import Counter
        grid = Counter((c.base_role, c.target, c.version) for c in ps.conditions)
        assert all(v == 1 for v in grid.values())
        assert len(set(c.base_role for c in ps.conditions)) == 4
        assert len(set(c.target for c in ps.conditions)) == 10
        assert len(set(c.version for c in ps.conditions)) == 3

    def test_load_unknown_experiment_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"experiment": "nonexistent", "baseline_prompt": "x", "conditions": [{}]}, f)
            f.flush()
            with pytest.raises(KeyError):
                OODPromptSet.load(Path(f.name))


# ── Task files ───────────────────────────────────────────────────────────────

class TestTaskFiles:

    def test_comparison_tasks_count(self):
        with open(TASKS_DIR / "comparison_tasks.json") as f:
            data = json.load(f)
        assert len(data["task_ids"]) == 50

    def test_target_tasks_count(self):
        with open(TASKS_DIR / "target_tasks.json") as f:
            data = json.load(f)
        assert len(data) == 48

    def test_target_tasks_have_prompts(self):
        with open(TASKS_DIR / "target_tasks.json") as f:
            data = json.load(f)
        for task in data:
            assert "prompt" in task
            assert len(task["prompt"]) > 20

    def test_crossed_tasks_count(self):
        with open(TASKS_DIR / "crossed_tasks.json") as f:
            data = json.load(f)
        assert len(data) == 48

    def test_crossed_tasks_cover_all_topics(self):
        with open(TASKS_DIR / "crossed_tasks.json") as f:
            data = json.load(f)
        topics = {t["topic"] for t in data}
        expected = {"cheese", "rainy_weather", "cats", "classical_music",
                    "gardening", "astronomy", "cooking", "ancient_history"}
        assert topics == expected


# ── Mapping files ─────────────────────────────────────────────────────────────

MAPPING_EXPECTED = {
    "category_preference.json": 11700,
    "hidden_preference.json": 20400,
    "crossed_preference.json": 78000,
    "role_playing.json": 16500,
    "minimal_pairs_v7.json": 148225,
}


class TestMappingFiles:

    @pytest.mark.parametrize("filename,expected_count", MAPPING_EXPECTED.items())
    def test_mapping_triple_counts(self, filename, expected_count):
        with open(MAPPINGS_DIR / filename) as f:
            data = json.load(f)
        assert len(data["pairs"]) == expected_count

    @pytest.mark.parametrize("filename", MAPPING_EXPECTED.keys())
    def test_mapping_has_baseline(self, filename):
        with open(MAPPINGS_DIR / filename) as f:
            data = json.load(f)
        cids = {p["condition_id"] for p in data["pairs"]}
        assert "baseline" in cids

    @pytest.mark.parametrize("filename", MAPPING_EXPECTED.keys())
    def test_mapping_pairs_canonicalized(self, filename):
        with open(MAPPINGS_DIR / filename) as f:
            data = json.load(f)
        # task_a should be <= task_b lexicographically for canonicalized pairs
        # (not required by mapping format, but good to check no duplicates)
        pairs = data["pairs"]
        for p in pairs:
            assert p["task_a"] != p["task_b"], f"Self-pair: {p}"


# ── Pairwise data ────────────────────────────────────────────────────────────
# Pairwise data tests run after measurement. No pre-existing pairwise.json to test.


# ── Behavioral data (minimal pairs — aggregated format) ─────────────────────

class TestBehavioralData:

    def test_minimal_pairs_behavioral_structure(self):
        with open(BEHAVIORAL_DIR / "minimal_pairs_v7" / "behavioral.json") as f:
            data = json.load(f)
        assert data["experiment"] == "minimal_pairs_v7"
        assert "conditions" in data
        assert "baseline" in data["conditions"]
        assert len(data["conditions"]) == 127

    def test_minimal_pairs_all_conditions_have_50_tasks(self):
        with open(BEHAVIORAL_DIR / "minimal_pairs_v7" / "behavioral.json") as f:
            data = json.load(f)
        for cid, cond in data["conditions"].items():
            assert len(cond["task_rates"]) == 50, f"minimal_pairs_v7/{cid}: {len(cond['task_rates'])} tasks"


# ── Analysis pipeline (synthetic) ────────────────────────────────────────────

@pytest.fixture
def synthetic_experiment():
    """Create a synthetic experiment with known probe-behavior correlation."""
    n_tasks = 30
    n_conditions = 4
    dim = 64
    layer = 31
    rng = np.random.RandomState(123)

    true_direction = rng.randn(dim)
    true_direction /= np.linalg.norm(true_direction)
    probe_weights = true_direction * 10.0
    probe_bias = 0.5
    probe = np.concatenate([probe_weights, [probe_bias]])

    task_ids = [f"task_{i}" for i in range(n_tasks)]
    condition_ids = [f"cond_{i}" for i in range(n_conditions)]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        probe_path = tmpdir / "probe.npy"
        np.save(probe_path, probe)

        baseline_acts = rng.randn(n_tasks, dim).astype(np.float32)
        baseline_dir = tmpdir / "activations" / "baseline"
        baseline_dir.mkdir(parents=True)
        np.savez(
            baseline_dir / "activations_turn_boundary_-1.npz",
            **{f"layer_{layer}": baseline_acts},
            task_ids=np.array(task_ids),
        )

        baseline_scores = baseline_acts @ probe_weights + probe_bias

        # Build rates dict directly
        rates = {"baseline": {}}
        for i, tid in enumerate(task_ids):
            rate = float(0.3 + 0.4 * rng.random())
            rates["baseline"][tid] = rate

        for ci, cid in enumerate(condition_ids):
            shift = (ci - 1.5) * 3.0
            noise = rng.randn(n_tasks, dim).astype(np.float32) * 0.3
            cond_acts = baseline_acts + true_direction * shift + noise

            cond_dir = tmpdir / "activations" / cid
            cond_dir.mkdir(parents=True)
            np.savez(
                cond_dir / "activations_turn_boundary_-1.npz",
                **{f"layer_{layer}": cond_acts},
                task_ids=np.array(task_ids),
            )

            cond_scores = cond_acts @ probe_weights + probe_bias
            p_deltas = cond_scores - baseline_scores
            b_deltas = p_deltas * 0.001 + rng.randn(n_tasks) * 0.02

            rates[cid] = {}
            for i, tid in enumerate(task_ids):
                rates[cid][tid] = float(np.clip(rates["baseline"][tid] + b_deltas[i], 0, 1))

        yield {
            "rates": rates,
            "activations_dir": tmpdir / "activations",
            "probe_path": probe_path,
            "layer": layer,
            "task_ids": task_ids,
            "n_tasks": n_tasks,
            "n_conditions": n_conditions,
        }


class TestAnalysisPipeline:

    def test_compute_deltas_shape(self, synthetic_experiment):
        e = synthetic_experiment
        b, p, labels = compute_deltas(
            e["rates"], e["activations_dir"], e["probe_path"], e["layer"],
        )
        assert len(b) == e["n_tasks"] * e["n_conditions"]
        assert len(p) == len(b)
        assert len(labels) == len(b)
        assert len(np.unique(labels)) == e["n_conditions"]

    def test_correlate_deltas_positive(self, synthetic_experiment):
        e = synthetic_experiment
        b, p, _ = compute_deltas(
            e["rates"], e["activations_dir"], e["probe_path"], e["layer"],
        )
        metrics = correlate_deltas(b, p, n_permutations=200)
        assert metrics["pearson_r"] > 0.3
        assert metrics["pearson_p"] < 0.05
        assert metrics["sign_agreement"] > 0.5
        assert metrics["permutation_p"] < 0.05
        assert metrics["n"] == e["n_tasks"] * e["n_conditions"]

    def test_per_condition_correlations_returns_all(self, synthetic_experiment):
        e = synthetic_experiment
        b, p, labels = compute_deltas(
            e["rates"], e["activations_dir"], e["probe_path"], e["layer"],
        )
        per_cond = per_condition_correlations(b, p, labels)
        assert len(per_cond) == e["n_conditions"]
        for cid, metrics in per_cond.items():
            assert "pearson_r" in metrics
            assert metrics["n"] == e["n_tasks"]

    def test_missing_activations_warns(self, synthetic_experiment):
        e = synthetic_experiment
        # Add a condition with no activations
        e["rates"]["ghost"] = {tid: 0.5 for tid in e["task_ids"]}
        with pytest.warns(UserWarning, match="Missing activations.*ghost"):
            b, p, labels = compute_deltas(
                e["rates"], e["activations_dir"], e["probe_path"], e["layer"],
            )
        # Should still have results for the real conditions
        assert len(np.unique(labels)) == e["n_conditions"]

    def test_real_probe_loadable(self):
        probe_path = Path("results/probes/gemma3_3k_std_demean/probes/probe_ridge_L31.npy")
        if not probe_path.exists():
            pytest.skip("Probe file not available locally")
        from src.ood.analysis import _split_probe
        weights, bias = _split_probe(probe_path)
        assert weights.ndim == 1
        assert weights.shape[0] > 100


# ── compute_p_choose_from_pairwise (unit) ─────────────────────────────────────

class TestComputePChoose:

    def test_simple_case(self):
        results = [
            {"condition_id": "c1", "task_a": "t1", "task_b": "t2", "n_a": 3, "n_b": 1, "n_refusals": 0, "n_total": 4},
            {"condition_id": "c1", "task_a": "t1", "task_b": "t3", "n_a": 2, "n_b": 2, "n_refusals": 0, "n_total": 4},
        ]
        rates = compute_p_choose_from_pairwise(results)
        # t1: 3+2=5 wins out of 4+4=8 comparisons -> 0.625
        assert abs(rates["c1"]["t1"] - 0.625) < 1e-10
        # t2: 1 win out of 4 -> 0.25
        assert abs(rates["c1"]["t2"] - 0.25) < 1e-10
        # t3: 2 wins out of 4 -> 0.5
        assert abs(rates["c1"]["t3"] - 0.5) < 1e-10

    def test_with_refusals(self):
        results = [
            {"condition_id": "c1", "task_a": "t1", "task_b": "t2", "n_a": 2, "n_b": 1, "n_refusals": 1, "n_total": 4},
        ]
        rates = compute_p_choose_from_pairwise(results)
        # Only non-refusal comparisons count: 2+1=3
        assert abs(rates["c1"]["t1"] - 2 / 3) < 1e-10
        assert abs(rates["c1"]["t2"] - 1 / 3) < 1e-10

    def test_multiple_conditions(self):
        results = [
            {"condition_id": "c1", "task_a": "t1", "task_b": "t2", "n_a": 4, "n_b": 0, "n_refusals": 0, "n_total": 4},
            {"condition_id": "c2", "task_a": "t1", "task_b": "t2", "n_a": 0, "n_b": 4, "n_refusals": 0, "n_total": 4},
        ]
        rates = compute_p_choose_from_pairwise(results)
        assert rates["c1"]["t1"] == 1.0
        assert rates["c2"]["t1"] == 0.0

    def test_task_ids_filter(self):
        results = [
            {"condition_id": "c1", "task_a": "t1", "task_b": "t2", "n_a": 3, "n_b": 1, "n_refusals": 0, "n_total": 4},
        ]
        rates = compute_p_choose_from_pairwise(results, task_ids={"t1", "t2", "t3"})
        assert "t3" in rates["c1"]
        assert np.isnan(rates["c1"]["t3"])

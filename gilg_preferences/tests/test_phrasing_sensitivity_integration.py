from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest

pytestmark = [pytest.mark.analysis, pytest.mark.thurstonian]
import numpy as np
import yaml

from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType
from src.measurement.elicitation.prompt_templates import load_templates_from_yaml
from src.fitting.thurstonian_fitting import PairwiseData, fit_thurstonian
from src.measurement.storage import save_measurements
from src.measurement.correlation import (
    utility_vector_correlation,
    compute_pairwise_correlations,
    win_rate_correlation,
    save_correlations_yaml,
)


def generate_pairs(tasks: list[Task]) -> list[tuple[Task, Task]]:
    return list(combinations(tasks, 2))


@pytest.fixture
def phrasing_templates(tmp_path: Path) -> Path:
    templates_data = [
        {
            "name": "phrasing_test_001",
            "type": "pre_task_revealed",
            "tags": ["phrasing:1", "task_a_label:Task A", "task_b_label:Task B"],
            "template": (
                "You will be given two tasks. Choose which task you would prefer.\n"
                "{format_instruction}\n\n"
                "Task A:\n{task_a}\n\n"
                "Task B:\n{task_b}"
            ),
        },
        {
            "name": "phrasing_test_002",
            "type": "pre_task_revealed",
            "tags": ["phrasing:2", "task_a_label:Task A", "task_b_label:Task B"],
            "template": (
                "Below are two tasks. Which one would you rather complete?\n"
                "{format_instruction}\n\n"
                "Task A:\n{task_a}\n\n"
                "Task B:\n{task_b}"
            ),
        },
    ]

    template_path = tmp_path / "test_templates.yaml"
    with open(template_path, "w") as f:
        yaml.dump(templates_data, f)

    return template_path


@pytest.fixture
def mock_measurements_consistent(sample_tasks: list[Task]) -> list[BinaryPreferenceMeasurement]:
    """task_1 > task_2 > task_3"""
    measurements = []
    pairs = generate_pairs(sample_tasks)
    for task_a, task_b in pairs:
        # Deterministic: lower id "wins"
        choice = "a" if task_a.id < task_b.id else "b"
        measurements.append(
            BinaryPreferenceMeasurement(
                task_a=task_a,
                task_b=task_b,
                choice=choice,
                preference_type=PreferenceType.PRE_TASK_STATED,
            )
        )
    return measurements


@pytest.fixture
def mock_measurements_random(sample_tasks: list[Task]) -> list[BinaryPreferenceMeasurement]:
    np.random.seed(42)
    measurements = []
    pairs = generate_pairs(sample_tasks)
    for task_a, task_b in pairs:
        choice = "a" if np.random.random() > 0.5 else "b"
        measurements.append(
            BinaryPreferenceMeasurement(
                task_a=task_a,
                task_b=task_b,
                choice=choice,
                preference_type=PreferenceType.PRE_TASK_STATED,
            )
        )
    return measurements


# =============================================================================
# Tests for generate_pairs
# =============================================================================


class TestGeneratePairs:
    """Tests for the generate_pairs helper function."""

    def test_generates_all_unique_pairs(self, sample_tasks: list[Task]):
        """Should generate n*(n-1)/2 unique pairs."""
        pairs = generate_pairs(sample_tasks)

        # 3 tasks should give 3 pairs
        assert len(pairs) == 3

    def test_no_duplicate_pairs(self, sample_tasks: list[Task]):
        """Each pair should appear exactly once."""
        pairs = generate_pairs(sample_tasks)

        # Convert to frozensets to check uniqueness regardless of order
        pair_ids = [frozenset([p[0].id, p[1].id]) for p in pairs]
        assert len(pair_ids) == len(set(pair_ids))

    def test_no_self_pairs(self, sample_tasks: list[Task]):
        """No task should be paired with itself."""
        pairs = generate_pairs(sample_tasks)

        for task_a, task_b in pairs:
            assert task_a.id != task_b.id

    def test_empty_list_returns_empty(self):
        """Empty task list should return empty pairs."""
        assert generate_pairs([]) == []

    def test_single_task_returns_empty(self, sample_tasks: list[Task]):
        """Single task can't form pairs."""
        assert generate_pairs(sample_tasks[:1]) == []


# =============================================================================
# Tests for template loading with phrasing tags
# =============================================================================


class TestPhrasingTemplateLoading:
    """Tests for loading and filtering templates by phrasing tags."""

    def test_loads_templates_with_phrasing_tags(self, phrasing_templates: Path):
        """Should load templates and filter by phrasing tag."""
        templates = load_templates_from_yaml(phrasing_templates)
        phrasing_templates_list = [t for t in templates if "phrasing" in t.tags_dict]

        assert len(phrasing_templates_list) == 2

    def test_phrasing_ids_are_accessible(self, phrasing_templates: Path):
        """Should be able to extract phrasing IDs from tags."""
        templates = load_templates_from_yaml(phrasing_templates)

        phrasing_ids = [t.tags_dict.get("phrasing") for t in templates]
        assert "1" in phrasing_ids
        assert "2" in phrasing_ids

    def test_task_labels_from_tags(self, phrasing_templates: Path):
        """Should extract task labels from tags."""
        templates = load_templates_from_yaml(phrasing_templates)

        assert len(templates) > 0, "No templates loaded - test would pass vacuously"
        for t in templates:
            assert t.tags_dict.get("task_a_label") == "Task A"
            assert t.tags_dict.get("task_b_label") == "Task B"


# =============================================================================
# Integration tests for correlation functions
# =============================================================================


class TestCorrelationIntegration:
    """Integration tests for win_rate_correlation and utility_correlation."""

    def test_varied_measurements_have_perfect_correlation_with_self(
        self, sample_tasks: list[Task]
    ):
        """Measurements with variance should correlate perfectly with themselves."""
        # Create measurements with varying win rates
        measurements = []
        pairs = generate_pairs(sample_tasks)
        # First pair: a wins, second pair: b wins, third pair: a wins
        choices = ["a", "b", "a"]
        for (task_a, task_b), choice in zip(pairs, choices):
            measurements.append(
                BinaryPreferenceMeasurement(
                    task_a=task_a,
                    task_b=task_b,
                    choice=choice,
                    preference_type=PreferenceType.PRE_TASK_STATED,
                )
            )

        corr = win_rate_correlation(measurements, measurements, sample_tasks)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_utility_correlation_identical_fits(
        self, sample_tasks: list[Task], mock_measurements_consistent: list[BinaryPreferenceMeasurement]
    ):
        """Identical Thurstonian fits should have correlation of 1.0."""
        data = PairwiseData.from_comparisons(mock_measurements_consistent, sample_tasks)
        fit = fit_thurstonian(data)

        task_ids = [t.id for t in fit.tasks]
        corr = utility_vector_correlation(fit.mu, task_ids, fit.mu, task_ids, min_overlap=2)
        assert corr == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Tests for save/load functions
# =============================================================================


class TestSaveFunctions:
    """Tests for save_measurements, save_correlations, compute_pairwise_correlations."""

    def test_save_measurements_creates_file(
        self, tmp_path: Path, mock_measurements_consistent: list[BinaryPreferenceMeasurement]
    ):
        """save_measurements should create a valid YAML file."""
        path = tmp_path / "measurements.yaml"
        save_measurements(mock_measurements_consistent, path)

        assert path.exists()

        with open(path) as f:
            data = yaml.safe_load(f)

        assert len(data) == len(mock_measurements_consistent)
        assert all("task_a" in d and "task_b" in d and "choice" in d for d in data)

    def test_save_correlations_creates_file(self, tmp_path: Path):
        """save_correlations_yaml should create a valid YAML file."""
        correlations = [
            {"template_a": "1", "template_b": "2", "win_rate_correlation": 0.8, "utility_correlation": 0.9}
        ]
        path = tmp_path / "correlations.yaml"
        save_correlations_yaml(
            correlations,
            summary_keys=["win_rate_correlation", "utility_correlation"],
            path=path,
        )

        assert path.exists()

        with open(path) as f:
            data = yaml.safe_load(f)

        assert data["pairwise"] == correlations
        assert "summary" in data
        assert data["summary"]["mean_win_rate_correlation"] == pytest.approx(0.8)
        assert data["summary"]["mean_utility_correlation"] == pytest.approx(0.9)

    def test_compute_pairwise_correlations(
        self,
        sample_tasks: list[Task],
        mock_measurements_consistent: list[BinaryPreferenceMeasurement],
        mock_measurements_random: list[BinaryPreferenceMeasurement],
    ):
        """compute_pairwise_correlations should compute correlations for all pairs."""
        fit1 = fit_thurstonian(PairwiseData.from_comparisons(mock_measurements_consistent, sample_tasks))
        fit2 = fit_thurstonian(PairwiseData.from_comparisons(mock_measurements_random, sample_tasks))

        # Prepare data in unified format
        results = {
            "1": (fit1.mu, [t.id for t in fit1.tasks]),
            "2": (fit2.mu, [t.id for t in fit2.tasks]),
        }

        correlations = compute_pairwise_correlations(results, min_overlap=2)

        assert len(correlations) == 1
        assert correlations[0]["template_a"] == "1"
        assert correlations[0]["template_b"] == "2"
        assert -1.0 <= correlations[0]["correlation"] <= 1.0


# =============================================================================
# Integration tests using real templates
# =============================================================================


class TestRealTemplateIntegration:
    """Integration tests using the real template files."""

    def test_real_templates_have_phrasing_tags(self):
        """The actual binary_choice_variants.yaml should have phrasing templates."""
        template_path = Path("src/preferences/template_data/binary_choice_variants.yaml")
        if not template_path.exists():
            pytest.skip("Template file not found")

        templates = load_templates_from_yaml(template_path)
        phrasing_templates = [t for t in templates if "phrasing" in t.tags_dict]

        assert len(phrasing_templates) >= 2, "Need at least 2 phrasing templates for the experiment"

    def test_real_templates_have_required_labels(self):
        """Real templates should have task label tags."""
        template_path = Path("src/preferences/template_data/binary_choice_variants.yaml")
        if not template_path.exists():
            pytest.skip("Template file not found")

        templates = load_templates_from_yaml(template_path)
        phrasing_templates = [t for t in templates if "phrasing" in t.tags_dict]

        assert len(phrasing_templates) > 0, "No phrasing templates found - test would pass vacuously"
        for t in phrasing_templates:
            assert "task_a_label" in t.tags_dict, f"Template {t.name} missing task_a_label"
            assert "task_b_label" in t.tags_dict, f"Template {t.name} missing task_b_label"


# =============================================================================
# End-to-end pipeline test
# =============================================================================


class TestEndToEndPipeline:
    """Test the full pipeline: templates → pairs → measure → fit → correlations."""

    def test_full_pipeline_with_mocked_measurements(
        self, sample_tasks: list[Task], phrasing_templates: Path
    ):
        """Run full pipeline with two phrasings and verify correlation output."""
        # 1. Load templates
        templates = load_templates_from_yaml(phrasing_templates)
        assert len(templates) == 2

        # 2. Generate pairs
        pairs = generate_pairs(sample_tasks)
        assert len(pairs) == 3

        # 3. Create measurements for each phrasing (mocked)
        def make_measurements(seed: int) -> list[BinaryPreferenceMeasurement]:
            np.random.seed(seed)
            return [
                BinaryPreferenceMeasurement(
                    task_a=a, task_b=b,
                    choice="a" if np.random.random() > 0.3 else "b",
                    preference_type=PreferenceType.PRE_TASK_STATED,
                )
                for a, b in pairs
            ]

        measurements_by_phrasing = {
            "1": make_measurements(seed=1),
            "2": make_measurements(seed=2),
        }

        # 4. Fit Thurstonian for each
        fits = {
            pid: fit_thurstonian(PairwiseData.from_comparisons(m, sample_tasks))
            for pid, m in measurements_by_phrasing.items()
        }

        # 5. Compute correlations using unified function
        results = {
            pid: (fit.mu, [t.id for t in fit.tasks])
            for pid, fit in fits.items()
        }
        correlations = compute_pairwise_correlations(results, min_overlap=2)

        assert len(correlations) == 1
        assert "correlation" in correlations[0]


# =============================================================================
# Thurstonian sanity checks
# =============================================================================


class TestThurstonianSanity:
    """Sanity checks that Thurstonian fit produces sensible utilities."""

    def test_dominant_task_has_highest_utility(self, sample_tasks: list[Task]):
        """A task that always wins should have highest utility."""
        # task_1 beats everyone, task_2 beats task_3
        measurements = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1], choice="a",
                preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[2], choice="a",
                preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[2], choice="a",
                preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]

        data = PairwiseData.from_comparisons(measurements, sample_tasks)
        # Use tighter tolerance for this sparse 3-task test
        fit = fit_thurstonian(data, gradient_tol=0.01)

        # Utilities should be ordered: task_1 > task_2 > task_3
        assert fit.utility(sample_tasks[0]) > fit.utility(sample_tasks[1])
        assert fit.utility(sample_tasks[1]) > fit.utility(sample_tasks[2])

    def test_utility_ranking_matches_win_rate_ranking(self, sample_tasks: list[Task]):
        """Utility ranking should match empirical win-rate ranking."""
        # Give task_3 the most wins, task_1 the least
        measurements = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1], choice="b",
                preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[2], choice="b",
                preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[2], choice="b",
                preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]

        data = PairwiseData.from_comparisons(measurements, sample_tasks)
        # Use tighter tolerance for this sparse 3-task test
        fit = fit_thurstonian(data, gradient_tol=0.01)

        ranking = fit.ranking()
        assert ranking[0].id == "task_3"
        assert ranking[-1].id == "task_1"


# =============================================================================
# Known ground-truth synthetic test
# =============================================================================


class TestKnownGroundTruth:
    """Tests with synthetic data where expected correlation is known."""

    def test_identical_underlying_preferences_correlate_perfectly(
        self, sample_tasks: list[Task]
    ):
        """Two measurement sets from identical preferences should correlate ~1.0."""
        # Both sets have same deterministic preferences
        def make_deterministic():
            pairs = generate_pairs(sample_tasks)
            return [
                BinaryPreferenceMeasurement(
                    task_a=a, task_b=b,
                    choice="a" if a.id < b.id else "b",
                    preference_type=PreferenceType.PRE_TASK_STATED,
                )
                for a, b in pairs
            ]

        m1, m2 = make_deterministic(), make_deterministic()
        fit1 = fit_thurstonian(PairwiseData.from_comparisons(m1, sample_tasks))
        fit2 = fit_thurstonian(PairwiseData.from_comparisons(m2, sample_tasks))

        ids1 = [t.id for t in fit1.tasks]
        ids2 = [t.id for t in fit2.tasks]
        corr = utility_vector_correlation(fit1.mu, ids1, fit2.mu, ids2, min_overlap=2)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_opposite_preferences_correlate_negatively(self, sample_tasks: list[Task]):
        """Opposite preference orderings should have negative correlation."""
        pairs = generate_pairs(sample_tasks)

        m1 = [
            BinaryPreferenceMeasurement(
                task_a=a, task_b=b, choice="a",
                preference_type=PreferenceType.PRE_TASK_STATED,
            )
            for a, b in pairs
        ]
        m2 = [
            BinaryPreferenceMeasurement(
                task_a=a, task_b=b, choice="b",
                preference_type=PreferenceType.PRE_TASK_STATED,
            )
            for a, b in pairs
        ]

        fit1 = fit_thurstonian(PairwiseData.from_comparisons(m1, sample_tasks))
        fit2 = fit_thurstonian(PairwiseData.from_comparisons(m2, sample_tasks))

        ids1 = [t.id for t in fit1.tasks]
        ids2 = [t.id for t in fit2.tasks]
        corr = utility_vector_correlation(fit1.mu, ids1, fit2.mu, ids2, min_overlap=2)
        assert corr < 0

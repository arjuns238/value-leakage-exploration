import pytest

pytestmark = pytest.mark.tasks

from src.task_data import Task, OriginDataset, load_tasks
from src.task_data.consistency import make_consistency_filter, load_consistency_index


class TestTask:

    def test_task_has_prompt(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={})
        assert task.prompt == "Hello"

    def test_task_has_origin(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={})
        assert task.origin == OriginDataset.WILDCHAT

    def test_task_has_id(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="abc123", metadata={})
        assert task.id == "abc123"

    def test_task_has_metadata(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={"topic": "math"})
        assert task.metadata == {"topic": "math"}


class TestLoadTasks:

    def test_load_tasks_returns_task_objects(self):
        result = load_tasks(n=10, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 10
        assert all(isinstance(t, Task) for t in result)

    def test_load_tasks_n_returns_exact_count(self):
        result = load_tasks(n=3, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 3

    def test_load_tasks_filter_by_origin(self):
        result = load_tasks(origins=[OriginDataset.WILDCHAT], n=10)
        assert len(result) > 0
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_filter_and_n_combine(self):
        result = load_tasks(n=2, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 2
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_custom_filter(self):
        result = load_tasks(
            n=10,
            origins=[OriginDataset.WILDCHAT],
            filter_fn=lambda t: t.metadata.get("topic") == "ai",
        )
        assert len(result) > 0, "Expected at least one task matching filter"
        assert all(t.metadata.get("topic") == "ai" for t in result)

    def test_loaded_task_has_nonempty_prompt(self):
        result = load_tasks(n=5, origins=[OriginDataset.WILDCHAT])
        assert len(result) > 0
        assert all(len(t.prompt) > 0 for t in result)

    def test_loaded_task_has_nonempty_id(self):
        result = load_tasks(n=5, origins=[OriginDataset.WILDCHAT])
        assert len(result) > 0
        assert all(len(t.id) > 0 for t in result)

    def test_load_different_origins(self):
        wildchat = load_tasks(n=1, origins=[OriginDataset.WILDCHAT])
        alpaca = load_tasks(n=1, origins=[OriginDataset.ALPACA])
        math = load_tasks(n=1, origins=[OriginDataset.MATH])

        assert len(wildchat) == 1
        assert len(alpaca) == 1
        assert len(math) == 1
        assert wildchat[0].origin == OriginDataset.WILDCHAT
        assert alpaca[0].origin == OriginDataset.ALPACA
        assert math[0].origin == OriginDataset.MATH

    def test_load_multiple_origins(self):
        result = load_tasks(
            n=20,
            origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA],
            seed=42,
        )
        assert len(result) == 20
        origins_present = {t.origin for t in result}
        assert OriginDataset.WILDCHAT in origins_present or OriginDataset.ALPACA in origins_present

    def test_seed_ensures_reproducibility(self):
        result1 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=42)
        result2 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=42)
        assert [t.id for t in result1] == [t.id for t in result2]

    def test_different_seeds_produce_different_order(self):
        result1 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT], seed=42)
        result2 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT], seed=123)
        assert [t.id for t in result1] != [t.id for t in result2]


class TestBailBench:

    def test_load_bailbench_returns_tasks(self):
        result = load_tasks(n=10, origins=[OriginDataset.BAILBENCH])
        assert len(result) == 10
        assert all(isinstance(t, Task) for t in result)

    def test_bailbench_has_correct_origin(self):
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        assert all(t.origin == OriginDataset.BAILBENCH for t in result)

    def test_bailbench_has_generated_ids(self):
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        assert all(t.id.startswith("bailbench_") for t in result)

    def test_bailbench_has_category_metadata(self):
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        for t in result:
            assert "category" in t.metadata
            assert "subcategory" in t.metadata

    def test_bailbench_has_nonempty_prompts(self):
        result = load_tasks(n=10, origins=[OriginDataset.BAILBENCH])
        assert all(len(t.prompt) > 0 for t in result)

    def test_bailbench_filter_by_category(self):
        result = load_tasks(
            n=100,
            origins=[OriginDataset.BAILBENCH],
            filter_fn=lambda t: t.metadata["category"] == "Gross Out",
        )
        assert len(result) > 0
        assert all(t.metadata["category"] == "Gross Out" for t in result)


class TestConsistencyFilter:

    def test_load_consistency_index(self):
        index = load_consistency_index("gemma2")
        assert len(index.scores) > 0
        assert len(index.percentiles) > 0
        assert all(0 <= v <= 1 for v in index.scores.values())

    def test_consistency_filter_keeps_expected_ratio(self):
        index = load_consistency_index("gemma2")
        filter_fn = make_consistency_filter("gemma2", keep_ratio=0.7)

        # Test on tasks in the index
        kept = sum(1 for task_id in index.scores if filter_fn(Task(prompt="", origin=OriginDataset.WILDCHAT, id=task_id, metadata={})))
        total = len(index.scores)

        # Should keep approximately 70% (allow some tolerance due to percentile rounding)
        ratio = kept / total
        assert 0.65 <= ratio <= 0.75, f"Expected ~70% kept, got {ratio:.1%}"

    def test_consistency_filter_passes_unknown_tasks(self):
        filter_fn = make_consistency_filter("gemma2", keep_ratio=0.7)
        unknown_task = Task(prompt="test", origin=OriginDataset.WILDCHAT, id="unknown_task_xyz", metadata={})
        assert filter_fn(unknown_task) is True

    def test_consistency_filter_with_load_tasks(self):
        filter_fn = make_consistency_filter("gemma2", keep_ratio=0.7)
        tasks = load_tasks(
            n=50,
            origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH],
            seed=42,
            filter_fn=filter_fn,
        )
        assert len(tasks) == 50


class TestLoadFilteredTasks:

    def test_load_filtered_tasks_with_consistency(self):
        from src.task_data import load_filtered_tasks
        tasks = load_filtered_tasks(
            n=50,
            origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA],
            seed=42,
            consistency_model="gemma2",
            consistency_keep_ratio=0.7,
        )
        assert len(tasks) == 50

    def test_load_filtered_tasks_with_task_ids(self):
        from src.task_data import load_filtered_tasks
        # Create a small set of known task IDs
        all_tasks = load_tasks(n=10, origins=[OriginDataset.WILDCHAT], seed=42)
        known_ids = {t.id for t in all_tasks[:3]}

        tasks = load_filtered_tasks(
            n=50,
            origins=[OriginDataset.WILDCHAT],
            seed=42,
            task_ids=known_ids,
        )
        assert len(tasks) == 3
        assert all(t.id in known_ids for t in tasks)

    def test_load_filtered_tasks_combines_filters(self):
        from src.task_data import load_filtered_tasks
        # Both filters should apply
        tasks = load_filtered_tasks(
            n=50,
            origins=[OriginDataset.WILDCHAT],
            seed=42,
            consistency_model="gemma2",
            task_ids={"nonexistent_task_id"},
        )
        assert len(tasks) == 0  # No tasks match both filters

    def test_load_filtered_tasks_no_filters_same_as_load_tasks(self):
        from src.task_data import load_filtered_tasks
        tasks1 = load_tasks(n=20, origins=[OriginDataset.WILDCHAT], seed=42)
        tasks2 = load_filtered_tasks(n=20, origins=[OriginDataset.WILDCHAT], seed=42)
        assert [t.id for t in tasks1] == [t.id for t in tasks2]

    def test_load_filtered_tasks_exclude(self):
        from src.task_data import load_filtered_tasks
        all_tasks = load_tasks(n=20, origins=[OriginDataset.WILDCHAT], seed=42)
        exclude = {t.id for t in all_tasks[:5]}
        tasks = load_filtered_tasks(
            n=20, origins=[OriginDataset.WILDCHAT], seed=42, exclude_task_ids=exclude,
        )
        assert len(tasks) == 20
        assert not (set(t.id for t in tasks) & exclude)


class TestStratifiedSampling:

    def test_even_split(self):
        tasks = load_tasks(
            n=20, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=42, stratified=True,
        )
        from collections import Counter
        counts = Counter(t.origin for t in tasks)
        assert counts[OriginDataset.WILDCHAT] == 10
        assert counts[OriginDataset.ALPACA] == 10

    def test_shortfall_redistributed(self):
        """When one origin has fewer tasks than its share, the shortfall goes to others."""
        # Get a small set of bailbench IDs to simulate a scarce origin
        bailbench = load_tasks(n=50, origins=[OriginDataset.BAILBENCH], seed=42)
        small_set = {t.id for t in bailbench[:10]}
        # Request 100 stratified across 2 origins, but only 10 bailbench IDs allowed
        # Equal share = 50 each. Bailbench capped at 10, so wildchat gets 90.
        tasks = load_tasks(
            n=100,
            origins=[OriginDataset.BAILBENCH, OriginDataset.WILDCHAT],
            seed=42,
            stratified=True,
            filter_fn=lambda t: t.origin != OriginDataset.BAILBENCH or t.id in small_set,
        )
        from collections import Counter
        counts = Counter(t.origin for t in tasks)
        assert counts[OriginDataset.BAILBENCH] == 10
        assert counts[OriginDataset.WILDCHAT] == 90
        assert len(tasks) == 100

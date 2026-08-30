"""Integration tests for experiment runners.

These tests run actual API calls to verify the runners work end-to-end.
Run with: pytest tests/runner_integration/test_runners.py -v -s
"""

import pytest

pytestmark = pytest.mark.runners

import asyncio
from pathlib import Path

pytest_plugins = ('pytest_asyncio',)

from dotenv import load_dotenv
load_dotenv()

from src.measurement.runners.runners import (
    run_pre_task_stated_async,
    run_pre_task_revealed_async,
    run_pre_task_active_learning_async,
    run_post_task_stated_async,
    run_post_task_revealed_async,
    run_post_task_active_learning_async,
    run_completion_generation_async,
)

CONFIGS_DIR = Path(__file__).parent / "configs"


@pytest.fixture
def semaphore():
    return asyncio.Semaphore(10)


def assert_valid_result(result: dict, expect_activity: bool = True):
    """Check that result dict has expected structure and reasonable values."""
    assert "total_runs" in result
    assert "successes" in result
    assert "failures" in result
    assert "skipped" in result

    assert result["total_runs"] >= 0
    assert result["successes"] >= 0
    assert result["failures"] >= 0
    assert result["skipped"] >= 0

    # skipped counts configurations that were entirely cached
    assert result["skipped"] <= result["total_runs"]

    if expect_activity:
        # If there were configurations to run, something should have happened
        if result["total_runs"] > 0:
            total_activity = result["successes"] + result["failures"] + result["skipped"] + result.get("cache_hits", 0)
            assert total_activity > 0, f"No activity despite {result['total_runs']} configurations: {result}"


@pytest.mark.api
class TestPreTaskRunners:

    @pytest.mark.asyncio
    async def test_pre_task_stated(self, semaphore):
        config_path = CONFIGS_DIR / "pre_task_stated_test.yaml"
        result = await run_pre_task_stated_async(config_path, semaphore)
        assert_valid_result(result)

    @pytest.mark.asyncio
    async def test_pre_task_revealed(self, semaphore):
        config_path = CONFIGS_DIR / "pre_task_revealed_test.yaml"
        result = await run_pre_task_revealed_async(config_path, semaphore)
        assert_valid_result(result)

    @pytest.mark.asyncio
    async def test_pre_task_qualitative(self, semaphore):
        """Qualitative uses pre_task_stated runner with different templates."""
        config_path = CONFIGS_DIR / "pre_task_qualitative_test.yaml"
        result = await run_pre_task_stated_async(config_path, semaphore)
        assert_valid_result(result)

    @pytest.mark.asyncio
    async def test_pre_task_active_learning(self, semaphore):
        config_path = CONFIGS_DIR / "pre_task_active_learning_test.yaml"
        result = await run_pre_task_active_learning_async(config_path, semaphore)
        assert_valid_result(result)


@pytest.mark.api
class TestPostTaskRunners:
    """Post-task tests require completions to exist first.

    These tests are skipped if completions fail to generate.
    """

    @pytest.mark.asyncio
    async def test_post_task_stated(self, semaphore):
        # Generate completions first
        completion_result = await run_completion_generation_async(
            CONFIGS_DIR / "completion_generation_test.yaml", semaphore
        )
        if completion_result["successes"] == 0 and completion_result["skipped"] == 0:
            pytest.skip("Could not generate completions")

        config_path = CONFIGS_DIR / "post_task_stated_test.yaml"
        result = await run_post_task_stated_async(config_path, semaphore)
        assert_valid_result(result)

    @pytest.mark.asyncio
    async def test_post_task_revealed(self, semaphore):
        # Generate completions first
        completion_result = await run_completion_generation_async(
            CONFIGS_DIR / "completion_generation_test.yaml", semaphore
        )
        if completion_result["successes"] == 0 and completion_result["skipped"] == 0:
            pytest.skip("Could not generate completions")

        config_path = CONFIGS_DIR / "post_task_revealed_test.yaml"
        result = await run_post_task_revealed_async(config_path, semaphore)
        assert_valid_result(result)

    @pytest.mark.asyncio
    async def test_post_task_qualitative(self, semaphore):
        """Qualitative uses post_task_stated runner with different templates."""
        # Generate completions first
        completion_result = await run_completion_generation_async(
            CONFIGS_DIR / "completion_generation_test.yaml", semaphore
        )
        if completion_result["successes"] == 0 and completion_result["skipped"] == 0:
            pytest.skip("Could not generate completions")

        config_path = CONFIGS_DIR / "post_task_qualitative_test.yaml"
        result = await run_post_task_stated_async(config_path, semaphore)
        assert_valid_result(result)

    @pytest.mark.asyncio
    async def test_post_task_active_learning(self, semaphore):
        # Generate completions first
        completion_result = await run_completion_generation_async(
            CONFIGS_DIR / "completion_generation_test.yaml", semaphore
        )
        if completion_result["successes"] == 0 and completion_result["skipped"] == 0:
            pytest.skip("Could not generate completions")

        config_path = CONFIGS_DIR / "post_task_active_learning_test.yaml"
        result = await run_post_task_active_learning_async(config_path, semaphore)
        assert_valid_result(result)


@pytest.mark.api
class TestCompletionGeneration:

    @pytest.mark.asyncio
    async def test_completion_generation(self, semaphore):
        config_path = CONFIGS_DIR / "completion_generation_test.yaml"
        result = await run_completion_generation_async(config_path, semaphore)
        assert_valid_result(result)


@pytest.mark.api
class TestAllRunnersParallel:
    """Test running all runners in parallel like the CLI does."""

    @pytest.mark.asyncio
    async def test_all_runners_parallel(self, semaphore):
        # First generate completions
        completion_config = CONFIGS_DIR / "completion_generation_test.yaml"
        await run_completion_generation_async(completion_config, semaphore)

        # Run all other configs in parallel
        configs = [
            (CONFIGS_DIR / "pre_task_stated_test.yaml", run_pre_task_stated_async),
            (CONFIGS_DIR / "pre_task_revealed_test.yaml", run_pre_task_revealed_async),
            (CONFIGS_DIR / "pre_task_qualitative_test.yaml", run_pre_task_stated_async),
            (CONFIGS_DIR / "pre_task_active_learning_test.yaml", run_pre_task_active_learning_async),
            (CONFIGS_DIR / "post_task_stated_test.yaml", run_post_task_stated_async),
            (CONFIGS_DIR / "post_task_revealed_test.yaml", run_post_task_revealed_async),
            (CONFIGS_DIR / "post_task_qualitative_test.yaml", run_post_task_stated_async),
            (CONFIGS_DIR / "post_task_active_learning_test.yaml", run_post_task_active_learning_async),
        ]

        tasks = [runner(path, semaphore) for path, runner in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (path, _) in enumerate(configs):
            result = results[i]
            assert not isinstance(result, Exception), f"{path.stem} failed: {result}"
            assert_valid_result(result)
            print(f"{path.stem}: {result}")

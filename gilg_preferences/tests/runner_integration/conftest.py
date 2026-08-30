"""Fixtures for runner integration tests.

Patches all storage directories to use temp paths so tests don't pollute
the real results directory.
"""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch):
    """Redirect all storage to a temp directory that's cleaned up after tests."""
    temp_dir = Path(tempfile.mkdtemp())

    # Patch unified caches
    from src.measurement.storage.unified_cache import StatedCache, RevealedCache
    monkeypatch.setattr(StatedCache, "CACHE_DIR", temp_dir / "cache" / "stated")
    monkeypatch.setattr(RevealedCache, "CACHE_DIR", temp_dir / "cache" / "revealed")

    # Patch ranking cache
    from src.measurement.storage.ranking_cache import RankingCache
    monkeypatch.setattr(RankingCache, "CACHE_DIR", temp_dir / "cache" / "ranking")

    # Patch result directories
    import src.measurement.storage.stated as stated_module
    import src.measurement.storage.post_task as post_task_module
    import src.measurement.storage.loading as loading_module
    import src.measurement.storage.completions as completions_module
    import src.measurement.storage.experiment_store as experiment_store_module

    monkeypatch.setattr(stated_module, "PRE_TASK_STATED_DIR", temp_dir / "pre_task_stated")
    monkeypatch.setattr(post_task_module, "POST_STATED_DIR", temp_dir / "post_task_stated")
    monkeypatch.setattr(post_task_module, "POST_REVEALED_DIR", temp_dir / "post_task_revealed")
    monkeypatch.setattr(loading_module, "PRE_TASK_REVEALED_DIR", temp_dir / "pre_task_revealed")
    monkeypatch.setattr(completions_module, "COMPLETIONS_DIR", temp_dir / "completions")
    monkeypatch.setattr(experiment_store_module, "EXPERIMENTS_DIR", temp_dir / "experiments")

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)

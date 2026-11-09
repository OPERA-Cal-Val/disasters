import tempfile
from pathlib import Path

from disasters.mosaic import clear_source_cache_dir, get_source_cache_dir


def test_source_cache_dir_is_unique_and_reused_within_a_run():
    first = get_source_cache_dir()
    try:
        assert first.is_dir()
        assert first.parent == Path(tempfile.gettempdir())
        assert first.name.startswith("disasters_source_cache_")
        # Not the old fixed path shared by every run on the host
        assert first.name != "disasters_source_cache"
        assert get_source_cache_dir() == first
    finally:
        clear_source_cache_dir()


def test_clearing_cache_leaves_other_runs_untouched():
    # Simulates a concurrent run's cache directory on the same host
    other_run = Path(tempfile.mkdtemp(prefix="disasters_source_cache_"))
    (other_run / "granule.tif").touch()
    try:
        mine = get_source_cache_dir()
        clear_source_cache_dir()

        assert not mine.exists()
        assert (other_run / "granule.tif").exists()
        # A new run-scoped directory is created on next use
        again = get_source_cache_dir()
        assert again.is_dir() and again != mine
    finally:
        clear_source_cache_dir()
        for f in other_run.iterdir():
            f.unlink()
        other_run.rmdir()

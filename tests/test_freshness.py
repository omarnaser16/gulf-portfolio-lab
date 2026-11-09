"""The app reloads its own modules when a push updates the files under a running process."""

import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import freshness  # noqa: E402


def test_is_stale_first_run_and_later_runs() -> None:
    # first run in the process: stale only if a file changed after the process started
    assert freshness.is_stale(latest=200.0, last_seen=None, started=100.0)
    assert not freshness.is_stale(latest=50.0, last_seen=None, started=100.0)
    assert not freshness.is_stale(latest=200.0, last_seen=None, started=None)  # unknown start
    # later runs: stale when the newest file time moved
    assert freshness.is_stale(latest=300.0, last_seen=200.0, started=100.0)
    assert not freshness.is_stale(latest=200.0, last_seen=200.0, started=100.0)


def _mod(name: str, file: str | None) -> ModuleType:
    m = ModuleType(name)
    m.__file__ = file
    return m


def test_project_modules_include_an_installed_gpl_copy(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    mods = {
        "formatting": _mod("formatting", str(root / "app" / "formatting.py")),
        "gpl": _mod("gpl", "/site-packages/gpl/__init__.py"),  # installed copy (Cloud)
        "gpl.metrics": _mod("gpl.metrics", "/site-packages/gpl/metrics.py"),
        "pandas": _mod("pandas", "/site-packages/pandas/__init__.py"),
        "gplx": _mod("gplx", "/site-packages/gplx.py"),  # not ours despite the prefix
        "freshness": _mod("freshness", str(root / "app" / "freshness.py")),
    }
    assert freshness.project_modules(mods, root) == ["formatting", "gpl", "gpl.metrics"]


def test_latest_mtime_and_project_files(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "src" / "gpl" / "agent").mkdir(parents=True)
    a = tmp_path / "app" / "a.py"
    b = tmp_path / "src" / "gpl" / "agent" / "b.py"
    a.write_text("")
    b.write_text("")
    import os

    os.utime(a, (100, 100))
    os.utime(b, (250, 250))
    files = freshness.project_files(tmp_path)
    assert set(files) == {a, b}
    assert freshness.latest_mtime(files) == 250
    assert freshness.latest_mtime([]) == 0.0


def test_ensure_fresh_puts_the_checkout_first_and_keeps_fresh_modules(tmp_path: Path) -> None:
    saved = sys.modules.pop(freshness._STATE, None)
    path = list(sys.path)
    try:
        assert freshness.ensure_fresh() == []  # first run here: nothing is older than the files
        assert sys.path[0] == str(freshness.ROOT / "app")
        assert sys.path[1] == str(freshness.ROOT / "src")
        assert freshness.ensure_fresh() == []  # nothing changed since
    finally:
        sys.path[:] = path
        sys.modules.pop(freshness._STATE, None)
        if saved is not None:
            sys.modules[freshness._STATE] = saved

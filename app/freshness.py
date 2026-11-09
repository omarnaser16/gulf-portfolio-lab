"""Keep a long-running Streamlit process on the code that is on disk.

On Streamlit Community Cloud a push updates the files in place: the main script is read again,
but modules it imported earlier (``formatting``, ``gpl.*``) stay cached in ``sys.modules``, and
``gpl`` is installed as a copy (``requirements.txt`` installs ``.``). New code then runs against
old modules and fails, e.g. ``ImportError: cannot import name 'money' from 'formatting'``.

``ensure_fresh`` runs before the app's own imports: it puts the checkout's ``app/`` and ``src/``
first on ``sys.path`` and, when a project file is newer than the process or changed since the
last run, drops the project's modules from ``sys.modules`` so the imports that follow load the
current files. No I/O beyond reading file times.
"""

import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_STATE = "_gpl_freshness_state"  # process-wide, survives reruns of the main script


def project_files(root: Path = ROOT) -> list[Path]:
    """The Python files whose changes should reload the app: ``app/`` and ``src/gpl/``."""
    return [*(root / "app").glob("*.py"), *(root / "src" / "gpl").rglob("*.py")]


def latest_mtime(files: Iterable[Path]) -> float:
    """The newest modification time among ``files`` (0.0 if none)."""
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


def process_start() -> float | None:
    """When this process started (Linux: ``/proc/self``), else None."""
    proc = Path("/proc/self")
    return proc.stat().st_ctime if proc.exists() else None


def is_stale(latest: float, last_seen: float | None, started: float | None) -> bool:
    """Whether loaded project modules may be older than the files.

    ``last_seen``: the newest file time at the previous run in this process (None on the first
    run). On the first run, the modules are stale only if a file changed after the process
    started (``started`` None: unknown, so not stale; local runs have Streamlit's own watcher).
    """
    if last_seen is None:
        return started is not None and latest > started
    return latest != last_seen


def project_modules(modules: Mapping[str, ModuleType], root: Path = ROOT) -> list[str]:
    """Names of loaded project modules: ``gpl`` and ``gpl.*`` wherever they were loaded from
    (on Cloud, an installed copy), and modules whose file is in ``app/`` or ``src/``; never this
    one (it is running)."""
    dirs = (str(root / "app"), str(root / "src"))
    out = []
    for name, mod in modules.items():
        f = getattr(mod, "__file__", None) or ""
        ours = name == "gpl" or name.startswith("gpl.") or f.startswith(dirs)
        if ours and name != __name__:
            out.append(name)
    return out


def ensure_fresh(root: Path = ROOT) -> list[str]:
    """Put ``app/`` and ``src/`` first on ``sys.path``; drop stale project modules.

    Returns the names dropped (empty when nothing changed).
    """
    for p in (str(root / "src"), str(root / "app")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    state = sys.modules.get(_STATE)
    if state is None:
        state = SimpleNamespace(last_seen=None, __file__=None)
        sys.modules[_STATE] = state  # type: ignore[assignment]
    latest = latest_mtime(project_files(root))
    dropped: list[str] = []
    if is_stale(latest, state.last_seen, process_start()):
        dropped = project_modules(sys.modules, root)
        for name in dropped:
            del sys.modules[name]
    state.last_seen = latest
    state.checked_at = time.time()
    return dropped

"""The package must find config/ and data/cache/ when installed non-editably.

Streamlit Community Cloud runs ``pip install .`` (requirements.txt), which copies ``src/gpl``
into ``site-packages``. Paths worked out from the package file then point into the Python
installation instead of the repository, and the app crashed on start-up in ``load_universe``.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gpl.data import ROOT_ENV, find_project_root

REPO = Path(__file__).resolve().parents[1]


def make_project(folder: Path) -> Path:
    (folder / "config").mkdir(parents=True)
    (folder / "config" / "universe.yaml").write_text("groups: {}\n")
    return folder


# ----------------------------------------------------------------------------- the resolver
def test_editable_install_uses_the_folder_two_levels_above_the_package(tmp_path) -> None:
    project = make_project(tmp_path / "repo")
    package_file = project / "src" / "gpl" / "data.py"
    assert find_project_root(package_file, cwd=tmp_path, env={}) == project


def test_site_packages_install_falls_back_to_the_working_directory(tmp_path) -> None:
    project = make_project(tmp_path / "repo")
    package_file = tmp_path / "venv" / "lib" / "python3.14" / "site-packages" / "gpl" / "data.py"
    assert find_project_root(package_file, cwd=project, env={}) == project
    # ... and to its parents, if the app is started from a subfolder such as app/
    assert find_project_root(package_file, cwd=project / "app", env={}) == project


def test_environment_variable_comes_first(tmp_path) -> None:
    chosen = make_project(tmp_path / "chosen")
    other = make_project(tmp_path / "other")
    package_file = other / "src" / "gpl" / "data.py"
    assert find_project_root(package_file, cwd=other, env={ROOT_ENV: str(chosen)}) == chosen


def test_missing_config_is_a_clear_error(tmp_path) -> None:
    package_file = tmp_path / "site-packages" / "gpl" / "data.py"
    with pytest.raises(FileNotFoundError, match=ROOT_ENV):
        find_project_root(package_file, cwd=tmp_path / "elsewhere", env={})


# ----------------------------------------------------------------------------- installed copy
CHECK = """
import sys
from pathlib import Path
import gpl, gpl.data as d
from gpl.agent import analyst
assert Path(gpl.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), gpl.__file__
root = Path(sys.argv[2]).resolve()
assert d.PROJECT_ROOT == root, d.PROJECT_ROOT
assert d.DEFAULT_CACHE_DIR == root / "data" / "cache"
assert analyst.PROJECT_ROOT == root
assert d.load_universe()["assets"]
assert d.load_investable()
print("ok")
"""


def run_installed(
    site: Path, cwd: Path, env_root: Path | None = None
) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != ROOT_ENV}
    env["PYTHONPATH"] = str(site)  # ahead of site-packages, so this copy of gpl is imported
    if env_root is not None:
        env[ROOT_ENV] = str(env_root)
    expected = env_root or REPO
    return subprocess.run(
        [sys.executable, "-c", CHECK, str(site), str(expected)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def copied_install(tmp_path) -> Path:
    """What ``pip install .`` puts in site-packages for this pure-Python package: a copy of gpl."""
    site = tmp_path / "site-packages"
    skip = shutil.ignore_patterns("__pycache__")
    shutil.copytree(REPO / "src" / "gpl", site / "gpl", ignore=skip)
    return site


def test_installed_copy_finds_the_repository_config_from_the_repo_folder(copied_install) -> None:
    result = run_installed(copied_install, cwd=REPO)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_installed_copy_finds_the_config_from_the_app_folder(copied_install) -> None:
    result = run_installed(copied_install, cwd=REPO / "app")
    assert result.returncode == 0, result.stderr


def test_installed_copy_honours_the_environment_variable(copied_install, tmp_path) -> None:
    project = tmp_path / "project"
    shutil.copytree(REPO / "config", project / "config")
    result = run_installed(copied_install, cwd=tmp_path, env_root=project)
    assert result.returncode == 0, result.stderr


def test_installed_copy_outside_the_repository_fails_clearly(copied_install, tmp_path) -> None:
    result = run_installed(copied_install, cwd=tmp_path)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr and ROOT_ENV in result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("setuptools") is None,
    reason="building the wheel offline needs setuptools (installed in CI)",
)
def test_real_pip_install_finds_the_repository_config(tmp_path) -> None:
    site = tmp_path / "site-packages"
    build = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
         "--target", str(site), str(REPO)],
        capture_output=True,
        text=True,
        timeout=300,
    )  # fmt: skip
    assert build.returncode == 0, build.stderr
    assert not (site / "gpl" / "__editable__").exists()
    result = run_installed(site, cwd=REPO)
    assert result.returncode == 0, result.stderr

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SNIPPET = (
    f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
    "from codeflash.main import main; "
    "main()"
)


def run_codeflash(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252:strict"
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, "-c", MAIN_SNIPPET, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="cp1252",
        errors="replace",
        env=env,
    )


def test_optimize_help_does_not_crash_with_cp1252_output() -> None:
    result = run_codeflash(["optimize", "--help"], cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr
    assert "UnicodeEncodeError" not in result.stderr
    assert "--trace-only" in result.stdout


def test_show_config_does_not_crash_with_cp1252_output(tmp_path: Path) -> None:
    project_root = tmp_path / "python_project"
    project_root.mkdir()
    (project_root / "demo_pkg").mkdir()
    (project_root / "tests").mkdir()
    (project_root / "demo_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "tests" / "test_demo.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo-project'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "init"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    result = run_codeflash(["--show-config"], cwd=project_root)

    assert result.returncode == 0, result.stderr
    assert "UnicodeEncodeError" not in result.stderr
    assert "Codeflash Configuration" in result.stdout
    assert "Auto-detected (not saved)" in result.stdout

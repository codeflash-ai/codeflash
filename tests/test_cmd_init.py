import os
import tempfile
from pathlib import Path

import pytest

from codeflash.cli_cmds.cmd_init import (
    CLISetupInfo,
    VsCodeSetupInfo,
    configure_pyproject_toml,
    get_formatter_cmds,
    get_valid_subdirs,
)
from codeflash.cli_cmds.validators import PyprojectTomlValidator


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname).resolve()


def test_is_valid_pyproject_toml_with_empty_config(temp_dir: Path) -> None:
    with (temp_dir / "pyproject.toml").open(mode="w") as f:
        f.write(
            """[tool.codeflash]
"""
        )
        f.flush()
        validator = PyprojectTomlValidator()
        result = validator.validate(str(temp_dir / "pyproject.toml"))
        assert not result.is_valid
        assert "Missing required field: 'module_root'" in result.failure_descriptions[0]


def test_is_valid_pyproject_toml_with_incorrect_module_root(temp_dir: Path) -> None:
    with (temp_dir / "pyproject.toml").open(mode="w") as f:
        f.write(
            """[tool.codeflash]
module-root = "invalid_directory"
"""
        )
        f.flush()
        validator = PyprojectTomlValidator()
        result = validator.validate(str(temp_dir / "pyproject.toml"))
        assert not result.is_valid
        assert "Invalid 'module_root'" in result.failure_descriptions[0]


def test_is_valid_pyproject_toml_with_incorrect_tests_root(temp_dir: Path) -> None:
    with (temp_dir / "pyproject.toml").open(mode="w") as f:
        f.write(
            """[tool.codeflash]
module-root = "."
tests-root = "incorrect_tests_root"
"""
        )
        f.flush()
        validator = PyprojectTomlValidator()
        result = validator.validate(str(temp_dir / "pyproject.toml"))
        assert not result.is_valid
        assert "Invalid 'tests_root'" in result.failure_descriptions[0]


def test_is_valid_pyproject_toml_with_valid_config(temp_dir: Path) -> None:
    with (temp_dir / "pyproject.toml").open(mode="w") as f:
        os.makedirs(temp_dir / "tests")
        f.write(
            """[tool.codeflash]
module-root = "."
tests-root = "tests"
"""
        )
        f.flush()
        validator = PyprojectTomlValidator()
        result = validator.validate(str(temp_dir / "pyproject.toml"))
        assert result.is_valid


def test_get_formatter_cmd(temp_dir: Path) -> None:
    assert get_formatter_cmds("black") == ["black $file"]
    assert get_formatter_cmds("ruff") == ["ruff check --exit-zero --fix $file", "ruff format $file"]
    assert get_formatter_cmds("disabled") == ["disabled"]
    assert get_formatter_cmds("don't use a formatter") == ["disabled"]


def test_configure_pyproject_toml_for_cli(temp_dir: Path) -> None:
    pyproject_path = temp_dir / "pyproject.toml"

    with (pyproject_path).open(mode="w") as f:
        f.write("")
        f.flush()
        os.mkdir(temp_dir / "tests")
        config = CLISetupInfo(
            module_root=".",
            tests_root="tests",
            benchmarks_root=None,
            ignore_paths=[],
            formatter="black",
            git_remote="origin",
            enable_telemetry=False,
        )

        success = configure_pyproject_toml(config, pyproject_path)
        assert success

        config_content = pyproject_path.read_text()
        assert (
            config_content
            == """[tool.codeflash]
# All paths are relative to this pyproject.toml's directory.
module-root = "."
tests-root = "tests"
ignore-paths = []
disable-telemetry = true
formatter-cmds = ["black $file"]
"""
        )
        validator = PyprojectTomlValidator()
        result = validator.validate(str(pyproject_path))
        assert result.is_valid


def test_configure_pyproject_toml_for_vscode_with_empty_config(temp_dir: Path) -> None:
    pyproject_path = temp_dir / "pyproject.toml"

    with (pyproject_path).open(mode="w") as f:
        f.write("")
        f.flush()
        os.mkdir(temp_dir / "tests")
        config = VsCodeSetupInfo(module_root=".", tests_root="tests", formatter="black")

        success = configure_pyproject_toml(config, pyproject_path)
        assert success

        config_content = pyproject_path.read_text()
        assert (
            config_content
            == """[tool.codeflash]
module-root = "."
tests-root = "tests"
formatter-cmds = ["black $file"]
"""
        )
        validator = PyprojectTomlValidator()
        result = validator.validate(str(pyproject_path))
        assert result.is_valid


def test_configure_pyproject_toml_for_vscode_with_existing_config(temp_dir: Path) -> None:
    pyproject_path = temp_dir / "pyproject.toml"

    with (pyproject_path).open(mode="w") as f:
        f.write("""[tool.codeflash]
module-root = "codeflash"
tests-root = "tests"
benchmarks-root = "tests/benchmarks"
formatter-cmds = ["disabled"]
""")
        f.flush()
        os.mkdir(temp_dir / "tests")
        config = VsCodeSetupInfo(module_root=".", tests_root="tests", formatter="disabled")

        success = configure_pyproject_toml(config, pyproject_path)
        assert success

        config_content = pyproject_path.read_text()
        # the benchmarks-root shouldn't get overwritten
        assert (
            config_content
            == """[tool.codeflash]
module-root = "."
tests-root = "tests"
benchmarks-root = "tests/benchmarks"
formatter-cmds = ["disabled"]
"""
        )
        validator = PyprojectTomlValidator()
        result = validator.validate(str(pyproject_path))
        assert result.is_valid


def test_get_valid_subdirs(temp_dir: Path) -> None:
    os.mkdir(temp_dir / "dir1")
    os.mkdir(temp_dir / "dir2")
    os.mkdir(temp_dir / "__pycache__")
    os.mkdir(temp_dir / ".git")
    os.mkdir(temp_dir / "tests")

    dirs = get_valid_subdirs(temp_dir)
    assert "tests" in dirs
    assert "dir1" in dirs
    assert "dir2" in dirs

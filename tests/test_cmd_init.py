import pytest
import tempfile
from pathlib import Path
from codeflash.cli_cmds.cmd_init import (
    is_valid_pyproject_toml,
    configure_pyproject_toml,
    CLISetupInfo,
    get_formatter_cmds,
    VsCodeSetupInfo,
    get_valid_subdirs,
)
import os


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
        valid, _, _message = is_valid_pyproject_toml(temp_dir / "pyproject.toml")
        assert not valid
        assert _message == "Missing required field: 'module_root'"

def test_is_valid_pyproject_toml_with_incorrect_module_root(temp_dir: Path) -> None:
    with (temp_dir / "pyproject.toml").open(mode="w") as f:
        wrong_module_root = temp_dir / "invalid_directory"
        f.write(
            f"""[tool.codeflash]
module-root = "invalid_directory"
"""
        )
        f.flush()
        valid, config, _message = is_valid_pyproject_toml(temp_dir / "pyproject.toml")
        assert not valid
        assert _message == f"Invalid 'module_root': directory does not exist at {wrong_module_root}"


def test_is_valid_pyproject_toml_with_incorrect_tests_root(temp_dir: Path) -> None:
    with (temp_dir / "pyproject.toml").open(mode="w") as f:
        wrong_tests_root = temp_dir / "incorrect_tests_root"
        f.write(
            f"""[tool.codeflash]
module-root = "."
tests-root = "incorrect_tests_root"
"""
        )
        f.flush()
        valid, config, _message = is_valid_pyproject_toml(temp_dir / "pyproject.toml")
        assert not valid
        assert _message == f"Invalid 'tests_root': directory does not exist at {wrong_tests_root}"


def test_is_valid_pyproject_toml_with_valid_config(temp_dir: Path) -> None:
    with (temp_dir / "pyproject.toml").open(mode="w") as f:
        os.makedirs(temp_dir / "tests")
        f.write(
            """[tool.codeflash]
module-root = "."
tests-root = "tests"
test-framework = "pytest"
"""
        )
        f.flush()
        valid, config, _message = is_valid_pyproject_toml(temp_dir / "pyproject.toml")
        assert valid

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
            test_framework="pytest",
            ignore_paths=[],
            formatter="black",
            git_remote="origin",
            enable_telemetry=False,
        )

        success = configure_pyproject_toml(config, pyproject_path)
        assert success

        config_content = pyproject_path.read_text()
        assert """[tool.codeflash]
# All paths are relative to this pyproject.toml's directory.
module-root = "."
tests-root = "tests"
test-framework = "pytest"
ignore-paths = []
disable-telemetry = true
formatter-cmds = ["black $file"]
""" == config_content
        valid, _, _ = is_valid_pyproject_toml(pyproject_path)
        assert valid

def test_configure_pyproject_toml_for_vscode_with_empty_config(temp_dir: Path) -> None:

    pyproject_path = temp_dir / "pyproject.toml"

    with (pyproject_path).open(mode="w") as f:
        f.write("")
        f.flush()
        os.mkdir(temp_dir / "tests")
        config = VsCodeSetupInfo(
            module_root=".",
            tests_root="tests",
            test_framework="pytest",
            formatter="black",
        )

        success = configure_pyproject_toml(config, pyproject_path)
        assert success

        config_content = pyproject_path.read_text()
        assert """[tool.codeflash]
module-root = "."
tests-root = "tests"
test-framework = "pytest"
formatter-cmds = ["black $file"]
""" == config_content
        valid, _, _ = is_valid_pyproject_toml(pyproject_path)
        assert valid

def test_configure_pyproject_toml_for_vscode_with_existing_config(temp_dir: Path) -> None:
    pyproject_path = temp_dir / "pyproject.toml"
    
    with (pyproject_path).open(mode="w") as f:
        f.write("""[tool.codeflash]
module-root = "codeflash"
tests-root = "tests"
benchmarks-root = "tests/benchmarks"
test-framework = "pytest"
formatter-cmds = ["disabled"]
""")
        f.flush()
        os.mkdir(temp_dir / "tests")
        config = VsCodeSetupInfo(
            module_root=".",
            tests_root="tests",
            test_framework="pytest",
            formatter="disabled",
        )

        success = configure_pyproject_toml(config, pyproject_path)
        assert success

        config_content = pyproject_path.read_text()
        # the benchmarks-root shouldn't get overwritten
        assert """[tool.codeflash]
module-root = "."
tests-root = "tests"
benchmarks-root = "tests/benchmarks"
test-framework = "pytest"
formatter-cmds = ["disabled"]
""" == config_content
        valid, _, _ = is_valid_pyproject_toml(pyproject_path)
        assert valid

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

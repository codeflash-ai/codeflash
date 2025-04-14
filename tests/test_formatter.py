import os
import tempfile
from pathlib import Path

import pytest

from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.formatter import format_code, sort_imports


def test_remove_duplicate_imports():
    """Test that duplicate imports are removed when should_sort_imports is True."""
    original_code = "import os\nimport os\n"
    new_code = sort_imports(original_code)
    assert new_code == "import os\n"


def test_remove_multiple_duplicate_imports():
    """Test that multiple duplicate imports are removed when should_sort_imports is True."""
    original_code = "import sys\nimport os\nimport sys\n"

    new_code = sort_imports(original_code)
    assert new_code == "import os\nimport sys\n"


def test_sorting_imports():
    """Test that imports are sorted when should_sort_imports is True."""
    original_code = "import sys\nimport unittest\nimport os\n"

    new_code = sort_imports(original_code)
    assert new_code == "import os\nimport sys\nimport unittest\n"


def test_sort_imports_without_formatting():
    """Test that imports are sorted when formatting is disabled and should_sort_imports is True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.py"
        with tmp_path.open("w") as tmp:
            tmp.write("import sys\nimport unittest\nimport os\n")

        new_code = format_code(formatter_cmds=["disabled"], path=tmp_path)
        assert new_code is not None
        new_code = sort_imports(new_code)
        assert new_code == "import os\nimport sys\nimport unittest\n"


def test_dedup_and_sort_imports_deduplicates():
    original_code = """
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    expected = """
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    actual = sort_imports(original_code)

    assert actual == expected


def test_dedup_and_sort_imports_sorts_and_deduplicates():
    original_code = """
import os
import sys
import json
import os


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    expected = """
import json
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    actual = sort_imports(original_code)

    assert actual == expected


def test_formatter_cmds_non_existent():
    """Test that default formatter-cmds is used when it doesn't exist in the toml."""
    config_data = """
[tool.codeflash]
module-root = "src"
tests-root = "tests"
test-framework = "pytest"
ignore-paths = []
"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "pyproject.toml"
        with tmp_path.open("w") as tmp:
            tmp.write(config_data)

        config, _ = parse_config_file(tmp_path)
        assert config["formatter_cmds"] == ["black $file"]

    try:
        import black
    except ImportError:
        pytest.skip("black is not installed")

    original_code = b"""
import os
import sys
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.py"
        with tmp_path.open("wb") as tmp:
            tmp.write(original_code)
            tmp.flush()

        actual = format_code(formatter_cmds=["black $file"], path=tmp_path.resolve())
        assert actual == expected


def test_formatter_black():
    try:
        import black
    except ImportError:
        pytest.skip("black is not installed")
    original_code = b"""
import os
import sys    
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.py"
        with tmp_path.open("wb") as tmp:
            tmp.write(original_code)
            tmp.flush()

        actual = format_code(formatter_cmds=["black $file"], path=Path(tmp_path))
        assert actual == expected


def test_formatter_ruff():
    try:
        import ruff  # type: ignore
    except ImportError:
        pytest.skip("ruff is not installed")
    original_code = b"""
import os
import sys    
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.py"
        with tmp_path.open("wb") as tmp:
            tmp.write(original_code)
            tmp.flush()

        actual = format_code(
            formatter_cmds=["ruff check --exit-zero --fix $file", "ruff format $file"], path=Path(tmp_path)
        )
        assert actual == expected


def test_formatter_error():
    original_code = """
import os
import sys
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = original_code
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.py"
        with tmp_path.open("w") as tmp:
            tmp.write(original_code)

        with pytest.raises(FileNotFoundError):
            format_code(formatter_cmds=["exit 1"], path=tmp_path.resolve())

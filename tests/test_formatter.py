import argparse
import os
import tempfile
from pathlib import Path

import pytest
import shutil

from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.formatter import format_code, sort_imports

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

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
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"import sys\nimport unittest\nimport os\n")
        tmp.flush()
        tmp_path = Path(tmp.name)

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

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        tmp.write(config_data.encode())
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        config, _ = parse_config_file(tmp_path)
        assert config["formatter_cmds"] == ["black $file"]
    finally:
        os.remove(tmp_path)

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
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

        actual = format_code(formatter_cmds=["black $file"], path=Path(tmp_path))
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
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

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
    with tempfile.NamedTemporaryFile(suffix=".py") as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

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
    with tempfile.NamedTemporaryFile("w") as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name
        with pytest.raises(FileNotFoundError):
            format_code(formatter_cmds=["exit 1"], path=Path(tmp_path))


def _run_formatting_test(source_filename: str, should_content_change: bool):
    try:
        import ruff  # type: ignore
    except ImportError:
        pytest.skip("ruff is not installed")
    with tempfile.TemporaryDirectory() as test_dir_str:
        test_dir = Path(test_dir_str)
        this_file = Path(__file__).resolve()
        repo_root_dir = this_file.parent.parent
        source_file = repo_root_dir / "code_to_optimize" / source_filename

        original = source_file.read_text()
        target_path = test_dir / "target.py"
        
        shutil.copy2(source_file, target_path)

        function_to_optimize = FunctionToOptimize(
            function_name="process_data", 
            parents=[], 
            file_path=target_path
        )

        test_cfg = TestConfig(
            tests_root=test_dir,
            project_root_path=test_dir,
            test_framework="pytest",
            tests_project_rootdir=test_dir,
        )

        args = argparse.Namespace(
            disable_imports_sorting=False,
            formatter_cmds=[
                "ruff check --exit-zero --fix $file",
                "ruff format $file"
            ],
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            args=args,
        )
        
        optimizer.reformat_code_and_helpers(
            helper_functions=[],
            path=target_path,
            original_code=optimizer.function_to_optimize_source_code,
        )
        
        content = target_path.read_text()
        if should_content_change:
            assert content != original, f"Expected content to change for {source_filename}"
        else:
            assert content == original, f"Expected content to remain unchanged for {source_filename}"


def test_formatting_file_with_many_diffs():
    """Test that files with many formatting errors are skipped (content unchanged)."""
    _run_formatting_test("many_formatting_errors.py", should_content_change=False)


def test_formatting_file_with_few_diffs():
    """Test that files with few formatting errors are formatted (content changed)."""
    _run_formatting_test("few_formatting_errors.py", should_content_change=True)

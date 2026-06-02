from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from codeflash.discovery.functions_to_optimize import (
    get_all_files_and_functions,
    get_functions_within_lines,
)


def test_prefilter_skips_test_files(tmp_path: Path) -> None:
    """Files in tests_root should be skipped before read_text() is called."""
    module_root = tmp_path / "src"
    module_root.mkdir()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()

    source_file = module_root / "app.py"
    source_file.write_text("def compute():\n    return 1\n", encoding="utf-8")

    test_file = tests_root / "test_app.py"
    test_file.write_text("def test_compute():\n    return True\n", encoding="utf-8")

    with patch("codeflash.discovery.functions_to_optimize.get_files_for_language") as mock_get_files:
        mock_get_files.return_value = [source_file, test_file]
        result = get_all_files_and_functions(
            module_root, ignore_paths=[], tests_root=tests_root, module_root=module_root
        )

    assert source_file in result
    assert test_file not in result


def test_prefilter_skips_ignored_paths(tmp_path: Path) -> None:
    """Files in ignore_paths should be skipped before read_text() is called."""
    module_root = tmp_path / "src"
    module_root.mkdir()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    ignored_dir = module_root / "vendor"
    ignored_dir.mkdir()

    source_file = module_root / "app.py"
    source_file.write_text("def compute():\n    return 1\n", encoding="utf-8")

    vendor_file = ignored_dir / "lib.py"
    vendor_file.write_text("def helper():\n    return 2\n", encoding="utf-8")

    with patch("codeflash.discovery.functions_to_optimize.get_files_for_language") as mock_get_files:
        mock_get_files.return_value = [source_file, vendor_file]
        result = get_all_files_and_functions(
            module_root, ignore_paths=[ignored_dir], tests_root=tests_root, module_root=module_root
        )

    assert source_file in result
    assert vendor_file not in result


def test_prefilter_skips_files_outside_module_root(tmp_path: Path) -> None:
    """Files outside module_root should be skipped before read_text() is called."""
    module_root = tmp_path / "src"
    module_root.mkdir()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    source_file = module_root / "app.py"
    source_file.write_text("def compute():\n    return 1\n", encoding="utf-8")

    outside_file = other_dir / "stray.py"
    outside_file.write_text("def stray():\n    return 3\n", encoding="utf-8")

    with patch("codeflash.discovery.functions_to_optimize.get_files_for_language") as mock_get_files:
        mock_get_files.return_value = [source_file, outside_file]
        result = get_all_files_and_functions(
            module_root, ignore_paths=[], tests_root=tests_root, module_root=module_root
        )

    assert source_file in result
    assert outside_file not in result


def test_prefilter_disabled_without_params(tmp_path: Path) -> None:
    """Without tests_root/module_root, no prefiltering occurs (backward compat)."""
    module_root = tmp_path / "src"
    module_root.mkdir()

    source_file = module_root / "app.py"
    source_file.write_text("def compute():\n    return 1\n", encoding="utf-8")

    with patch("codeflash.discovery.functions_to_optimize.get_files_for_language") as mock_get_files:
        mock_get_files.return_value = [source_file]
        result = get_all_files_and_functions(module_root, ignore_paths=[])

    assert source_file in result


def test_prefilter_in_get_functions_within_lines(tmp_path: Path) -> None:
    """get_functions_within_lines should skip test files when prefilter params are provided."""
    module_root = tmp_path / "src"
    module_root.mkdir()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()

    source_file = module_root / "app.py"
    source_file.write_text("def compute():\n    return 1\n", encoding="utf-8")

    test_file = tests_root / "test_app.py"
    test_file.write_text("def test_compute():\n    return True\n", encoding="utf-8")

    modified_lines = {
        str(source_file): [1, 2],
        str(test_file): [1, 2],
    }

    result = get_functions_within_lines(
        modified_lines, tests_root=tests_root, ignore_paths=[], module_root=module_root
    )

    assert source_file in result
    assert test_file not in result


def test_prefilter_avoids_reading_skipped_files(tmp_path: Path) -> None:
    """Verify that find_all_functions_in_file is NOT called for prefiltered files (the core perf win)."""
    module_root = tmp_path / "src"
    module_root.mkdir()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()

    source_file = module_root / "app.py"
    source_file.write_text("def compute():\n    return 1\n", encoding="utf-8")

    test_file = tests_root / "test_app.py"
    test_file.write_text("def test_compute():\n    return True\n", encoding="utf-8")

    with (
        patch("codeflash.discovery.functions_to_optimize.get_files_for_language") as mock_get_files,
        patch("codeflash.discovery.functions_to_optimize.find_all_functions_in_file") as mock_find,
    ):
        mock_get_files.return_value = [source_file, test_file]
        mock_find.return_value = {}
        get_all_files_and_functions(
            module_root, ignore_paths=[], tests_root=tests_root, module_root=module_root
        )

    # find_all_functions_in_file (which does read_text) should only be called for source_file
    assert mock_find.call_count == 1
    mock_find.assert_called_once_with(source_file)


def test_prefilter_skips_submodule_paths(tmp_path: Path) -> None:
    """Submodule paths should be skipped by prefilter."""
    module_root = tmp_path / "src"
    module_root.mkdir()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    submodule_dir = module_root / "vendor_submodule"
    submodule_dir.mkdir()

    source_file = module_root / "app.py"
    source_file.write_text("def compute():\n    return 1\n", encoding="utf-8")

    submodule_file = submodule_dir / "lib.py"
    submodule_file.write_text("def helper():\n    return 2\n", encoding="utf-8")

    with (
        patch("codeflash.discovery.functions_to_optimize.get_files_for_language") as mock_get_files,
        patch(
            "codeflash.discovery.functions_to_optimize.ignored_submodule_paths", return_value=[submodule_dir]
        ),
    ):
        mock_get_files.return_value = [source_file, submodule_file]
        result = get_all_files_and_functions(
            module_root, ignore_paths=[], tests_root=tests_root, module_root=module_root
        )

    assert source_file in result
    assert submodule_file not in result

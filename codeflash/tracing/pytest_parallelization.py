from __future__ import annotations

import os
from math import ceil
from pathlib import Path
from random import shuffle


def pytest_split(
    arguments: list[str], num_splits: int | None = None, limit: int | None = None
) -> tuple[list[list[str]] | None, list[str] | None]:
    """Split pytest test files from a directory into N roughly equal groups for parallel execution.

    Args:
        arguments: List of arguments passed to pytest
        num_splits: Number of groups to split tests into. If None, uses CPU count.
        limit: Maximum number of test files to process. If None, processes all files.

    Returns:
        List of lists, where each inner list contains test file paths for one group.
        Returns single list with all tests if number of test files < CPU cores.

    """
    try:
        import pytest

        parser = pytest.Parser()

        pytest_args = parser.parse_known_args(arguments)
        test_paths = getattr(pytest_args, "file_or_dir", None)
        if not test_paths:
            return None, None

    except ImportError:
        return None, None
    test_files_set: set[str] = set()

    # Find all test_*.py files recursively in the directory
    for test_path in test_paths:
        _test_path = Path(test_path)
        if not _test_path.exists():
            return None, None
        if _test_path.is_dir():
            # Find all test files matching the pattern test_*.py
            test_files_set.update(map(str, _test_path.rglob("test_*.py")))
            test_files_set.update(map(str, _test_path.rglob("*_test.py")))
        elif _test_path.is_file():
            test_files_set.add(str(_test_path))

    if not test_files_set:
        return [[]], None

    # Determine number of splits
    if num_splits is None:
        num_splits = os.cpu_count() or 4

    # randomize to increase chances of all splits being balanced
    test_files = list(test_files_set)
    shuffle(test_files)

    # Apply limit if specified
    if limit is not None and limit > 0:
        test_files = test_files[:limit]

    # Ensure each split has at least 4 test files
    # If we have fewer test files than 4 * num_splits, reduce num_splits
    max_possible_splits = len(test_files) // 4
    if max_possible_splits == 0:
        return [test_files], test_paths

    num_splits = min(num_splits, max_possible_splits)

    # Calculate chunk size (round up to ensure all files are included)
    total_files = len(test_files)
    chunk_size = ceil(total_files / num_splits)

    # Initialize result groups
    result_groups: list[list[str]] = [[] for _ in range(num_splits)]

    # Distribute files across groups
    for i, test_file in enumerate(test_files):
        group_index = i // chunk_size
        # Ensure we don't exceed the number of groups (edge case handling)
        if group_index >= num_splits:
            group_index = num_splits - 1
        result_groups[group_index].append(test_file)

    return result_groups, test_paths

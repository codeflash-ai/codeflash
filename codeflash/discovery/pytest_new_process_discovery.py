from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)
# This script should not have any relation to the codeflash package, be careful with imports
cwd = sys.argv[1]
tests_root = sys.argv[2]
pickle_path = sys.argv[3]
collected_tests = []
pytest_rootdir = None
sys.path.insert(1, cwd)


class PytestCollectionPlugin:
    def __init__(self) -> None:
        self.pytest_rootdir: Path | None = None

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        collected_tests.extend(session.items)
        self.pytest_rootdir = session.config.rootdir  # type: ignore

    def pytest_load_initial_conftests(self, early_config: pytest.Config) -> None:
        """Disable any plugins that may be present during test collection."""
        plugin_names = {dist.project_name for _, dist in early_config.pluginmanager.list_plugin_distinfo()}
        for plugin_name in plugin_names:
            early_config.pluginmanager.set_blocked(plugin_name)


def parse_pytest_collection_results(pytest_tests: list[Any]) -> list[dict[str, str]]:
    test_results = []
    for test in pytest_tests:
        test_class = None
        if test.cls:
            test_class = test.parent.name
        test_results.append({"test_file": str(test.path), "test_class": test_class, "test_function": test.name})
    return test_results


if __name__ == "__main__":
    plugin = PytestCollectionPlugin()
    try:
        exitcode = pytest.main([tests_root, "-pno:logging", "--collect-only", "-m", "not skip"], plugins=[plugin])
    except Exception as e:
        logger.exception("Failed to collect tests", exc_info=e)
        exitcode = -1
    tests = parse_pytest_collection_results(collected_tests)

    with Path(pickle_path).open("wb") as f:
        pickle.dump((exitcode, tests, plugin.pytest_rootdir), f, protocol=pickle.HIGHEST_PROTOCOL)

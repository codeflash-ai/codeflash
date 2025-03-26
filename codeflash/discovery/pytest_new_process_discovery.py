import sys
from typing import Any

# This script should not have any relation to the codeflash package, be careful with imports
cwd = sys.argv[1]
tests_root = sys.argv[2]
pickle_path = sys.argv[3]
collected_tests = []
pytest_rootdir = None
sys.path.insert(1, str(cwd))


class PytestCollectionPlugin:
    def pytest_collection_finish(self, session) -> None:
        global pytest_rootdir
        collected_tests.extend(session.items)
        pytest_rootdir = session.config.rootdir

    def pytest_collection_modifyitems(config, items):
        skip_benchmark = pytest.mark.skip(reason="Skipping benchmark tests")
        for item in items:
            if "benchmark" in item.fixturenames:
                item.add_marker(skip_benchmark)


def parse_pytest_collection_results(pytest_tests: list[Any]) -> list[dict[str, str]]:
    test_results = []
    for test in pytest_tests:
        test_class = None
        if test.cls:
            test_class = test.parent.name
        test_results.append({"test_file": str(test.path), "test_class": test_class, "test_function": test.name})
    return test_results


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    try:
        exitcode = pytest.main(
            [tests_root, "-p no:logging", "--collect-only", "-m", "not skip"], plugins=[PytestCollectionPlugin()]
        )
    except Exception as e:  # noqa: BLE001
        print(f"Failed to collect tests: {e!s}")  # noqa: T201
        exitcode = -1
    tests = parse_pytest_collection_results(collected_tests)
    import pickle

    with Path(pickle_path).open("wb") as f:
        pickle.dump((exitcode, tests, pytest_rootdir), f, protocol=pickle.HIGHEST_PROTOCOL)

# ruff: noqa
import sys
from pathlib import Path
from typing import Any
import pickle


# This script should not have any relation to the codeflash package, be careful with imports
cwd = sys.argv[1]
tests_root = sys.argv[2]
pickle_path = sys.argv[3]
collected_tests = []
pytest_rootdir = None
sys.path.insert(1, str(cwd))


def parse_pytest_collection_results(pytest_tests: list[Any]) -> list[dict[str, str]]:
    test_results = []
    for test in pytest_tests:
        test_class = None
        if test.cls:
            test_class = test.parent.name
        test_results.append({"test_file": str(test.path), "test_class": test_class, "test_function": test.name})
    return test_results


class PytestCollectionPlugin:
    def pytest_collection_finish(self, session) -> None:
        global pytest_rootdir, collected_tests

        collected_tests.extend(session.items)
        pytest_rootdir = session.config.rootdir

        # Write results immediately since pytest.main() will exit after this callback, not always with a success code
        tests = parse_pytest_collection_results(collected_tests)
        exit_code = getattr(session.config, "exitstatus", 0)
        with Path(pickle_path).open("wb") as f:
            pickle.dump((exit_code, tests, pytest_rootdir), f, protocol=pickle.HIGHEST_PROTOCOL)

    def pytest_collection_modifyitems(self, items) -> None:
        skip_benchmark = pytest.mark.skip(reason="Skipping benchmark tests")
        for item in items:
            if "benchmark" in item.fixturenames:
                item.add_marker(skip_benchmark)


if __name__ == "__main__":
    import pytest

    try:
        pytest.main(
            [tests_root, "-p", "no:logging", "--collect-only", "-m", "not skip", "-p", "no:codeflash-benchmark"],
            plugins=[PytestCollectionPlugin()],
        )
    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        try:
            with Path(pickle_path).open("wb") as f:
                pickle.dump((-1, [], None), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as pickle_error:
            print(f"Failed to write failure pickle: {pickle_error!s}", file=sys.stderr)

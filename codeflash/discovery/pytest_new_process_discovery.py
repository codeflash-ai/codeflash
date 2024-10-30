import os
import pickle
import sys

# This script should not have any relation to the codeflash package, be careful with imports
cwd = sys.argv[1]
tests_root = sys.argv[2]
pickle_path = sys.argv[3]
os.chdir(cwd)
collected_tests = []
pytest_rootdir = None
tests = []
sys.path.insert(1, str(cwd))


class PytestCollectionPlugin:
    def pytest_collection_finish(self, session) -> None:
        global pytest_rootdir
        collected_tests.extend(session.items)
        pytest_rootdir = session.config.rootdir


def parse_pytest_collection_results(pytest_tests: list[any]) -> list[dict[str, str]]:
    test_results: list[list[str]] = []
    for test in pytest_tests:
        test_class = None
        if test.cls:
            test_class = test.parent.name
        test_results.append({"test_file": str(test.path), "test_class": test_class, "test_function": test.name})
    return test_results


if __name__ == "__main__":
    import pytest

    try:
        exitcode = pytest.main(
            [tests_root, "-pno:logging", "--collect-only", "-m", "not skip"], plugins=[PytestCollectionPlugin()]
        )
    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        exitcode = -1
    tests = parse_pytest_collection_results(collected_tests)
    with open(pickle_path, "wb") as f:
        pickle.dump((exitcode, tests, pytest_rootdir), f, protocol=pickle.HIGHEST_PROTOCOL)

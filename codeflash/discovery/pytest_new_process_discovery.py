# ruff: noqa
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
        
        # Write the pickle file here because pytest.main() will call sys.exit() after this callback
        try:
            from pathlib import Path
            import pickle
            
            tests = parse_pytest_collection_results(collected_tests)
            exitcode = 0  # Collection was successful if we reach this point
            
            with Path(pickle_path).open("wb") as f:
                pickle.dump((exitcode, tests, pytest_rootdir), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Failed to write pickle file: {e!s}")

    def pytest_collection_modifyitems(self, items) -> None:
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
    import pytest

    try:
        # Note: pytest.main() will call sys.exit() after collection finishes,
        # so the pickle file is written in the pytest_collection_finish callback above
        pytest.main(
            [tests_root, "-p", "no:logging", "--collect-only", "-m", "not skip", "-p", "no:codeflash-benchmark"],
            plugins=[PytestCollectionPlugin()],
        )
    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        # If there's an exception, write an error pickle file
        try:
            from pathlib import Path
            import pickle
            with Path(pickle_path).open("wb") as f:
                pickle.dump((-1, [], None), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass  # If we can't write the error file, there's not much we can do

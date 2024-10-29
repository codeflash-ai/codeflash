import os
import sys
from multiprocessing import Queue

#from codeflash.models.models import TestsInFile
#from codeflash.verification.test_results import TestType


def run_pytest_discovery_new_process(queue: Queue, cwd: str, tests_root: str) -> tuple[int, list] | None:
    sys.modules.pop("returns")
    import pytest

    os.chdir(cwd)
    collected_tests = []
    pytest_rootdir = None
    tests = []
    sys.path.insert(1, str(cwd))

    class PytestCollectionPlugin:
        def pytest_collection_finish(self, session) -> None:
            nonlocal pytest_rootdir
            collected_tests.extend(session.items)
            pytest_rootdir = session.config.rootdir

    try:
        exitcode = pytest.main(
            [tests_root, "--collect-only", "-pno:terminal", "-m", "not skip"], plugins=[PytestCollectionPlugin()]
        )
    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        exitcode = -1
        queue.put((exitcode, tests, pytest_rootdir))
    #tests = parse_pytest_collection_results(collected_tests)
    queue.put((exitcode, collected_tests, pytest_rootdir))


# def parse_pytest_collection_results(pytest_tests: str) -> list[TestsInFile]:
#     test_results: list[TestsInFile] = []
#     for test in pytest_tests:
#         test_class = None
#         test_file_path = str(test.path)
#         if test.cls:
#             test_class = test.parent.name
#         test_type = TestType.REPLAY_TEST if "__replay_test" in test_file_path else TestType.EXISTING_UNIT_TEST
#         test_results.append(
#             TestsInFile(
#                 test_file=str(test.path),
#                 test_class=test_class,
#                 test_function=test.name,
#                 test_suite=None,  # not used in pytest until now
#                 test_type=test_type,
#             )
#         )
#     return test_results
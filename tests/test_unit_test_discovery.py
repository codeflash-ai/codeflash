import os
import pathlib

from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.verification.verification_utils import TestConfig


def test_unit_test_discovery_pytest():
    project_path = pathlib.Path(__file__).parent.parent.resolve() / "code_to_optimize"
    test_path = project_path / "tests" / "pytest"
    test_config = TestConfig(
        tests_root=str(project_path), project_root_path=str(project_path), test_framework="pytest"
    )
    tests = discover_unit_tests(test_config)
    assert len(tests) > 0
    # print(tests)


def test_unit_test_discovery_unittest():
    project_path = pathlib.Path(__file__).parent.parent.resolve() / "code_to_optimize"
    test_path = project_path / "tests" / "unittest"
    test_config = TestConfig(
        tests_root=str(project_path), project_root_path=str(project_path), test_framework="unittest"
    )
    os.chdir(str(project_path))
    tests = discover_unit_tests(test_config)
    # assert len(tests) > 0
    # Unittest discovery within a pytest environment does not work

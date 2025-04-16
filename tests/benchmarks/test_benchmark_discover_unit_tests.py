from pathlib import Path

from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.verification.verification_utils import TestConfig


def test_benchmark_code_to_optimize_test_discovery(benchmark) -> None:
    project_path = Path(__file__).parent.parent.parent.resolve() / "code_to_optimize"
    tests_path = project_path / "tests" / "pytest"
    test_config = TestConfig(
        tests_root=tests_path,
        project_root_path=project_path,
        test_framework="pytest",
        tests_project_rootdir=tests_path.parent,
    )
    benchmark(discover_unit_tests, test_config)
def test_benchmark_codeflash_test_discovery(benchmark) -> None:
    project_path = Path(__file__).parent.parent.parent.resolve() / "codeflash"
    tests_path = project_path / "tests"
    test_config = TestConfig(
        tests_root=tests_path,
        project_root_path=project_path,
        test_framework="pytest",
        tests_project_rootdir=tests_path.parent,
    )
    benchmark(discover_unit_tests, test_config)

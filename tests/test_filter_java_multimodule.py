"""Tests for Java multi-module project test detection.

This test suite specifically addresses the bug where production code in Java
multi-module projects was incorrectly filtered as test code.

We use Python files for testing since the path-based logic is language-agnostic.
"""

import tempfile
import unittest.mock
from pathlib import Path

from codeflash.discovery.functions_to_optimize import filter_functions, find_all_functions_in_file


def test_filter_java_production_code_in_src_main_java():
    """Test that src/main/java files are NEVER filtered as tests."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create production code in src/main/java (using .py for testing)
        src_main_java = temp_dir / "src" / "main" / "java" / "com" / "example"
        src_main_java.mkdir(parents=True)

        production_file = src_main_java / "service.py"
        with production_file.open("w") as f:
            f.write("""
def calculate():
    return 42
""")

        # Discover functions
        discovered = find_all_functions_in_file(production_file)

        # Tests root is in a separate directory (typical Gradle structure)
        tests_root = temp_dir / "src" / "test" / "java"
        tests_root.mkdir(parents=True)

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                discovered,
                tests_root=tests_root,
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # CRITICAL: Production code in src/main/java should NOT be filtered
        assert production_file in filtered, (
            f"Production code in src/main/java was incorrectly filtered! "
            f"Expected {production_file} in filtered results."
        )
        assert count == 1, f"Expected 1 function, got {count}"
        assert filtered[production_file][0].function_name == "calculate"


def test_filter_java_test_code_in_src_test_java():
    """Test that src/test/java files ARE filtered as tests."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create test code in src/test/java
        src_test_java = temp_dir / "src" / "test" / "java" / "com" / "example"
        src_test_java.mkdir(parents=True)

        test_file = src_test_java / "test_service.py"
        with test_file.open("w") as f:
            f.write("""
def test_calculate():
    return True
""")

        discovered = find_all_functions_in_file(test_file)

        tests_root = temp_dir / "src" / "test" / "java"

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                discovered,
                tests_root=tests_root,
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # Test code in src/test/java SHOULD be filtered
        assert test_file not in filtered, (
            f"Test code in src/test/java was NOT filtered! "
            f"Should have been removed but found in: {filtered.keys()}"
        )
        assert count == 0, f"Expected 0 functions (all filtered), got {count}"


def test_filter_java_test_in_src_main_test():
    """Test that src/main/test files ARE filtered as tests (edge case)."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Edge case: test directory under src/main/test
        src_main_test = temp_dir / "src" / "main" / "test" / "com" / "example"
        src_main_test.mkdir(parents=True)

        test_file = src_main_test / "edge_test.py"
        with test_file.open("w") as f:
            f.write("""
def test_something():
    return True
""")

        discovered = find_all_functions_in_file(test_file)

        tests_root = temp_dir / "src" / "test" / "java"
        tests_root.mkdir(parents=True)

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                discovered,
                tests_root=tests_root,
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # Files in src/main/test should be filtered (they have "test" in path)
        assert test_file not in filtered, (
            f"Test code in src/main/test was NOT filtered! "
            f"Should have been removed but found in: {filtered.keys()}"
        )


def test_filter_java_kotlin_scala_production_code():
    """Test that src/main/{kotlin,scala,resources} are NOT filtered."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create Kotlin production code
        src_main_kotlin = temp_dir / "src" / "main" / "kotlin" / "com" / "example"
        src_main_kotlin.mkdir(parents=True)
        kotlin_file = src_main_kotlin / "service.py"
        with kotlin_file.open("w") as f:
            f.write("def calculate(): return 42")

        # Create Scala production code
        src_main_scala = temp_dir / "src" / "main" / "scala" / "com" / "example"
        src_main_scala.mkdir(parents=True)
        scala_file = src_main_scala / "service.py"
        with scala_file.open("w") as f:
            f.write("def calculate(): return 42")

        # Discover functions
        discovered_kotlin = find_all_functions_in_file(kotlin_file)
        discovered_scala = find_all_functions_in_file(scala_file)
        all_discovered = {**discovered_kotlin, **discovered_scala}

        tests_root = temp_dir / "src" / "test" / "java"
        tests_root.mkdir(parents=True)

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                all_discovered,
                tests_root=tests_root,
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # All production code should remain
        assert kotlin_file in filtered, "Kotlin production code was filtered!"
        assert scala_file in filtered, "Scala production code was filtered!"
        assert count == 2


def test_filter_java_multimodule_elasticsearch_scenario():
    """Test the exact Elasticsearch multi-module scenario that was failing.

    This reproduces the bug where 6,708 production functions were incorrectly
    filtered as "test functions" when running from a submodule.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Simulate Elasticsearch structure:
        # elasticsearch/
        #   server/
        #     src/main/java/org/elasticsearch/search/MultiValueMode.py
        #     src/test/java/org/elasticsearch/search/MultiValueModeTests.py

        server_module = temp_dir / "server"
        server_module.mkdir()

        # Production code
        src_main = server_module / "src" / "main" / "java" / "org" / "elasticsearch" / "search"
        src_main.mkdir(parents=True)
        production_file = src_main / "MultiValueMode.py"
        with production_file.open("w") as f:
            f.write("""
def pick(values):
    return 42
""")

        # Test code
        src_test = server_module / "src" / "test" / "java" / "org" / "elasticsearch" / "search"
        src_test.mkdir(parents=True)
        test_file = src_test / "MultiValueModeTests.py"
        with test_file.open("w") as f:
            f.write("""
def test_pick():
    return True
""")

        # Discover functions
        discovered_prod = find_all_functions_in_file(production_file)
        discovered_test = find_all_functions_in_file(test_file)
        all_discovered = {**discovered_prod, **discovered_test}

        # When running from server/ submodule
        tests_root = server_module / "src" / "test" / "java"

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                all_discovered,
                tests_root=tests_root,
                ignore_paths=[],
                project_root=server_module,
                module_root=server_module,
            )

        # CRITICAL: Production code should NOT be filtered
        # Test code SHOULD be filtered
        assert production_file in filtered, (
            f"Production code in src/main/java was incorrectly filtered! "
            f"This is the Elasticsearch bug. Expected {production_file} in results."
        )
        assert test_file not in filtered, (
            f"Test code in src/test/java was NOT filtered! "
            f"Expected {test_file} to be removed."
        )
        assert count == 1, f"Expected 1 production function, got {count}"


def test_filter_java_testfixtures_directory():
    """Test that src/testFixtures is correctly identified as test code."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Gradle's testFixtures directory
        test_fixtures = temp_dir / "src" / "testFixtures" / "java" / "com" / "example"
        test_fixtures.mkdir(parents=True)

        fixture_file = test_fixtures / "fixture.py"
        with fixture_file.open("w") as f:
            f.write("""
def create_mock():
    return None
""")

        discovered = find_all_functions_in_file(fixture_file)

        tests_root = temp_dir / "src" / "test" / "java"
        tests_root.mkdir(parents=True)

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                discovered,
                tests_root=tests_root,
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # testFixtures should be filtered (it's test-related code)
        assert fixture_file not in filtered, (
            f"TestFixtures code was NOT filtered! "
            f"Should have been removed but found in: {filtered.keys()}"
        )
        assert count == 0

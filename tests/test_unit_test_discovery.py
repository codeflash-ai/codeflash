import os
import tempfile
from pathlib import Path

from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.verification.verification_utils import TestConfig


def test_unit_test_discovery_pytest():
    project_path = Path(__file__).parent.parent.resolve() / "code_to_optimize"
    tests_path = project_path / "tests" / "pytest"
    test_config = TestConfig(
        tests_root=tests_path,
        project_root_path=project_path,
        test_framework="pytest",
    )
    tests = discover_unit_tests(test_config)
    assert len(tests) > 0
    # print(tests)


def test_unit_test_discovery_unittest():
    project_path = Path(__file__).parent.parent.resolve() / "code_to_optimize"
    test_path = project_path / "tests" / "unittest"
    test_config = TestConfig(
        tests_root=project_path,
        project_root_path=project_path,
        test_framework="unittest",
    )
    os.chdir(project_path)
    tests = discover_unit_tests(test_config)
    # assert len(tests) > 0
    # Unittest discovery within a pytest environment does not work


def test_discover_tests_pytest_with_temp_dir_root():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a dummy test file
        test_file_path = Path(tmpdirname) / "test_dummy.py"
        test_file_content = (
            "import pytest\n"
            "from dummy_code import dummy_function\n\n"
            "def test_dummy_function():\n"
            "    assert dummy_function() is True\n"
            "@pytest.mark.parametrize('param', [True])\n"
            "def test_dummy_parametrized_function(param):\n"
            "    assert dummy_function() is True\n"
        )
        test_file_path.write_text(test_file_content)
        path_obj_tempdirname = Path(tmpdirname)

        # Create a file that the test file is testing
        code_file_path = path_obj_tempdirname / "dummy_code.py"
        code_file_content = "def dummy_function():\n    return True\n"
        code_file_path.write_text(code_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tempdirname,
            project_root_path=path_obj_tempdirname,
            test_framework="pytest",
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the dummy test file is discovered
        assert len(discovered_tests) == 1
        assert len(discovered_tests["dummy_code.dummy_function"]) == 2
        assert discovered_tests["dummy_code.dummy_function"][0].test_file == test_file_path
        assert discovered_tests["dummy_code.dummy_function"][1].test_file == test_file_path
        assert {
            discovered_tests["dummy_code.dummy_function"][0].test_function,
            discovered_tests["dummy_code.dummy_function"][1].test_function,
        } == {
            "test_dummy_parametrized_function[True]",
            "test_dummy_function",
        }


def test_discover_tests_pytest_with_multi_level_dirs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create multi-level directories
        level1_dir = path_obj_tmpdirname / "level1"
        level2_dir = level1_dir / "level2"
        level2_dir.mkdir(parents=True)

        # Create code files at each level
        root_code_file_path = path_obj_tmpdirname / "root_code.py"
        root_code_file_content = "def root_function():\n    return True\n"
        root_code_file_path.write_text(root_code_file_content)

        level1_code_file_path = level1_dir / "level1_code.py"
        level1_code_file_content = "def level1_function():\n    return True\n"
        level1_code_file_path.write_text(level1_code_file_content)

        level2_code_file_path = level2_dir / "level2_code.py"
        level2_code_file_content = "def level2_function():\n    return True\n"
        level2_code_file_path.write_text(level2_code_file_content)

        # Create a test file at the root level
        root_test_file_path = path_obj_tmpdirname / "test_root.py"
        root_test_file_content = (
            "from root_code import root_function\n\n"
            "def test_root_function():\n"
            "    assert True\n"
            "    assert root_function() is True\n"
        )
        root_test_file_path.write_text(root_test_file_content)

        # Create a test file at level 1
        level1_test_file_path = level1_dir / "test_level1.py"
        level1_test_file_content = (
            "from level1_code import level1_function\n\n"
            "def test_level1_function():\n"
            "    assert True\n"
            "    assert level1_function() is True\n"
        )
        level1_test_file_path.write_text(level1_test_file_content)

        # Create a test file at level 2
        level2_test_file_path = level2_dir / "test_level2.py"
        level2_test_file_content = (
            "from level2_code import level2_function\n\n"
            "def test_level2_function():\n"
            "    assert True\n"
            "    assert level2_function() is True\n"
        )
        level2_test_file_path.write_text(level2_test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test files at all levels are discovered
        assert len(discovered_tests) == 3
        assert discovered_tests["root_code.root_function"][0].test_file == root_test_file_path
        assert discovered_tests["level1.level1_code.level1_function"][0].test_file == level1_test_file_path

        assert (
            discovered_tests["level1.level2.level2_code.level2_function"][0].test_file
            == level2_test_file_path
        )


def test_discover_tests_pytest_dirs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create multi-level directories
        level1_dir = Path(tmpdirname) / "level1"
        level2_dir = level1_dir / "level2"
        level2_dir.mkdir(parents=True)
        level3_dir = level1_dir / "level3"
        level3_dir.mkdir(parents=True)

        # Create code files at each level
        root_code_file_path = path_obj_tmpdirname / "root_code.py"
        root_code_file_content = "def root_function():\n    return True\n"
        root_code_file_path.write_text(root_code_file_content)

        level1_code_file_path = level1_dir / "level1_code.py"
        level1_code_file_content = "def level1_function():\n    return True\n"
        level1_code_file_path.write_text(level1_code_file_content)

        level2_code_file_path = level2_dir / "level2_code.py"
        level2_code_file_content = "def level2_function():\n    return True\n"
        level2_code_file_path.write_text(level2_code_file_content)

        level3_code_file_path = level3_dir / "level3_code.py"
        level3_code_file_content = "def level3_function():\n    return True\n"
        level3_code_file_path.write_text(level3_code_file_content)

        # Create a test file at the root level
        root_test_file_path = path_obj_tmpdirname / "test_root.py"
        root_test_file_content = (
            "from root_code import root_function\n\n"
            "def test_root_function():\n"
            "    assert True\n"
            "    assert root_function() is True\n"
        )
        root_test_file_path.write_text(root_test_file_content)

        # Create a test file at level 1
        level1_test_file_path = level1_dir / "test_level1.py"
        level1_test_file_content = (
            "from level1_code import level1_function\n\n"
            "def test_level1_function():\n"
            "    assert True\n"
            "    assert level1_function() is True\n"
        )
        level1_test_file_path.write_text(level1_test_file_content)

        # Create a test file at level 2
        level2_test_file_path = level2_dir / "test_level2.py"
        level2_test_file_content = (
            "from level2_code import level2_function\n\n"
            "def test_level2_function():\n"
            "    assert True\n"
            "    assert level2_function() is True\n"
        )
        level2_test_file_path.write_text(level2_test_file_content)

        level3_test_file_path = level3_dir / "test_level3.py"
        level3_test_file_content = (
            "from level3_code import level3_function\n\n"
            "def test_level3_function():\n"
            "    assert True\n"
            "    assert level3_function() is True\n"
        )
        level3_test_file_path.write_text(level3_test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test files at all levels are discovered
        assert len(discovered_tests) == 4
        assert discovered_tests["root_code.root_function"][0].test_file == root_test_file_path
        assert discovered_tests["level1.level1_code.level1_function"][0].test_file == level1_test_file_path
        assert (
            discovered_tests["level1.level2.level2_code.level2_function"][0].test_file
            == level2_test_file_path
        )

        assert (
            discovered_tests["level1.level3.level3_code.level3_function"][0].test_file
            == level3_test_file_path
        )


def test_discover_tests_pytest_with_class():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create a code file with a class
        code_file_path = path_obj_tmpdirname / "some_class_code.py"
        code_file_content = "class SomeClass:\n    def some_method(self):\n        return True\n"
        code_file_path.write_text(code_file_content)

        # Create a test file with a test class and a test method
        test_file_path = path_obj_tmpdirname / "test_some_class.py"
        test_file_content = (
            "from some_class_code import SomeClass\n\n"
            "def test_some_method():\n"
            "    instance = SomeClass()\n"
            "    assert instance.some_method() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test class and method are discovered
        assert len(discovered_tests) == 1
        assert discovered_tests["some_class_code.SomeClass.some_method"][0].test_file == test_file_path


def test_discover_tests_pytest_with_double_nested_directories():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create nested directories
        nested_dir = path_obj_tmpdirname / "nested" / "more_nested"
        nested_dir.mkdir(parents=True)

        # Create a code file with a class in the nested directory
        code_file_path = nested_dir / "nested_class_code.py"
        code_file_content = "class NestedClass:\n    def nested_method(self):\n        return True\n"
        code_file_path.write_text(code_file_content)

        # Create a test file with a test class and a test method in the nested directory
        test_file_path = nested_dir / "test_nested_class.py"
        test_file_content = (
            "from nested_class_code import NestedClass\n\n"
            "def test_nested_method():\n"
            "    instance = NestedClass()\n"
            "    assert instance.nested_method() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test class and method are discovered
        assert len(discovered_tests) == 1
        assert (
            discovered_tests["nested.more_nested.nested_class_code.NestedClass.nested_method"][0].test_file
            == test_file_path
        )


def test_discover_tests_with_code_in_dir_and_test_in_subdir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create a directory for the code file
        code_dir = path_obj_tmpdirname / "code"
        code_dir.mkdir()

        # Create a code file in the code directory
        code_file_path = code_dir / "some_code.py"
        code_file_content = "def some_function():\n    return True\n"
        code_file_path.write_text(code_file_content)

        # Create a subdirectory for the test file within the code directory
        test_subdir = code_dir / "tests"
        test_subdir.mkdir()

        # Create a test file in the test subdirectory
        test_file_path = test_subdir / "test_some_code.py"
        test_file_content = (
            "import sys\n"
            "import os\n"
            "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n"
            "from some_code import some_function\n\n"
            "def test_some_function():\n"
            "    assert some_function() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the code directory as the root
        test_config = TestConfig(
            tests_root=test_subdir,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test file is discovered and associated with the code file
        assert len(discovered_tests) == 1
        assert discovered_tests["code.some_code.some_function"][0].test_file == test_file_path


def test_discover_tests_pytest_with_nested_class():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path_obj_tmpdirname = Path(tmpdirname)
        # Create a code file with a nested class
        code_file_path = path_obj_tmpdirname / "nested_class_code.py"
        code_file_content = (
            "class OuterClass:\n"
            "    class InnerClass:\n"
            "        def inner_method(self):\n"
            "            return True\n"
        )
        code_file_path.write_text(code_file_content)

        # Create a test file with a test for the nested class method
        test_file_path = path_obj_tmpdirname / "test_nested_class.py"
        test_file_content = (
            "from nested_class_code import OuterClass\n\n"
            "def test_inner_method():\n"
            "    instance = OuterClass.InnerClass()\n"
            "    assert instance.inner_method() is True\n"
        )
        test_file_path.write_text(test_file_content)

        # Create a TestConfig with the temporary directory as the root
        test_config = TestConfig(
            tests_root=path_obj_tmpdirname,
            project_root_path=path_obj_tmpdirname,
            test_framework="pytest",
        )

        # Discover tests
        discovered_tests = discover_unit_tests(test_config)

        # Check if the test for the nested class method is discovered
        assert len(discovered_tests) == 1
        assert (
            discovered_tests["nested_class_code.OuterClass.InnerClass.inner_method"][0].test_file
            == test_file_path
        )

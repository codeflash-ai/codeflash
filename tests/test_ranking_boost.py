from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.discovery.discover_unit_tests import existing_unit_test_count
from codeflash.models.function_types import FunctionToOptimize
from codeflash.models.models import CodePosition, FunctionCalledInTest, TestsInFile
from codeflash.models.test_type import TestType


def make_func(name: str, project_root: Path) -> FunctionToOptimize:
    return FunctionToOptimize(function_name=name, file_path=project_root / "mod.py")


def make_test(test_type: TestType, test_name: str = "test_something") -> FunctionCalledInTest:
    return FunctionCalledInTest(
        tests_in_file=TestsInFile(
            test_file=Path("/tests/test_mod.py"), test_class=None, test_function=test_name, test_type=test_type
        ),
        position=CodePosition(line_no=1, col_no=0),
    )


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "mod.py").write_text("def foo(): pass\ndef bar(): pass\ndef baz(): pass\n")
    return root


def test_no_tests(project_root: Path) -> None:
    func = make_func("foo", project_root)
    assert existing_unit_test_count(func, project_root, {}) == 0


def test_no_matching_key(project_root: Path) -> None:
    func = make_func("foo", project_root)
    tests = {"other_module.bar": {make_test(TestType.EXISTING_UNIT_TEST)}}
    assert existing_unit_test_count(func, project_root, tests) == 0


def test_only_replay_tests(project_root: Path) -> None:
    func = make_func("foo", project_root)
    key = func.qualified_name_with_modules_from_root(project_root)
    tests = {key: {make_test(TestType.REPLAY_TEST)}}
    assert existing_unit_test_count(func, project_root, tests) == 0


def test_single_existing_test(project_root: Path) -> None:
    func = make_func("foo", project_root)
    key = func.qualified_name_with_modules_from_root(project_root)
    tests = {key: {make_test(TestType.EXISTING_UNIT_TEST)}}
    assert existing_unit_test_count(func, project_root, tests) == 1


def test_multiple_existing_tests(project_root: Path) -> None:
    func = make_func("foo", project_root)
    key = func.qualified_name_with_modules_from_root(project_root)
    tests = {
        key: {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_three"),
        }
    }
    assert existing_unit_test_count(func, project_root, tests) == 3


def test_mixed_test_types(project_root: Path) -> None:
    func = make_func("foo", project_root)
    key = func.qualified_name_with_modules_from_root(project_root)
    tests = {
        key: {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.REPLAY_TEST, "test_replay"),
            make_test(TestType.GENERATED_REGRESSION, "test_gen"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
        }
    }
    assert existing_unit_test_count(func, project_root, tests) == 2


def test_truthiness_for_boolean_usage(project_root: Path) -> None:
    func = make_func("foo", project_root)
    key = func.qualified_name_with_modules_from_root(project_root)
    assert not existing_unit_test_count(func, project_root, {})
    assert existing_unit_test_count(func, project_root, {key: {make_test(TestType.EXISTING_UNIT_TEST)}})


def test_functions_with_more_tests_rank_higher(project_root: Path) -> None:
    funcs = [make_func(name, project_root) for name in ("foo", "bar", "baz")]
    function_to_tests: dict[str, set[FunctionCalledInTest]] = {
        funcs[0].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
        },
        funcs[1].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_three"),
        },
        # baz has no tests
    }

    ranked = sorted(
        funcs,
        key=lambda f: -existing_unit_test_count(f, project_root, function_to_tests),
    )

    assert ranked[0].function_name == "bar"  # 3 tests
    assert ranked[1].function_name == "foo"  # 1 test
    assert ranked[2].function_name == "baz"  # 0 tests


def test_stable_sort_preserves_order_for_equal_counts(project_root: Path) -> None:
    funcs = [make_func(name, project_root) for name in ("foo", "bar", "baz")]
    function_to_tests: dict[str, set[FunctionCalledInTest]] = {
        f.qualified_name_with_modules_from_root(project_root): {make_test(TestType.EXISTING_UNIT_TEST)} for f in funcs
    }

    ranked = sorted(
        funcs,
        key=lambda f: -existing_unit_test_count(f, project_root, function_to_tests),
    )

    assert [f.function_name for f in ranked] == ["foo", "bar", "baz"]

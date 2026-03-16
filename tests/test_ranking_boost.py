from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.discovery.discover_unit_tests import existing_unit_test_count
from codeflash.models.function_types import FunctionToOptimize
from codeflash.models.models import CodePosition, FunctionCalledInTest, TestsInFile
from codeflash.models.test_type import TestType
from codeflash.optimization.optimizer import Optimizer


def make_func(name: str, project_root: Path) -> FunctionToOptimize:
    return FunctionToOptimize(function_name=name, file_path=project_root / "mod.py")


def make_test(test_type: TestType, test_name: str = "test_something") -> FunctionCalledInTest:
    return FunctionCalledInTest(
        tests_in_file=TestsInFile(
            test_file=Path("/tests/test_mod.py"), test_class=None, test_function=test_name, test_type=test_type
        ),
        position=CodePosition(line_no=1, col_no=0),
    )


def build_test_count_cache(
    funcs: list[FunctionToOptimize], project_root: Path, function_to_tests: dict[str, set[FunctionCalledInTest]]
) -> dict[tuple[Path, str], int]:
    return {
        (func.file_path, func.qualified_name): existing_unit_test_count(func, project_root, function_to_tests)
        for func in funcs
    }


def make_optimizer(project_root: Path) -> Optimizer:
    def _noop_display_global_ranking(*_args: object, **_kwargs: object) -> None:
        return None

    optimizer = Optimizer.__new__(Optimizer)
    optimizer.args = Namespace(project_root=project_root)
    optimizer.display_global_ranking = _noop_display_global_ranking
    return optimizer


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
            make_test(TestType.EXISTING_UNIT_TEST, "test_one")
        },
        funcs[1].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_three"),
        },
        # baz has no tests
    }

    ranked = sorted(funcs, key=lambda f: -existing_unit_test_count(f, project_root, function_to_tests))

    assert ranked[0].function_name == "bar"  # 3 tests
    assert ranked[1].function_name == "foo"  # 1 test
    assert ranked[2].function_name == "baz"  # 0 tests


def test_stable_sort_preserves_order_for_equal_counts(project_root: Path) -> None:
    funcs = [make_func(name, project_root) for name in ("foo", "bar", "baz")]
    function_to_tests: dict[str, set[FunctionCalledInTest]] = {
        f.qualified_name_with_modules_from_root(project_root): {make_test(TestType.EXISTING_UNIT_TEST)} for f in funcs
    }

    ranked = sorted(funcs, key=lambda f: -existing_unit_test_count(f, project_root, function_to_tests))

    assert [f.function_name for f in ranked] == ["foo", "bar", "baz"]


def test_parametrized_tests_deduplication(project_root: Path) -> None:
    func = make_func("foo", project_root)
    key = func.qualified_name_with_modules_from_root(project_root)
    tests = {
        key: {
            make_test(TestType.EXISTING_UNIT_TEST, "test_foo[0]"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_foo[1]"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_foo[2]"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_bar"),
        }
    }
    assert existing_unit_test_count(func, project_root, tests) == 2


def test_trace_ranking_keeps_addressable_time_primary_over_test_count(project_root: Path, tmp_path: Path) -> None:
    optimizer = make_optimizer(project_root)
    funcs = [make_func(name, project_root) for name in ("foo", "bar", "baz")]
    trace_file = tmp_path / "trace.db"
    trace_file.touch()

    ranked_functions = [funcs[0], funcs[1], funcs[2]]
    addressable_times = {"foo": 100.0, "bar": 20.0, "baz": 5.0}
    function_to_tests: dict[str, set[FunctionCalledInTest]] = {
        funcs[1].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_three"),
        }
    }

    class FakeRanker:
        def __init__(self, _trace_file: Path) -> None:
            pass

        def rank_functions(self, _functions: list[FunctionToOptimize]) -> list[FunctionToOptimize]:
            return ranked_functions

        def get_function_addressable_time(self, function: FunctionToOptimize) -> float:
            return addressable_times[function.function_name]

    with patch("codeflash.benchmarking.function_ranker.FunctionRanker", FakeRanker):
        ranked = optimizer.rank_all_functions_globally(
            {project_root / "mod.py": funcs},
            trace_file,
            test_count_cache=build_test_count_cache(funcs, project_root, function_to_tests),
        )

    assert [func.function_name for _, func in ranked] == ["foo", "bar", "baz"]


def test_trace_ranking_uses_test_count_as_tiebreaker(project_root: Path, tmp_path: Path) -> None:
    optimizer = make_optimizer(project_root)
    funcs = [make_func(name, project_root) for name in ("foo", "bar", "baz")]
    trace_file = tmp_path / "trace.db"
    trace_file.touch()

    ranked_functions = [funcs[0], funcs[1], funcs[2]]
    addressable_times = {"foo": 100.0, "bar": 100.0, "baz": 5.0}
    function_to_tests: dict[str, set[FunctionCalledInTest]] = {
        funcs[0].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one")
        },
        funcs[1].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_three"),
        },
    }

    class FakeRanker:
        def __init__(self, _trace_file: Path) -> None:
            pass

        def rank_functions(self, _functions: list[FunctionToOptimize]) -> list[FunctionToOptimize]:
            return ranked_functions

        def get_function_addressable_time(self, function: FunctionToOptimize) -> float:
            return addressable_times[function.function_name]

    with patch("codeflash.benchmarking.function_ranker.FunctionRanker", FakeRanker):
        ranked = optimizer.rank_all_functions_globally(
            {project_root / "mod.py": funcs},
            trace_file,
            test_count_cache=build_test_count_cache(funcs, project_root, function_to_tests),
        )

    assert [func.function_name for _, func in ranked] == ["bar", "foo", "baz"]


def test_dependency_count_ranking_keeps_callee_count_primary(project_root: Path) -> None:
    optimizer = make_optimizer(project_root)
    funcs = [make_func(name, project_root) for name in ("foo", "bar")]
    function_to_tests: dict[str, set[FunctionCalledInTest]] = {
        funcs[1].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_three"),
        }
    }

    class FakeResolver:
        def count_callees_per_function(self, _mapping: dict[Path, set[str]]) -> dict[tuple[Path, str], int]:
            return {(project_root / "mod.py", "foo"): 5, (project_root / "mod.py", "bar"): 1}

    ranked = optimizer.rank_by_dependency_count(
        [(project_root / "mod.py", funcs[0]), (project_root / "mod.py", funcs[1])],
        FakeResolver(),
        test_count_cache=build_test_count_cache(funcs, project_root, function_to_tests),
    )

    assert [func.function_name for _, func in ranked] == ["foo", "bar"]


def test_dependency_count_ranking_uses_test_count_as_tiebreaker(project_root: Path) -> None:
    optimizer = make_optimizer(project_root)
    funcs = [make_func(name, project_root) for name in ("foo", "bar")]
    function_to_tests: dict[str, set[FunctionCalledInTest]] = {
        funcs[0].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one")
        },
        funcs[1].qualified_name_with_modules_from_root(project_root): {
            make_test(TestType.EXISTING_UNIT_TEST, "test_one"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_two"),
            make_test(TestType.EXISTING_UNIT_TEST, "test_three"),
        },
    }

    class FakeResolver:
        def count_callees_per_function(self, _mapping: dict[Path, set[str]]) -> dict[tuple[Path, str], int]:
            return {(project_root / "mod.py", "foo"): 2, (project_root / "mod.py", "bar"): 2}

    ranked = optimizer.rank_by_dependency_count(
        [(project_root / "mod.py", funcs[0]), (project_root / "mod.py", funcs[1])],
        FakeResolver(),
        test_count_cache=build_test_count_cache(funcs, project_root, function_to_tests),
    )

    assert [func.function_name for _, func in ranked] == ["bar", "foo"]

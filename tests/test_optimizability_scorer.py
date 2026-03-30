from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeflash.discovery.optimizability_scorer import (
    DEFAULT_OPTIMIZABILITY_THRESHOLD,
    OptimizabilityScore,
    filter_by_optimizability,
    score_function_optimizability,
)
from codeflash.models.function_types import FunctionToOptimize


def _make_func(
    name: str = "my_func",
    start: int = 1,
    end: int = 20,
    language: str = "python",
    file_path: Path | None = None,
) -> FunctionToOptimize:
    return FunctionToOptimize(
        function_name=name,
        file_path=file_path or Path("/fake/module.py"),
        starting_line=start,
        ending_line=end,
        language=language,
    )


class TestOptimizabilityScore:
    def test_is_optimizable_above_threshold(self) -> None:
        score = OptimizabilityScore("f", 0.5, "loops")
        assert score.is_optimizable

    def test_is_not_optimizable_below_threshold(self) -> None:
        score = OptimizabilityScore("f", 0.0, "too small")
        assert not score.is_optimizable

    def test_threshold_boundary(self) -> None:
        score = OptimizabilityScore("f", DEFAULT_OPTIMIZABILITY_THRESHOLD, "boundary")
        assert score.is_optimizable


class TestScorePythonFunction:
    def test_tiny_function_scores_zero(self) -> None:
        source = "def f():\n    return 1\n"
        func = _make_func(start=1, end=2)
        result = score_function_optimizability(func, source)
        assert result.score == 0.0
        assert "too small" in result.reason

    def test_function_with_loops_scores_high(self) -> None:
        source = "\n".join([
            "def process(data):",
            "    result = []",
            "    for item in data:",
            "        for sub in item:",
            "            result.append(sub * 2)",
            "    return result",
            "",
            "",
            "",
            "",
        ])
        func = _make_func(start=1, end=6)
        result = score_function_optimizability(func, source)
        assert result.score >= 0.3
        assert "loop" in result.reason or "nested" in result.reason

    def test_simple_delegation_scores_low(self) -> None:
        source = "\n".join([
            "def wrapper(x):",
            "    return other_func(x)",
        ])
        func = _make_func(start=1, end=2)
        result = score_function_optimizability(func, source)
        assert result.score < DEFAULT_OPTIMIZABILITY_THRESHOLD

    def test_comprehension_contributes(self) -> None:
        source = "\n".join([
            "def transform(data):",
            "    a = [x * 2 for x in data]",
            "    b = {k: v for k, v in pairs}",
            "    return a, b",
            "",
            "",
        ])
        func = _make_func(start=1, end=4)
        result = score_function_optimizability(func, source)
        assert result.score > 0

    def test_recursive_function_scores_well(self) -> None:
        source = "\n".join([
            "def fibonacci(n):",
            "    if n <= 1:",
            "        return n",
            "    return fibonacci(n - 1) + fibonacci(n - 2)",
            "",
            "",
        ])
        func = _make_func(name="fibonacci", start=1, end=4)
        result = score_function_optimizability(func, source)
        assert result.score >= DEFAULT_OPTIMIZABILITY_THRESHOLD
        assert "recursive" in result.reason

    def test_large_function_gets_size_bonus(self) -> None:
        lines = ["def big_func():"]
        for i in range(50):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_0")
        source = "\n".join(lines)
        func = _make_func(start=1, end=52)
        result = score_function_optimizability(func, source)
        assert result.score > 0.1

    def test_unknown_bounds_gets_neutral_score(self) -> None:
        func = _make_func(start=None, end=None)
        result = score_function_optimizability(func, "def f(): pass")
        assert result.score == 0.5


class TestScoreByHeuristics:
    def test_js_function_with_loops(self) -> None:
        source = "\n".join([
            "function processData(arr) {",
            "  const result = [];",
            "  for (let i = 0; i < arr.length; i++) {",
            "    result.push(arr[i] * 2);",
            "  }",
            "  return result;",
            "",
            "",
        ])
        func = _make_func(start=1, end=6, language="javascript")
        result = score_function_optimizability(func, source)
        assert result.score >= DEFAULT_OPTIMIZABILITY_THRESHOLD

    def test_js_tiny_function_scores_low(self) -> None:
        source = "function getId() { return this.id; }"
        func = _make_func(start=1, end=1, language="javascript")
        result = score_function_optimizability(func, source)
        assert result.score == 0.0


class TestFilterByOptimizability:
    def test_filters_low_score_functions(self, tmp_path: Path) -> None:
        # Create a file with a tiny function and a complex function
        source = "\n".join([
            "def tiny():",
            "    return 1",
            "",
            "def complex_func(data):",
            "    result = []",
            "    for item in data:",
            "        for sub in item:",
            "            result.append(sub * 2)",
            "    return result",
        ])
        file = tmp_path / "module.py"
        file.write_text(source, encoding="utf-8")

        tiny = _make_func(name="tiny", start=1, end=2, file_path=file)
        complex_fn = _make_func(name="complex_func", start=4, end=9, file_path=file)

        functions = {file: [tiny, complex_fn]}
        filtered, skipped = filter_by_optimizability(functions)
        assert skipped >= 1
        # complex_func should survive
        assert any(f.function_name == "complex_func" for f in filtered.get(file, []))

    def test_empty_input(self) -> None:
        filtered, skipped = filter_by_optimizability({})
        assert filtered == {}
        assert skipped == 0

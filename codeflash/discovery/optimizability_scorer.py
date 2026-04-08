from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize

MIN_LINES_FOR_OPTIMIZATION = 3
DEFAULT_OPTIMIZABILITY_THRESHOLD = 0.15


@dataclass(frozen=True)
class OptimizabilityScore:
    function_name: str
    score: float
    reason: str

    @property
    def is_optimizable(self) -> bool:
        return self.score >= DEFAULT_OPTIMIZABILITY_THRESHOLD


def score_function_optimizability(func: FunctionToOptimize, source: str | None = None) -> OptimizabilityScore:
    """Score a function's optimization potential using fast static analysis.

    Returns a score between 0.0 (not worth optimizing) and 1.0 (high potential).
    """
    if func.starting_line is None or func.ending_line is None:
        return OptimizabilityScore(func.qualified_name, 0.5, "unknown bounds")

    num_lines = func.ending_line - func.starting_line + 1
    if num_lines < MIN_LINES_FOR_OPTIMIZATION:
        return OptimizabilityScore(func.qualified_name, 0.0, f"too small ({num_lines} lines)")

    if source is None:
        try:
            source = func.file_path.read_text(encoding="utf-8")
        except OSError:
            return OptimizabilityScore(func.qualified_name, 0.5, "could not read source")

    func_source = _extract_function_source(source, func.starting_line, func.ending_line)
    if func_source is None:
        return OptimizabilityScore(func.qualified_name, 0.5, "could not extract source")

    if func.language == "python":
        return _score_python_function(func, func_source, num_lines)
    if func.language in ("javascript", "typescript"):
        return _score_by_heuristics(func, func_source, num_lines)
    if func.language == "java":
        return _score_by_heuristics(func, func_source, num_lines)
    return OptimizabilityScore(func.qualified_name, 0.5, "unknown language")


def _extract_function_source(full_source: str, start_line: int, end_line: int) -> str | None:
    lines = full_source.splitlines()
    if start_line < 1 or end_line > len(lines):
        return None
    return "\n".join(lines[start_line - 1 : end_line])


def _score_python_function(func: FunctionToOptimize, func_source: str, num_lines: int) -> OptimizabilityScore:
    try:
        tree = ast.parse(func_source)
    except SyntaxError:
        return _score_by_heuristics(func, func_source, num_lines)

    visitor = _PythonComplexityVisitor()
    visitor.visit(tree)

    score = 0.0
    reasons: list[str] = []

    # Size contribution (logarithmic, caps at ~0.3)
    size_score = min(0.3, num_lines / 100)
    score += size_score

    # Loop presence is a strong signal
    if visitor.loop_count > 0:
        loop_score = min(0.35, visitor.loop_count * 0.15)
        score += loop_score
        reasons.append(f"{visitor.loop_count} loop(s)")

    # Comprehension/generator expressions
    if visitor.comprehension_count > 0:
        score += min(0.15, visitor.comprehension_count * 0.08)
        reasons.append(f"{visitor.comprehension_count} comprehension(s)")

    # Nested loops are a very strong signal
    if visitor.max_loop_depth >= 2:
        score += 0.2
        reasons.append(f"nested loops (depth {visitor.max_loop_depth})")

    # Recursion
    if visitor.has_recursion:
        score += 0.15
        reasons.append("recursive")

    # Mathematical operations (sorting, searching patterns)
    if visitor.math_op_count > 0:
        score += min(0.1, visitor.math_op_count * 0.03)
        reasons.append("math ops")

    # Data structure operations
    if visitor.collection_op_count > 0:
        score += min(0.1, visitor.collection_op_count * 0.03)
        reasons.append("collection ops")

    # Penalty: mostly I/O (file, network, DB)
    if visitor.io_call_count > 0 and visitor.loop_count == 0:
        io_ratio = visitor.io_call_count / max(1, visitor.total_call_count)
        if io_ratio > 0.5:
            score *= 0.3
            reasons.append(f"I/O dominated ({visitor.io_call_count} I/O calls)")

    # Penalty: simple delegation (just calls another function)
    if num_lines <= 5 and visitor.total_call_count == 1 and visitor.loop_count == 0:
        score *= 0.2
        reasons.append("simple delegation")

    # Penalty: only string formatting / logging
    if visitor.is_mostly_string_ops:
        score *= 0.3
        reasons.append("mostly string/logging ops")

    score = min(1.0, max(0.0, score))
    reason = ", ".join(reasons) if reasons else f"{num_lines} lines"
    return OptimizabilityScore(func.qualified_name, score, reason)


def _score_by_heuristics(func: FunctionToOptimize, func_source: str, num_lines: int) -> OptimizabilityScore:
    """Language-agnostic heuristic scoring using text patterns."""
    score = 0.0
    reasons: list[str] = []

    # Size
    score += min(0.3, num_lines / 100)

    # Loop keywords
    loop_keywords = ("for ", "for(", "while ", "while(", ".forEach(", ".map(", ".filter(", ".reduce(")
    loop_count = sum(1 for kw in loop_keywords if kw in func_source)
    if loop_count > 0:
        score += min(0.35, loop_count * 0.12)
        reasons.append(f"loop patterns ({loop_count})")

    # Sorting/searching
    sort_patterns = ("sort(", "sorted(", ".sort(", "binarySearch", "indexOf", "Collections.sort")
    if any(p in func_source for p in sort_patterns):
        score += 0.15
        reasons.append("sort/search ops")

    # Nested structure (indentation depth as proxy)
    max_indent = max((len(line) - len(line.lstrip()) for line in func_source.splitlines() if line.strip()), default=0)
    if max_indent > 16:  # roughly 4+ levels of nesting
        score += 0.15
        reasons.append("deep nesting")

    # Simple delegation penalty
    if num_lines <= 5:
        score *= 0.3
        reasons.append("very small")

    score = min(1.0, max(0.0, score))
    reason = ", ".join(reasons) if reasons else f"{num_lines} lines"
    return OptimizabilityScore(func.qualified_name, score, reason)


class _PythonComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loop_count = 0
        self.max_loop_depth = 0
        self.comprehension_count = 0
        self.has_recursion = False
        self.math_op_count = 0
        self.collection_op_count = 0
        self.io_call_count = 0
        self.total_call_count = 0
        self.is_mostly_string_ops = False
        self._current_loop_depth = 0
        self._current_func_name: str | None = None
        self._string_op_count = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._current_func_name is None:
            self._current_func_name = node.name
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._current_func_name is None:
            self._current_func_name = node.name
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loop_count += 1
        self._current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self._current_loop_depth)
        self.generic_visit(node)
        self._current_loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_count += 1
        self._current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self._current_loop_depth)
        self.generic_visit(node)
        self._current_loop_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.total_call_count += 1
        func_name = _get_call_name(node)
        if func_name:
            # Recursion detection
            if func_name == self._current_func_name:
                self.has_recursion = True
            # I/O patterns
            io_names = {"open", "read", "write", "send", "recv", "connect", "execute", "fetch", "request", "urlopen"}
            if func_name in io_names or func_name.startswith(("requests.", "urllib")):
                self.io_call_count += 1
            # Math patterns
            math_names = {"sum", "min", "max", "abs", "pow", "sqrt", "log", "exp", "ceil", "floor"}
            if func_name in math_names or "numpy" in func_name or "np." in func_name or "math." in func_name:
                self.math_op_count += 1
            # Collection operations
            collection_names = {"sorted", "reversed", "enumerate", "zip", "filter", "map", "reduce"}
            if func_name in collection_names:
                self.collection_op_count += 1
            # String ops
            string_names = {"format", "join", "split", "replace", "strip", "lower", "upper", "encode", "decode"}
            if func_name in string_names or func_name.endswith((".format", ".join")):
                self._string_op_count += 1

        self.generic_visit(node)

        # After visiting all nodes, check if mostly string ops
        if self.total_call_count > 0:
            self.is_mostly_string_ops = (self._string_op_count / self.total_call_count) > 0.7

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Mult, ast.Pow, ast.MatMult, ast.FloorDiv)):
            self.math_op_count += 1
        self.generic_visit(node)


def _get_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def filter_by_optimizability(
    functions: dict[Path, list[FunctionToOptimize]], threshold: float = DEFAULT_OPTIMIZABILITY_THRESHOLD
) -> tuple[dict[Path, list[FunctionToOptimize]], int]:
    """Filter functions by optimizability score, returning only those above threshold."""
    filtered: dict[Path, list[FunctionToOptimize]] = {}
    skipped_count = 0

    for file_path, funcs in functions.items():
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError:
            filtered[file_path] = funcs
            continue

        kept: list[FunctionToOptimize] = []
        for func in funcs:
            result = score_function_optimizability(func, source)
            if result.is_optimizable:
                kept.append(func)
            else:
                skipped_count += 1
                logger.debug(f"Skipping {func.qualified_name} (score={result.score:.2f}, reason: {result.reason})")

        if kept:
            filtered[file_path] = kept

    if skipped_count > 0:
        logger.info(f"Pre-screening: skipped {skipped_count} low-optimizability function(s)")

    return filtered, skipped_count

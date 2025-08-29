from __future__ import annotations

import ast
import logging
import os
from collections import defaultdict
from pathlib import Path


class CallInfo:
    """Information about a function call."""

    def __init__(self, caller: str, callee: str, line_number: int) -> None:
        self.caller = caller
        self.callee = callee
        self.line_number = line_number


class FunctionInfo:
    """Information about a function definition."""

    def __init__(self, name: str, file_path: str, line_number: int, full_name: str) -> None:
        self.name = name
        self.file_path = file_path
        self.line_number = line_number
        self.full_name = full_name  # e.g., "module::ClassName.method_name."


class CallGraph:
    """Simple call graph."""

    def __init__(self, root_path: str) -> None:
        self.root_path = Path(root_path).resolve()
        self.functions: dict[str, FunctionInfo] = {}  # full_name -> FunctionInfo
        self.call_edges: list[CallInfo] = []  # List of all function calls
        self.call_count: dict[str, int] = defaultdict(int)  # function_name -> call_count
        self.callers: dict[str, set[str]] = defaultdict(set)  # callee -> set of callers
        self.callees: dict[str, set[str]] = defaultdict(set)  # caller -> set of callees

    def parse_project(self) -> None:
        """Parse all Python files in the project."""
        for root, dirs, files in os.walk(str(self.root_path)):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "node_modules"]]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        self._parse_file(file_path)
                    except Exception as e:
                        logging.warning("Failed to parse %s: %s", file_path, e)

        self._resolve_calls()

    def _parse_file(self, file_path: Path) -> None:
        """Parse a single Python file."""
        with file_path.open(encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError as e:
                logging.warning("Syntax error in %s: %s", file_path, e)
                return

        visitor = FunctionVisitor(file_path)
        visitor.visit(tree)

        # Add functions to our registry
        for func_info in visitor.functions:
            self.functions[func_info.full_name] = func_info

        # Add calls
        self.call_edges.extend(visitor.calls)

    def _resolve_calls(self) -> None:
        """Resolve function calls and build the call graph."""
        for call in self.call_edges:
            # Try to find the actual function being called
            resolved_callee = self._resolve_function_name(call.callee)
            if resolved_callee:
                self.call_count[resolved_callee] += 1
                self.callers[resolved_callee].add(call.caller)
                self.callees[call.caller].add(resolved_callee)

    def _resolve_function_name(self, call_name: str) -> str | None:
        """Try to resolve a function call to an actual function definition."""
        # Direct match
        if call_name in self.functions:
            return call_name

        # Try to find by function name only (simple case)
        for full_name, func_info in self.functions.items():
            if func_info.name == call_name or full_name.endswith("." + call_name):
                return full_name

        return None

    def get_call_count(self, function_name: str) -> int:
        """Get the number of times a function is called."""
        # Try exact match first
        if function_name in self.call_count:
            return self.call_count[function_name]

        # Try to find by partial name
        for full_name in self.call_count:
            if full_name.endswith("." + function_name) or full_name == function_name:
                return self.call_count[full_name]

        return 0

    def get_total_call_count(self, function_name: str) -> int:
        """Get total number of times a function is called, including indirect calls through its callers."""
        resolved_name = self._resolve_function_name(function_name)
        if not resolved_name:
            return 0

        visited = set()

        def _count(func: str) -> int:
            if func in visited:
                return 0
            visited.add(func)

            # direct calls
            total = self.call_count.get(func, 0)

            # add calls from all callers
            for caller in self.callers.get(func, []):
                total += _count(caller)

            return total

        return _count(resolved_name)

    def get_function_callers(self, function_name: str) -> set[str]:
        """Get all functions that call the given function."""
        resolved_name = self._resolve_function_name(function_name)
        if resolved_name:
            return self.callers.get(resolved_name, set())
        return set()

    def get_function_callees(self, function_name: str) -> set[str]:
        """Get all functions called by the given function."""
        resolved_name = self._resolve_function_name(function_name)
        if resolved_name:
            return self.callees.get(resolved_name, set())
        return set()

    def build_call_chain(self, root_function: str, max_depth: int = 5) -> dict:
        """Build a call chain starting from root_function."""
        resolved_root = self._resolve_function_name(root_function)
        if not resolved_root:
            return {}

        visited = set()

        def _build_chain(func_name: str, depth: int) -> dict:
            if depth > max_depth or func_name in visited:
                return {}

            visited.add(func_name)
            callees = self.get_function_callees(func_name)

            chain = {"function": func_name, "call_count": self.get_call_count(func_name), "calls": {}}

            for callee in callees:
                chain["calls"][callee] = _build_chain(callee, depth + 1)

            return chain

        return _build_chain(resolved_root, 0)


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function definitions and calls."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.module_name = self._get_module_name(file_path)
        self.functions: list[FunctionInfo] = []
        self.calls: list[CallInfo] = []
        self.current_class = None
        self.current_function = None

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        return file_path.resolve().as_posix()

    def visit_ClassDef(self, node: ast.AST) -> None:
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.AST) -> None:
        """Visit function definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AST) -> None:
        """Visit async function definition."""
        self._visit_function(node)

    def _visit_function(self, node: ast.AST) -> None:
        """Handle both sync and async function definitions."""
        # Build full function name
        name = getattr(node, "name", None)
        lineno = getattr(node, "lineno", None)
        if self.current_class:
            full_name = f"{self.module_name}::{self.current_class}.{name}"
        else:
            full_name = f"{self.module_name}::{name}"

        func_info = FunctionInfo(name=name, file_path=str(self.file_path), line_number=lineno, full_name=full_name)  # type: ignore[arg-type]
        self.functions.append(func_info)

        # Visit function body to find calls
        old_function = self.current_function
        self.current_function = full_name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Call(self, node: ast.AST) -> None:
        """Visit function call."""
        if self.current_function:
            call_name = self._get_call_name(node.func)  # type: ignore[arg-type]
            if call_name:
                call_info = CallInfo(caller=self.current_function, callee=call_name, line_number=node.lineno)  # type: ignore[arg-type]
                self.calls.append(call_info)

        self.generic_visit(node)

    def _get_call_name(self, func_node: ast.AST) -> str | None:
        """Extract function name from call node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            # Handle method calls like obj.method()
            return func_node.attr
        return None

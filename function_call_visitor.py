"""AST Visitor to count function calls and identify calls within loops.

This module provides a visitor that can track calls to specific functions,
including regular functions, methods, classmethods, and staticmethods.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CallInfo:
    """Information about a function call."""

    line: int
    col: int
    call_text: str
    in_loop: bool
    loop_type: Optional[str] = None  # 'for', 'while', or nested combinations
    file_path: Optional[str] = None

    def __repr__(self):
        loop_info = f" (in {self.loop_type} loop)" if self.in_loop else ""
        file_info = f"{self.file_path}:" if self.file_path else ""
        return f"{file_info}{self.line}:{self.col} - {self.call_text}{loop_info}"


class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor to count and track function calls.

    Handles:
    - Regular function calls: func()
    - Method calls: obj.method()
    - Class method calls: Class.method()
    - Static method calls: Class.static_method()
    - Nested attribute calls: module.submodule.func()
    """

    def __init__(self, target_functions: list[str], file_path: Optional[str] = None):
        """Initialize the visitor.

        Args:
            target_functions: list of function names to track. Can be:
                - Simple names: ['print', 'len']
                - Qualified names: ['os.path.join', 'numpy.array']
                - Method names: ['append', 'extend'] (will match any obj.append())
            file_path: Optional path to the file being analyzed

        """
        self.target_functions = set(target_functions)
        self.file_path = file_path
        self.calls: list[CallInfo] = []
        self.loop_stack: list[str] = []  # Track nested loops
        self._source_lines: Optional[list[str]] = None

    def set_source(self, source: str):
        """Set the source code for better call text extraction."""
        self._source_lines = source.splitlines()

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the full name of the called function."""
        if isinstance(node.func, ast.Name):
            # Simple function call: func()
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            # Method or qualified call: obj.method() or module.func()
            parts = []
            current = node.func

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)
                full_name = ".".join(reversed(parts))

                # Check if we should track this call
                # Match exact qualified names or just the method name
                if full_name in self.target_functions:
                    return full_name

                # Also check if just the method name matches
                # (for tracking all calls to a method regardless of object)
                method_name = parts[0]  # The rightmost part is the method
                if method_name in self.target_functions:
                    return full_name

                # Check partial matches (e.g., 'path.join' matches 'os.path.join')
                for target in self.target_functions:
                    if full_name.endswith(target) or target.endswith(full_name):
                        return full_name

        return None

    def _get_call_text(self, node: ast.Call) -> str:
        """Get a string representation of the call."""
        if self._source_lines and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            try:
                if node.lineno == node.end_lineno:
                    line = self._source_lines[node.lineno - 1]
                    if hasattr(node, "col_offset") and hasattr(node, "end_col_offset"):
                        return line[node.col_offset:node.end_col_offset]
                else:
                    # Multi-line call
                    lines = []
                    for i in range(node.lineno - 1, node.end_lineno):
                        if i < len(self._source_lines):
                            if i == node.lineno - 1:
                                lines.append(self._source_lines[i][node.col_offset:])
                            elif i == node.end_lineno - 1:
                                lines.append(self._source_lines[i][:node.end_col_offset])
                            else:
                                lines.append(self._source_lines[i])
                    return " ".join(line.strip() for line in lines)
            except (IndexError, AttributeError):
                pass

        # Fallback to reconstructing from AST
        return ast.unparse(node) if hasattr(ast, "unparse") else self._get_call_name(node) + "(...)"

    def _in_loop(self) -> bool:
        """Check if we're currently inside a loop."""
        return len(self.loop_stack) > 0

    def _get_loop_type(self) -> Optional[str]:
        """Get the current loop type(s)."""
        if not self.loop_stack:
            return None
        if len(self.loop_stack) == 1:
            return self.loop_stack[0]
        return " -> ".join(self.loop_stack)  # Show nested loops

    def visit_Call(self, node: ast.Call):
        """Visit a function call node."""
        call_name = self._get_call_name(node)

        if call_name:
            # Check if this matches any of our target functions
            should_track = False

            # Direct match
            if call_name in self.target_functions:
                should_track = True
            else:
                # Check if just the method/function name matches
                simple_name = call_name.split(".")[-1]
                if simple_name in self.target_functions:
                    should_track = True
                else:
                    # Check for partial qualified matches
                    for target in self.target_functions:
                        if "." in target:
                            # For qualified targets, check if call matches the end
                            if call_name.endswith("." + target.split(".")[-1]):
                                should_track = True
                                break

            if should_track:
                call_info = CallInfo(
                    line=node.lineno,
                    col=node.col_offset,
                    call_text=self._get_call_text(node),
                    in_loop=self._in_loop(),
                    loop_type=self._get_loop_type(),
                    file_path=self.file_path
                )
                self.calls.append(call_info)

        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Visit a for loop."""
        self.loop_stack.append("for")
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_While(self, node: ast.While):
        """Visit a while loop."""
        self.loop_stack.append("while")
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_AsyncFor(self, node: ast.AsyncFor):
        """Visit an async for loop."""
        self.loop_stack.append("async for")
        self.generic_visit(node)
        self.loop_stack.pop()

    def get_summary(self) -> dict:
        """Get a summary of the calls found."""
        total_calls = len(self.calls)
        calls_in_loops = [c for c in self.calls if c.in_loop]
        calls_outside_loops = [c for c in self.calls if not c.in_loop]

        return {
            "total_calls": total_calls,
            "calls_in_loops": len(calls_in_loops),
            "calls_outside_loops": len(calls_outside_loops),
            "all_calls": self.calls,
            "loop_calls": calls_in_loops,
            "non_loop_calls": calls_outside_loops
        }


def analyze_file(file_path: str, target_functions: list[str]) -> dict:
    """Analyze a Python file for function calls.

    Args:
        file_path: Path to the Python file
        target_functions: list of function names to track

    Returns:
        dictionary with call statistics and details

    """
    with Path.open(file_path) as f:
        source = f.read()

    tree = ast.parse(source, filename=file_path)
    visitor = FunctionCallVisitor(target_functions, file_path)
    visitor.set_source(source)
    visitor.visit(tree)

    return visitor.get_summary()


def analyze_code(source: str, target_functions: list[str], file_path: Optional[str] = None) -> dict:
    """Analyze Python source code for function calls.

    Args:
        source: Python source code as string
        target_functions: list of function names to track
        file_path: Optional file path for reference

    Returns:
        dictionary with call statistics and details

    """
    tree = ast.parse(source)
    visitor = FunctionCallVisitor(target_functions, file_path)
    visitor.set_source(source)
    visitor.visit(tree)

    return visitor.get_summary()


if __name__ == "__main__":
    # Example usage
    example_code = """
import os
import numpy as np

def process_data(data):
    print("Starting processing")
    result = []

    for item in data:
        print(f"Processing {item}")
        value = len(item)
        result.append(value)

        for i in range(3):
            print(f"Inner loop {i}")
            np.array([1, 2, 3])

    while len(result) < 10:
        print("Adding more items")
        result.append(0)

    os.path.join("dir", "file")
    print("Done")
    return result

class DataProcessor:
    def process(self, items):
        for item in items:
            self.validate(item)
            print(f"Item: {item}")

    def validate(self, item):
        if len(item) > 0:
            print("Valid")

    @classmethod
    def create(cls):
        print("Creating processor")
        return cls()

    @staticmethod
    def utility():
        print("Utility function")
"""

    # Track multiple functions
    targets = ["print", "len", "np.array", "os.path.join", "append", "validate"]
    results = analyze_code(example_code, targets, "example.py")

    print("Function Call Analysis Results")
    print("=" * 50)
    print(f"Total calls found: {results['total_calls']}")
    print(f"Calls in loops: {results['calls_in_loops']}")
    print(f"Calls outside loops: {results['calls_outside_loops']}")
    print("\nAll calls:")
    print("-" * 50)
    for call in results["all_calls"]:
        print(f"  {call}")

    if results["loop_calls"]:
        print("\nCalls within loops:")
        print("-" * 50)
        for call in results["loop_calls"]:
            print(f"  {call}")

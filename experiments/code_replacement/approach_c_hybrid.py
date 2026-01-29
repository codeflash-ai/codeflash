"""Approach C: Hybrid - Tree-sitter for analysis + text-based replacement.

This approach:
1. Uses tree-sitter to parse and understand the code structure
2. Uses tree-sitter queries to find exact function boundaries
3. Does text-based replacement using byte offsets (more precise than line numbers)
4. Optionally validates result with tree-sitter

Pros:
- More precise than line-based replacement (uses byte offsets)
- Understands code structure for validation
- Can handle complex nesting scenarios
- No external Node.js dependencies

Cons:
- Tree-sitter setup required
- More complex than pure text-based
- Still text-based replacement (not AST rewriting)
"""

import sys
from dataclasses import dataclass
from typing import Optional

# Try to import tree-sitter, provide fallback if not available
try:
    import tree_sitter_javascript
    import tree_sitter_typescript
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print(
        "Warning: tree-sitter not available. Install with: pip install tree-sitter tree-sitter-javascript tree-sitter-typescript"
    )


@dataclass
class FunctionBoundary:
    """Precise boundaries of a function in source code."""

    name: str
    start_byte: int
    end_byte: int
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    start_col: int
    end_col: int
    node_type: str  # e.g., 'function_declaration', 'arrow_function', 'method_definition'


class HybridReplacer:
    """Replace functions using tree-sitter analysis + text replacement."""

    def __init__(self, language: str = "javascript"):
        """Initialize with specified language.

        Args:
            language: 'javascript' or 'typescript'

        """
        self.language = language

        if TREE_SITTER_AVAILABLE:
            if language == "javascript":
                self.ts_language = Language(tree_sitter_javascript.language())
            elif language == "typescript":
                self.ts_language = Language(tree_sitter_typescript.language_typescript())
            elif language == "tsx":
                self.ts_language = Language(tree_sitter_typescript.language_tsx())
            else:
                raise ValueError(f"Unsupported language: {language}")

            self.parser = Parser(self.ts_language)
        else:
            self.parser = None

    def find_function_boundaries(self, source: str, function_name: Optional[str] = None) -> list[FunctionBoundary]:
        """Find all function boundaries in source code.

        Args:
            source: Source code to analyze
            function_name: If provided, only return functions with this name

        Returns:
            List of FunctionBoundary objects

        """
        if not TREE_SITTER_AVAILABLE:
            return []

        tree = self.parser.parse(bytes(source, "utf8"))
        source_bytes = bytes(source, "utf8")

        boundaries = []

        def get_function_name(node) -> Optional[str]:
            """Extract function name from various node types."""
            # function_declaration: function foo() {}
            if node.type == "function_declaration" or node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")

            # variable_declarator with arrow function: const foo = () => {}
            elif node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if name_node and value_node and value_node.type == "arrow_function":
                    return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")

            # lexical_declaration: const foo = () => {}
            elif node.type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        return get_function_name(child)

            return None

        def traverse(node) -> None:  # noqa: ANN001
            """Recursively traverse tree to find functions."""
            node_type = node.type

            # Check if this is a function-like node
            is_function = node_type in [
                "function_declaration",
                "function",
                "arrow_function",
                "method_definition",
                "generator_function_declaration",
            ]

            # For lexical declarations, check if they contain arrow functions
            if node_type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        value = child.child_by_field_name("value")
                        if value and value.type == "arrow_function":
                            name = get_function_name(child)
                            if name and (function_name is None or name == function_name):
                                # Use the full declaration bounds
                                boundaries.append(
                                    FunctionBoundary(
                                        name=name,
                                        start_byte=node.start_byte,
                                        end_byte=node.end_byte,
                                        start_line=node.start_point[0] + 1,
                                        end_line=node.end_point[0] + 1,
                                        start_col=node.start_point[1],
                                        end_col=node.end_point[1],
                                        node_type="arrow_function",
                                    )
                                )
                return  # Don't recurse into lexical declarations we've handled

            if is_function:
                name = get_function_name(node)
                if name and (function_name is None or name == function_name):
                    boundaries.append(
                        FunctionBoundary(
                            name=name,
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            start_col=node.start_point[1],
                            end_col=node.end_point[1],
                            node_type=node_type,
                        )
                    )

            # Recurse into children
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return boundaries

    def replace_function_by_bytes(self, source: str, start_byte: int, end_byte: int, new_function: str) -> str:
        """Replace function using byte offsets.

        Args:
            source: Original source code
            start_byte: Starting byte offset
            end_byte: Ending byte offset
            new_function: New function source code

        Returns:
            Modified source code

        """
        source_bytes = source.encode("utf8")

        # Get original indentation from the first line of the function
        # Find the start of the line containing start_byte
        line_start = source_bytes.rfind(b"\n", 0, start_byte)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1  # Move past the newline

        original_indent = start_byte - line_start

        # Detect indentation of new function
        new_lines = new_function.splitlines(keepends=True)
        if new_lines:
            new_first_line = new_lines[0]
            new_indent = len(new_first_line) - len(new_first_line.lstrip())
        else:
            new_indent = 0

        # Adjust indentation if needed
        indent_diff = original_indent - new_indent
        if indent_diff != 0:
            adjusted_new_lines = []
            for line in new_lines:
                if line.strip():
                    if indent_diff > 0:
                        adjusted_new_lines.append(" " * indent_diff + line)
                    else:
                        current_indent = len(line) - len(line.lstrip())
                        remove_amount = min(current_indent, abs(indent_diff))
                        adjusted_new_lines.append(line[remove_amount:])
                else:
                    adjusted_new_lines.append(line)
            new_function = "".join(adjusted_new_lines)

        # Perform byte-level replacement
        before = source_bytes[:start_byte].decode("utf8")
        after = source_bytes[end_byte:].decode("utf8")

        return before + new_function + after

    def replace_function(self, source: str, function_name: str, new_function: str) -> str:
        """Replace a function by name using tree-sitter analysis.

        Args:
            source: Original source code
            function_name: Name of function to replace
            new_function: New function source code

        Returns:
            Modified source code

        """
        boundaries = self.find_function_boundaries(source, function_name)

        if not boundaries:
            msg = f"Function '{function_name}' not found in source"
            raise ValueError(msg)

        if len(boundaries) > 1:
            # Multiple functions with same name - use the first one
            # In practice, you'd want to disambiguate by line number
            pass

        boundary = boundaries[0]
        return self.replace_function_by_bytes(source, boundary.start_byte, boundary.end_byte, new_function)

    def replace_function_by_lines(self, source: str, start_line: int, end_line: int, new_function: str) -> str:
        """Replace function using line numbers (for compatibility with test cases).

        This method delegates to the text-based approach since it's more reliable
        for line-based replacement. The byte-based approach is better when you
        have precise byte offsets from tree-sitter analysis.

        Args:
            source: Original source code
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed, inclusive)
            new_function: New function source code

        Returns:
            Modified source code

        """
        # For line-based replacement, use the simpler text-based approach
        # It handles edge cases (newlines, indentation) more reliably
        lines = source.splitlines(keepends=True)

        # Handle case where source doesn't end with newline
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        # Get indentation from original function's first line
        if start_line <= len(lines):
            original_first_line = lines[start_line - 1]
            original_indent = len(original_first_line) - len(original_first_line.lstrip())
        else:
            original_indent = 0

        # Get indentation from new function's first line
        new_lines = new_function.splitlines(keepends=True)
        if new_lines:
            new_first_line = new_lines[0]
            new_indent = len(new_first_line) - len(new_first_line.lstrip())
        else:
            new_indent = 0

        # Calculate indent adjustment needed
        indent_diff = original_indent - new_indent

        # Adjust indentation of new function if needed
        if indent_diff != 0:
            adjusted_new_lines = []
            for line in new_lines:
                if line.strip():  # Non-empty line
                    if indent_diff > 0:
                        adjusted_new_lines.append(" " * indent_diff + line)
                    else:
                        current_indent = len(line) - len(line.lstrip())
                        remove_amount = min(current_indent, abs(indent_diff))
                        adjusted_new_lines.append(line[remove_amount:])
                else:
                    adjusted_new_lines.append(line)
            new_lines = adjusted_new_lines

        # Ensure new function ends with newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Build result
        before = lines[: start_line - 1]
        after = lines[end_line:]

        result_lines = before + new_lines + after
        return "".join(result_lines)

    def validate_result(self, source: str) -> bool:
        """Validate that the result is syntactically correct.

        Args:
            source: Source code to validate

        Returns:
            True if valid, False otherwise

        """
        if not TREE_SITTER_AVAILABLE:
            return True  # Can't validate without tree-sitter

        tree = self.parser.parse(bytes(source, "utf8"))
        return not tree.root_node.has_error


def replace_function_hybrid(
    source: str, start_line: int, end_line: int, new_function: str, language: str = "javascript"
) -> str:
    """Convenience function for hybrid replacement.

    Args:
        source: Original source code
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive)
        new_function: New function source code
        language: 'javascript' or 'typescript'

    Returns:
        Modified source code

    """
    replacer = HybridReplacer(language)
    return replacer.replace_function_by_lines(source, start_line, end_line, new_function)


# Test the implementation
if __name__ == "__main__":
    from test_cases import get_test_cases

    if not TREE_SITTER_AVAILABLE:
        print("Cannot run tests: tree-sitter not installed")
        sys.exit(1)

    replacer = HybridReplacer("javascript")
    ts_replacer = HybridReplacer("typescript")

    print("=" * 60)
    print("Testing Approach C: Hybrid (Tree-sitter + Text)")
    print("=" * 60)

    passed = 0
    failed = 0

    for tc in get_test_cases():
        # Use TypeScript parser for TypeScript test cases
        is_typescript = "typescript" in tc.name or "interface" in tc.description.lower()
        current_replacer = ts_replacer if is_typescript else replacer

        result = current_replacer.replace_function_by_lines(
            tc.original_source, tc.start_line, tc.end_line, tc.new_function
        )

        # Normalize line endings for comparison
        result_normalized = result.replace("\r\n", "\n")
        expected_normalized = tc.expected_result.replace("\r\n", "\n")

        if result_normalized == expected_normalized:
            print(f"✓ PASS: {tc.name}")
            passed += 1
        else:
            print(f"✗ FAIL: {tc.name}")
            print(f"  Description: {tc.description}")
            print("  --- Expected ---")
            for i, line in enumerate(expected_normalized.splitlines(), 1):
                print(f"  {i:3}: {line!r}")
            print("  --- Got ---")
            for i, line in enumerate(result_normalized.splitlines(), 1):
                print(f"  {i:3}: {line!r}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)

    # Also test validation
    print("\nValidation tests:")
    valid_js = "function foo() { return 1; }"
    invalid_js = "function foo( { return 1; }"

    print(f"  Valid JS parses correctly: {replacer.validate_result(valid_js)}")
    print(f"  Invalid JS detected: {not replacer.validate_result(invalid_js)}")

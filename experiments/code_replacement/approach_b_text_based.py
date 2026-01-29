"""Approach B: Text-based code replacement using line numbers.

This approach:
1. Uses tree-sitter to find function boundaries (line numbers)
2. Does direct text replacement using those line numbers
3. Optionally runs a formatter to clean up the result

Pros:
- No external dependencies beyond tree-sitter
- Works entirely in Python
- Fast execution
- Simple implementation

Cons:
- May have issues with indentation in edge cases
- Doesn't understand AST structure during replacement
- Relies on accurate line numbers from tree-sitter
"""

from dataclasses import dataclass


@dataclass
class FunctionLocation:
    """Location of a function in source code."""

    name: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    start_byte: int
    end_byte: int


class TextBasedReplacer:
    """Replace functions using text-based line manipulation."""

    def replace_function(self, source: str, start_line: int, end_line: int, new_function: str) -> str:
        """Replace function at given line range with new function code.

        Args:
            source: Original source code
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed, inclusive)
            new_function: New function source code

        Returns:
            Modified source code

        """
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
                        # Add indentation
                        adjusted_new_lines.append(" " * indent_diff + line)
                    else:
                        # Remove indentation (careful not to remove too much)
                        current_indent = len(line) - len(line.lstrip())
                        remove_amount = min(current_indent, abs(indent_diff))
                        adjusted_new_lines.append(line[remove_amount:])
                else:
                    adjusted_new_lines.append(line)
            new_lines = adjusted_new_lines

        # Ensure new function ends with newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Build result: before + new function + after
        before = lines[: start_line - 1]
        after = lines[end_line:]

        result_lines = before + new_lines + after
        return "".join(result_lines)

    def replace_function_preserve_context(
        self,
        source: str,
        start_line: int,
        end_line: int,
        new_function: str,
        preserve_leading_empty_lines: bool = True,
        preserve_trailing_empty_lines: bool = True,
    ) -> str:
        """Replace function while preserving surrounding whitespace context.

        Args:
            source: Original source code
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed, inclusive)
            new_function: New function source code
            preserve_leading_empty_lines: Keep empty lines before function
            preserve_trailing_empty_lines: Keep empty lines after function

        Returns:
            Modified source code

        """
        lines = source.splitlines(keepends=True)

        # Handle case where source doesn't end with newline
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        # Find actual content boundaries (skip empty lines at start/end of function)
        actual_start = start_line
        actual_end = end_line

        # Prepare new function lines
        new_lines = new_function.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Auto-detect and adjust indentation
        if lines and start_line <= len(lines):
            original_first_line = lines[start_line - 1]
            original_indent = len(original_first_line) - len(original_first_line.lstrip())

            if new_lines:
                new_first_line = new_lines[0]
                new_indent = len(new_first_line) - len(new_first_line.lstrip())
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
                    new_lines = adjusted_new_lines

        # Build result
        before = lines[: actual_start - 1]
        after = lines[actual_end:]

        result_lines = before + new_lines + after
        return "".join(result_lines)


def replace_function_text_based(source: str, start_line: int, end_line: int, new_function: str) -> str:
    """Convenience function for text-based replacement.

    Args:
        source: Original source code
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive)
        new_function: New function source code

    Returns:
        Modified source code

    """
    replacer = TextBasedReplacer()
    return replacer.replace_function(source, start_line, end_line, new_function)


# Test the implementation
if __name__ == "__main__":
    from test_cases import get_test_cases

    replacer = TextBasedReplacer()

    print("=" * 60)
    print("Testing Approach B: Text-Based Replacement")
    print("=" * 60)

    passed = 0
    failed = 0

    for tc in get_test_cases():
        result = replacer.replace_function(tc.original_source, tc.start_line, tc.end_line, tc.new_function)

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

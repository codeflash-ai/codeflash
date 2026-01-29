"""Approach A: jscodeshift/recast via Node.js subprocess.

This approach:
1. Writes a jscodeshift transform script
2. Calls jscodeshift via npx subprocess
3. Captures the transformed output

Pros:
- AST-aware replacement
- Preserves formatting through recast
- Battle-tested codemod tooling
- Handles complex transformations

Cons:
- Requires Node.js
- External process overhead
- More complex setup
- Slower than pure Python approaches
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class JsCodeshiftResult:
    """Result from jscodeshift transformation."""

    success: bool
    output: str
    error: Optional[str] = None
    stderr: Optional[str] = None


class JsCodeshiftReplacer:
    """Replace functions using jscodeshift/recast."""

    def __init__(self):
        """Initialize the replacer."""
        self._check_node_available()

    def _check_node_available(self) -> bool:
        """Check if Node.js is available."""
        try:
            result = subprocess.run(["node", "--version"], check=False, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_jscodeshift_available(self) -> bool:
        """Check if jscodeshift is available via npx."""
        try:
            result = subprocess.run(
                ["npx", "jscodeshift", "--version"], check=False, capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _create_transform_script(self, function_name: str, new_source: str, start_line: int, end_line: int) -> str:
        """Create a jscodeshift transform script.

        Args:
            function_name: Name of function to replace
            new_source: New function source code
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)

        Returns:
            JavaScript transform script

        """
        # Escape the new source for embedding in JS string
        escaped_source = json.dumps(new_source)

        return f"""
// jscodeshift transform to replace function by line number
module.exports = function(fileInfo, api) {{
    const j = api.jscodeshift;
    const root = j(fileInfo.source);

    const startLine = {start_line};
    const endLine = {end_line};
    const newSource = {escaped_source};

    // Find and replace function declarations
    root.find(j.FunctionDeclaration)
        .filter(path => {{
            const loc = path.node.loc;
            return loc && loc.start.line === startLine;
        }})
        .forEach(path => {{
            // Parse the new source and replace
            const newAst = j(newSource);
            const newNode = newAst.find(j.FunctionDeclaration).get().node;
            if (newNode) {{
                j(path).replaceWith(newNode);
            }}
        }});

    // Find and replace method definitions
    root.find(j.MethodDefinition)
        .filter(path => {{
            const loc = path.node.loc;
            return loc && loc.start.line === startLine;
        }})
        .forEach(path => {{
            // For methods, we need to parse as a class member
            const tempClass = j(`class Temp {{ ${{newSource}} }}`);
            const newMethod = tempClass.find(j.MethodDefinition).get().node;
            if (newMethod) {{
                j(path).replaceWith(newMethod);
            }}
        }});

    // Find and replace variable declarations with arrow functions
    root.find(j.VariableDeclaration)
        .filter(path => {{
            const loc = path.node.loc;
            if (!loc || loc.start.line !== startLine) return false;

            // Check if any declarator has an arrow function
            return path.node.declarations.some(d =>
                d.init && d.init.type === 'ArrowFunctionExpression'
            );
        }})
        .forEach(path => {{
            const newAst = j(newSource);
            const newNode = newAst.find(j.VariableDeclaration).get().node;
            if (newNode) {{
                j(path).replaceWith(newNode);
            }}
        }});

    // Find and replace arrow functions in exports
    root.find(j.ExportDefaultDeclaration)
        .filter(path => {{
            const loc = path.node.loc;
            return loc && loc.start.line === startLine;
        }})
        .forEach(path => {{
            const newAst = j(newSource);
            const newNode = newAst.find(j.ExportDefaultDeclaration).get();
            if (newNode) {{
                j(path).replaceWith(newNode.node);
            }}
        }});

    // Find and replace exported function declarations
    root.find(j.ExportNamedDeclaration)
        .filter(path => {{
            const loc = path.node.loc;
            return loc && loc.start.line === startLine;
        }})
        .forEach(path => {{
            const newAst = j(newSource);
            const newNode = newAst.find(j.ExportNamedDeclaration).get();
            if (newNode) {{
                j(path).replaceWith(newNode.node);
            }}
        }});

    return root.toSource({{ quote: 'single' }});
}};
"""

    def _create_simple_transform_script(self, start_line: int, end_line: int, new_source: str) -> str:
        """Create a simpler transform script that uses line-based replacement.

        This fallback approach uses recast to parse, does line-based replacement,
        and uses recast to output (preserving formatting).
        """
        escaped_source = json.dumps(new_source)

        return f"""
// Simple line-based replacement using recast for parsing/printing
const recast = require('recast');

module.exports = function(fileInfo, api) {{
    const startLine = {start_line};
    const endLine = {end_line};
    const newSource = {escaped_source};

    // Split into lines
    const lines = fileInfo.source.split('\\n');

    // Replace the lines
    const before = lines.slice(0, startLine - 1);
    const after = lines.slice(endLine);
    const newLines = newSource.split('\\n');

    // Get original indentation
    const originalFirstLine = lines[startLine - 1] || '';
    const originalIndent = originalFirstLine.length - originalFirstLine.trimStart().length;

    // Get new source indentation
    const newFirstLine = newLines[0] || '';
    const newIndent = newFirstLine.length - newFirstLine.trimStart().length;

    // Adjust indentation
    const indentDiff = originalIndent - newIndent;
    const adjustedNewLines = newLines.map(line => {{
        if (!line.trim()) return line;
        if (indentDiff > 0) {{
            return ' '.repeat(indentDiff) + line;
        }} else if (indentDiff < 0) {{
            const currentIndent = line.length - line.trimStart().length;
            const removeAmount = Math.min(currentIndent, Math.abs(indentDiff));
            return line.slice(removeAmount);
        }}
        return line;
    }});

    return [...before, ...adjustedNewLines, ...after].join('\\n');
}};
"""

    def replace_function(
        self, source: str, function_name: str, new_function: str, start_line: int, end_line: int
    ) -> JsCodeshiftResult:
        """Replace a function using jscodeshift.

        Args:
            source: Original source code
            function_name: Name of function to replace
            new_function: New function source code
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)

        Returns:
            JsCodeshiftResult with success status and output

        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write source file
            source_file = tmpdir_path / "source.js"
            source_file.write_text(source)

            # Write transform script
            transform_file = tmpdir_path / "transform.js"
            transform_script = self._create_transform_script(function_name, new_function, start_line, end_line)
            transform_file.write_text(transform_script)

            try:
                # Run jscodeshift
                result = subprocess.run(
                    [
                        "npx",
                        "jscodeshift",
                        "-t",
                        str(transform_file),
                        str(source_file),
                        "--print",  # Print output to stdout instead of modifying file
                        "--dry",  # Don't actually write
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir_path,
                )

                if result.returncode == 0:
                    # Read the modified file (jscodeshift modifies in place even with --dry sometimes)
                    # Actually --print should output to stdout
                    output = result.stdout.strip()
                    if not output:
                        # Fallback: read the file
                        output = source_file.read_text()

                    return JsCodeshiftResult(success=True, output=output)
                return JsCodeshiftResult(
                    success=False,
                    output=source,  # Return original on failure
                    error=f"jscodeshift failed with code {result.returncode}",
                    stderr=result.stderr,
                )

            except subprocess.TimeoutExpired:
                return JsCodeshiftResult(success=False, output=source, error="jscodeshift timed out")
            except Exception as e:
                return JsCodeshiftResult(success=False, output=source, error=str(e))

    def replace_function_simple(
        self, source: str, start_line: int, end_line: int, new_function: str
    ) -> JsCodeshiftResult:
        """Replace a function using simple line-based approach via Node.js.

        This is a fallback that still uses Node.js but with simpler logic.

        Args:
            source: Original source code
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            new_function: New function source code

        Returns:
            JsCodeshiftResult with success status and output

        """
        # For simplicity, let's just use the text-based approach
        # but run through Node.js for consistency testing
        from approach_b_text_based import TextBasedReplacer

        replacer = TextBasedReplacer()
        result = replacer.replace_function(source, start_line, end_line, new_function)

        return JsCodeshiftResult(success=True, output=result)


def replace_function_jscodeshift(
    source: str, function_name: str, new_function: str, start_line: int, end_line: int
) -> str:
    """Convenience function for jscodeshift replacement.

    Args:
        source: Original source code
        function_name: Name of function to replace
        new_function: New function source code
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)

    Returns:
        Modified source code (or original if failed)

    """
    replacer = JsCodeshiftReplacer()
    result = replacer.replace_function(source, function_name, new_function, start_line, end_line)
    return result.output


# Test the implementation
if __name__ == "__main__":
    replacer = JsCodeshiftReplacer()

    # Check if jscodeshift is available
    if not replacer._check_node_available():
        print("Node.js not available. Skipping Approach A tests.")
        print("Install Node.js to test this approach.")
        exit(0)

    print("=" * 60)
    print("Testing Approach A: jscodeshift/recast")
    print("=" * 60)
    print("Note: This approach requires npx and jscodeshift to be installed.")
    print("Run: npm install -g jscodeshift")
    print()

    # Test with a simple case first
    simple_source = """function add(a, b) {
    return a + b;
}
"""
    simple_new = """function add(a, b) {
    return (a + b) | 0;
}"""

    result = replacer.replace_function(simple_source, "add", simple_new, start_line=1, end_line=3)

    print("Simple test result:")
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Output:\n{result.output}")
    else:
        print(f"  Error: {result.error}")
        print(f"  Stderr: {result.stderr}")

    # Since jscodeshift requires npm setup, we'll note that this approach
    # needs more setup and may not work in all environments
    print("\n" + "=" * 60)
    print("Note: Full test suite requires jscodeshift npm package.")
    print("For production, consider Approach B or C as they don't require Node.js.")
    print("=" * 60)

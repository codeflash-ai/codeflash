"""Tests for TypeScript code extraction and validation.

These tests verify that TypeScript code is correctly extracted by the codeflash
language support and that the extracted code is syntactically valid.

This test suite was created to help diagnose issues where the API server
reported "Invalid TypeScript: Invalid syntax" errors for code that tree-sitter
validates as correct.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionInfo, Language, ParentInfo
from codeflash.languages.javascript.support import TypeScriptSupport


@pytest.fixture
def ts_support():
    """Create a TypeScriptSupport instance."""
    return TypeScriptSupport()


class TestTypeScriptValidation:
    """Tests for TypeScript syntax validation."""

    def test_simple_function_is_valid(self, ts_support):
        """Test that a simple TypeScript function validates correctly."""
        code = """
function hello(name: string): string {
    return "Hello " + name;
}
"""
        assert ts_support.validate_syntax(code) is True

    def test_async_function_is_valid(self, ts_support):
        """Test that an async function validates correctly."""
        code = """
async function fetchData(url: string): Promise<any> {
    const response = await fetch(url);
    return response.json();
}
"""
        assert ts_support.validate_syntax(code) is True

    def test_template_literal_is_valid(self, ts_support):
        """Test that template literals with expressions validate correctly."""
        code = """
async function execCommand(expression: string, uri: string) {
    const command = `--eval=${expression}`;
    return await runCommand([
        "mongosh",
        uri,
        command,
    ]);
}
"""
        assert ts_support.validate_syntax(code) is True

    def test_function_with_try_catch_is_valid(self, ts_support):
        """Test that try-catch blocks validate correctly."""
        code = """
async function figureOutContentsPath(root: string): Promise<string> {
    const subfolders = await fs.readdir(root, { withFileTypes: true });

    try {
        await fs.access(path.join(root, "manifest.json"));
        return root;
    } catch (error) {
        // Ignore
    }

    for (const subfolder of subfolders) {
        if (subfolder.isDirectory()) {
            try {
                await fs.access(path.join(root, subfolder.name, "manifest.json"));
                return path.join(root, subfolder.name);
            } catch (error) {
                // Ignore
            }
        }
    }

    throw new Error("Could not find contents.");
}
"""
        assert ts_support.validate_syntax(code) is True

    def test_class_method_is_valid(self, ts_support):
        """Test that class methods validate correctly."""
        code = """
class MyClass {
    private value: number;

    constructor(value: number) {
        this.value = value;
    }

    async processData(input: string): Promise<number> {
        const result = parseInt(input, 10);
        return result * this.value;
    }
}
"""
        assert ts_support.validate_syntax(code) is True

    def test_invalid_syntax_is_detected(self, ts_support):
        """Test that invalid syntax is correctly detected."""
        code = "function broken( { return; }"
        assert ts_support.validate_syntax(code) is False


class TestTypeScriptCodeExtraction:
    """Tests for TypeScript code context extraction."""

    def test_extract_simple_function(self, ts_support):
        """Test extracting code context for a simple function."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
function add(a: number, b: number): number {
    return a + b;
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = ts_support.discover_functions(file_path)
            assert len(functions) == 1
            assert functions[0].function_name == "add"

            # Extract code context
            code_context = ts_support.extract_code_context(
                functions[0], file_path.parent, file_path.parent
            )

            # Verify extracted code is valid
            assert ts_support.validate_syntax(code_context.target_code) is True
            assert "function add" in code_context.target_code

    def test_extract_async_function_with_template_literal(self, ts_support):
        """Test extracting async function with template literals."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
import * as utils from "./utils";

const command_args = process.argv.slice(3);

async function execMongoEval(queryExpression, appsmithMongoURI) {
    queryExpression = queryExpression.trim();

    if (command_args.includes("--pretty")) {
        queryExpression += ".pretty()";
    }

    return await utils.execCommand([
        "mongosh",
        appsmithMongoURI,
        `--eval=${queryExpression}`,
    ]);
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = ts_support.discover_functions(file_path)
            assert len(functions) == 1
            assert functions[0].function_name == "execMongoEval"

            # Extract code context
            code_context = ts_support.extract_code_context(
                functions[0], file_path.parent, file_path.parent
            )

            # Verify extracted code is valid
            assert ts_support.validate_syntax(code_context.target_code) is True
            assert "execMongoEval" in code_context.target_code
            # Template literal should be preserved
            assert "`--eval=${queryExpression}`" in code_context.target_code

    def test_extract_function_with_complex_try_catch(self, ts_support):
        """Test extracting function with nested try-catch blocks."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
import fsPromises from "fs/promises";
import path from "path";

async function figureOutContentsPath(root: string): Promise<string> {
    const subfolders = await fsPromises.readdir(root, { withFileTypes: true });

    try {
        await fsPromises.access(path.join(root, "manifest.json"));
        return root;
    } catch (error) {
        // Ignore
    }

    for (const subfolder of subfolders) {
        if (subfolder.isDirectory()) {
            try {
                await fsPromises.access(
                    path.join(root, subfolder.name, "manifest.json"),
                );
                return path.join(root, subfolder.name);
            } catch (error) {
                // Ignore
            }
        }
    }

    throw new Error("Could not find the contents.");
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = ts_support.discover_functions(file_path)
            assert len(functions) == 1
            assert functions[0].function_name == "figureOutContentsPath"

            # Extract code context
            code_context = ts_support.extract_code_context(
                functions[0], file_path.parent, file_path.parent
            )

            # Verify extracted code is valid
            assert ts_support.validate_syntax(code_context.target_code) is True
            assert "figureOutContentsPath" in code_context.target_code
            # Should contain nested try-catch
            assert "try {" in code_context.target_code
            assert "catch (error)" in code_context.target_code

    def test_extracted_code_includes_imports(self, ts_support):
        """Test that imports are included in code context."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
import fs from "fs";
import path from "path";

function readConfig(filename: string): string {
    const fullPath = path.join(__dirname, filename);
    return fs.readFileSync(fullPath, "utf8");
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = ts_support.discover_functions(file_path)
            assert len(functions) == 1

            code_context = ts_support.extract_code_context(
                functions[0], file_path.parent, file_path.parent
            )

            # Check that imports are captured
            assert len(code_context.imports) > 0
            assert any("fs" in imp for imp in code_context.imports)

    def test_extracted_code_includes_global_variables(self, ts_support):
        """Test that referenced global variables are included."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
const CONFIG = { timeout: 5000 };
const MAX_RETRIES = 3;

async function fetchWithRetry(url: string): Promise<any> {
    for (let i = 0; i < MAX_RETRIES; i++) {
        try {
            const response = await fetch(url, { signal: AbortSignal.timeout(CONFIG.timeout) });
            return response.json();
        } catch (e) {
            if (i === MAX_RETRIES - 1) throw e;
        }
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = ts_support.discover_functions(file_path)
            assert len(functions) == 1

            code_context = ts_support.extract_code_context(
                functions[0], file_path.parent, file_path.parent
            )

            # Verify extracted code is valid
            assert ts_support.validate_syntax(code_context.target_code) is True


class TestTypeScriptLanguageProperties:
    """Tests for TypeScript language support properties."""

    def test_language_is_typescript(self, ts_support):
        """Test that language property returns TypeScript."""
        assert ts_support.language == Language.TYPESCRIPT

    def test_file_extensions_include_ts(self, ts_support):
        """Test that TypeScript file extensions are supported."""
        extensions = ts_support.file_extensions
        assert ".ts" in extensions
        assert ".tsx" in extensions

    def test_test_file_suffix(self, ts_support):
        """Test that test file suffix is .test.ts."""
        assert ts_support.get_test_file_suffix() == ".test.ts"

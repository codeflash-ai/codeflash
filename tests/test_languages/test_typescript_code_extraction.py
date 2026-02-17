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
export function add(a: number, b: number): number {
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

export async function execMongoEval(queryExpression, appsmithMongoURI) {
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

export async function figureOutContentsPath(root: string): Promise<string> {
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

export function readConfig(filename: string): string {
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

export async function fetchWithRetry(url: string): Promise<any> {
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


class TestSameClassHelperExtraction:
    """Tests for same-class helper method extraction.

    When a class method calls other methods from the same class, those helper
    methods should be included inside the class wrapper (not appended outside),
    because they may use class-specific syntax like 'private'.
    """

    def test_private_helper_method_inside_class_wrapper(self, ts_support):
        """Test that private helper methods are included inside the class wrapper."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            # Export the class and add return statements so discover_functions finds the methods
            f.write("""
export class EndpointGroup {
    private endpoints: any[] = [];

    constructor() {
        this.endpoints = [];
    }

    post(path: string, handler: Function): EndpointGroup {
        this.addEndpoint("POST", path, handler);
        return this;
    }

    private addEndpoint(method: string, path: string, handler: Function): void {
        this.endpoints.push({ method, path, handler });
        return;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            # Discover the 'post' method
            functions = ts_support.discover_functions(file_path)
            post_method = None
            for func in functions:
                if func.function_name == "post":
                    post_method = func
                    break

            assert post_method is not None, "post method should be discovered"

            # Extract code context
            code_context = ts_support.extract_code_context(
                post_method, file_path.parent, file_path.parent
            )

            # The extracted code should be syntactically valid
            assert ts_support.validate_syntax(code_context.target_code) is True, (
                f"Extracted code should be valid TypeScript:\n{code_context.target_code}"
            )

            # Both post and addEndpoint should be inside the class
            assert "class EndpointGroup" in code_context.target_code
            assert "post(" in code_context.target_code
            assert "private addEndpoint" in code_context.target_code

            # The private method should be inside the class, not outside
            # Check that addEndpoint appears BEFORE the closing brace of the class
            class_end_index = code_context.target_code.rfind("}")
            add_endpoint_index = code_context.target_code.find("addEndpoint")
            assert add_endpoint_index < class_end_index, (
                "addEndpoint should be inside the class wrapper"
            )

    def test_multiple_private_helpers_inside_class(self, ts_support):
        """Test that multiple private helpers are all included inside the class."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
export class Router {
    private routes: Map<string, any> = new Map();

    addRoute(path: string, handler: Function): boolean {
        const normalizedPath = this.normalizePath(path);
        this.validatePath(normalizedPath);
        this.routes.set(normalizedPath, handler);
        return true;
    }

    private normalizePath(path: string): string {
        return path.toLowerCase().trim();
    }

    private validatePath(path: string): boolean {
        if (!path.startsWith("/")) {
            throw new Error("Path must start with /");
        }
        return true;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            # Discover the 'addRoute' method
            functions = ts_support.discover_functions(file_path)
            add_route_method = None
            for func in functions:
                if func.function_name == "addRoute":
                    add_route_method = func
                    break

            assert add_route_method is not None

            code_context = ts_support.extract_code_context(
                add_route_method, file_path.parent, file_path.parent
            )

            # Should be valid TypeScript
            assert ts_support.validate_syntax(code_context.target_code) is True

            # All methods should be inside the class
            assert "private normalizePath" in code_context.target_code
            assert "private validatePath" in code_context.target_code

    def test_same_class_helpers_filtered_from_helper_list(self, ts_support):
        """Test that same-class helpers are not duplicated in the helpers list."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
export class Calculator {
    add(a: number, b: number): number {
        return this.compute(a, b, "+");
    }

    private compute(a: number, b: number, op: string): number {
        if (op === "+") return a + b;
        return 0;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = ts_support.discover_functions(file_path)
            add_method = None
            for func in functions:
                if func.function_name == "add":
                    add_method = func
                    break

            assert add_method is not None

            code_context = ts_support.extract_code_context(
                add_method, file_path.parent, file_path.parent
            )

            # 'compute' should be in target_code (inside class)
            assert "compute" in code_context.target_code

            # 'compute' should NOT be in helper_functions (would be duplicate)
            helper_names = [h.name for h in code_context.helper_functions]
            assert "compute" not in helper_names, (
                "Same-class helper 'compute' should not be in helper_functions list"
            )


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

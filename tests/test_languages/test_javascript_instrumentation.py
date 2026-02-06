"""Tests for JavaScript instrumentation (line profiling and tracing).

This module tests the line profiling and tracing instrumentation for JavaScript code.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.javascript.line_profiler import JavaScriptLineProfiler
from codeflash.languages.javascript.tracer import JavaScriptTracer
from codeflash.models.models import FunctionParent


def make_func(name: str, class_name: str | None = None) -> FunctionToOptimize:
    """Helper to create FunctionToOptimize for testing."""
    parents = [FunctionParent(name=class_name, type="ClassDef")] if class_name else []
    return FunctionToOptimize(
        function_name=name,
        file_path=Path("/test/file.js"),
        parents=parents,
        language="javascript",
    )


class TestJavaScriptLineProfiler:
    """Tests for JavaScript line profiling instrumentation."""

    def test_line_profiler_initialization(self):
        """Test line profiler can be initialized."""
        output_file = Path("/tmp/test_profile.json")
        profiler = JavaScriptLineProfiler(output_file)

        assert profiler.output_file == output_file
        assert profiler.profiler_var == "__codeflash_line_profiler__"

    def test_line_profiler_generates_init_code(self):
        """Test line profiler generates initialization code."""
        output_file = Path("/tmp/test_profile.json")
        profiler = JavaScriptLineProfiler(output_file)

        init_code = profiler._generate_profiler_init()

        assert profiler.profiler_var in init_code
        assert "hit" in init_code  # Changed from recordLine to hit
        assert "save" in init_code
        assert output_file.as_posix() in init_code

    def test_line_profiler_instruments_simple_function(self):
        """Test line profiler can instrument a simple function."""
        source = """
function add(a, b) {
    const result = a + b;
    return result;
}
"""

        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            file_path = Path(f.name)

        func_info = FunctionInfo(
            function_name="add", file_path=file_path, starting_line=2, ending_line=5, language="javascript"
        )

        output_file = Path("/tmp/test_profile.json")
        profiler = JavaScriptLineProfiler(output_file)

        instrumented = profiler.instrument_source(source, file_path, [func_info])

        # Check that profiler initialization is added
        assert profiler.profiler_var in instrumented
        assert "hit" in instrumented  # Changed from recordLine to hit

        # Clean up
        file_path.unlink()

    def test_line_profiler_parse_results_empty(self):
        """Test parsing results when file doesn't exist."""
        output_file = Path("/tmp/nonexistent_profile.json")
        results = JavaScriptLineProfiler.parse_results(output_file)

        assert results["timings"] == {}
        assert results["unit"] == 1e-9


class TestJavaScriptTracer:
    """Tests for JavaScript function tracing instrumentation."""

    def test_tracer_initialization(self):
        """Test tracer can be initialized."""
        output_db = Path("/tmp/test_traces.db")
        tracer = JavaScriptTracer(output_db)

        assert tracer.output_db == output_db
        assert tracer.tracer_var == "__codeflash_tracer__"

    def test_tracer_generates_init_code(self):
        """Test tracer generates initialization code."""
        output_db = Path("/tmp/test_traces.db")
        tracer = JavaScriptTracer(output_db)

        init_code = tracer._generate_tracer_init()

        assert tracer.tracer_var in init_code
        assert "serialize" in init_code
        assert "wrap" in init_code
        assert output_db.as_posix() in init_code

    def test_tracer_instruments_simple_function(self):
        """Test tracer can instrument a simple function."""
        source = """
function multiply(x, y) {
    return x * y;
}
"""

        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            file_path = Path(f.name)

        func_info = FunctionInfo(
            function_name="multiply", file_path=file_path, starting_line=2, ending_line=4, language="javascript"
        )

        output_db = Path("/tmp/test_traces.db")
        tracer = JavaScriptTracer(output_db)

        instrumented = tracer.instrument_source(source, file_path, [func_info])

        # Check that tracer initialization is added
        assert tracer.tracer_var in instrumented
        assert "wrap" in instrumented

        # Clean up
        file_path.unlink()

    def test_tracer_parse_results_empty(self):
        """Test parsing results when file doesn't exist."""
        output_db = Path("/tmp/nonexistent_traces.db")
        results = JavaScriptTracer.parse_results(output_db)

        assert results == []


class TestJavaScriptSupportInstrumentation:
    """Integration tests for JavaScript support instrumentation methods."""

    def test_javascript_support_instrument_for_behavior(self):
        """Test JavaScriptSupport.instrument_for_behavior method."""
        from codeflash.languages import get_language_support

        js_support = get_language_support(Language.JAVASCRIPT)

        source = """
function greet(name) {
    return "Hello, " + name;
}
"""

        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            file_path = Path(f.name)

        func_info = FunctionInfo(
            function_name="greet", file_path=file_path, starting_line=2, ending_line=4, language="javascript"
        )

        output_file = file_path.parent / ".codeflash" / "traces.db"
        instrumented = js_support.instrument_for_behavior(source, [func_info], output_file=output_file)

        assert "__codeflash_tracer__" in instrumented
        assert "wrap" in instrumented

        # Clean up
        file_path.unlink()

    def test_javascript_support_instrument_for_line_profiling(self):
        """Test JavaScriptSupport.instrument_source_for_line_profiler method."""
        from codeflash.languages import get_language_support

        js_support = get_language_support(Language.JAVASCRIPT)

        source = """
function square(n) {
    const result = n * n;
    return result;
}
"""

        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            file_path = Path(f.name)

        func_info = FunctionInfo(
            function_name="square", file_path=file_path, starting_line=2, ending_line=5, language="javascript"
        )

        output_file = file_path.parent / ".codeflash" / "line_profile.json"
        # instrument_source_for_line_profiler modifies the file directly
        result = js_support.instrument_source_for_line_profiler(
            func_info=func_info, line_profiler_output_file=output_file
        )

        assert result is True
        # Read the instrumented code from the file
        instrumented = file_path.read_text()
        assert "__codeflash_line_profiler__" in instrumented
        assert "hit" in instrumented  # Changed from recordLine to hit

        # Clean up
        file_path.unlink()


class TestImportStyleValidation:
    """Tests for import style validation and fixing."""

    def test_fix_named_import_for_default_export_commonjs(self):
        """Test fixing named require to default when source uses default export."""
        from codeflash.languages.javascript.instrument import validate_and_fix_import_style

        # Source file with default export (module.exports = function)
        source = """
module.exports = function decrypt(data) {
    return data;
}
"""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            source_path = Path(f.name)

        # Test code using wrong import style (named import for default export)
        test_code = f"""
const {{ decrypt }} = require('{source_path.as_posix()}');

test('decrypt works', () => {{
    expect(decrypt('hello')).toBe('hello');
}});
"""

        fixed_code = validate_and_fix_import_style(test_code, source_path, "decrypt")

        # Should be fixed to default import
        assert f"const decrypt = require('{source_path.as_posix()}')" in fixed_code
        assert "{ decrypt }" not in fixed_code

        # Clean up
        source_path.unlink()

    def test_fix_named_import_for_default_export_esm(self):
        """Test fixing named import to default when source uses default export."""
        from codeflash.languages.javascript.instrument import validate_and_fix_import_style

        # Source file with default export
        source = """
export default function decrypt(data) {
    return data;
}
"""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            source_path = Path(f.name)

        # Test code using wrong import style
        test_code = f"""
import {{ decrypt }} from '{source_path.as_posix()}';

test('decrypt works', () => {{
    expect(decrypt('hello')).toBe('hello');
}});
"""

        fixed_code = validate_and_fix_import_style(test_code, source_path, "decrypt")

        # Should be fixed to default import
        assert f"import decrypt from '{source_path.as_posix()}'" in fixed_code
        assert "{ decrypt }" not in fixed_code

        # Clean up
        source_path.unlink()

    def test_fix_default_import_for_named_export(self):
        """Test fixing default import to named when source uses named export."""
        from codeflash.languages.javascript.instrument import validate_and_fix_import_style

        # Source file with named export
        source = """
function decrypt(data) {
    return data;
}
module.exports = { decrypt };
"""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            source_path = Path(f.name)

        # Test code using wrong import style (default import for named export)
        test_code = f"""
const decrypt = require('{source_path.as_posix()}');

test('decrypt works', () => {{
    expect(decrypt('hello')).toBe('hello');
}});
"""

        fixed_code = validate_and_fix_import_style(test_code, source_path, "decrypt")

        # Should be fixed to named import
        assert f"const {{ decrypt }} = require('{source_path.as_posix()}')" in fixed_code

        # Clean up
        source_path.unlink()

    def test_no_change_when_import_correct(self):
        """Test that correct imports are not modified."""
        from codeflash.languages.javascript.instrument import validate_and_fix_import_style

        # Source file with named export
        source = """
export function decrypt(data) {
    return data;
}
"""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            source_path = Path(f.name)

        # Test code with correct import style
        test_code = f"""
import {{ decrypt }} from '{source_path.as_posix()}';

test('decrypt works', () => {{
    expect(decrypt('hello')).toBe('hello');
}});
"""

        fixed_code = validate_and_fix_import_style(test_code, source_path, "decrypt")

        # Should not be changed
        assert fixed_code == test_code

        # Clean up
        source_path.unlink()


class TestClassMethodInstrumentation:
    """Tests for class method instrumentation."""

    def test_instrument_method_call_on_instance(self):
        """Test that method calls on instances are correctly instrumented."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = """
const calc = new Calculator();
const result = calc.fibonacci(10);
console.log(result);
"""
        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="Calculator"), capture_func="capture"
        )

        # Should transform calc.fibonacci(10) to codeflash.capture(..., calc.fibonacci.bind(calc), 10)
        assert "codeflash.capture('Calculator.fibonacci'" in transformed
        assert "calc.fibonacci.bind(calc)" in transformed
        assert counter == 1

    def test_instrument_expect_with_method_call(self):
        """Test that expect() with method calls are correctly instrumented."""
        from codeflash.languages.javascript.instrument import transform_expect_calls

        code = """
test('fibonacci works', () => {
    const calc = new FibonacciCalculator();
    expect(calc.fibonacci(10)).toBe(55);
});
"""
        transformed, counter = transform_expect_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="FibonacciCalculator"), capture_func="capture"
        )

        # Should transform expect(calc.fibonacci(10)) to
        # expect(codeflash.capture(..., calc.fibonacci.bind(calc), 10))
        assert "codeflash.capture('FibonacciCalculator.fibonacci'" in transformed
        assert "calc.fibonacci.bind(calc)" in transformed
        assert ".toBe(55)" in transformed
        assert counter == 1

    def test_instrument_expect_with_method_removes_assertion(self):
        """Test that expect() with method calls are correctly instrumented with assertion removal."""
        from codeflash.languages.javascript.instrument import transform_expect_calls

        code = """
test('fibonacci works', () => {
    const calc = new FibonacciCalculator();
    expect(calc.fibonacci(10)).toBe(55);
});
"""
        transformed, counter = transform_expect_calls(
            code=code,
            function_to_optimize=make_func("fibonacci", class_name="FibonacciCalculator"),
            capture_func="capture",
            remove_assertions=True,
        )

        # Should remove expect wrapper and assertion
        assert "codeflash.capture('FibonacciCalculator.fibonacci'" in transformed
        assert "calc.fibonacci.bind(calc)" in transformed
        assert ".toBe(55)" not in transformed  # Assertion removed
        assert "expect(" not in transformed  # expect wrapper removed
        assert counter == 1

    def test_does_not_instrument_function_definition(self):
        """Test that function definitions are NOT transformed."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = """
class FibonacciCalculator {
    fibonacci(n) {
        if (n <= 1) return n;
        return this.fibonacci(n - 1) + this.fibonacci(n - 2);
    }
}
"""
        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="FibonacciCalculator"), capture_func="capture"
        )

        # The method definition should NOT be transformed
        # Only the recursive calls this.fibonacci(...) should potentially be transformed
        assert "fibonacci(n) {" in transformed  # Method definition unchanged
        assert counter >= 0  # May or may not transform the recursive calls

    def test_does_not_instrument_prototype_assignment(self):
        """Test that prototype assignments are NOT transformed."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = """
FibonacciCalculator.prototype.fibonacci = function(n) {
    if (n <= 1) return n;
    return this.fibonacci(n - 1) + this.fibonacci(n - 2);
};
"""
        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="FibonacciCalculator"), capture_func="capture"
        )

        # The prototype assignment should NOT be transformed
        # It should still have the original pattern
        assert "FibonacciCalculator.prototype.fibonacci = function(n)" in transformed

    def test_instrument_multiple_method_calls(self):
        """Test that multiple method calls are correctly instrumented."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = """
const calc = new Calculator();
const a = calc.fibonacci(5);
const b = calc.fibonacci(10);
const sum = a + b;
"""
        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="Calculator"), capture_func="capture"
        )

        # Should transform both calls
        assert transformed.count("codeflash.capture") == 2
        assert counter == 2

    def test_instrument_this_method_call(self):
        """Test that this.method() calls are correctly instrumented."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = """
class Wrapper {
    callFibonacci(n) {
        return this.fibonacci(n);
    }
}
"""
        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="Wrapper"), capture_func="capture"
        )

        # Should transform this.fibonacci(n)
        assert "codeflash.capture('Wrapper.fibonacci'" in transformed
        assert "this.fibonacci.bind(this)" in transformed
        assert counter == 1

    def test_full_instrumentation_produces_valid_syntax(self):
        """Test that full instrumentation produces syntactically valid JavaScript."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import Language
        from codeflash.languages.javascript.instrument import _instrument_js_test_code

        js_support = get_language_support(Language.JAVASCRIPT)

        test_code = """
const { FibonacciCalculator } = require('../fibonacci_class');

describe('FibonacciCalculator', () => {
    let calc;

    beforeEach(() => {
        calc = new FibonacciCalculator();
    });

    test('fibonacci returns correct values', () => {
        expect(calc.fibonacci(0)).toBe(0);
        expect(calc.fibonacci(1)).toBe(1);
        expect(calc.fibonacci(10)).toBe(55);
    });

    test('standalone call', () => {
        const result = calc.fibonacci(5);
        expect(result).toBe(5);
    });
});
"""
        instrumented = _instrument_js_test_code(
            code=test_code,
            function_to_optimize=make_func("fibonacci", class_name="FibonacciCalculator"),
            test_file_path="test.js",
            mode="behavior",
        )

        # Check that codeflash import was added
        assert "codeflash" in instrumented

        # Check that method calls were instrumented
        assert "codeflash.capture" in instrumented

        # Check that the instrumented code is valid JavaScript
        assert js_support.validate_syntax(instrumented) is True, f"Invalid syntax:\n{instrumented}"

    def test_instrumentation_preserves_test_structure(self):
        """Test that instrumentation preserves the test structure."""
        from codeflash.languages.javascript.instrument import _instrument_js_test_code

        test_code = """
const { Calculator } = require('../calculator');

describe('Calculator', () => {
    test('add works', () => {
        const calc = new Calculator();
        expect(calc.add(1, 2)).toBe(3);
    });
});
"""
        instrumented = _instrument_js_test_code(
            code=test_code, function_to_optimize=make_func("add", class_name="Calculator"), test_file_path="test.js", mode="behavior"
        )

        # describe and test structure should be preserved
        assert "describe('Calculator'" in instrumented
        assert "test('add works'" in instrumented
        assert "beforeEach" in instrumented or "beforeEach" not in test_code  # Only if it was there

        # Method call should be instrumented
        assert "codeflash.capture('Calculator.add'" in instrumented
        assert "calc.add.bind(calc)" in instrumented

    def test_instrumentation_with_async_methods(self):
        """Test instrumentation with async method calls."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = """
const api = new ApiClient();
const data = await api.fetchData('http://example.com');
console.log(data);
"""
        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fetchData", class_name="ApiClient"), capture_func="capture"
        )

        # Should preserve await
        assert "await codeflash.capture" in transformed
        assert "api.fetchData.bind(api)" in transformed
        assert counter == 1


class TestInstrumentationFullStringEquality:
    """Tests with full string equality for precise verification."""

    def test_standalone_method_call_exact_output(self):
        """Test exact output of standalone method call instrumentation."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = "    calc.fibonacci(10);"

        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="Calculator"), capture_func="capture"
        )

        expected = "    codeflash.capture('Calculator.fibonacci', '1', calc.fibonacci.bind(calc), 10);"
        assert transformed == expected, f"Expected:\n{expected}\nGot:\n{transformed}"
        assert counter == 1

    def test_expect_method_call_exact_output(self):
        """Test exact output of expect() method call instrumentation."""
        from codeflash.languages.javascript.instrument import transform_expect_calls

        code = "    expect(calc.fibonacci(10)).toBe(55);"

        transformed, counter = transform_expect_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="Calculator"), capture_func="capture"
        )

        expected = "    expect(codeflash.capture('Calculator.fibonacci', '1', calc.fibonacci.bind(calc), 10)).toBe(55);"
        assert transformed == expected, f"Expected:\n{expected}\nGot:\n{transformed}"
        assert counter == 1

    def test_expect_method_call_remove_assertions_exact_output(self):
        """Test exact output when removing assertions."""
        from codeflash.languages.javascript.instrument import transform_expect_calls

        code = "    expect(calc.fibonacci(10)).toBe(55);"

        transformed, counter = transform_expect_calls(
            code=code,
            function_to_optimize=make_func("fibonacci", class_name="Calculator"),
            capture_func="capture",
            remove_assertions=True,
        )

        expected = "    codeflash.capture('Calculator.fibonacci', '1', calc.fibonacci.bind(calc), 10);"
        assert transformed == expected, f"Expected:\n{expected}\nGot:\n{transformed}"
        assert counter == 1

    def test_standalone_function_call_no_object_prefix(self):
        """Test that standalone function calls (no object) work correctly."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = "    fibonacci(10);"

        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci"), capture_func="capture"
        )

        expected = "    codeflash.capture('fibonacci', '1', fibonacci, 10);"
        assert transformed == expected, f"Expected:\n{expected}\nGot:\n{transformed}"
        assert counter == 1

    def test_this_method_call_exact_output(self):
        """Test exact output for this.method() call."""
        from codeflash.languages.javascript.instrument import transform_standalone_calls

        code = "        return this.fibonacci(n - 1);"

        transformed, counter = transform_standalone_calls(
            code=code, function_to_optimize=make_func("fibonacci", class_name="Class"), capture_func="capture"
        )

        expected = "        return codeflash.capture('Class.fibonacci', '1', this.fibonacci.bind(this), n - 1);"
        assert transformed == expected, f"Expected:\n{expected}\nGot:\n{transformed}"
        assert counter == 1


class TestFixImportsInsideTestBlocks:
    """Tests for fix_imports_inside_test_blocks function."""

    def test_fix_named_import_inside_test_block(self):
        """Test fixing named import inside test function."""
        from codeflash.languages.javascript.instrument import fix_imports_inside_test_blocks

        code = """
test('should work', () => {
    const mock = jest.fn();
    import { foo } from '../src/module';
    expect(foo()).toBe(true);
});
"""
        fixed = fix_imports_inside_test_blocks(code)

        assert "const { foo } = require('../src/module');" in fixed
        assert "import { foo }" not in fixed

    def test_fix_default_import_inside_test_block(self):
        """Test fixing default import inside test function."""
        from codeflash.languages.javascript.instrument import fix_imports_inside_test_blocks

        code = """
test('should work', () => {
    env.isTest.mockReturnValue(false);
    import queuesModule from '../src/queue/queue';
    expect(queuesModule).toBeDefined();
});
"""
        fixed = fix_imports_inside_test_blocks(code)

        assert "const queuesModule = require('../src/queue/queue');" in fixed
        assert "import queuesModule from" not in fixed

    def test_fix_namespace_import_inside_test_block(self):
        """Test fixing namespace import inside test function."""
        from codeflash.languages.javascript.instrument import fix_imports_inside_test_blocks

        code = """
test('should work', () => {
    import * as utils from '../src/utils';
    expect(utils.foo()).toBe(true);
});
"""
        fixed = fix_imports_inside_test_blocks(code)

        assert "const utils = require('../src/utils');" in fixed
        assert "import * as utils" not in fixed

    def test_preserve_top_level_imports(self):
        """Test that top-level imports are not modified."""
        from codeflash.languages.javascript.instrument import fix_imports_inside_test_blocks

        code = """
import { jest, describe, test, expect } from '@jest/globals';
import { foo } from '../src/module';

describe('test suite', () => {
    test('should work', () => {
        expect(foo()).toBe(true);
    });
});
"""
        fixed = fix_imports_inside_test_blocks(code)

        # Top-level imports should remain unchanged
        assert "import { jest, describe, test, expect } from '@jest/globals';" in fixed
        assert "import { foo } from '../src/module';" in fixed

    def test_empty_code(self):
        """Test handling empty code."""
        from codeflash.languages.javascript.instrument import fix_imports_inside_test_blocks

        assert fix_imports_inside_test_blocks("") == ""
        assert fix_imports_inside_test_blocks("   ") == "   "


class TestFixJestMockPaths:
    """Tests for fix_jest_mock_paths function."""

    def test_fix_mock_path_when_source_relative(self):
        """Test fixing mock path that's relative to source file."""
        from codeflash.languages.javascript.instrument import fix_jest_mock_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            src_dir = Path(tmpdir) / "src" / "queue"
            tests_dir = Path(tmpdir) / "tests"
            env_file = Path(tmpdir) / "src" / "environment.ts"

            src_dir.mkdir(parents=True)
            tests_dir.mkdir(parents=True)
            env_file.parent.mkdir(parents=True, exist_ok=True)
            env_file.write_text("export const env = {};")

            source_file = src_dir / "queue.ts"
            source_file.write_text("import env from '../environment';")

            test_file = tests_dir / "test_queue.test.ts"

            # Test code with incorrect mock path (relative to source, not test)
            test_code = """
import { jest, describe, test, expect } from '@jest/globals';
jest.mock('../environment');
jest.mock('../redis/utils');

describe('queue', () => {
    test('works', () => {});
});
"""
            fixed = fix_jest_mock_paths(test_code, test_file, source_file, tests_dir)

            # Should fix the path to be relative to the test file
            assert "jest.mock('../src/environment')" in fixed

    def test_preserve_valid_mock_path(self):
        """Test that valid mock paths are not modified."""
        from codeflash.languages.javascript.instrument import fix_jest_mock_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            src_dir = Path(tmpdir) / "src"
            tests_dir = Path(tmpdir) / "tests"

            src_dir.mkdir(parents=True)
            tests_dir.mkdir(parents=True)

            # Create the file being mocked at the correct location
            mock_file = src_dir / "utils.ts"
            mock_file.write_text("export const utils = {};")

            source_file = src_dir / "main.ts"
            source_file.write_text("")
            test_file = tests_dir / "test_main.test.ts"

            # Test code with correct mock path (valid from test location)
            test_code = """
jest.mock('../src/utils');

describe('main', () => {
    test('works', () => {});
});
"""
            fixed = fix_jest_mock_paths(test_code, test_file, source_file, tests_dir)

            # Should keep the path unchanged since it's valid
            assert "jest.mock('../src/utils')" in fixed

    def test_fix_doMock_path(self):
        """Test fixing jest.doMock path."""
        from codeflash.languages.javascript.instrument import fix_jest_mock_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure: src/queue/queue.ts imports ../environment (-> src/environment.ts)
            src_dir = Path(tmpdir) / "src"
            queue_dir = src_dir / "queue"
            tests_dir = Path(tmpdir) / "tests"
            env_file = src_dir / "environment.ts"

            queue_dir.mkdir(parents=True)
            tests_dir.mkdir(parents=True)
            env_file.write_text("export const env = {};")

            source_file = queue_dir / "queue.ts"
            source_file.write_text("")
            test_file = tests_dir / "test_queue.test.ts"

            # From src/queue/queue.ts, ../environment resolves to src/environment.ts
            # Test file is at tests/test_queue.test.ts
            # So the correct mock path from test should be ../src/environment
            test_code = """
jest.doMock('../environment', () => ({ isTest: jest.fn() }));
"""
            fixed = fix_jest_mock_paths(test_code, test_file, source_file, tests_dir)

            # Should fix the doMock path
            assert "jest.doMock('../src/environment'" in fixed

    def test_empty_code(self):
        """Test handling empty code."""
        from codeflash.languages.javascript.instrument import fix_jest_mock_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            source_file = Path(tmpdir) / "src" / "main.ts"
            test_file = tests_dir / "test.ts"

            assert fix_jest_mock_paths("", test_file, source_file, tests_dir) == ""
            assert fix_jest_mock_paths("   ", test_file, source_file, tests_dir) == "   "
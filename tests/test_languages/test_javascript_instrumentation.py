"""
Tests for JavaScript instrumentation (line profiling and tracing).

This module tests the line profiling and tracing instrumentation for JavaScript code.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.javascript.line_profiler import JavaScriptLineProfiler
from codeflash.languages.javascript.tracer import JavaScriptTracer


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
            name="add",
            file_path=file_path,
            start_line=2,
            end_line=5,
            language=Language.JAVASCRIPT,
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
            name="multiply",
            file_path=file_path,
            start_line=2,
            end_line=4,
            language=Language.JAVASCRIPT,
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
            name="greet",
            file_path=file_path,
            start_line=2,
            end_line=4,
            language=Language.JAVASCRIPT,
        )

        output_file = file_path.parent / ".codeflash" / "traces.db"
        instrumented = js_support.instrument_for_behavior(
            source, [func_info], output_file=output_file
        )

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
            name="square",
            file_path=file_path,
            start_line=2,
            end_line=5,
            language=Language.JAVASCRIPT,
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
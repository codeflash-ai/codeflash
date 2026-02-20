"""Tests for Java test discovery with type-resolved method call matching."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.languages.java.parser import get_java_analyzer
from codeflash.languages.java.test_discovery import (
    _build_field_type_map,
    _build_local_type_map,
    _build_static_import_map,
    _extract_imports,
    _match_test_to_functions,
    _resolve_method_calls_in_range,
    discover_all_tests,
    discover_tests,
    find_tests_for_function,
    get_test_class_for_source_class,
    is_test_file,
)
from codeflash.models.function_types import FunctionParent, FunctionToOptimize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_func(name: str, class_name: str, file_path: Path | None = None) -> FunctionToOptimize:
    """Create a minimal FunctionToOptimize for testing."""
    return FunctionToOptimize(
        function_name=name,
        file_path=file_path or Path("src/main/java/com/example/Dummy.java"),
        parents=[FunctionParent(name=class_name, type="ClassDef")],
        starting_line=1,
        ending_line=10,
        is_method=True,
        language="java",
    )


def make_test_method(
    name: str, class_name: str, starting_line: int, ending_line: int, file_path: Path | None = None,
) -> FunctionToOptimize:
    return FunctionToOptimize(
        function_name=name,
        file_path=file_path or Path("src/test/java/com/example/DummyTest.java"),
        parents=[FunctionParent(name=class_name, type="ClassDef")],
        starting_line=starting_line,
        ending_line=ending_line,
        is_method=True,
        language="java",
    )


@pytest.fixture
def analyzer():
    return get_java_analyzer()


# ===================================================================
# _build_local_type_map
# ===================================================================


class TestBuildLocalTypeMap:
    def test_basic_declaration(self, analyzer):
        source = """\
class Foo {
    void test() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 5, analyzer)
        assert type_map == {"calc": "Calculator"}

    def test_multiple_declarations(self, analyzer):
        source = """\
class Foo {
    void test() {
        Calculator calc = new Calculator();
        Buffer buf = new Buffer(10);
        calc.add(1, 2);
        buf.read();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 7, analyzer)
        assert type_map == {"calc": "Calculator", "buf": "Buffer"}

    def test_generic_type_stripped(self, analyzer):
        source = """\
class Foo {
    void test() {
        List<String> items = new ArrayList<>();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 4, analyzer)
        assert type_map == {"items": "List"}

    def test_var_inferred_from_constructor(self, analyzer):
        source = """\
class Foo {
    void test() {
        var calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 5, analyzer)
        assert type_map == {"calc": "Calculator"}

    def test_var_not_inferred_from_method_call(self, analyzer):
        source = """\
class Foo {
    void test() {
        var result = getResult();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 4, analyzer)
        assert type_map == {}

    def test_declaration_outside_range_excluded(self, analyzer):
        source = """\
class Foo {
    void setup() {
        Calculator calc = new Calculator();
    }
    void test() {
        Buffer buf = new Buffer(10);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        # Only the test() method range (lines 5-7)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 5, 7, analyzer)
        assert "calc" not in type_map
        assert type_map == {"buf": "Buffer"}


# ===================================================================
# _build_field_type_map
# ===================================================================


class TestBuildFieldTypeMap:
    def test_basic_field(self, analyzer):
        source = """\
class CalculatorTest {
    private Calculator calculator;

    void testAdd() {
        calculator.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_field_type_map(tree.root_node, source_bytes, analyzer, "CalculatorTest")
        assert type_map == {"calculator": "Calculator"}

    def test_multiple_fields(self, analyzer):
        source = """\
class CalculatorTest {
    private Calculator calculator;
    private Buffer buffer;

    void testAdd() {}
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_field_type_map(tree.root_node, source_bytes, analyzer, "CalculatorTest")
        assert type_map == {"calculator": "Calculator", "buffer": "Buffer"}

    def test_wrong_class_excluded(self, analyzer):
        source = """\
class OtherTest {
    private Calculator calculator;
}
class CalculatorTest {
    private Buffer buffer;
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_field_type_map(tree.root_node, source_bytes, analyzer, "CalculatorTest")
        assert type_map == {"buffer": "Buffer"}

    def test_generic_field_stripped(self, analyzer):
        source = """\
class MyTest {
    private List<String> items;
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_field_type_map(tree.root_node, source_bytes, analyzer, "MyTest")
        assert type_map == {"items": "List"}


# ===================================================================
# _build_static_import_map
# ===================================================================


class TestBuildStaticImportMap:
    def test_specific_static_import(self, analyzer):
        source = """\
import static com.example.Calculator.add;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        static_map = _build_static_import_map(tree.root_node, source_bytes, analyzer)
        assert static_map == {"add": "Calculator"}

    def test_multiple_static_imports(self, analyzer):
        source = """\
import static com.example.Calculator.add;
import static com.example.Calculator.subtract;
import static com.example.MathUtils.square;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        static_map = _build_static_import_map(tree.root_node, source_bytes, analyzer)
        assert static_map == {"add": "Calculator", "subtract": "Calculator", "square": "MathUtils"}

    def test_wildcard_static_import_excluded(self, analyzer):
        source = """\
import static com.example.Calculator.*;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        static_map = _build_static_import_map(tree.root_node, source_bytes, analyzer)
        assert static_map == {}

    def test_regular_import_excluded(self, analyzer):
        source = """\
import com.example.Calculator;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        static_map = _build_static_import_map(tree.root_node, source_bytes, analyzer)
        assert static_map == {}


# ===================================================================
# _extract_imports
# ===================================================================


class TestExtractImports:
    def test_regular_import(self, analyzer):
        source = """\
import com.example.Calculator;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)
        assert imports == {"Calculator"}

    def test_static_import_extracts_class(self, analyzer):
        source = """\
import static com.example.Calculator.add;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)
        assert imports == {"Calculator"}

    def test_wildcard_regular_import_excluded(self, analyzer):
        source = """\
import com.example.*;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)
        assert imports == set()

    def test_static_wildcard_extracts_class(self, analyzer):
        source = """\
import static com.example.Calculator.*;
class Foo {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)
        assert imports == {"Calculator"}


# ===================================================================
# _resolve_method_calls_in_range
# ===================================================================


class TestResolveMethodCallsInRange:
    def test_instance_method_via_local_variable(self, analyzer):
        source = """\
class FooTest {
    void testAdd() {
        Calculator calc = new Calculator();
        int result = calc.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 5, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved

    def test_static_method_call(self, analyzer):
        source = """\
class FooTest {
    void testAdd() {
        int result = Calculator.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        assert "Calculator.add" in resolved

    def test_static_import_call(self, analyzer):
        source = """\
import static com.example.Calculator.add;
class FooTest {
    void testAdd() {
        int result = add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        static_map = {"add": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 3, 5, analyzer, {}, static_map,
        )
        assert "Calculator.add" in resolved

    def test_new_expression_method_call(self, analyzer):
        source = """\
class FooTest {
    void testAdd() {
        int result = new Calculator().add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        assert "Calculator.add" in resolved

    def test_field_access_via_this(self, analyzer):
        source = """\
class FooTest {
    Calculator calculator;
    void testAdd() {
        this.calculator.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calculator": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 3, 5, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved

    def test_unresolvable_call_not_included(self, analyzer):
        source = """\
class FooTest {
    void testSomething() {
        someUnknown.doStuff();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        # someUnknown is lowercase and not in type_map → not resolved
        assert len(resolved) == 0

    def test_assertion_methods_not_resolved_without_import(self, analyzer):
        source = """\
class FooTest {
    void testAdd() {
        assertEquals(3, result);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        # assertEquals has no receiver, and not in static_import_map
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        assert len(resolved) == 0

    def test_multiple_different_receivers(self, analyzer):
        source = """\
class FooTest {
    void testBoth() {
        Calculator calc = new Calculator();
        Buffer buf = new Buffer(10);
        calc.add(1, 2);
        buf.read();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator", "buf": "Buffer"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 7, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved
        assert "Buffer.read" in resolved

    def test_calls_outside_range_excluded(self, analyzer):
        source = """\
class FooTest {
    void setUp() {
        Calculator calc = new Calculator();
        calc.init();
    }
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 6, 9, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved
        assert "Calculator.init" not in resolved


# ===================================================================
# _match_test_to_functions (the core matching function)
# ===================================================================


class TestMatchTestToFunctions:
    def test_basic_instance_method_match(self, analyzer):
        test_source = """\
import com.example.Calculator;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        int result = calc.add(1, 2);
        assertEquals(3, result);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 5, 10)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]

    def test_static_method_match(self, analyzer):
        test_source = """\
import com.example.MathUtils;
import org.junit.jupiter.api.Test;

class MathUtilsTest {
    @Test
    void testSquare() {
        int result = MathUtils.square(5);
        assertEquals(25, result);
    }
}
"""
        func_map = {"MathUtils.square": make_func("square", "MathUtils")}
        test_method = make_test_method("testSquare", "MathUtilsTest", 5, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["MathUtils.square"]

    def test_static_import_match(self, analyzer):
        test_source = """\
import static com.example.MathUtils.square;
import org.junit.jupiter.api.Test;

class MathUtilsTest {
    @Test
    void testSquare() {
        int result = square(5);
        assertEquals(25, result);
    }
}
"""
        func_map = {"MathUtils.square": make_func("square", "MathUtils")}
        test_method = make_test_method("testSquare", "MathUtilsTest", 5, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["MathUtils.square"]

    def test_field_variable_match(self, analyzer):
        test_source = """\
import com.example.Calculator;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    private Calculator calculator;

    @Test
    void testAdd() {
        int result = calculator.add(1, 2);
        assertEquals(3, result);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 7, 11)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]

    def test_no_false_positive_from_import_only(self, analyzer):
        """Importing a class should NOT match all its methods if they're not called."""
        test_source = """\
import com.example.Calculator;
import org.junit.jupiter.api.Test;

class SomeTest {
    @Test
    void testSomethingElse() {
        int x = 42;
        assertEquals(42, x);
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "Calculator.subtract": make_func("subtract", "Calculator"),
        }
        test_method = make_test_method("testSomethingElse", "SomeTest", 5, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == []

    def test_no_false_positive_from_test_class_naming(self, analyzer):
        """CalculatorTest should NOT match all Calculator methods automatically."""
        test_source = """\
import com.example.Calculator;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "Calculator.subtract": make_func("subtract", "Calculator"),
            "Calculator.multiply": make_func("multiply", "Calculator"),
        }
        test_method = make_test_method("testAdd", "CalculatorTest", 5, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        # Should only match add, not subtract or multiply
        assert matched == ["Calculator.add"]

    def test_multiple_methods_called_in_single_test(self, analyzer):
        test_source = """\
import com.example.Calculator;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testOperations() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
        calc.subtract(5, 3);
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "Calculator.subtract": make_func("subtract", "Calculator"),
            "Calculator.multiply": make_func("multiply", "Calculator"),
        }
        test_method = make_test_method("testOperations", "CalculatorTest", 5, 10)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert "Calculator.add" in matched
        assert "Calculator.subtract" in matched
        assert "Calculator.multiply" not in matched

    def test_different_classes_in_one_test(self, analyzer):
        test_source = """\
import com.example.Calculator;
import com.example.Buffer;
import org.junit.jupiter.api.Test;

class IntegrationTest {
    @Test
    void testFlow() {
        Calculator calc = new Calculator();
        Buffer buf = new Buffer(10);
        calc.add(1, 2);
        buf.read();
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "Buffer.read": make_func("read", "Buffer"),
            "Buffer.write": make_func("write", "Buffer"),
        }
        test_method = make_test_method("testFlow", "IntegrationTest", 6, 12)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert "Calculator.add" in matched
        assert "Buffer.read" in matched
        assert "Buffer.write" not in matched

    def test_new_expression_inline(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        int result = new Calculator().add(1, 2);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 4, 7)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]

    def test_var_type_inference(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        var calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 4, 8)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]

    def test_method_not_in_function_map_not_matched(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
        calc.toString();
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 4, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        # toString is resolved to Calculator.toString but it's not in function_map
        assert matched == ["Calculator.add"]

    def test_this_field_access(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    private Calculator calculator;

    @Test
    void testAdd() {
        this.calculator.add(1, 2);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 6, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]

    def test_empty_test_method(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testNothing() {
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testNothing", "CalculatorTest", 4, 6)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == []

    def test_unresolvable_receiver_not_matched(self, analyzer):
        """Method calls on unresolvable receivers should produce no match."""
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        getCalculator().add(1, 2);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 4, 7)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        # getCalculator() returns unknown type → can't resolve → no match
        assert matched == []

    def test_local_variable_shadows_field(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    private Buffer calculator;

    @Test
    void testAdd() {
        Calculator calculator = new Calculator();
        calculator.add(1, 2);
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "Buffer.add": make_func("add", "Buffer"),
        }
        test_method = make_test_method("testAdd", "CalculatorTest", 6, 10)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        # Local Calculator declaration shadows the Buffer field
        assert "Calculator.add" in matched
        assert "Buffer.add" not in matched


# ===================================================================
# discover_tests (integration test with real file I/O)
# ===================================================================


class TestDiscoverTests:
    def test_basic_integration(self, tmp_path, analyzer):
        """Full pipeline: write test file to disk, discover tests, verify mapping."""
        test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
        test_dir.mkdir(parents=True)

        test_file = test_dir / "CalculatorTest.java"
        test_file.write_text("""\
package com.example;

import com.example.Calculator;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        int result = calc.add(1, 2);
        assertEquals(3, result);
    }

    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        int result = calc.subtract(5, 3);
        assertEquals(2, result);
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("add", "Calculator"),
            make_func("subtract", "Calculator"),
            make_func("multiply", "Calculator"),
        ]

        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "Calculator.add" in result
        assert len(result["Calculator.add"]) == 1
        assert result["Calculator.add"][0].test_name == "testAdd"

        assert "Calculator.subtract" in result
        assert len(result["Calculator.subtract"]) == 1
        assert result["Calculator.subtract"][0].test_name == "testSubtract"

        # multiply is never called → should not appear
        assert "Calculator.multiply" not in result

    def test_static_method_integration(self, tmp_path, analyzer):
        test_dir = tmp_path / "src" / "test" / "java"
        test_dir.mkdir(parents=True)

        test_file = test_dir / "MathUtilsTest.java"
        test_file.write_text("""\
package com.example;

import com.example.MathUtils;
import org.junit.jupiter.api.Test;

class MathUtilsTest {
    @Test
    void testSquare() {
        int result = MathUtils.square(5);
    }

    @Test
    void testAbs() {
        int result = MathUtils.abs(-3);
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("square", "MathUtils"),
            make_func("abs", "MathUtils"),
            make_func("pow", "MathUtils"),
        ]

        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "MathUtils.square" in result
        assert result["MathUtils.square"][0].test_name == "testSquare"

        assert "MathUtils.abs" in result
        assert result["MathUtils.abs"][0].test_name == "testAbs"

        assert "MathUtils.pow" not in result

    def test_field_based_integration(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        test_file = test_dir / "CalculatorTest.java"
        test_file.write_text("""\
package com.example;

import com.example.Calculator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

class CalculatorTest {
    private Calculator calculator;

    @BeforeEach
    void setUp() {
        calculator = new Calculator();
    }

    @Test
    void testAdd() {
        calculator.add(1, 2);
    }

    @Test
    void testMultiply() {
        calculator.multiply(3, 4);
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("add", "Calculator"),
            make_func("subtract", "Calculator"),
            make_func("multiply", "Calculator"),
        ]

        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "Calculator.add" in result
        assert result["Calculator.add"][0].test_name == "testAdd"

        assert "Calculator.multiply" in result
        assert result["Calculator.multiply"][0].test_name == "testMultiply"

        # subtract is never called
        assert "Calculator.subtract" not in result


# ===================================================================
# Additional _build_local_type_map tests
# ===================================================================


class TestBuildLocalTypeMapExtended:
    def test_enhanced_for_loop_variable(self, analyzer):
        source = """\
class Foo {
    void test() {
        for (Calculator calc : calculators) {
            calc.add(1, 2);
        }
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 6, analyzer)
        assert type_map == {"calc": "Calculator"}

    def test_declaration_without_initializer(self, analyzer):
        source = """\
class Foo {
    void test() {
        Calculator calc;
        calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 6, analyzer)
        assert type_map == {"calc": "Calculator"}

    def test_var_with_generic_constructor(self, analyzer):
        source = """\
class Foo {
    void test() {
        var list = new ArrayList<String>();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 4, analyzer)
        assert type_map == {"list": "ArrayList"}

    def test_multiple_declarators_same_line(self, analyzer):
        source = """\
class Foo {
    void test() {
        int a = 1, b = 2;
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 4, analyzer)
        assert type_map == {"a": "int", "b": "int"}

    def test_nested_generic_type(self, analyzer):
        source = """\
class Foo {
    void test() {
        Map<String, List<Integer>> map = new HashMap<>();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 4, analyzer)
        assert type_map == {"map": "Map"}

    def test_interface_typed_variable(self, analyzer):
        source = """\
class Foo {
    void test() {
        Runnable task = new MyTask();
        task.run();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_local_type_map(tree.root_node, source_bytes, 2, 5, analyzer)
        assert type_map == {"task": "Runnable"}


# ===================================================================
# Additional _build_field_type_map tests
# ===================================================================


class TestBuildFieldTypeMapExtended:
    def test_field_with_initializer(self, analyzer):
        source = """\
class MyTest {
    private Calculator calc = new Calculator();
    void testAdd() {}
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_field_type_map(tree.root_node, source_bytes, analyzer, "MyTest")
        assert type_map == {"calc": "Calculator"}

    def test_static_field(self, analyzer):
        source = """\
class MyTest {
    private static Calculator shared = new Calculator();
    void testAdd() {}
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_field_type_map(tree.root_node, source_bytes, analyzer, "MyTest")
        assert type_map == {"shared": "Calculator"}

    def test_null_class_name(self, analyzer):
        source = """\
class MyTest {
    private Calculator calc;
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = _build_field_type_map(tree.root_node, source_bytes, analyzer, None)
        assert type_map == {}


# ===================================================================
# Additional _resolve_method_calls_in_range tests
# ===================================================================


class TestResolveMethodCallsExtended:
    def test_cast_expression(self, analyzer):
        source = """\
class FooTest {
    void testCast() {
        Object obj = new Calculator();
        ((Calculator) obj).add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 5, analyzer, {"obj": "Object"}, {},
        )
        assert "Calculator.add" in resolved

    def test_method_call_inside_if(self, analyzer):
        source = """\
class FooTest {
    void testConditional() {
        Calculator calc = new Calculator();
        if (true) {
            calc.add(1, 2);
        }
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 7, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved

    def test_method_call_inside_try_catch(self, analyzer):
        source = """\
class FooTest {
    void testTryCatch() {
        Calculator calc = new Calculator();
        try {
            calc.add(1, 2);
        } catch (Exception e) {
            calc.reset();
        }
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 9, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved
        assert "Calculator.reset" in resolved

    def test_method_call_inside_loop(self, analyzer):
        source = """\
class FooTest {
    void testLoop() {
        Calculator calc = new Calculator();
        for (int i = 0; i < 10; i++) {
            calc.add(i, 1);
        }
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 7, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved

    def test_method_call_inside_lambda(self, analyzer):
        source = """\
class FooTest {
    void testLambda() {
        Calculator calc = new Calculator();
        Runnable r = () -> calc.add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 5, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved

    def test_duplicate_calls_resolved_once(self, analyzer):
        source = """\
class FooTest {
    void testDup() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
        calc.add(3, 4);
        calc.add(5, 6);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 7, analyzer, type_map, {},
        )
        # resolved is a set, so duplicates are naturally deduplicated
        assert resolved == {"Calculator.add", "Calculator.Calculator", "Calculator.<init>"}

    def test_same_method_name_different_classes(self, analyzer):
        source = """\
class FooTest {
    void testBoth() {
        Calculator calc = new Calculator();
        Buffer buf = new Buffer(10);
        calc.add(1, 2);
        buf.add("data");
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator", "buf": "Buffer"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 7, analyzer, type_map, {},
        )
        assert "Calculator.add" in resolved
        assert "Buffer.add" in resolved
        # Also includes constructor refs: Calculator.Calculator, Calculator.<init>, Buffer.Buffer, Buffer.<init>
        assert "Calculator.Calculator" in resolved
        assert "Buffer.Buffer" in resolved

    def test_chained_method_call_partial_resolution(self, analyzer):
        """Only the outermost receiver-resolved call should match; chained return types are unknown."""
        source = """\
class FooTest {
    void testChain() {
        Calculator calc = new Calculator();
        calc.getResult().toString();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"calc": "Calculator"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 5, analyzer, type_map, {},
        )
        # calc.getResult() resolves to Calculator.getResult
        assert "Calculator.getResult" in resolved
        # toString() is called on the return of getResult() which is unresolvable
        # (method_invocation as object node returns None)
        assert "Calculator.toString" not in resolved

    def test_super_method_call_not_resolved(self, analyzer):
        source = """\
class FooTest {
    void testSuper() {
        super.setup();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        assert len(resolved) == 0

    def test_this_method_call_not_resolved(self, analyzer):
        """Calling this.someHelperMethod() should not produce a source match."""
        source = """\
class FooTest {
    void testHelper() {
        this.helperMethod();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        # this is not a field_access with a field that's in the type map, so not resolved
        assert len(resolved) == 0

    def test_method_call_on_method_return_not_resolved(self, analyzer):
        source = """\
class FooTest {
    void testFactory() {
        getCalculator().add(1, 2);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        # getCalculator() returns a method_invocation node as object, can't resolve
        assert "Calculator.add" not in resolved

    def test_new_expression_with_generics(self, analyzer):
        source = """\
class FooTest {
    void testGeneric() {
        new ArrayList<String>().add("hello");
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        assert "ArrayList.add" in resolved

    def test_assertion_via_static_import_mapped_to_assertions_class(self, analyzer):
        """JUnit assertEquals via static import resolves to Assertions.assertEquals, not source."""
        source = """\
import static org.junit.jupiter.api.Assertions.assertEquals;
class FooTest {
    void testAssert() {
        assertEquals(1, 1);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        static_map = {"assertEquals": "Assertions"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 3, 5, analyzer, {}, static_map,
        )
        assert "Assertions.assertEquals" in resolved
        assert len(resolved) == 1

    def test_constructor_call_detected(self, analyzer):
        """``new ClassName(...)`` should emit ClassName.ClassName and ClassName.<init>."""
        source = """\
class FooTest {
    void testCreate() {
        Calculator calc = new Calculator();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        assert "Calculator.Calculator" in resolved
        assert "Calculator.<init>" in resolved

    def test_constructor_inside_method_arg(self, analyzer):
        """Constructor used as argument: ``list.add(new BatchRead(...))``."""
        source = """\
class FooTest {
    void testBatch() {
        List<BatchRead> records = new ArrayList<BatchRead>();
        records.add(new BatchRead(new Key("ns", "set", "k1"), true));
        records.add(new BatchRead(new Key("ns", "set", "k2"), false));
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_map = {"records": "List"}
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 6, analyzer, type_map, {},
        )
        assert "BatchRead.BatchRead" in resolved
        assert "BatchRead.<init>" in resolved
        assert "Key.Key" in resolved
        assert "Key.<init>" in resolved
        assert "List.add" in resolved

    def test_constructor_with_generics_stripped(self, analyzer):
        source = """\
class FooTest {
    void testGeneric() {
        HashMap<String, Integer> map = new HashMap<String, Integer>();
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        resolved = _resolve_method_calls_in_range(
            tree.root_node, source_bytes, 2, 4, analyzer, {}, {},
        )
        assert "HashMap.HashMap" in resolved
        assert "HashMap.<init>" in resolved


# ===================================================================
# Additional _match_test_to_functions tests
# ===================================================================


class TestMatchTestToFunctionsExtended:
    def test_same_method_name_different_classes_precise(self, analyzer):
        """When two classes have methods with the same name, only the actually called one matches."""
        test_source = """\
import org.junit.jupiter.api.Test;

class MyTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "MathUtils.add": make_func("add", "MathUtils"),
        }
        test_method = make_test_method("testAdd", "MyTest", 4, 8)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]
        assert "MathUtils.add" not in matched

    def test_call_inside_assert(self, analyzer):
        """A source method call wrapped in an assertion should still be matched."""
        test_source = """\
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testAdd", "CalculatorTest", 5, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]

    def test_multiple_tests_different_methods_same_class(self, analyzer):
        """Two test methods in the same source text should each match only the methods they call."""
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }

    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        calc.subtract(5, 3);
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "Calculator.subtract": make_func("subtract", "Calculator"),
        }
        test_add = make_test_method("testAdd", "CalculatorTest", 4, 8)
        test_sub = make_test_method("testSubtract", "CalculatorTest", 10, 14)

        matched_add = _match_test_to_functions(test_add, test_source, func_map, analyzer)
        matched_sub = _match_test_to_functions(test_sub, test_source, func_map, analyzer)

        assert matched_add == ["Calculator.add"]
        assert matched_sub == ["Calculator.subtract"]

    def test_builder_pattern(self, analyzer):
        """Builder-pattern chaining: only the first-level call resolves."""
        test_source = """\
import org.junit.jupiter.api.Test;

class BuilderTest {
    @Test
    void testBuild() {
        ConfigBuilder builder = new ConfigBuilder();
        builder.setName("test").setValue(42).build();
    }
}
"""
        func_map = {
            "ConfigBuilder.setName": make_func("setName", "ConfigBuilder"),
            "ConfigBuilder.setValue": make_func("setValue", "ConfigBuilder"),
            "ConfigBuilder.build": make_func("build", "ConfigBuilder"),
        }
        test_method = make_test_method("testBuild", "BuilderTest", 4, 8)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        # setName is called directly on builder (resolved via type_map)
        assert "ConfigBuilder.setName" in matched
        # setValue and build are chained on the return of setName - unresolvable
        assert "ConfigBuilder.setValue" not in matched
        assert "ConfigBuilder.build" not in matched

    def test_method_call_inside_enhanced_for(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class ProcessorTest {
    @Test
    void testProcessAll() {
        for (Processor proc : processors) {
            proc.process();
        }
    }
}
"""
        func_map = {"Processor.process": make_func("process", "Processor")}
        test_method = make_test_method("testProcessAll", "ProcessorTest", 4, 9)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Processor.process"]

    def test_cast_expression_match(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class ServiceTest {
    @Test
    void testCast() {
        Object obj = getService();
        ((Calculator) obj).add(1, 2);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testCast", "ServiceTest", 4, 8)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]

    def test_method_called_multiple_times_matched_once(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testRepeated() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
        calc.add(3, 4);
        calc.add(5, 6);
    }
}
"""
        func_map = {"Calculator.add": make_func("add", "Calculator")}
        test_method = make_test_method("testRepeated", "CalculatorTest", 4, 10)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == ["Calculator.add"]
        assert len(matched) == 1

    def test_mixed_static_and_instance_calls(self, analyzer):
        test_source = """\
import static com.example.MathUtils.abs;
import org.junit.jupiter.api.Test;

class MixedTest {
    @Test
    void testMixed() {
        Calculator calc = new Calculator();
        int sum = calc.add(1, abs(-2));
        int result = MathUtils.square(sum);
    }
}
"""
        func_map = {
            "Calculator.add": make_func("add", "Calculator"),
            "MathUtils.abs": make_func("abs", "MathUtils"),
            "MathUtils.square": make_func("square", "MathUtils"),
        }
        test_method = make_test_method("testMixed", "MixedTest", 5, 10)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert "Calculator.add" in matched
        assert "MathUtils.abs" in matched
        assert "MathUtils.square" in matched
        assert len(matched) == 3

    def test_no_match_when_function_map_empty(self, analyzer):
        test_source = """\
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
"""
        func_map: dict[str, FunctionToOptimize] = {}
        test_method = make_test_method("testAdd", "CalculatorTest", 4, 8)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert matched == []

    def test_constructor_matched(self, analyzer):
        """new ClassName() should match the constructor in the function map."""
        test_source = """\
import org.junit.jupiter.api.Test;

class BatchReadTest {
    @Test
    void testBatchRead() {
        List<BatchRead> records = new ArrayList<BatchRead>();
        records.add(new BatchRead(new Key("ns", "set", "k1"), true));
    }
}
"""
        func_map = {"BatchRead.BatchRead": make_func("BatchRead", "BatchRead")}
        test_method = make_test_method("testBatchRead", "BatchReadTest", 4, 8)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert "BatchRead.BatchRead" in matched

    def test_constructor_init_convention_matched(self, analyzer):
        """new ClassName() should also match <init> naming convention."""
        test_source = """\
import org.junit.jupiter.api.Test;

class BatchReadTest {
    @Test
    void testCreate() {
        BatchRead br = new BatchRead(key, true);
    }
}
"""
        func_map = {"BatchRead.<init>": make_func("<init>", "BatchRead")}
        test_method = make_test_method("testCreate", "BatchReadTest", 4, 7)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert "BatchRead.<init>" in matched

    def test_constructor_does_not_match_unrelated_methods(self, analyzer):
        """new BatchRead() should not cause BatchRead.read to match."""
        test_source = """\
import org.junit.jupiter.api.Test;

class SomeTest {
    @Test
    void testCreate() {
        BatchRead br = new BatchRead(key, true);
    }
}
"""
        func_map = {
            "BatchRead.BatchRead": make_func("BatchRead", "BatchRead"),
            "BatchRead.read": make_func("read", "BatchRead"),
        }
        test_method = make_test_method("testCreate", "SomeTest", 4, 7)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert "BatchRead.BatchRead" in matched
        assert "BatchRead.read" not in matched

    def test_aerospike_batch_read_complex_pattern(self, analyzer):
        """Real-world pattern from aerospike: multiple constructors as method arguments."""
        test_source = """\
import com.aerospike.client.BatchRead;
import com.aerospike.client.Key;
import org.junit.Test;

class TestAsyncBatch {
    @Test
    void asyncBatchReadComplex() {
        String[] bins = new String[] {"binname"};
        List<BatchRead> records = new ArrayList<BatchRead>();
        records.add(new BatchRead(new Key("ns", "set", "k1"), bins));
        records.add(new BatchRead(new Key("ns", "set", "k2"), true));
        records.add(new BatchRead(new Key("ns", "set", "k3"), false));
    }
}
"""
        func_map = {
            "BatchRead.BatchRead": make_func("BatchRead", "BatchRead"),
            "Key.Key": make_func("Key", "Key"),
            "BatchWrite.BatchWrite": make_func("BatchWrite", "BatchWrite"),
        }
        test_method = make_test_method("asyncBatchReadComplex", "TestAsyncBatch", 6, 14)
        matched = _match_test_to_functions(test_method, test_source, func_map, analyzer)
        assert "BatchRead.BatchRead" in matched
        assert "Key.Key" in matched
        assert "BatchWrite.BatchWrite" not in matched


# ===================================================================
# Additional discover_tests integration tests
# ===================================================================


class TestDiscoverTestsExtended:
    def test_tests_suffix_naming(self, tmp_path, analyzer):
        """*Tests.java pattern should be discovered."""
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTests.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTests {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)
        assert "Calculator.add" in result

    def test_test_prefix_naming(self, tmp_path, analyzer):
        """Test*.java pattern should be discovered."""
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "TestCalculator.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class TestCalculator {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)
        assert "Calculator.add" in result

    def test_empty_test_directory(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)
        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)
        assert result == {}

    def test_same_function_tested_multiple_methods_in_one_file(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAddPositive() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }

    @Test
    void testAddNegative() {
        Calculator calc = new Calculator();
        calc.add(-1, -2);
    }

    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        calc.subtract(5, 3);
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("add", "Calculator"),
            make_func("subtract", "Calculator"),
        ]
        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "Calculator.add" in result
        assert len(result["Calculator.add"]) == 2
        test_names = {t.test_name for t in result["Calculator.add"]}
        assert test_names == {"testAddPositive", "testAddNegative"}

        assert "Calculator.subtract" in result
        assert len(result["Calculator.subtract"]) == 1

    def test_same_function_tested_across_multiple_files(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        (test_dir / "IntegrationTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class IntegrationTest {
    @Test
    void testIntegration() {
        Calculator calc = new Calculator();
        calc.add(10, 20);
    }
}
""", encoding="utf-8")

        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "Calculator.add" in result
        assert len(result["Calculator.add"]) == 2
        test_names = {t.test_name for t in result["Calculator.add"]}
        assert test_names == {"testAdd", "testIntegration"}

    def test_parameterized_test_annotation(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class CalculatorTest {
    @ParameterizedTest
    @CsvSource({"1, 2, 3", "4, 5, 9"})
    void testAdd(int a, int b, int expected) {
        Calculator calc = new Calculator();
        calc.add(a, b);
    }
}
""", encoding="utf-8")

        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)
        assert "Calculator.add" in result
        assert result["Calculator.add"][0].test_name == "testAdd"

    def test_nested_test_directories(self, tmp_path, analyzer):
        deep_dir = tmp_path / "test" / "com" / "example" / "deep"
        deep_dir.mkdir(parents=True)

        (deep_dir / "NestedTest.java").write_text("""\
package com.example.deep;
import org.junit.jupiter.api.Test;

class NestedTest {
    @Test
    void testDeep() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)
        assert "Calculator.add" in result

    def test_var_integration(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        var calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)
        assert "Calculator.add" in result

    def test_no_source_functions(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        result = discover_tests(tmp_path, [], analyzer)
        assert result == {}

    def test_constructor_integration(self, tmp_path, analyzer):
        """Constructor calls should map to source constructors in the function map."""
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "BatchReadTest.java").write_text("""\
package com.aerospike.test;
import com.aerospike.client.BatchRead;
import com.aerospike.client.Key;
import org.junit.jupiter.api.Test;

class BatchReadTest {
    @Test
    void testBatchReadComplex() {
        List<BatchRead> records = new ArrayList<BatchRead>();
        records.add(new BatchRead(new Key("ns", "set", "k1"), true));
        records.add(new BatchRead(new Key("ns", "set", "k2"), false));
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("BatchRead", "BatchRead"),
            make_func("Key", "Key"),
            make_func("BatchWrite", "BatchWrite"),
        ]
        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "BatchRead.BatchRead" in result
        assert result["BatchRead.BatchRead"][0].test_name == "testBatchReadComplex"

        assert "Key.Key" in result
        assert result["Key.Key"][0].test_name == "testBatchReadComplex"

        assert "BatchWrite.BatchWrite" not in result


# ===================================================================
# Utility function tests
# ===================================================================


class TestIsTestFile:
    def test_test_suffix(self):
        assert is_test_file(Path("src/test/java/CalculatorTest.java")) is True

    def test_tests_suffix(self):
        assert is_test_file(Path("src/test/java/CalculatorTests.java")) is True

    def test_test_prefix(self):
        assert is_test_file(Path("src/test/java/TestCalculator.java")) is True

    def test_not_test_file(self):
        assert is_test_file(Path("src/main/java/Calculator.java")) is False

    def test_test_directory(self):
        assert is_test_file(Path("test/com/example/Anything.java")) is True

    def test_tests_directory(self):
        assert is_test_file(Path("tests/com/example/Anything.java")) is True

    def test_non_test_naming_outside_test_dir(self):
        assert is_test_file(Path("src/main/java/Helper.java")) is False


class TestGetTestClassForSourceClass:
    def test_finds_test_suffix(self, tmp_path):
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "CalculatorTest.java").write_text("class CalculatorTest {}", encoding="utf-8")

        result = get_test_class_for_source_class("Calculator", test_dir)
        assert result is not None
        assert result.name == "CalculatorTest.java"

    def test_finds_test_prefix(self, tmp_path):
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "TestCalculator.java").write_text("class TestCalculator {}", encoding="utf-8")

        result = get_test_class_for_source_class("Calculator", test_dir)
        assert result is not None
        assert result.name == "TestCalculator.java"

    def test_finds_tests_suffix(self, tmp_path):
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "CalculatorTests.java").write_text("class CalculatorTests {}", encoding="utf-8")

        result = get_test_class_for_source_class("Calculator", test_dir)
        assert result is not None
        assert result.name == "CalculatorTests.java"

    def test_not_found(self, tmp_path):
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        result = get_test_class_for_source_class("Calculator", test_dir)
        assert result is None

    def test_finds_in_subdirectory(self, tmp_path):
        test_dir = tmp_path / "test" / "com" / "example"
        test_dir.mkdir(parents=True)
        (test_dir / "CalculatorTest.java").write_text("class CalculatorTest {}", encoding="utf-8")

        result = get_test_class_for_source_class("Calculator", tmp_path / "test")
        assert result is not None
        assert result.name == "CalculatorTest.java"


class TestFindTestsForFunction:
    def test_basic(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        func = make_func("add", "Calculator")
        result = find_tests_for_function(func, tmp_path, analyzer)
        assert len(result) == 1
        assert result[0].test_name == "testAdd"

    def test_no_tests_found(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        func = make_func("add", "Calculator")
        result = find_tests_for_function(func, tmp_path, analyzer)
        assert result == []


class TestDiscoverAllTests:
    def test_basic(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {}

    @Test
    void testSubtract() {}
}
""", encoding="utf-8")

        all_tests = discover_all_tests(tmp_path, analyzer)
        names = {t.function_name for t in all_tests}
        assert names == {"testAdd", "testSubtract"}

    def test_empty_directory(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        all_tests = discover_all_tests(tmp_path, analyzer)
        assert all_tests == []

    def test_multiple_files(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "ATest.java").write_text("""\
import org.junit.jupiter.api.Test;
class ATest {
    @Test
    void testA() {}
}
""", encoding="utf-8")

        (test_dir / "BTest.java").write_text("""\
import org.junit.jupiter.api.Test;
class BTest {
    @Test
    void testB() {}
}
""", encoding="utf-8")

        all_tests = discover_all_tests(tmp_path, analyzer)
        names = {t.function_name for t in all_tests}
        assert names == {"testA", "testB"}
    def test_no_false_positive_import_only_integration(self, tmp_path, analyzer):
        """A test file that imports Calculator but never calls its methods should not match."""
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        test_file = test_dir / "SomeTest.java"
        test_file.write_text("""\
package com.example;

import com.example.Calculator;
import org.junit.jupiter.api.Test;

class SomeTest {
    @Test
    void testUnrelated() {
        int x = 42;
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("add", "Calculator"),
            make_func("subtract", "Calculator"),
        ]

        result = discover_tests(tmp_path, source_functions, analyzer)
        assert result == {}

    def test_multiple_test_files(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        (test_dir / "BufferTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class BufferTest {
    @Test
    void testRead() {
        Buffer buf = new Buffer(10);
        buf.read();
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("add", "Calculator"),
            make_func("read", "Buffer"),
            make_func("write", "Buffer"),
        ]

        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "Calculator.add" in result
        assert result["Calculator.add"][0].test_name == "testAdd"

        assert "Buffer.read" in result
        assert result["Buffer.read"][0].test_name == "testRead"

        assert "Buffer.write" not in result

    def test_test_file_deduplication(self, tmp_path, analyzer):
        """A file matching multiple patterns (e.g. FooTest.java) should not double-count."""
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        # This file matches *Test.java pattern
        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""", encoding="utf-8")

        source_functions = [make_func("add", "Calculator")]
        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "Calculator.add" in result
        # Should have exactly 1 test, not duplicated
        assert len(result["Calculator.add"]) == 1

    def test_static_import_integration(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "MathUtilsTest.java").write_text("""\
package com.example;
import static com.example.MathUtils.square;
import org.junit.jupiter.api.Test;

class MathUtilsTest {
    @Test
    void testSquare() {
        int result = square(5);
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("square", "MathUtils"),
            make_func("cube", "MathUtils"),
        ]

        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "MathUtils.square" in result
        assert "MathUtils.cube" not in result

    def test_one_test_calls_multiple_source_methods(self, tmp_path, analyzer):
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        (test_dir / "CalculatorTest.java").write_text("""\
package com.example;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testChainedOps() {
        Calculator calc = new Calculator();
        int a = calc.add(1, 2);
        int b = calc.multiply(a, 3);
    }
}
""", encoding="utf-8")

        source_functions = [
            make_func("add", "Calculator"),
            make_func("multiply", "Calculator"),
            make_func("subtract", "Calculator"),
        ]

        result = discover_tests(tmp_path, source_functions, analyzer)

        assert "Calculator.add" in result
        assert result["Calculator.add"][0].test_name == "testChainedOps"
        assert "Calculator.multiply" in result
        assert result["Calculator.multiply"][0].test_name == "testChainedOps"
        assert "Calculator.subtract" not in result

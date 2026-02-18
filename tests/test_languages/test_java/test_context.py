"""Tests for Java code context extraction."""

from pathlib import Path

import pytest

from codeflash.languages.base import FunctionFilterCriteria, Language, ParentInfo
from codeflash.languages.java.context import (
    TypeSkeleton,
    _extract_type_skeleton,
    extract_class_context,
    extract_code_context,
    extract_function_source,
    extract_read_only_context,
    find_helper_functions,
    get_java_imported_type_skeletons,
    _extract_public_method_signatures,
    _format_skeleton_for_context,
)
from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.import_resolver import JavaImportResolver, ResolvedImport
from codeflash.languages.java.parser import JavaImportInfo, get_java_analyzer


# Filter criteria that includes void methods
NO_RETURN_FILTER = FunctionFilterCriteria(require_return=False)


class TestExtractCodeContextBasic:
    """Tests for basic extract_code_context functionality."""

    def test_simple_method(self, tmp_path: Path):
        """Test extracting context for a simple method."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert context.target_file == java_file
        # Method is wrapped in class skeleton
        assert (
            context.target_code
            == """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        )
        assert context.imports == []
        assert context.helper_functions == []
        assert context.read_only_context == ""

    def test_method_with_javadoc(self, tmp_path: Path):
        """Test extracting context for method with Javadoc."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""public class Calculator {
    /**
     * Adds two numbers.
     * @param a first number
     * @param b second number
     * @return sum
     */
    public int add(int a, int b) {
        return a + b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert context.target_file == java_file
        assert (
            context.target_code
            == """public class Calculator {
    /**
     * Adds two numbers.
     * @param a first number
     * @param b second number
     * @return sum
     */
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        )
        assert context.imports == []
        assert context.helper_functions == []
        assert context.read_only_context == ""

    def test_static_method(self, tmp_path: Path):
        """Test extracting context for a static method."""
        java_file = tmp_path / "MathUtils.java"
        java_file.write_text("""public class MathUtils {
    public static int multiply(int a, int b) {
        return a * b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert context.target_file == java_file
        assert (
            context.target_code
            == """public class MathUtils {
    public static int multiply(int a, int b) {
        return a * b;
    }
}
"""
        )
        assert context.imports == []
        assert context.helper_functions == []
        assert context.read_only_context == ""

    def test_private_method(self, tmp_path: Path):
        """Test extracting context for a private method."""
        java_file = tmp_path / "Helper.java"
        java_file.write_text("""public class Helper {
    private int getValue() {
        return 42;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert context.target_file == java_file
        assert (
            context.target_code
            == """public class Helper {
    private int getValue() {
        return 42;
    }
}
"""
        )

    def test_protected_method(self, tmp_path: Path):
        """Test extracting context for a protected method."""
        java_file = tmp_path / "Base.java"
        java_file.write_text("""public class Base {
    protected int compute(int x) {
        return x * 2;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert context.target_file == java_file
        assert (
            context.target_code
            == """public class Base {
    protected int compute(int x) {
        return x * 2;
    }
}
"""
        )

    def test_synchronized_method(self, tmp_path: Path):
        """Test extracting context for a synchronized method."""
        java_file = tmp_path / "Counter.java"
        java_file.write_text("""public class Counter {
    public synchronized int getCount() {
        return count;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert (
            context.target_code
            == """public class Counter {
    public synchronized int getCount() {
        return count;
    }
}
"""
        )

    def test_method_with_throws(self, tmp_path: Path):
        """Test extracting context for a method with throws clause."""
        java_file = tmp_path / "FileHandler.java"
        java_file.write_text("""public class FileHandler {
    public String readFile(String path) throws IOException, FileNotFoundException {
        return Files.readString(Path.of(path));
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert (
            context.target_code
            == """public class FileHandler {
    public String readFile(String path) throws IOException, FileNotFoundException {
        return Files.readString(Path.of(path));
    }
}
"""
        )

    def test_method_with_varargs(self, tmp_path: Path):
        """Test extracting context for a method with varargs."""
        java_file = tmp_path / "Logger.java"
        java_file.write_text("""public class Logger {
    public String format(String... messages) {
        return String.join(", ", messages);
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert (
            context.target_code
            == """public class Logger {
    public String format(String... messages) {
        return String.join(", ", messages);
    }
}
"""
        )

    def test_void_method(self, tmp_path: Path):
        """Test extracting context for a void method."""
        java_file = tmp_path / "Printer.java"
        java_file.write_text("""public class Printer {
    public void print(String text) {
        System.out.println(text);
    }
}
""")
        functions = discover_functions_from_source(
            java_file.read_text(), file_path=java_file, filter_criteria=NO_RETURN_FILTER
        )
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert (
            context.target_code
            == """public class Printer {
    public void print(String text) {
        System.out.println(text);
    }
}
"""
        )

    def test_generic_return_type(self, tmp_path: Path):
        """Test extracting context for a method with generic return type."""
        java_file = tmp_path / "Container.java"
        java_file.write_text("""public class Container {
    public List<String> getNames() {
        return new ArrayList<>();
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert (
            context.target_code
            == """public class Container {
    public List<String> getNames() {
        return new ArrayList<>();
    }
}
"""
        )


class TestExtractCodeContextWithImports:
    """Tests for extract_code_context with various import types."""

    def test_with_package_and_imports(self, tmp_path: Path):
        """Test context extraction with package and imports."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""package com.example;

import java.util.List;

public class Calculator {
    private int base = 0;

    public int add(int a, int b) {
        return a + b + base;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        add_func = next((f for f in functions if f.function_name == "add"), None)
        assert add_func is not None

        context = extract_code_context(add_func, tmp_path)

        assert context.language == Language.JAVA
        assert context.target_file == java_file
        # Class skeleton includes fields
        assert (
            context.target_code
            == """public class Calculator {
    private int base = 0;
    public int add(int a, int b) {
        return a + b + base;
    }
}
"""
        )
        assert context.imports == ["import java.util.List;"]
        # Fields are in skeleton, so read_only_context is empty
        assert context.read_only_context == ""

    def test_with_static_imports(self, tmp_path: Path):
        """Test context extraction with static imports."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""package com.example;

import java.util.List;
import static java.lang.Math.PI;
import static java.lang.Math.sqrt;

public class Calculator {
    public double circleArea(double radius) {
        return PI * radius * radius;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert (
            context.target_code
            == """public class Calculator {
    public double circleArea(double radius) {
        return PI * radius * radius;
    }
}
"""
        )
        assert context.imports == [
            "import java.util.List;",
            "import static java.lang.Math.PI;",
            "import static java.lang.Math.sqrt;",
        ]

    def test_with_wildcard_imports(self, tmp_path: Path):
        """Test context extraction with wildcard imports."""
        java_file = tmp_path / "Processor.java"
        java_file.write_text("""package com.example;

import java.util.*;
import java.io.*;

public class Processor {
    public List<String> process(String input) {
        return Arrays.asList(input.split(","));
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        assert context.imports == ["import java.util.*;", "import java.io.*;"]

    def test_with_multiple_import_types(self, tmp_path: Path):
        """Test context extraction with various import types."""
        java_file = tmp_path / "Handler.java"
        java_file.write_text("""package com.example;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import static java.util.Collections.sort;
import static java.util.Collections.reverse;

public class Handler {
    public List<Integer> sortNumbers(List<Integer> nums) {
        sort(nums);
        return nums;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Handler {
    public List<Integer> sortNumbers(List<Integer> nums) {
        sort(nums);
        return nums;
    }
}
"""
        )
        assert context.imports == [
            "import java.util.List;",
            "import java.util.Map;",
            "import java.util.ArrayList;",
            "import static java.util.Collections.sort;",
            "import static java.util.Collections.reverse;",
        ]
        assert context.read_only_context == ""
        assert context.helper_functions == []


class TestExtractCodeContextWithFields:
    """Tests for extract_code_context with class fields.

    Note: When fields are included in the class skeleton (target_code),
    read_only_context should be empty to avoid duplication.
    """

    def test_with_instance_fields(self, tmp_path: Path):
        """Test context extraction with instance fields."""
        java_file = tmp_path / "Person.java"
        java_file.write_text("""public class Person {
    private String name;
    private int age;

    public String getName() {
        return name;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        # Class skeleton includes fields
        assert (
            context.target_code
            == """public class Person {
    private String name;
    private int age;
    public String getName() {
        return name;
    }
}
"""
        )
        # Fields are in skeleton, so read_only_context is empty (no duplication)
        assert context.read_only_context == ""
        assert context.imports == []
        assert context.helper_functions == []

    def test_with_static_fields(self, tmp_path: Path):
        """Test context extraction with static fields."""
        java_file = tmp_path / "Counter.java"
        java_file.write_text("""public class Counter {
    private static int instanceCount = 0;
    private static String prefix = "counter_";

    public int getCount() {
        return instanceCount;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Counter {
    private static int instanceCount = 0;
    private static String prefix = "counter_";
    public int getCount() {
        return instanceCount;
    }
}
"""
        )
        # Fields are in skeleton, so read_only_context is empty
        assert context.read_only_context == ""

    def test_with_final_fields(self, tmp_path: Path):
        """Test context extraction with final fields."""
        java_file = tmp_path / "Config.java"
        java_file.write_text("""public class Config {
    private final String name;
    private final int maxSize;

    public String getName() {
        return name;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Config {
    private final String name;
    private final int maxSize;
    public String getName() {
        return name;
    }
}
"""
        )
        assert context.read_only_context == ""

    def test_with_static_final_constants(self, tmp_path: Path):
        """Test context extraction with static final constants."""
        java_file = tmp_path / "Constants.java"
        java_file.write_text("""public class Constants {
    public static final double PI = 3.14159;
    public static final int MAX_VALUE = 100;
    private static final String PREFIX = "const_";

    public double getPI() {
        return PI;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Constants {
    public static final double PI = 3.14159;
    public static final int MAX_VALUE = 100;
    private static final String PREFIX = "const_";
    public double getPI() {
        return PI;
    }
}
"""
        )
        assert context.read_only_context == ""

    def test_with_volatile_fields(self, tmp_path: Path):
        """Test context extraction with volatile fields."""
        java_file = tmp_path / "ThreadSafe.java"
        java_file.write_text("""public class ThreadSafe {
    private volatile boolean running = true;
    private volatile int counter = 0;

    public boolean isRunning() {
        return running;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class ThreadSafe {
    private volatile boolean running = true;
    private volatile int counter = 0;
    public boolean isRunning() {
        return running;
    }
}
"""
        )
        assert context.read_only_context == ""

    def test_with_generic_fields(self, tmp_path: Path):
        """Test context extraction with generic type fields."""
        java_file = tmp_path / "Container.java"
        java_file.write_text("""public class Container {
    private List<String> names;
    private Map<String, Integer> scores;
    private Set<Long> ids;

    public List<String> getNames() {
        return names;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Container {
    private List<String> names;
    private Map<String, Integer> scores;
    private Set<Long> ids;
    public List<String> getNames() {
        return names;
    }
}
"""
        )
        assert context.read_only_context == ""

    def test_with_array_fields(self, tmp_path: Path):
        """Test context extraction with array fields."""
        java_file = tmp_path / "ArrayHolder.java"
        java_file.write_text("""public class ArrayHolder {
    private int[] numbers;
    private String[] names;
    private double[][] matrix;

    public int[] getNumbers() {
        return numbers;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class ArrayHolder {
    private int[] numbers;
    private String[] names;
    private double[][] matrix;
    public int[] getNumbers() {
        return numbers;
    }
}
"""
        )
        assert context.read_only_context == ""


class TestExtractCodeContextWithHelpers:
    """Tests for extract_code_context with helper functions."""

    def test_single_helper_method(self, tmp_path: Path):
        """Test context extraction with a single helper method."""
        java_file = tmp_path / "Processor.java"
        java_file.write_text("""public class Processor {
    public String process(String input) {
        return normalize(input);
    }

    private String normalize(String s) {
        return s.trim().toLowerCase();
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        process_func = next((f for f in functions if f.function_name == "process"), None)
        assert process_func is not None

        context = extract_code_context(process_func, tmp_path)

        assert context.language == Language.JAVA
        assert (
            context.target_code
            == """public class Processor {
    public String process(String input) {
        return normalize(input);
    }
}
"""
        )
        assert len(context.helper_functions) == 1
        assert context.helper_functions[0].name == "normalize"
        assert (
            context.helper_functions[0].source_code
            == "private String normalize(String s) {\n        return s.trim().toLowerCase();\n    }"
        )

    def test_multiple_helper_methods(self, tmp_path: Path):
        """Test context extraction with multiple helper methods."""
        java_file = tmp_path / "Processor.java"
        java_file.write_text("""public class Processor {
    public String process(String input) {
        String trimmed = trim(input);
        return upper(trimmed);
    }

    private String trim(String s) {
        return s.trim();
    }

    private String upper(String s) {
        return s.toUpperCase();
    }

    private String unused(String s) {
        return s;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        process_func = next((f for f in functions if f.function_name == "process"), None)
        assert process_func is not None

        context = extract_code_context(process_func, tmp_path)

        assert (
            context.target_code
            == """public class Processor {
    public String process(String input) {
        String trimmed = trim(input);
        return upper(trimmed);
    }
}
"""
        )
        assert context.read_only_context == ""
        assert context.imports == []
        helper_names = sorted([h.name for h in context.helper_functions])
        assert helper_names == ["trim", "upper"]

    def test_chained_helper_calls(self, tmp_path: Path):
        """Test context extraction with chained helper calls."""
        java_file = tmp_path / "Processor.java"
        java_file.write_text("""public class Processor {
    public String process(String input) {
        return normalize(input);
    }

    private String normalize(String s) {
        return sanitize(s).toLowerCase();
    }

    private String sanitize(String s) {
        return s.trim();
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        process_func = next((f for f in functions if f.function_name == "process"), None)
        assert process_func is not None

        context = extract_code_context(process_func, tmp_path)

        helper_names = [h.name for h in context.helper_functions]
        assert helper_names == ["normalize"]

    def test_no_helpers_when_none_called(self, tmp_path: Path):
        """Test context extraction when no helpers are called."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    private int unused(int x) {
        return x * 2;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        add_func = next((f for f in functions if f.function_name == "add"), None)
        assert add_func is not None

        context = extract_code_context(add_func, tmp_path)

        assert (
            context.target_code
            == """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        )
        assert context.helper_functions == []

    def test_static_helper_from_instance_method(self, tmp_path: Path):
        """Test context extraction with static helper called from instance method."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""public class Calculator {
    public int calculate(int x) {
        return staticHelper(x);
    }

    private static int staticHelper(int x) {
        return x * 2;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        calc_func = next((f for f in functions if f.function_name == "calculate"), None)
        assert calc_func is not None

        context = extract_code_context(calc_func, tmp_path)

        helper_names = [h.name for h in context.helper_functions]
        assert helper_names == ["staticHelper"]


class TestExtractCodeContextWithJavadoc:
    """Tests for extract_code_context with various Javadoc patterns."""

    def test_simple_javadoc(self, tmp_path: Path):
        """Test context extraction with simple Javadoc."""
        java_file = tmp_path / "Example.java"
        java_file.write_text("""public class Example {
    /** Simple description. */
    public void doSomething() {
    }
}
""")
        functions = discover_functions_from_source(
            java_file.read_text(), file_path=java_file, filter_criteria=NO_RETURN_FILTER
        )
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Example {
    /** Simple description. */
    public void doSomething() {
    }
}
"""
        )

    def test_javadoc_with_params(self, tmp_path: Path):
        """Test context extraction with Javadoc @param tags."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""public class Calculator {
    /**
     * Adds two numbers.
     * @param a the first number
     * @param b the second number
     */
    public int add(int a, int b) {
        return a + b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Calculator {
    /**
     * Adds two numbers.
     * @param a the first number
     * @param b the second number
     */
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        )

    def test_javadoc_with_return(self, tmp_path: Path):
        """Test context extraction with Javadoc @return tag."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""public class Calculator {
    /**
     * Computes the sum.
     * @return the sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Calculator {
    /**
     * Computes the sum.
     * @return the sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        )

    def test_javadoc_with_throws(self, tmp_path: Path):
        """Test context extraction with Javadoc @throws tag."""
        java_file = tmp_path / "Divider.java"
        java_file.write_text("""public class Divider {
    /**
     * Divides two numbers.
     * @throws ArithmeticException if divisor is zero
     * @throws IllegalArgumentException if inputs are negative
     */
    public double divide(double a, double b) {
        if (b == 0) throw new ArithmeticException();
        return a / b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Divider {
    /**
     * Divides two numbers.
     * @throws ArithmeticException if divisor is zero
     * @throws IllegalArgumentException if inputs are negative
     */
    public double divide(double a, double b) {
        if (b == 0) throw new ArithmeticException();
        return a / b;
    }
}
"""
        )

    def test_javadoc_multiline(self, tmp_path: Path):
        """Test context extraction with multi-paragraph Javadoc."""
        java_file = tmp_path / "Complex.java"
        java_file.write_text("""public class Complex {
    /**
     * This is a complex method.
     *
     * <p>It does many things:</p>
     * <ul>
     *   <li>First thing</li>
     *   <li>Second thing</li>
     * </ul>
     *
     * @param input the input value
     * @return the processed result
     */
    public String process(String input) {
        return input.toUpperCase();
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Complex {
    /**
     * This is a complex method.
     *
     * <p>It does many things:</p>
     * <ul>
     *   <li>First thing</li>
     *   <li>Second thing</li>
     * </ul>
     *
     * @param input the input value
     * @return the processed result
     */
    public String process(String input) {
        return input.toUpperCase();
    }
}
"""
        )


class TestExtractCodeContextWithGenerics:
    """Tests for extract_code_context with generic types."""

    def test_generic_method_type_parameter(self, tmp_path: Path):
        """Test context extraction with generic type parameter."""
        java_file = tmp_path / "Utils.java"
        java_file.write_text("""public class Utils {
    public <T> T identity(T value) {
        return value;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Utils {
    public <T> T identity(T value) {
        return value;
    }
}
"""
        )

    def test_bounded_type_parameter(self, tmp_path: Path):
        """Test context extraction with bounded type parameter."""
        java_file = tmp_path / "Statistics.java"
        java_file.write_text("""public class Statistics {
    public <T extends Number> double average(List<T> numbers) {
        double sum = 0;
        for (T num : numbers) {
            sum += num.doubleValue();
        }
        return sum / numbers.size();
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Statistics {
    public <T extends Number> double average(List<T> numbers) {
        double sum = 0;
        for (T num : numbers) {
            sum += num.doubleValue();
        }
        return sum / numbers.size();
    }
}
"""
        )

    def test_wildcard_type(self, tmp_path: Path):
        """Test context extraction with wildcard type."""
        java_file = tmp_path / "Printer.java"
        java_file.write_text("""public class Printer {
    public int countItems(List<?> items) {
        return items.size();
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Printer {
    public int countItems(List<?> items) {
        return items.size();
    }
}
"""
        )

    def test_bounded_wildcard_extends(self, tmp_path: Path):
        """Test context extraction with upper bounded wildcard."""
        java_file = tmp_path / "Aggregator.java"
        java_file.write_text("""public class Aggregator {
    public double sum(List<? extends Number> numbers) {
        double total = 0;
        for (Number n : numbers) {
            total += n.doubleValue();
        }
        return total;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Aggregator {
    public double sum(List<? extends Number> numbers) {
        double total = 0;
        for (Number n : numbers) {
            total += n.doubleValue();
        }
        return total;
    }
}
"""
        )

    def test_bounded_wildcard_super(self, tmp_path: Path):
        """Test context extraction with lower bounded wildcard."""
        java_file = tmp_path / "Filler.java"
        java_file.write_text("""public class Filler {
    public boolean fill(List<? super Integer> list, Integer value) {
        list.add(value);
        return true;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Filler {
    public boolean fill(List<? super Integer> list, Integer value) {
        list.add(value);
        return true;
    }
}
"""
        )

    def test_multiple_type_parameters(self, tmp_path: Path):
        """Test context extraction with multiple type parameters."""
        java_file = tmp_path / "Mapper.java"
        java_file.write_text("""public class Mapper {
    public <K, V> Map<V, K> invert(Map<K, V> map) {
        Map<V, K> result = new HashMap<>();
        for (Map.Entry<K, V> entry : map.entrySet()) {
            result.put(entry.getValue(), entry.getKey());
        }
        return result;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Mapper {
    public <K, V> Map<V, K> invert(Map<K, V> map) {
        Map<V, K> result = new HashMap<>();
        for (Map.Entry<K, V> entry : map.entrySet()) {
            result.put(entry.getValue(), entry.getKey());
        }
        return result;
    }
}
"""
        )

    def test_recursive_type_bound(self, tmp_path: Path):
        """Test context extraction with recursive type bound."""
        java_file = tmp_path / "Sorter.java"
        java_file.write_text("""public class Sorter {
    public <T extends Comparable<T>> T max(T a, T b) {
        return a.compareTo(b) > 0 ? a : b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Sorter {
    public <T extends Comparable<T>> T max(T a, T b) {
        return a.compareTo(b) > 0 ? a : b;
    }
}
"""
        )


class TestExtractCodeContextWithAnnotations:
    """Tests for extract_code_context with annotations."""

    def test_override_annotation(self, tmp_path: Path):
        """Test context extraction with @Override annotation."""
        java_file = tmp_path / "Child.java"
        java_file.write_text("""public class Child extends Parent {
    @Override
    public String toString() {
        return "Child";
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Child extends Parent {
    @Override
    public String toString() {
        return "Child";
    }
}
"""
        )

    def test_deprecated_annotation(self, tmp_path: Path):
        """Test context extraction with @Deprecated annotation."""
        java_file = tmp_path / "Legacy.java"
        java_file.write_text("""public class Legacy {
    @Deprecated
    public int oldMethod() {
        return 0;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Legacy {
    @Deprecated
    public int oldMethod() {
        return 0;
    }
}
"""
        )

    def test_suppress_warnings_annotation(self, tmp_path: Path):
        """Test context extraction with @SuppressWarnings annotation."""
        java_file = tmp_path / "Processor.java"
        java_file.write_text("""public class Processor {
    @SuppressWarnings("unchecked")
    public List process(Object input) {
        return (List) input;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Processor {
    @SuppressWarnings("unchecked")
    public List process(Object input) {
        return (List) input;
    }
}
"""
        )

    def test_multiple_annotations(self, tmp_path: Path):
        """Test context extraction with multiple annotations."""
        java_file = tmp_path / "Service.java"
        java_file.write_text("""public class Service {
    @Override
    @Deprecated
    @SuppressWarnings("deprecation")
    public String legacyMethod() {
        return "legacy";
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Service {
    @Override
    @Deprecated
    @SuppressWarnings("deprecation")
    public String legacyMethod() {
        return "legacy";
    }
}
"""
        )

    def test_annotation_with_array_value(self, tmp_path: Path):
        """Test context extraction with annotation array value."""
        java_file = tmp_path / "Handler.java"
        java_file.write_text("""public class Handler {
    @SuppressWarnings({"unchecked", "rawtypes"})
    public Object handle(Object input) {
        return input;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Handler {
    @SuppressWarnings({"unchecked", "rawtypes"})
    public Object handle(Object input) {
        return input;
    }
}
"""
        )


class TestExtractCodeContextWithInheritance:
    """Tests for extract_code_context with inheritance scenarios."""

    def test_method_in_subclass(self, tmp_path: Path):
        """Test context extraction for method in subclass."""
        java_file = tmp_path / "AdvancedCalc.java"
        java_file.write_text("""public class AdvancedCalc extends Calculator {
    public int multiply(int a, int b) {
        return a * b;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert context.language == Language.JAVA
        # Class skeleton includes extends clause
        assert (
            context.target_code
            == """public class AdvancedCalc extends Calculator {
    public int multiply(int a, int b) {
        return a * b;
    }
}
"""
        )

    def test_interface_implementation(self, tmp_path: Path):
        """Test context extraction for interface implementation."""
        java_file = tmp_path / "MyComparable.java"
        java_file.write_text("""public class MyComparable implements Comparable<MyComparable> {
    private int value;

    @Override
    public int compareTo(MyComparable other) {
        return Integer.compare(this.value, other.value);
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        # Class skeleton includes implements clause and fields
        assert (
            context.target_code
            == """public class MyComparable implements Comparable<MyComparable> {
    private int value;
    @Override
    public int compareTo(MyComparable other) {
        return Integer.compare(this.value, other.value);
    }
}
"""
        )
        # Fields are in skeleton, so read_only_context is empty (no duplication)
        assert context.read_only_context == ""

    def test_multiple_interfaces(self, tmp_path: Path):
        """Test context extraction for multiple interface implementations."""
        java_file = tmp_path / "MultiImpl.java"
        java_file.write_text("""public class MultiImpl implements Runnable, Comparable<MultiImpl> {
    public void run() {
        System.out.println("Running");
    }

    public int compareTo(MultiImpl other) {
        return 0;
    }
}
""")
        functions = discover_functions_from_source(
            java_file.read_text(), file_path=java_file, filter_criteria=NO_RETURN_FILTER
        )
        assert len(functions) == 2

        run_func = next((f for f in functions if f.function_name == "run"), None)
        assert run_func is not None

        context = extract_code_context(run_func, tmp_path)
        assert (
            context.target_code
            == """public class MultiImpl implements Runnable, Comparable<MultiImpl> {
    public void run() {
        System.out.println("Running");
    }
}
"""
        )

    def test_default_interface_method(self, tmp_path: Path):
        """Test context extraction for default interface method."""
        java_file = tmp_path / "MyInterface.java"
        java_file.write_text("""public interface MyInterface {
    default String greet() {
        return "Hello";
    }

    void doSomething();
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        greet_func = next((f for f in functions if f.function_name == "greet"), None)
        assert greet_func is not None

        context = extract_code_context(greet_func, tmp_path)

        # Interface methods are wrapped in interface skeleton
        assert (
            context.target_code
            == """public interface MyInterface {
    default String greet() {
        return "Hello";
    }
}
"""
        )
        assert context.read_only_context == ""


class TestExtractCodeContextWithInnerClasses:
    """Tests for extract_code_context with inner/nested classes."""

    def test_static_nested_class_method(self, tmp_path: Path):
        """Test context extraction for static nested class method."""
        java_file = tmp_path / "Container.java"
        java_file.write_text("""public class Container {
    public static class Nested {
        public int compute(int x) {
            return x * 2;
        }
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        compute_func = next((f for f in functions if f.function_name == "compute"), None)
        assert compute_func is not None

        context = extract_code_context(compute_func, tmp_path)

        # Inner class wrapped in outer class skeleton
        assert (
            context.target_code
            == """public class Container {
    public static class Nested {
        public int compute(int x) {
            return x * 2;
        }
    }
}
"""
        )
        assert context.read_only_context == ""

    def test_inner_class_method(self, tmp_path: Path):
        """Test context extraction for inner class method."""
        java_file = tmp_path / "Outer.java"
        java_file.write_text("""public class Outer {
    private int value = 10;

    public class Inner {
        public int getValue() {
            return value;
        }
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        get_func = next((f for f in functions if f.function_name == "getValue"), None)
        assert get_func is not None

        context = extract_code_context(get_func, tmp_path)

        # Inner class wrapped in outer class skeleton
        assert (
            context.target_code
            == """public class Outer {
    public class Inner {
        public int getValue() {
            return value;
        }
    }
}
"""
        )
        assert context.read_only_context == ""


class TestExtractCodeContextWithEnumAndInterface:
    """Tests for extract_code_context with enums and interfaces."""

    def test_enum_method(self, tmp_path: Path):
        """Test context extraction for enum method."""
        java_file = tmp_path / "Operation.java"
        java_file.write_text("""public enum Operation {
    ADD, SUBTRACT, MULTIPLY, DIVIDE;

    public int apply(int a, int b) {
        switch (this) {
            case ADD: return a + b;
            case SUBTRACT: return a - b;
            case MULTIPLY: return a * b;
            case DIVIDE: return a / b;
            default: throw new AssertionError();
        }
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        apply_func = next((f for f in functions if f.function_name == "apply"), None)
        assert apply_func is not None

        context = extract_code_context(apply_func, tmp_path)

        # Enum methods are wrapped in enum skeleton with constants
        assert (
            context.target_code
            == """public enum Operation {
    ADD, SUBTRACT, MULTIPLY, DIVIDE;

    public int apply(int a, int b) {
        switch (this) {
            case ADD: return a + b;
            case SUBTRACT: return a - b;
            case MULTIPLY: return a * b;
            case DIVIDE: return a / b;
            default: throw new AssertionError();
        }
    }
}
"""
        )
        assert context.read_only_context == ""

    def test_interface_default_method(self, tmp_path: Path):
        """Test context extraction for interface default method."""
        java_file = tmp_path / "Greeting.java"
        java_file.write_text("""public interface Greeting {
    default String greet(String name) {
        return "Hello, " + name;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        greet_func = next((f for f in functions if f.function_name == "greet"), None)
        assert greet_func is not None

        context = extract_code_context(greet_func, tmp_path)

        # Interface methods are wrapped in interface skeleton
        assert (
            context.target_code
            == """public interface Greeting {
    default String greet(String name) {
        return "Hello, " + name;
    }
}
"""
        )
        assert context.read_only_context == ""

    def test_interface_static_method(self, tmp_path: Path):
        """Test context extraction for interface static method."""
        java_file = tmp_path / "Factory.java"
        java_file.write_text("""public interface Factory {
    static Factory create() {
        return null;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        create_func = next((f for f in functions if f.function_name == "create"), None)
        assert create_func is not None

        context = extract_code_context(create_func, tmp_path)

        # Interface methods are wrapped in interface skeleton
        assert (
            context.target_code
            == """public interface Factory {
    static Factory create() {
        return null;
    }
}
"""
        )
        assert context.read_only_context == ""


class TestExtractCodeContextEdgeCases:
    """Tests for extract_code_context edge cases."""

    def test_empty_method(self, tmp_path: Path):
        """Test context extraction for empty method."""
        java_file = tmp_path / "Empty.java"
        java_file.write_text("""public class Empty {
    public void doNothing() {
    }
}
""")
        functions = discover_functions_from_source(
            java_file.read_text(), file_path=java_file, filter_criteria=NO_RETURN_FILTER
        )
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Empty {
    public void doNothing() {
    }
}
"""
        )

    def test_single_line_method(self, tmp_path: Path):
        """Test context extraction for single-line method."""
        java_file = tmp_path / "OneLiner.java"
        java_file.write_text("""public class OneLiner {
    public int get() { return 42; }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class OneLiner {
    public int get() { return 42; }
}
"""
        )

    def test_method_with_lambda(self, tmp_path: Path):
        """Test context extraction for method with lambda."""
        java_file = tmp_path / "Functional.java"
        java_file.write_text("""public class Functional {
    public List<String> filter(List<String> items) {
        return items.stream()
            .filter(s -> s != null && !s.isEmpty())
            .collect(Collectors.toList());
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Functional {
    public List<String> filter(List<String> items) {
        return items.stream()
            .filter(s -> s != null && !s.isEmpty())
            .collect(Collectors.toList());
    }
}
"""
        )

    def test_method_with_method_reference(self, tmp_path: Path):
        """Test context extraction for method with method reference."""
        java_file = tmp_path / "Printer.java"
        java_file.write_text("""public class Printer {
    public List<String> toUpper(List<String> items) {
        return items.stream().map(String::toUpperCase).collect(Collectors.toList());
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Printer {
    public List<String> toUpper(List<String> items) {
        return items.stream().map(String::toUpperCase).collect(Collectors.toList());
    }
}
"""
        )

    def test_deeply_nested_blocks(self, tmp_path: Path):
        """Test context extraction for method with deeply nested blocks."""
        java_file = tmp_path / "Nested.java"
        java_file.write_text("""public class Nested {
    public int deepMethod(int n) {
        int result = 0;
        if (n > 0) {
            for (int i = 0; i < n; i++) {
                while (i > 0) {
                    try {
                        if (i % 2 == 0) {
                            result += i;
                        }
                    } catch (Exception e) {
                        result = -1;
                    }
                    break;
                }
            }
        }
        return result;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Nested {
    public int deepMethod(int n) {
        int result = 0;
        if (n > 0) {
            for (int i = 0; i < n; i++) {
                while (i > 0) {
                    try {
                        if (i % 2 == 0) {
                            result += i;
                        }
                    } catch (Exception e) {
                        result = -1;
                    }
                    break;
                }
            }
        }
        return result;
    }
}
"""
        )

    def test_unicode_in_source(self, tmp_path: Path):
        """Test context extraction for method with unicode characters."""
        java_file = tmp_path / "Unicode.java"
        java_file.write_text("""public class Unicode {
    public String greet() {
        return "こんにちは世界";
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        assert len(functions) == 1

        context = extract_code_context(functions[0], tmp_path)

        assert (
            context.target_code
            == """public class Unicode {
    public String greet() {
        return "こんにちは世界";
    }
}
"""
        )

    def test_file_not_found(self, tmp_path: Path):
        """Test context extraction for missing file."""
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.models.function_types import FunctionParent

        missing_file = tmp_path / "NonExistent.java"
        func = FunctionToOptimize(
            function_name="test",
            file_path=missing_file,
            starting_line=1,
            ending_line=5,
            parents=[FunctionParent(name="Test", type="ClassDef")],
            language="java",
        )

        context = extract_code_context(func, tmp_path)

        assert context.target_code == ""
        assert context.language == Language.JAVA
        assert context.target_file == missing_file

    def test_max_helper_depth_zero(self, tmp_path: Path):
        """Test context extraction with max_helper_depth=0."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""public class Calculator {
    public int calculate(int x) {
        return helper(x);
    }

    private int helper(int x) {
        return x * 2;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        calc_func = next((f for f in functions if f.function_name == "calculate"), None)
        assert calc_func is not None

        context = extract_code_context(calc_func, tmp_path, max_helper_depth=0)

        # With max_depth=0, cross-file helpers should be empty, but same-file helpers are still found
        assert (
            context.target_code
            == """public class Calculator {
    public int calculate(int x) {
        return helper(x);
    }
}
"""
        )


class TestExtractCodeContextWithConstructor:
    """Tests for extract_code_context with constructors in class skeleton."""

    def test_class_with_constructor(self, tmp_path: Path):
        """Test context extraction includes constructor in skeleton."""
        java_file = tmp_path / "Person.java"
        java_file.write_text("""public class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        get_func = next((f for f in functions if f.function_name == "getName"), None)
        assert get_func is not None

        context = extract_code_context(get_func, tmp_path)

        # Class skeleton includes fields and constructor
        assert (
            context.target_code
            == """public class Person {
    private String name;
    private int age;
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    public String getName() {
        return name;
    }
}
"""
        )

    def test_class_with_multiple_constructors(self, tmp_path: Path):
        """Test context extraction includes all constructors in skeleton."""
        java_file = tmp_path / "Config.java"
        java_file.write_text("""public class Config {
    private String name;
    private int value;

    public Config() {
        this("default", 0);
    }

    public Config(String name) {
        this(name, 0);
    }

    public Config(String name, int value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        get_func = next((f for f in functions if f.function_name == "getName"), None)
        assert get_func is not None

        context = extract_code_context(get_func, tmp_path)

        # Class skeleton includes fields and all constructors
        assert (
            context.target_code
            == """public class Config {
    private String name;
    private int value;
    public Config() {
        this("default", 0);
    }
    public Config(String name) {
        this(name, 0);
    }
    public Config(String name, int value) {
        this.name = name;
        this.value = value;
    }
    public String getName() {
        return name;
    }
}
"""
        )


class TestExtractCodeContextFullIntegration:
    """Integration tests for extract_code_context with all components."""

    def test_full_context_with_all_components(self, tmp_path: Path):
        """Test context extraction with imports, fields, and helpers."""
        java_file = tmp_path / "Service.java"
        java_file.write_text("""package com.example;

import java.util.List;
import java.util.ArrayList;

public class Service {
    private static final String PREFIX = "service_";
    private List<String> history = new ArrayList<>();

    public String process(String input) {
        String result = transform(input);
        history.add(result);
        return result;
    }

    private String transform(String s) {
        return PREFIX + s.toUpperCase();
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        process_func = next((f for f in functions if f.function_name == "process"), None)
        assert process_func is not None

        context = extract_code_context(process_func, tmp_path)

        assert context.language == Language.JAVA
        assert context.target_file == java_file
        # Class skeleton includes fields
        assert (
            context.target_code
            == """public class Service {
    private static final String PREFIX = "service_";
    private List<String> history = new ArrayList<>();
    public String process(String input) {
        String result = transform(input);
        history.add(result);
        return result;
    }
}
"""
        )
        assert context.imports == ["import java.util.List;", "import java.util.ArrayList;"]
        # Fields are in skeleton, so read_only_context is empty (no duplication)
        assert context.read_only_context == ""
        assert len(context.helper_functions) == 1
        assert context.helper_functions[0].name == "transform"

    def test_complex_class_with_javadoc_and_annotations(self, tmp_path: Path):
        """Test context extraction for complex class with javadoc and annotations."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""package com.example.math;

import java.util.Objects;
import static java.lang.Math.sqrt;

public class Calculator {
    private double precision = 0.0001;

    /**
     * Calculates the square root using Newton's method.
     * @param n the number to calculate square root for
     * @return the approximate square root
     * @throws IllegalArgumentException if n is negative
     */
    @SuppressWarnings("unused")
    public double sqrtNewton(double n) {
        if (n < 0) throw new IllegalArgumentException();
        return approximate(n, n / 2);
    }

    private double approximate(double n, double guess) {
        double next = (guess + n / guess) / 2;
        if (Math.abs(guess - next) < precision) return next;
        return approximate(n, next);
    }
}
""")
        functions = discover_functions_from_source(java_file.read_text(), file_path=java_file)
        sqrt_func = next((f for f in functions if f.function_name == "sqrtNewton"), None)
        assert sqrt_func is not None

        context = extract_code_context(sqrt_func, tmp_path)

        assert context.language == Language.JAVA
        # Class skeleton includes fields and Javadoc
        assert (
            context.target_code
            == """public class Calculator {
    private double precision = 0.0001;
    /**
     * Calculates the square root using Newton's method.
     * @param n the number to calculate square root for
     * @return the approximate square root
     * @throws IllegalArgumentException if n is negative
     */
    @SuppressWarnings("unused")
    public double sqrtNewton(double n) {
        if (n < 0) throw new IllegalArgumentException();
        return approximate(n, n / 2);
    }
}
"""
        )
        assert context.imports == ["import java.util.Objects;", "import static java.lang.Math.sqrt;"]
        # Fields are in skeleton, so read_only_context is empty (no duplication)
        assert context.read_only_context == ""
        assert len(context.helper_functions) == 1
        assert context.helper_functions[0].name == "approximate"


class TestExtractClassContext:
    """Tests for extract_class_context."""

    def test_extract_class_with_imports(self, tmp_path: Path):
        """Test extracting full class context with imports."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""package com.example;

import java.util.List;
import java.util.ArrayList;

public class Calculator {
    private List<Integer> history = new ArrayList<>();

    public int add(int a, int b) {
        int result = a + b;
        history.add(result);
        return result;
    }
}
""")

        context = extract_class_context(java_file, "Calculator")

        assert (
            context
            == """package com.example;

import java.util.List;
import java.util.ArrayList;

public class Calculator {
    private List<Integer> history = new ArrayList<>();

    public int add(int a, int b) {
        int result = a + b;
        history.add(result);
        return result;
    }
}"""
        )

    def test_extract_class_not_found(self, tmp_path: Path):
        """Test extracting non-existent class returns empty string."""
        java_file = tmp_path / "Test.java"
        java_file.write_text("""public class Test {
    public void test() {}
}
""")

        context = extract_class_context(java_file, "NonExistent")

        assert context == ""

    def test_extract_class_missing_file(self, tmp_path: Path):
        """Test extracting from missing file returns empty string."""
        missing_file = tmp_path / "Missing.java"

        context = extract_class_context(missing_file, "Missing")

        assert context == ""


class TestExtractFunctionSourceStaleLineNumbers:
    """Tests for tree-sitter based function extraction resilience to stale line numbers.

    When running --all mode, a prior optimization may modify the source file,
    shifting line numbers for subsequent functions. The tree-sitter based
    extraction should still find the correct function by name.
    """

    def test_extraction_with_stale_line_numbers(self):
        """Verify extraction works when pre-computed line numbers no longer match the source."""
        # Original source: functionA at lines 2-4, functionB at lines 5-7
        original_source = """public class Utils {
    public int functionA() {
        return 1;
    }
    public int functionB() {
        return 2;
    }
}
"""
        analyzer = get_java_analyzer()
        functions = discover_functions_from_source(original_source, file_path=Path("Utils.java"))
        func_b = [f for f in functions if f.function_name == "functionB"][0]
        original_b_start = func_b.starting_line

        # Simulate a prior optimization adding lines to functionA
        modified_source = """public class Utils {
    public int functionA() {
        int x = 1;
        int y = 2;
        int z = 3;
        return x + y + z;
    }
    public int functionB() {
        return 2;
    }
}
"""
        # func_b still has the STALE line numbers from the original source
        # With tree-sitter, extraction should still work correctly
        result = extract_function_source(modified_source, func_b, analyzer=analyzer)
        assert "functionB" in result
        assert "return 2;" in result

    def test_extraction_without_analyzer_uses_line_numbers(self):
        """Without analyzer, extraction falls back to pre-computed line numbers."""
        source = """public class Utils {
    public int functionA() {
        return 1;
    }
    public int functionB() {
        return 2;
    }
}
"""
        functions = discover_functions_from_source(source, file_path=Path("Utils.java"))
        func_b = [f for f in functions if f.function_name == "functionB"][0]

        # Without analyzer, should still work with correct line numbers
        result = extract_function_source(source, func_b)
        assert "functionB" in result
        assert "return 2;" in result

    def test_extraction_with_javadoc_after_file_modification(self):
        """Verify Javadoc is included when using tree-sitter extraction on modified files."""
        original_source = """public class Utils {
    /** Adds two numbers. */
    public int add(int a, int b) {
        return a + b;
    }
    /** Subtracts two numbers. */
    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
        analyzer = get_java_analyzer()
        functions = discover_functions_from_source(original_source, file_path=Path("Utils.java"))
        func_sub = [f for f in functions if f.function_name == "subtract"][0]

        # Simulate prior optimization expanding the add method
        modified_source = """public class Utils {
    /** Adds two numbers. */
    public int add(int a, int b) {
        // Optimized with null check
        if (a == 0) return b;
        if (b == 0) return a;
        return a + b;
    }
    /** Subtracts two numbers. */
    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
        result = extract_function_source(modified_source, func_sub, analyzer=analyzer)
        assert "/** Subtracts two numbers. */" in result
        assert "public int subtract" in result
        assert "return a - b;" in result

    def test_extraction_with_overloaded_methods(self):
        """Verify correct overload is selected using line proximity."""
        source = """public class Utils {
    public int process(int x) {
        return x * 2;
    }
    public int process(int x, int y) {
        return x + y;
    }
}
"""
        analyzer = get_java_analyzer()
        functions = discover_functions_from_source(source, file_path=Path("Utils.java"))
        # Get the second overload (process(int, int))
        func_two_args = [f for f in functions if f.function_name == "process" and f.ending_line > 4][0]

        result = extract_function_source(source, func_two_args, analyzer=analyzer)
        assert "int x, int y" in result
        assert "return x + y;" in result

    def test_extraction_function_not_found_falls_back(self):
        """If tree-sitter can't find the method, fall back to line numbers."""
        source = """public class Utils {
    public int functionA() {
        return 1;
    }
}
"""
        analyzer = get_java_analyzer()
        functions = discover_functions_from_source(source, file_path=Path("Utils.java"))
        func_a = functions[0]

        # Create a copy with a non-existent name so tree-sitter can't find it
        from dataclasses import replace

        func_fake = replace(func_a, function_name="nonExistentMethod")

        # Should fall back to line-number extraction (which still works since source is unmodified)
        result = extract_function_source(source, func_fake, analyzer=analyzer)
        assert "functionA" in result
        assert "return 1;" in result


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "java_maven"


class TestGetJavaImportedTypeSkeletons:
    """Tests for get_java_imported_type_skeletons()."""

    def test_resolves_internal_imports(self):
        """Verify that project-internal imports are resolved and skeletons extracted."""
        project_root = FIXTURE_DIR
        module_root = FIXTURE_DIR / "src" / "main" / "java"
        analyzer = get_java_analyzer()

        source = (FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "Calculator.java").read_text()
        imports = analyzer.find_imports(source)

        result = get_java_imported_type_skeletons(imports, project_root, module_root, analyzer)

        # Should contain skeletons for MathHelper and Formatter (imported by Calculator)
        assert "MathHelper" in result
        assert "Formatter" in result

    def test_skeletons_contain_method_signatures(self):
        """Verify extracted skeletons include public method signatures."""
        project_root = FIXTURE_DIR
        module_root = FIXTURE_DIR / "src" / "main" / "java"
        analyzer = get_java_analyzer()

        source = (FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "Calculator.java").read_text()
        imports = analyzer.find_imports(source)

        result = get_java_imported_type_skeletons(imports, project_root, module_root, analyzer)

        # MathHelper should have its public static methods listed
        assert "add" in result
        assert "multiply" in result
        assert "factorial" in result

    def test_skips_external_imports(self):
        """Verify that standard library and external imports are skipped."""
        project_root = FIXTURE_DIR
        module_root = FIXTURE_DIR / "src" / "main" / "java"
        analyzer = get_java_analyzer()

        # DataProcessor has java.util.* imports but no internal project imports
        source = (FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "DataProcessor.java").read_text()
        imports = analyzer.find_imports(source)

        result = get_java_imported_type_skeletons(imports, project_root, module_root, analyzer)

        # No internal imports → empty result
        assert result == ""

    def test_deduplicates_imports(self):
        """Verify that the same type imported twice is only included once."""
        project_root = FIXTURE_DIR
        module_root = FIXTURE_DIR / "src" / "main" / "java"
        analyzer = get_java_analyzer()

        source = (FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "Calculator.java").read_text()
        imports = analyzer.find_imports(source)
        # Double the imports to simulate duplicates
        doubled_imports = imports + imports

        result = get_java_imported_type_skeletons(doubled_imports, project_root, module_root, analyzer)

        # Count occurrences of MathHelper — should appear exactly once
        assert result.count("class MathHelper") == 1

    def test_empty_imports_returns_empty(self):
        """Verify that empty import list returns empty string."""
        project_root = FIXTURE_DIR
        analyzer = get_java_analyzer()

        result = get_java_imported_type_skeletons([], project_root, None, analyzer)

        assert result == ""

    def test_respects_token_budget(self):
        """Verify that the function stops when token budget is exceeded."""
        project_root = FIXTURE_DIR
        module_root = FIXTURE_DIR / "src" / "main" / "java"
        analyzer = get_java_analyzer()

        source = (FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "Calculator.java").read_text()
        imports = analyzer.find_imports(source)

        # With a very small budget, should truncate output
        import codeflash.languages.java.context as ctx

        original_budget = ctx.IMPORTED_SKELETON_TOKEN_BUDGET
        try:
            ctx.IMPORTED_SKELETON_TOKEN_BUDGET = 1  # Very small budget
            result = get_java_imported_type_skeletons(imports, project_root, module_root, analyzer)
            # Should be empty since even a single skeleton exceeds 1 token
            assert result == ""
        finally:
            ctx.IMPORTED_SKELETON_TOKEN_BUDGET = original_budget


class TestExtractPublicMethodSignatures:
    """Tests for _extract_public_method_signatures()."""

    def test_extracts_public_methods(self):
        """Verify public method signatures are extracted."""
        source = """public class Foo {
    public int add(int a, int b) {
        return a + b;
    }
    private void secret() {}
    public static String format(double val) {
        return String.valueOf(val);
    }
}"""
        analyzer = get_java_analyzer()
        sigs = _extract_public_method_signatures(source, "Foo", analyzer)

        assert len(sigs) == 2
        assert any("add" in s for s in sigs)
        assert any("format" in s for s in sigs)
        # private method should not be included
        assert not any("secret" in s for s in sigs)

    def test_excludes_constructors(self):
        """Verify constructors are excluded from method signatures."""
        source = """public class Bar {
    public Bar(int x) { this.x = x; }
    public int getX() { return x; }
}"""
        analyzer = get_java_analyzer()
        sigs = _extract_public_method_signatures(source, "Bar", analyzer)

        assert len(sigs) == 1
        assert "getX" in sigs[0]
        assert not any("Bar(" in s for s in sigs)

    def test_empty_class_returns_empty(self):
        """Verify empty class returns no signatures."""
        source = """public class Empty {}"""
        analyzer = get_java_analyzer()
        sigs = _extract_public_method_signatures(source, "Empty", analyzer)

        assert sigs == []

    def test_filters_by_class_name(self):
        """Verify only methods from the specified class are returned."""
        source = """public class A {
    public int aMethod() { return 1; }
}
class B {
    public int bMethod() { return 2; }
}"""
        analyzer = get_java_analyzer()
        sigs_a = _extract_public_method_signatures(source, "A", analyzer)
        sigs_b = _extract_public_method_signatures(source, "B", analyzer)

        assert len(sigs_a) == 1
        assert "aMethod" in sigs_a[0]
        assert len(sigs_b) == 1
        assert "bMethod" in sigs_b[0]


class TestFormatSkeletonForContext:
    """Tests for _format_skeleton_for_context()."""

    def test_formats_basic_skeleton(self):
        """Verify basic skeleton formatting with fields and constructors."""
        source = """public class Widget {
    private int size;
    public Widget(int size) { this.size = size; }
    public int getSize() { return size; }
}"""
        analyzer = get_java_analyzer()
        skeleton = TypeSkeleton(
            type_declaration="public class Widget",
            type_javadoc=None,
            fields_code="    private int size;\n",
            constructors_code="    public Widget(int size) { this.size = size; }\n",
            enum_constants="",
            type_indent="",
            type_kind="class",
        )

        result = _format_skeleton_for_context(skeleton, source, "Widget", analyzer)

        assert result.startswith("public class Widget {")
        assert "private int size;" in result
        assert "Widget(int size)" in result
        assert "getSize" in result
        assert result.endswith("}")

    def test_formats_enum_skeleton(self):
        """Verify enum formatting includes constants."""
        source = """public enum Color {
    RED, GREEN, BLUE;
    public String lower() { return name().toLowerCase(); }
}"""
        analyzer = get_java_analyzer()
        skeleton = TypeSkeleton(
            type_declaration="public enum Color",
            type_javadoc=None,
            fields_code="",
            constructors_code="",
            enum_constants="RED, GREEN, BLUE",
            type_indent="",
            type_kind="enum",
        )

        result = _format_skeleton_for_context(skeleton, source, "Color", analyzer)

        assert "public enum Color {" in result
        assert "RED, GREEN, BLUE;" in result
        assert "lower" in result

    def test_formats_empty_class(self):
        """Verify formatting of a class with no fields or methods."""
        source = """public class Empty {}"""
        analyzer = get_java_analyzer()
        skeleton = TypeSkeleton(
            type_declaration="public class Empty",
            type_javadoc=None,
            fields_code="",
            constructors_code="",
            enum_constants="",
            type_indent="",
            type_kind="class",
        )

        result = _format_skeleton_for_context(skeleton, source, "Empty", analyzer)

        assert result == "public class Empty {\n}"


class TestGetJavaImportedTypeSkeletonsEdgeCases:
    """Additional edge case tests for get_java_imported_type_skeletons()."""

    def test_wildcard_imports_are_skipped(self):
        """Wildcard imports (e.g., import com.example.helpers.*) have class_name=None and should be skipped."""
        project_root = FIXTURE_DIR
        module_root = FIXTURE_DIR / "src" / "main" / "java"
        analyzer = get_java_analyzer()

        # Create a source with a wildcard import
        source = "package com.example;\nimport com.example.helpers.*;\npublic class Foo {}"
        imports = analyzer.find_imports(source)

        # Verify the import is wildcard
        assert any(imp.is_wildcard for imp in imports)

        result = get_java_imported_type_skeletons(imports, project_root, module_root, analyzer)

        # Wildcard imports can't resolve to a single class, so result should be empty
        assert result == ""

    def test_import_to_nonexistent_class_in_file(self):
        """When an import resolves to a file but the class doesn't exist in it, skeleton extraction returns None."""
        analyzer = get_java_analyzer()

        source = "package com.example;\npublic class Actual { public int x; }"
        # Try to extract a skeleton for a class that doesn't exist in this source
        skeleton = _extract_type_skeleton(source, "NonExistent", "", analyzer)

        assert skeleton is None

    def test_skeleton_output_is_well_formed(self):
        """Verify the skeleton string has proper Java-like structure with braces."""
        project_root = FIXTURE_DIR
        module_root = FIXTURE_DIR / "src" / "main" / "java"
        analyzer = get_java_analyzer()

        source = (FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "Calculator.java").read_text()
        imports = analyzer.find_imports(source)

        result = get_java_imported_type_skeletons(imports, project_root, module_root, analyzer)

        # Each skeleton block should be well-formed: starts with declaration {, ends with }
        for block in result.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            assert "{" in block, f"Skeleton block missing opening brace: {block[:50]}"
            assert block.endswith("}"), f"Skeleton block missing closing brace: {block[-50:]}"


class TestExtractPublicMethodSignaturesEdgeCases:
    """Additional edge case tests for _extract_public_method_signatures()."""

    def test_excludes_protected_and_package_private(self):
        """Verify protected and package-private methods are excluded."""
        source = """public class Visibility {
    public int publicMethod() { return 1; }
    protected int protectedMethod() { return 2; }
    int packagePrivateMethod() { return 3; }
    private int privateMethod() { return 4; }
}"""
        analyzer = get_java_analyzer()
        sigs = _extract_public_method_signatures(source, "Visibility", analyzer)

        assert len(sigs) == 1
        assert "publicMethod" in sigs[0]
        assert not any("protectedMethod" in s for s in sigs)
        assert not any("packagePrivateMethod" in s for s in sigs)
        assert not any("privateMethod" in s for s in sigs)

    def test_handles_overloaded_methods(self):
        """Verify all public overloads are extracted."""
        source = """public class Overloaded {
    public int process(int x) { return x; }
    public int process(int x, int y) { return x + y; }
    public String process(String s) { return s; }
}"""
        analyzer = get_java_analyzer()
        sigs = _extract_public_method_signatures(source, "Overloaded", analyzer)

        assert len(sigs) == 3
        # All should contain "process"
        assert all("process" in s for s in sigs)

    def test_handles_generic_methods(self):
        """Verify generic method signatures are extracted correctly."""
        source = """public class Generic {
    public <T> T identity(T value) { return value; }
    public <K, V> void putPair(K key, V value) {}
}"""
        analyzer = get_java_analyzer()
        sigs = _extract_public_method_signatures(source, "Generic", analyzer)

        assert len(sigs) == 2
        assert any("identity" in s for s in sigs)
        assert any("putPair" in s for s in sigs)


class TestFormatSkeletonRoundTrip:
    """Tests that verify _extract_type_skeleton → _format_skeleton_for_context produces valid output."""

    def test_round_trip_produces_valid_skeleton(self):
        """Extract a real skeleton and format it — verify the output is sensible."""
        source = """public class Service {
    private final String name;
    private int count;

    public Service(String name) {
        this.name = name;
        this.count = 0;
    }

    public String getName() {
        return name;
    }

    public void increment() {
        count++;
    }

    public int getCount() {
        return count;
    }

    private void reset() {
        count = 0;
    }
}"""
        analyzer = get_java_analyzer()
        skeleton = _extract_type_skeleton(source, "Service", "", analyzer)
        assert skeleton is not None

        result = _format_skeleton_for_context(skeleton, source, "Service", analyzer)

        # Should contain class declaration
        assert "public class Service {" in result
        # Should contain fields
        assert "name" in result
        assert "count" in result
        # Should contain constructor
        assert "Service(String name)" in result
        # Should contain public methods
        assert "getName" in result
        assert "getCount" in result
        # Should NOT contain private methods
        assert "reset" not in result
        # Should end properly
        assert result.strip().endswith("}")

    def test_round_trip_with_fixture_mathhelper(self):
        """Round-trip test using the real MathHelper fixture file."""
        source = (FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "helpers" / "MathHelper.java").read_text()
        analyzer = get_java_analyzer()

        skeleton = _extract_type_skeleton(source, "MathHelper", "", analyzer)
        assert skeleton is not None

        result = _format_skeleton_for_context(skeleton, source, "MathHelper", analyzer)

        assert "public class MathHelper {" in result
        # All public static methods should have signatures
        for method_name in ["add", "multiply", "factorial", "power", "isPrime", "gcd", "lcm"]:
            assert method_name in result, f"Expected method '{method_name}' in skeleton"
        assert result.strip().endswith("}")

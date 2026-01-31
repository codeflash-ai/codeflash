"""Tests for the Java tree-sitter parser utilities."""

import pytest

from codeflash.languages.java.parser import (
    JavaAnalyzer,
    JavaClassNode,
    JavaFieldInfo,
    JavaImportInfo,
    JavaMethodNode,
    get_java_analyzer,
)


class TestJavaAnalyzerBasic:
    """Basic tests for JavaAnalyzer initialization and parsing."""

    def test_get_java_analyzer(self):
        """Test that get_java_analyzer returns a JavaAnalyzer instance."""
        analyzer = get_java_analyzer()
        assert isinstance(analyzer, JavaAnalyzer)

    def test_parse_simple_class(self):
        """Test parsing a simple Java class."""
        analyzer = get_java_analyzer()
        source = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        tree = analyzer.parse(source)
        assert tree is not None
        assert tree.root_node is not None
        assert not tree.root_node.has_error

    def test_validate_syntax_valid(self):
        """Test syntax validation with valid code."""
        analyzer = get_java_analyzer()
        source = """
public class Test {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        assert analyzer.validate_syntax(source) is True

    def test_validate_syntax_invalid(self):
        """Test syntax validation with invalid code."""
        analyzer = get_java_analyzer()
        source = """
public class Test {
    public int add(int a, int b) {
        return a + b
    }  // Missing semicolon
}
"""
        assert analyzer.validate_syntax(source) is False


class TestMethodDiscovery:
    """Tests for method discovery functionality."""

    def test_find_simple_method(self):
        """Test finding a simple method."""
        analyzer = get_java_analyzer()
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        methods = analyzer.find_methods(source)
        assert len(methods) == 1
        assert methods[0].name == "add"
        assert methods[0].class_name == "Calculator"
        assert methods[0].is_public is True
        assert methods[0].is_static is False
        assert methods[0].return_type == "int"

    def test_find_multiple_methods(self):
        """Test finding multiple methods in a class."""
        analyzer = get_java_analyzer()
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }

    private int multiply(int a, int b) {
        return a * b;
    }
}
"""
        methods = analyzer.find_methods(source)
        assert len(methods) == 3
        method_names = {m.name for m in methods}
        assert method_names == {"add", "subtract", "multiply"}

    def test_find_methods_with_modifiers(self):
        """Test finding methods with various modifiers."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    public static void staticMethod() {}
    private void privateMethod() {}
    protected void protectedMethod() {}
    public synchronized void syncMethod() {}
    public abstract void abstractMethod();
}
"""
        methods = analyzer.find_methods(source)

        static_method = next((m for m in methods if m.name == "staticMethod"), None)
        assert static_method is not None
        assert static_method.is_static is True
        assert static_method.is_public is True

        private_method = next((m for m in methods if m.name == "privateMethod"), None)
        assert private_method is not None
        assert private_method.is_private is True

        sync_method = next((m for m in methods if m.name == "syncMethod"), None)
        assert sync_method is not None
        assert sync_method.is_synchronized is True

    def test_filter_private_methods(self):
        """Test filtering out private methods."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    public void publicMethod() {}
    private void privateMethod() {}
}
"""
        methods = analyzer.find_methods(source, include_private=False)
        assert len(methods) == 1
        assert methods[0].name == "publicMethod"

    def test_filter_static_methods(self):
        """Test filtering out static methods."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    public void instanceMethod() {}
    public static void staticMethod() {}
}
"""
        methods = analyzer.find_methods(source, include_static=False)
        assert len(methods) == 1
        assert methods[0].name == "instanceMethod"

    def test_method_with_javadoc(self):
        """Test finding method with Javadoc comment."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    /**
     * Adds two numbers together.
     * @param a first number
     * @param b second number
     * @return the sum
     */
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        methods = analyzer.find_methods(source)
        assert len(methods) == 1
        assert methods[0].javadoc_start_line is not None
        # Javadoc should start before the method
        assert methods[0].javadoc_start_line < methods[0].start_line


class TestClassDiscovery:
    """Tests for class discovery functionality."""

    def test_find_simple_class(self):
        """Test finding a simple class."""
        analyzer = get_java_analyzer()
        source = """
public class HelloWorld {
    public void sayHello() {}
}
"""
        classes = analyzer.find_classes(source)
        assert len(classes) == 1
        assert classes[0].name == "HelloWorld"
        assert classes[0].is_public is True

    def test_find_class_with_extends(self):
        """Test finding a class that extends another."""
        analyzer = get_java_analyzer()
        source = """
public class Child extends Parent {
    public void method() {}
}
"""
        classes = analyzer.find_classes(source)
        assert len(classes) == 1
        assert classes[0].name == "Child"
        assert classes[0].extends == "Parent"

    def test_find_class_with_implements(self):
        """Test finding a class that implements interfaces."""
        analyzer = get_java_analyzer()
        source = """
public class MyService implements Service, Runnable {
    public void run() {}
}
"""
        classes = analyzer.find_classes(source)
        assert len(classes) == 1
        assert classes[0].name == "MyService"
        assert "Service" in classes[0].implements or "Runnable" in classes[0].implements

    def test_find_abstract_class(self):
        """Test finding an abstract class."""
        analyzer = get_java_analyzer()
        source = """
public abstract class AbstractBase {
    public abstract void doSomething();
}
"""
        classes = analyzer.find_classes(source)
        assert len(classes) == 1
        assert classes[0].is_abstract is True

    def test_find_final_class(self):
        """Test finding a final class."""
        analyzer = get_java_analyzer()
        source = """
public final class ImmutableClass {
    private final int value;
}
"""
        classes = analyzer.find_classes(source)
        assert len(classes) == 1
        assert classes[0].is_final is True


class TestImportDiscovery:
    """Tests for import discovery functionality."""

    def test_find_simple_import(self):
        """Test finding a simple import."""
        analyzer = get_java_analyzer()
        source = """
import java.util.List;

public class Example {}
"""
        imports = analyzer.find_imports(source)
        assert len(imports) == 1
        assert "java.util.List" in imports[0].import_path
        assert imports[0].is_static is False
        assert imports[0].is_wildcard is False

    def test_find_wildcard_import(self):
        """Test finding a wildcard import."""
        analyzer = get_java_analyzer()
        source = """
import java.util.*;

public class Example {}
"""
        imports = analyzer.find_imports(source)
        assert len(imports) == 1
        assert imports[0].is_wildcard is True

    def test_find_static_import(self):
        """Test finding a static import."""
        analyzer = get_java_analyzer()
        source = """
import static java.lang.Math.PI;

public class Example {}
"""
        imports = analyzer.find_imports(source)
        assert len(imports) == 1
        assert imports[0].is_static is True

    def test_find_multiple_imports(self):
        """Test finding multiple imports."""
        analyzer = get_java_analyzer()
        source = """
import java.util.List;
import java.util.Map;
import java.io.File;

public class Example {}
"""
        imports = analyzer.find_imports(source)
        assert len(imports) == 3


class TestFieldDiscovery:
    """Tests for field discovery functionality."""

    def test_find_simple_field(self):
        """Test finding a simple field."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    private int count;
}
"""
        fields = analyzer.find_fields(source)
        assert len(fields) == 1
        assert fields[0].name == "count"
        assert fields[0].type_name == "int"
        assert fields[0].is_private is True

    def test_find_field_with_modifiers(self):
        """Test finding a field with various modifiers."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    private static final String CONSTANT = "value";
}
"""
        fields = analyzer.find_fields(source)
        assert len(fields) == 1
        assert fields[0].name == "CONSTANT"
        assert fields[0].is_static is True
        assert fields[0].is_final is True

    def test_find_multiple_fields_same_declaration(self):
        """Test finding multiple fields in same declaration."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    private int a, b, c;
}
"""
        fields = analyzer.find_fields(source)
        assert len(fields) == 3
        field_names = {f.name for f in fields}
        assert field_names == {"a", "b", "c"}


class TestMethodCalls:
    """Tests for method call detection."""

    def test_find_method_calls(self):
        """Test finding method calls within a method."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    public void caller() {
        helper();
        anotherHelper();
    }

    private void helper() {}
    private void anotherHelper() {}
}
"""
        methods = analyzer.find_methods(source)
        caller = next((m for m in methods if m.name == "caller"), None)
        assert caller is not None

        calls = analyzer.find_method_calls(source, caller)
        assert "helper" in calls
        assert "anotherHelper" in calls


class TestPackageExtraction:
    """Tests for package name extraction."""

    def test_get_package_name(self):
        """Test extracting package name."""
        analyzer = get_java_analyzer()
        source = """
package com.example.myapp;

public class Example {}
"""
        package = analyzer.get_package_name(source)
        assert package == "com.example.myapp"

    def test_get_package_name_simple(self):
        """Test extracting simple package name."""
        analyzer = get_java_analyzer()
        source = """
package mypackage;

public class Example {}
"""
        package = analyzer.get_package_name(source)
        assert package == "mypackage"

    def test_no_package(self):
        """Test when there's no package declaration."""
        analyzer = get_java_analyzer()
        source = """
public class Example {}
"""
        package = analyzer.get_package_name(source)
        assert package is None


class TestHasReturn:
    """Tests for return statement detection."""

    def test_has_return(self):
        """Test detecting return statement."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    public int getValue() {
        return 42;
    }
}
"""
        methods = analyzer.find_methods(source)
        assert len(methods) == 1
        assert analyzer.has_return_statement(methods[0], source) is True

    def test_void_method(self):
        """Test void method (no return needed)."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    public void doSomething() {
        System.out.println("Hello");
    }
}
"""
        methods = analyzer.find_methods(source)
        assert len(methods) == 1
        # void methods return False since they don't need return
        assert analyzer.has_return_statement(methods[0], source) is False


class TestComplexJavaCode:
    """Tests for complex Java code patterns."""

    def test_generic_method(self):
        """Test finding a method with generics."""
        analyzer = get_java_analyzer()
        source = """
public class Container<T> {
    public <U> U transform(T value, Function<T, U> transformer) {
        return transformer.apply(value);
    }
}
"""
        methods = analyzer.find_methods(source)
        assert len(methods) == 1
        assert methods[0].name == "transform"

    def test_nested_class(self):
        """Test finding methods in nested classes."""
        analyzer = get_java_analyzer()
        source = """
public class Outer {
    public void outerMethod() {}

    public static class Inner {
        public void innerMethod() {}
    }
}
"""
        methods = analyzer.find_methods(source)
        method_names = {m.name for m in methods}
        assert "outerMethod" in method_names
        assert "innerMethod" in method_names

    def test_annotation_on_method(self):
        """Test finding method with annotations."""
        analyzer = get_java_analyzer()
        source = """
public class Example {
    @Override
    public String toString() {
        return "Example";
    }

    @Deprecated
    @SuppressWarnings("unchecked")
    public void oldMethod() {}
}
"""
        methods = analyzer.find_methods(source)
        assert len(methods) == 2

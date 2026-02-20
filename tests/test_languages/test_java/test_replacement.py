"""Tests for Java code replacement.

Tests the high-level replacement functions using complete valid Java source files.
All optimized code is syntactically valid Java that could compile.
All assertions use exact string equality for rigorous verification.
"""

from pathlib import Path

import pytest

from codeflash.code_utils.code_replacer import (
    replace_function_definitions_for_language,
    replace_function_definitions_in_module,
)
from codeflash.models.function_types import FunctionParent
from codeflash.languages.base import Language
from codeflash.languages import current as language_current
from codeflash.models.models import CodeStringsMarkdown


@pytest.fixture
def java_language_context():
    """Set the current language to Java for the duration of the test."""
    original_language = language_current._current_language
    language_current._current_language = Language.JAVA
    yield
    language_current._current_language = original_language


class TestReplaceFunctionDefinitionsInModule:
    """Tests for replace_function_definitions_in_module with Java."""

    def test_replace_simple_method(self, tmp_path: Path, java_language_context):
        """Test replacing a simple method in a Java class."""
        java_file = tmp_path / "Calculator.java"
        original_code = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Calculator {{
    public int add(int a, int b) {{
        return Math.addExact(a, b);
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_in_module(
            function_names=["add"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            preexisting_objects=set(),
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Calculator {
    public int add(int a, int b) {
        return Math.addExact(a, b);
    }
}
"""
        assert new_code == expected

    def test_replace_method_preserves_other_methods(self, tmp_path: Path, java_language_context):
        """Test that replacing one method preserves other methods."""
        java_file = tmp_path / "Calculator.java"
        original_code = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Calculator {{
    public int add(int a, int b) {{
        return Integer.sum(a, b);
    }}

    public int subtract(int a, int b) {{
        return a - b;
    }}

    public int multiply(int a, int b) {{
        return a * b;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_in_module(
            function_names=["add"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            preexisting_objects=set(),
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Calculator {
    public int add(int a, int b) {
        return Integer.sum(a, b);
    }

    public int subtract(int a, int b) {
        return a - b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"""
        assert new_code == expected

    def test_replace_method_with_javadoc(self, tmp_path: Path, java_language_context):
        """Test replacing a method that has Javadoc comments."""
        java_file = tmp_path / "MathUtils.java"
        original_code = """public class MathUtils {
    /**
     * Calculates the factorial.
     * @param n the number
     * @return factorial of n
     */
    public long factorial(int n) {
        if (n <= 1) return 1;
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class MathUtils {{
    /**
     * Calculates the factorial (optimized).
     * @param n the number
     * @return factorial of n
     */
    public long factorial(int n) {{
        if (n <= 1) return 1;
        long result = 1;
        for (int i = 2; i <= n; i++) {{
            result = Math.multiplyExact(result, i);
        }}
        return result;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_in_module(
            function_names=["factorial"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            preexisting_objects=set(),
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class MathUtils {
    /**
     * Calculates the factorial (optimized).
     * @param n the number
     * @return factorial of n
     */
    public long factorial(int n) {
        if (n <= 1) return 1;
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result = Math.multiplyExact(result, i);
        }
        return result;
    }
}
"""
        assert new_code == expected

    def test_no_change_when_code_identical(self, tmp_path: Path, java_language_context):
        """Test that no change is made when optimized code is identical."""
        java_file = tmp_path / "Identity.java"
        original_code = """public class Identity {
    public int getValue() {
        return 42;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Identity {{
    public int getValue() {{
        return 42;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_in_module(
            function_names=["getValue"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            preexisting_objects=set(),
            project_root_path=tmp_path,
        )

        assert result is False
        new_code = java_file.read_text(encoding="utf-8")
        assert new_code == original_code


class TestReplaceFunctionDefinitionsForLanguage:
    """Tests for replace_function_definitions_for_language with Java."""

    def test_replace_static_method(self, tmp_path: Path):
        """Test replacing a static method."""
        java_file = tmp_path / "Utils.java"
        original_code = """public class Utils {
    public static int square(int n) {
        return n * n;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Utils {{
    public static int square(int n) {{
        return Math.multiplyExact(n, n);
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["square"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Utils {
    public static int square(int n) {
        return Math.multiplyExact(n, n);
    }
}
"""
        assert new_code == expected

    def test_replace_method_with_annotations(self, tmp_path: Path):
        """Test replacing a method with annotations."""
        java_file = tmp_path / "Service.java"
        original_code = """public class Service {
    @Override
    public String process(String input) {
        return input.trim();
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Service {{
    @Override
    public String process(String input) {{
        return input == null ? "" : input.strip();
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["process"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Service {
    @Override
    public String process(String input) {
        return input == null ? "" : input.strip();
    }
}
"""
        assert new_code == expected

    def test_replace_method_in_interface(self, tmp_path: Path):
        """Test replacing a default method in an interface."""
        java_file = tmp_path / "Processor.java"
        original_code = """public interface Processor {
    default String process(String input) {
        return input.toUpperCase();
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public interface Processor {{
    default String process(String input) {{
        return input == null ? null : input.toUpperCase();
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["process"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public interface Processor {
    default String process(String input) {
        return input == null ? null : input.toUpperCase();
    }
}
"""
        assert new_code == expected

    def test_replace_method_in_enum(self, tmp_path: Path):
        """Test replacing a method in an enum."""
        java_file = tmp_path / "Color.java"
        original_code = """public enum Color {
    RED, GREEN, BLUE;

    public String getCode() {
        return name().substring(0, 1);
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public enum Color {{
    RED, GREEN, BLUE;

    public String getCode() {{
        return String.valueOf(name().charAt(0));
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["getCode"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public enum Color {
    RED, GREEN, BLUE;

    public String getCode() {
        return String.valueOf(name().charAt(0));
    }
}
"""
        assert new_code == expected

    def test_replace_generic_method(self, tmp_path: Path):
        """Test replacing a method with generics."""
        java_file = tmp_path / "Container.java"
        original_code = """import java.util.List;
import java.util.ArrayList;

public class Container<T> {
    private List<T> items = new ArrayList<>();

    public List<T> getItems() {
        List<T> copy = new ArrayList<>();
        for (T item : items) {
            copy.add(item);
        }
        return copy;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
import java.util.List;
import java.util.ArrayList;

public class Container<T> {{
    private List<T> items = new ArrayList<>();

    public List<T> getItems() {{
        return new ArrayList<>(items);
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["getItems"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """import java.util.List;
import java.util.ArrayList;

public class Container<T> {
    private List<T> items = new ArrayList<>();

    public List<T> getItems() {
        return new ArrayList<>(items);
    }
}
"""
        assert new_code == expected

    def test_replace_method_with_throws(self, tmp_path: Path):
        """Test replacing a method with throws clause."""
        java_file = tmp_path / "FileReader.java"
        original_code = """import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class FileReader {
    public String readFile(String path) throws IOException {
        return new String(Files.readAllBytes(Path.of(path)));
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class FileReader {{
    public String readFile(String path) throws IOException {{
        return Files.readString(Path.of(path));
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["readFile"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class FileReader {
    public String readFile(String path) throws IOException {
        return Files.readString(Path.of(path));
    }
}
"""
        assert new_code == expected


class TestRealWorldOptimizationScenarios:
    """Real-world optimization scenarios with complete valid Java code."""

    def test_optimize_string_concatenation(self, tmp_path: Path):
        """Test optimizing string concatenation to StringBuilder."""
        java_file = tmp_path / "StringJoiner.java"
        original_code = """public class StringJoiner {
    public String buildString(String[] items) {
        String result = "";
        for (String item : items) {
            result = result + item;
        }
        return result;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class StringJoiner {{
    public String buildString(String[] items) {{
        StringBuilder sb = new StringBuilder();
        for (String item : items) {{
            sb.append(item);
        }}
        return sb.toString();
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["buildString"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class StringJoiner {
    public String buildString(String[] items) {
        StringBuilder sb = new StringBuilder();
        for (String item : items) {
            sb.append(item);
        }
        return sb.toString();
    }
}
"""
        assert new_code == expected

    def test_optimize_list_iteration(self, tmp_path: Path):
        """Test optimizing list iteration with streams."""
        java_file = tmp_path / "ListProcessor.java"
        original_code = """import java.util.List;

public class ListProcessor {
    public int sumList(List<Integer> numbers) {
        int sum = 0;
        for (int i = 0; i < numbers.size(); i++) {
            sum += numbers.get(i);
        }
        return sum;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
import java.util.List;

public class ListProcessor {{
    public int sumList(List<Integer> numbers) {{
        return numbers.stream().mapToInt(Integer::intValue).sum();
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["sumList"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """import java.util.List;

public class ListProcessor {
    public int sumList(List<Integer> numbers) {
        return numbers.stream().mapToInt(Integer::intValue).sum();
    }
}
"""
        assert new_code == expected

    def test_optimize_null_checks(self, tmp_path: Path):
        """Test optimizing null checks with Objects utility."""
        java_file = tmp_path / "NullChecker.java"
        original_code = """public class NullChecker {
    public boolean isEqual(String s1, String s2) {
        if (s1 == null && s2 == null) {
            return true;
        }
        if (s1 == null || s2 == null) {
            return false;
        }
        return s1.equals(s2);
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
import java.util.Objects;

public class NullChecker {{
    public boolean isEqual(String s1, String s2) {{
        return Objects.equals(s1, s2);
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["isEqual"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class NullChecker {
    public boolean isEqual(String s1, String s2) {
        return Objects.equals(s1, s2);
    }
}
"""
        assert new_code == expected

    def test_optimize_collection_creation(self, tmp_path: Path):
        """Test optimizing collection creation with factory methods."""
        java_file = tmp_path / "CollectionFactory.java"
        original_code = """import java.util.ArrayList;
import java.util.List;

public class CollectionFactory {
    public List<String> createList() {
        List<String> list = new ArrayList<>();
        list.add("one");
        list.add("two");
        list.add("three");
        return list;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
import java.util.ArrayList;
import java.util.List;

public class CollectionFactory {{
    public List<String> createList() {{
        return List.of("one", "two", "three");
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["createList"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """import java.util.ArrayList;
import java.util.List;

public class CollectionFactory {
    public List<String> createList() {
        return List.of("one", "two", "three");
    }
}
"""
        assert new_code == expected


class TestMultipleClassesAndMethods:
    """Tests for files with multiple classes or multiple methods being optimized."""

    def test_replace_method_in_first_class(self, tmp_path: Path):
        """Test replacing a method in the first class when multiple classes exist."""
        java_file = tmp_path / "MultiClass.java"
        original_code = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}

class Helper {
    public int helper() {
        return 0;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Calculator {{
    public int add(int a, int b) {{
        return Math.addExact(a, b);
    }}
}}

class Helper {{
    public int helper() {{
        return 0;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["add"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Calculator {
    public int add(int a, int b) {
        return Math.addExact(a, b);
    }
}

class Helper {
    public int helper() {
        return 0;
    }
}
"""
        assert new_code == expected

    def test_replace_multiple_methods(self, tmp_path: Path):
        """Test replacing multiple methods in the same class."""
        java_file = tmp_path / "MathOps.java"
        original_code = """public class MathOps {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class MathOps {{
    public int add(int a, int b) {{
        return Math.addExact(a, b);
    }}

    public int subtract(int a, int b) {{
        return Math.subtractExact(a, b);
    }}

    public int multiply(int a, int b) {{
        return a * b;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["add", "subtract"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class MathOps {
    public int add(int a, int b) {
        return Math.addExact(a, b);
    }

    public int subtract(int a, int b) {
        return Math.subtractExact(a, b);
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"""
        assert new_code == expected


class TestNestedClasses:
    """Tests for nested class scenarios."""

    def test_replace_method_in_nested_class(self, tmp_path: Path):
        """Test replacing a method in a nested class."""
        java_file = tmp_path / "Outer.java"
        original_code = """public class Outer {
    public int outerMethod() {
        return 1;
    }

    public static class Inner {
        public int innerMethod() {
            return 2;
        }
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Outer {{
    public int outerMethod() {{
        return 1;
    }}

    public static class Inner {{
        public int innerMethod() {{
            return 2 + 0;
        }}
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["innerMethod"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Outer {
    public int outerMethod() {
        return 1;
    }

    public static class Inner {
        public int innerMethod() {
            return 2 + 0;
        }
    }
}
"""
        assert new_code == expected


class TestPreservesStructure:
    """Tests that verify code structure is preserved during replacement."""

    def test_preserves_fields_and_constructors(self, tmp_path: Path):
        """Test that fields and constructors are preserved."""
        java_file = tmp_path / "Counter.java"
        original_code = """public class Counter {
    private int count;
    private final int max;

    public Counter(int max) {
        this.count = 0;
        this.max = max;
    }

    public int increment() {
        if (count < max) {
            count++;
        }
        return count;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Counter {{
    private int count;
    private final int max;

    public Counter(int max) {{
        this.count = 0;
        this.max = max;
    }}

    public int increment() {{
        return count < max ? ++count : count;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["increment"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Counter {
    private int count;
    private final int max;

    public Counter(int max) {
        this.count = 0;
        this.max = max;
    }

    public int increment() {
        return count < max ? ++count : count;
    }
}
"""
        assert new_code == expected


class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_empty_optimized_code_returns_false(self, tmp_path: Path):
        """Test that empty optimized code returns False."""
        java_file = tmp_path / "Empty.java"
        original_code = """public class Empty {
    public int getValue() {
        return 42;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = """```java:Empty.java
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["getValue"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is False
        new_code = java_file.read_text(encoding="utf-8")
        assert new_code == original_code

    def test_function_not_found_returns_false(self, tmp_path: Path):
        """Test that function not found returns False."""
        java_file = tmp_path / "NotFound.java"
        original_code = """public class NotFound {
    public int getValue() {
        return 42;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class NotFound {{
    public int nonExistent() {{
        return 0;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["nonExistent"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is False

    def test_unicode_in_code(self, tmp_path: Path):
        """Test handling of unicode characters in code."""
        java_file = tmp_path / "Unicode.java"
        original_code = """public class Unicode {
    public String greet() {
        return "Hello";
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Unicode {{
    public String greet() {{
        return "こんにちは";
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["greet"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Unicode {
    public String greet() {
        return "こんにちは";
    }
}
"""
        assert new_code == expected


class TestOptimizationWithStaticFields:
    """Tests for optimizations that add new static fields to the class."""

    def test_add_static_lookup_table(self, tmp_path: Path):
        """Test optimization that adds a static lookup table."""
        java_file = tmp_path / "Buffer.java"
        original_code = """public class Buffer {
    public static String bytesToHexString(byte[] buf, int offset, int length) {
        StringBuilder sb = new StringBuilder(length * 2);
        for (int i = offset; i < length; i++) {
            sb.append(String.format("%02x", buf[i]));
        }
        return sb.toString();
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # Optimization adds a static lookup table
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Buffer {{
    private static final char[] HEX_DIGITS = "0123456789abcdef".toCharArray();

    public static String bytesToHexString(byte[] buf, int offset, int length) {{
        StringBuilder sb = new StringBuilder(length * 2);
        for (int i = offset; i < length; i++) {{
            int v = buf[i] & 0xFF;
            sb.append(HEX_DIGITS[v >>> 4]);
            sb.append(HEX_DIGITS[v & 0x0F]);
        }}
        return sb.toString();
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["bytesToHexString"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Buffer {
    private static final char[] HEX_DIGITS = "0123456789abcdef".toCharArray();

    public static String bytesToHexString(byte[] buf, int offset, int length) {
        StringBuilder sb = new StringBuilder(length * 2);
        for (int i = offset; i < length; i++) {
            int v = buf[i] & 0xFF;
            sb.append(HEX_DIGITS[v >>> 4]);
            sb.append(HEX_DIGITS[v & 0x0F]);
        }
        return sb.toString();
    }
}
"""
        assert new_code == expected

    def test_add_precomputed_array(self, tmp_path: Path):
        """Test optimization that adds a precomputed static array."""
        java_file = tmp_path / "Encoder.java"
        original_code = """public class Encoder {
    public static String byteToHex(byte b) {
        return String.format("%02x", b);
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # Optimization with precomputed byte-to-hex lookup
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Encoder {{
    private static final String[] BYTE_TO_HEX = createByteToHex();

    private static String[] createByteToHex() {{
        String[] map = new String[256];
        for (int i = 0; i < 256; i++) {{
            map[i] = String.format("%02x", i);
        }}
        return map;
    }}

    public static String byteToHex(byte b) {{
        return BYTE_TO_HEX[b & 0xFF];
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["byteToHex"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Encoder {
    private static final String[] BYTE_TO_HEX = createByteToHex();

    private static String[] createByteToHex() {
        String[] map = new String[256];
        for (int i = 0; i < 256; i++) {
            map[i] = String.format("%02x", i);
        }
        return map;
    }

    public static String byteToHex(byte b) {
        return BYTE_TO_HEX[b & 0xFF];
    }
}
"""
        assert new_code == expected

    def test_preserve_existing_fields(self, tmp_path: Path):
        """Test that existing fields are preserved when adding new ones."""
        java_file = tmp_path / "Calculator.java"
        original_code = """public class Calculator {
    private static final int MAX_VALUE = 1000;

    public int calculate(int n) {
        int result = 0;
        for (int i = 0; i < n; i++) {
            result += i;
        }
        return result;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # Optimization adds a new static field
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Calculator {{
    private static final int MAX_VALUE = 1000;
    private static final int[] PRECOMPUTED = precompute();

    private static int[] precompute() {{
        int[] arr = new int[1001];
        for (int i = 1; i <= 1000; i++) {{
            arr[i] = arr[i-1] + i - 1;
        }}
        return arr;
    }}

    public int calculate(int n) {{
        if (n <= 1000) {{
            return PRECOMPUTED[n];
        }}
        int result = PRECOMPUTED[1000];
        for (int i = 1000; i < n; i++) {{
            result += i;
        }}
        return result;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["calculate"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Calculator {
    private static final int MAX_VALUE = 1000;
    private static final int[] PRECOMPUTED = precompute();

    private static int[] precompute() {
        int[] arr = new int[1001];
        for (int i = 1; i <= 1000; i++) {
            arr[i] = arr[i-1] + i - 1;
        }
        return arr;
    }

    public int calculate(int n) {
        if (n <= 1000) {
            return PRECOMPUTED[n];
        }
        int result = PRECOMPUTED[1000];
        for (int i = 1000; i < n; i++) {
            result += i;
        }
        return result;
    }
}
"""
        assert new_code == expected


class TestOptimizationWithHelperMethods:
    """Tests for optimizations that add new helper methods."""

    def test_add_private_helper_method(self, tmp_path: Path):
        """Test optimization that adds a private helper method."""
        java_file = tmp_path / "StringUtils.java"
        original_code = """public class StringUtils {
    public static String reverse(String s) {
        char[] chars = s.toCharArray();
        int left = 0;
        int right = chars.length - 1;
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
        return new String(chars);
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # Optimization extracts swap logic to helper
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class StringUtils {{
    private static void swap(char[] arr, int i, int j) {{
        char temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }}

    public static String reverse(String s) {{
        char[] chars = s.toCharArray();
        for (int i = 0, j = chars.length - 1; i < j; i++, j--) {{
            swap(chars, i, j);
        }}
        return new String(chars);
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["reverse"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class StringUtils {
    private static void swap(char[] arr, int i, int j) {
        char temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    public static String reverse(String s) {
        char[] chars = s.toCharArray();
        for (int i = 0, j = chars.length - 1; i < j; i++, j--) {
            swap(chars, i, j);
        }
        return new String(chars);
    }
}
"""
        assert new_code == expected

    def test_add_multiple_helpers(self, tmp_path: Path):
        """Test optimization that adds multiple helper methods."""
        java_file = tmp_path / "MathUtils.java"
        original_code = """public class MathUtils {
    public static int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # Optimization adds multiple helper methods
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class MathUtils {{
    private static int abs(int x) {{
        return x < 0 ? -x : x;
    }}

    private static int gcdInternal(int a, int b) {{
        return b == 0 ? a : gcdInternal(b, a % b);
    }}

    public static int gcd(int a, int b) {{
        return gcdInternal(abs(a), abs(b));
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["gcd"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class MathUtils {
    private static int abs(int x) {
        return x < 0 ? -x : x;
    }

    private static int gcdInternal(int a, int b) {
        return b == 0 ? a : gcdInternal(b, a % b);
    }

    public static int gcd(int a, int b) {
        return gcdInternal(abs(a), abs(b));
    }
}
"""
        assert new_code == expected


class TestOptimizationWithFieldsAndHelpers:
    """Tests for optimizations that add both static fields and helper methods."""

    def test_add_field_and_helper_together(self, tmp_path: Path):
        """Test optimization that adds both a static field and helper method."""
        java_file = tmp_path / "Fibonacci.java"
        original_code = """public class Fibonacci {
    public static long fib(int n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # Optimization with memoization using static field and helper
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public class Fibonacci {{
    private static final long[] CACHE = new long[100];
    private static final boolean[] COMPUTED = new boolean[100];

    private static long fibMemo(int n) {{
        if (n <= 1) return n;
        if (n < 100 && COMPUTED[n]) return CACHE[n];
        long result = fibMemo(n - 1) + fibMemo(n - 2);
        if (n < 100) {{
            CACHE[n] = result;
            COMPUTED[n] = true;
        }}
        return result;
    }}

    public static long fib(int n) {{
        return fibMemo(n);
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["fib"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public class Fibonacci {
    private static final long[] CACHE = new long[100];
    private static final boolean[] COMPUTED = new boolean[100];

    private static long fibMemo(int n) {
        if (n <= 1) return n;
        if (n < 100 && COMPUTED[n]) return CACHE[n];
        long result = fibMemo(n - 1) + fibMemo(n - 2);
        if (n < 100) {
            CACHE[n] = result;
            COMPUTED[n] = true;
        }
        return result;
    }

    public static long fib(int n) {
        return fibMemo(n);
    }
}
"""
        assert new_code == expected

    def test_real_world_bytes_to_hex_optimization(self, tmp_path: Path):
        """Test the actual bytesToHexString optimization pattern from aerospike."""
        java_file = tmp_path / "Buffer.java"
        original_code = """package com.example;

public final class Buffer {
    public static String bytesToHexString(byte[] buf, int offset, int length) {
        StringBuilder sb = new StringBuilder(length * 2);

        for (int i = offset; i < length; i++) {
            sb.append(String.format("%02x", buf[i]));
        }
        return sb.toString();
    }

    public static int otherMethod() {
        return 42;
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # The actual optimization pattern generated by the AI
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
package com.example;

public final class Buffer {{
    private static final String[] BYTE_TO_HEX = createByteToHex();

    private static String[] createByteToHex() {{
        String[] map = new String[256];
        for (int b = -128; b <= 127; b++) {{
            map[b + 128] = String.format("%02x", (byte) b);
        }}
        return map;
    }}

    public static String bytesToHexString(byte[] buf, int offset, int length) {{
        StringBuilder sb = new StringBuilder(length * 2);

        for (int i = offset; i < length; i++) {{
            sb.append(BYTE_TO_HEX[buf[i] + 128]);
        }}
        return sb.toString();
    }}

    public static int otherMethod() {{
        return 42;
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        result = replace_function_definitions_for_language(
            function_names=["bytesToHexString"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """package com.example;

public final class Buffer {
    private static final String[] BYTE_TO_HEX = createByteToHex();

    private static String[] createByteToHex() {
        String[] map = new String[256];
        for (int b = -128; b <= 127; b++) {
            map[b + 128] = String.format("%02x", (byte) b);
        }
        return map;
    }

    public static String bytesToHexString(byte[] buf, int offset, int length) {
        StringBuilder sb = new StringBuilder(length * 2);

        for (int i = offset; i < length; i++) {
            sb.append(BYTE_TO_HEX[buf[i] + 128]);
        }
        return sb.toString();
    }

    public static int otherMethod() {
        return 42;
    }
}
"""
        assert new_code == expected


class TestOverloadedMethods:
    """Tests for handling overloaded methods (same name, different signatures)."""

    def test_replace_specific_overload_by_line_number(self, tmp_path: Path):
        """Test replacing a specific overload when multiple exist."""
        java_file = tmp_path / "Buffer.java"
        original_code = """public final class Buffer {
    public static String bytesToHexString(byte[] buf) {
        if (buf == null || buf.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder(buf.length * 2);
        for (int i = 0; i < buf.length; i++) {
            sb.append(String.format("%02x", buf[i]));
        }
        return sb.toString();
    }

    public static String bytesToHexString(byte[] buf, int offset, int length) {
        StringBuilder sb = new StringBuilder(length * 2);
        for (int i = offset; i < length; i++) {
            sb.append(String.format("%02x", buf[i]));
        }
        return sb.toString();
    }
}
"""
        java_file.write_text(original_code, encoding="utf-8")

        # Optimization only for the 3-argument version
        optimized_markdown = f"""```java:{java_file.relative_to(tmp_path)}
public final class Buffer {{
    private static final char[] HEX_CHARS = {{'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'}};

    public static String bytesToHexString(byte[] buf, int offset, int length) {{
        char[] out = new char[(length - offset) * 2];
        for (int i = offset, j = 0; i < length; i++) {{
            int v = buf[i] & 0xFF;
            out[j++] = HEX_CHARS[v >>> 4];
            out[j++] = HEX_CHARS[v & 0x0F];
        }}
        return new String(out);
    }}
}}
```"""

        optimized_code = CodeStringsMarkdown.parse_markdown_code(optimized_markdown, expected_language="java")

        # Create FunctionToOptimize with line info for the 3-arg version (lines 13-18)
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize, FunctionParent

        function_to_optimize = FunctionToOptimize(
            function_name="bytesToHexString",
            file_path=java_file,
            starting_line=13,  # Line where 3-arg version starts (1-indexed)
            ending_line=18,
            parents=[FunctionParent(name="Buffer", type="ClassDef")],
            qualified_name="Buffer.bytesToHexString",
            is_method=True,
        )

        result = replace_function_definitions_for_language(
            function_names=["bytesToHexString"],
            optimized_code=optimized_code,
            module_abspath=java_file,
            project_root_path=tmp_path,
            function_to_optimize=function_to_optimize,
        )

        assert result is True
        new_code = java_file.read_text(encoding="utf-8")
        expected = """public final class Buffer {
    private static final char[] HEX_CHARS = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

    public static String bytesToHexString(byte[] buf) {
        if (buf == null || buf.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder(buf.length * 2);
        for (int i = 0; i < buf.length; i++) {
            sb.append(String.format("%02x", buf[i]));
        }
        return sb.toString();
    }

    public static String bytesToHexString(byte[] buf, int offset, int length) {
        char[] out = new char[(length - offset) * 2];
        for (int i = offset, j = 0; i < length; i++) {
            int v = buf[i] & 0xFF;
            out[j++] = HEX_CHARS[v >>> 4];
            out[j++] = HEX_CHARS[v & 0x0F];
        }
        return new String(out);
    }
}
"""
        assert new_code == expected

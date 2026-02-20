"""End-to-end integration tests for Java pipeline.

Tests the full optimization pipeline for Java:
- Function discovery
- Code context extraction
- Test discovery
- Code replacement
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import find_all_functions_in_file, get_files_for_language
from codeflash.languages.base import Language


class TestJavaFunctionDiscovery:
    """Tests for Java function discovery in the main pipeline."""

    @pytest.fixture
    def java_project_dir(self):
        """Get the Java sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        java_dir = project_root / "code_to_optimize" / "java"
        if not java_dir.exists():
            pytest.skip("code_to_optimize/java directory not found")
        return java_dir

    def test_discover_functions_in_bubble_sort(self, java_project_dir):
        """Test discovering functions in BubbleSort.java."""
        sort_file = java_project_dir / "src" / "main" / "java" / "com" / "example" / "BubbleSort.java"
        if not sort_file.exists():
            pytest.skip("BubbleSort.java not found")

        functions = find_all_functions_in_file(sort_file)

        assert sort_file in functions
        func_list = functions[sort_file]

        # Should find the sorting methods
        func_names = {f.function_name for f in func_list}
        assert "bubbleSort" in func_names
        assert "bubbleSortDescending" in func_names
        assert "insertionSort" in func_names
        assert "selectionSort" in func_names
        assert "isSorted" in func_names

        # All should be Java methods
        for func in func_list:
            assert func.language == "java"

    def test_discover_functions_in_calculator(self, java_project_dir):
        """Test discovering functions in Calculator.java."""
        calc_file = java_project_dir / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        if not calc_file.exists():
            pytest.skip("Calculator.java not found")

        functions = find_all_functions_in_file(calc_file)

        assert calc_file in functions
        func_list = functions[calc_file]

        func_names = {f.function_name for f in func_list}
        assert "add" in func_names or len(func_names) > 0  # Should find at least some methods

    def test_get_java_files(self, java_project_dir):
        """Test getting Java files from directory."""
        source_dir = java_project_dir / "src" / "main" / "java"
        files = get_files_for_language(source_dir, Language.JAVA)

        # Should find .java files
        java_files = [f for f in files if f.suffix == ".java"]
        assert len(java_files) >= 5  # BubbleSort, Calculator, etc.


class TestJavaCodeContext:
    """Tests for Java code context extraction."""

    @pytest.fixture
    def java_project_dir(self):
        """Get the Java sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        java_dir = project_root / "code_to_optimize" / "java"
        if not java_dir.exists():
            pytest.skip("code_to_optimize/java directory not found")
        return java_dir

    def test_extract_code_context_for_java(self, java_project_dir):
        """Test extracting code context for a Java method."""
        from codeflash.languages.python.context.code_context_extractor import get_code_optimization_context
        from codeflash.languages import current as lang_current
        from codeflash.languages.base import Language

        # Force set language to Java for proper context extraction routing
        lang_current._current_language = Language.JAVA

        sort_file = java_project_dir / "src" / "main" / "java" / "com" / "example" / "BubbleSort.java"
        if not sort_file.exists():
            pytest.skip("BubbleSort.java not found")

        functions = find_all_functions_in_file(sort_file)
        func_list = functions[sort_file]

        # Find the bubbleSort method
        bubble_func = next((f for f in func_list if f.function_name == "bubbleSort"), None)
        assert bubble_func is not None

        # Extract code context
        context = get_code_optimization_context(bubble_func, java_project_dir)

        # Verify context structure
        assert context.read_writable_code is not None
        assert context.read_writable_code.language == "java"
        assert len(context.read_writable_code.code_strings) > 0

        # The code should contain the method
        code = context.read_writable_code.code_strings[0].code
        assert "bubbleSort" in code


class TestJavaCodeReplacement:
    """Tests for Java code replacement."""

    def test_replace_method_in_java_file(self):
        """Test replacing a method in a Java file."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo, Language, ParentInfo

        original_source = """package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"""

        new_method = """public int add(int a, int b) {
        // Optimized version
        return a + b;
    }"""

        java_support = get_language_support(Language.JAVA)

        # Create FunctionInfo for the add method with parent class
        func_info = FunctionInfo(
            function_name="add",
            file_path=Path("/tmp/Calculator.java"),
            starting_line=4,
            ending_line=6,
            language=Language.JAVA,
            parents=(ParentInfo(name="Calculator", type="ClassDef"),),
        )

        result = java_support.replace_function(original_source, func_info, new_method)

        # Verify the method was replaced
        assert "// Optimized version" in result
        assert "multiply" in result  # Other method should still be there


class TestJavaTestDiscovery:
    """Tests for Java test discovery."""

    @pytest.fixture
    def java_project_dir(self):
        """Get the Java sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        java_dir = project_root / "code_to_optimize" / "java"
        if not java_dir.exists():
            pytest.skip("code_to_optimize/java directory not found")
        return java_dir

    def test_discover_junit_tests(self, java_project_dir):
        """Test discovering JUnit tests for Java methods."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo, Language, ParentInfo

        java_support = get_language_support(Language.JAVA)
        test_root = java_project_dir / "src" / "test" / "java"

        if not test_root.exists():
            pytest.skip("test directory not found")

        # Create FunctionInfo for bubbleSort method with parent class
        sort_file = java_project_dir / "src" / "main" / "java" / "com" / "example" / "BubbleSort.java"
        func_info = FunctionInfo(
            function_name="bubbleSort",
            file_path=sort_file,
            starting_line=14,
            ending_line=37,
            language=Language.JAVA,
            parents=(ParentInfo(name="BubbleSort", type="ClassDef"),),
        )

        # Discover tests
        tests = java_support.discover_tests(test_root, [func_info])

        # Should find tests for bubbleSort
        assert func_info.qualified_name in tests or "bubbleSort" in str(tests)


class TestJavaPipelineIntegration:
    """Integration tests for the full Java pipeline."""

    def test_function_to_optimize_has_correct_fields(self):
        """Test that FunctionToOptimize from Java has all required fields."""
        with tempfile.NamedTemporaryFile(suffix=".java", mode="w", delete=False) as f:
            f.write("""package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }

    public static int multiply(int x, int y) {
        return x * y;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            # Should find class methods
            assert len(functions.get(file_path, [])) >= 3

            # Check instance method
            add_fn = next((fn for fn in functions[file_path] if fn.function_name == "add"), None)
            assert add_fn is not None
            assert add_fn.language == "java"
            assert len(add_fn.parents) == 1
            assert add_fn.parents[0].name == "Calculator"

            # Check static method
            multiply_fn = next((fn for fn in functions[file_path] if fn.function_name == "multiply"), None)
            assert multiply_fn is not None
            assert multiply_fn.language == "java"

    def test_code_strings_markdown_uses_java_tag(self):
        """Test that CodeStringsMarkdown uses java for code blocks."""
        from codeflash.models.models import CodeString, CodeStringsMarkdown

        code_strings = CodeStringsMarkdown(
            code_strings=[
                CodeString(
                    code="public int add(int a, int b) { return a + b; }",
                    file_path=Path("Calculator.java"),
                    language="java",
                )
            ],
            language="java",
        )

        markdown = code_strings.markdown
        assert "```java" in markdown


class TestJavaProjectDetection:
    """Tests for Java project detection."""

    @pytest.fixture
    def java_project_dir(self):
        """Get the Java sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        java_dir = project_root / "code_to_optimize" / "java"
        if not java_dir.exists():
            pytest.skip("code_to_optimize/java directory not found")
        return java_dir

    def test_detect_maven_project(self, java_project_dir):
        """Test detecting Maven project structure."""
        from codeflash.languages.java.config import detect_java_project

        config = detect_java_project(java_project_dir)

        assert config is not None
        assert config.source_root is not None
        assert config.test_root is not None
        assert config.has_junit5 is True


class TestJavaCompilation:
    """Tests for Java compilation."""

    @pytest.fixture
    def java_project_dir(self):
        """Get the Java sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        java_dir = project_root / "code_to_optimize" / "java"
        if not java_dir.exists():
            pytest.skip("code_to_optimize/java directory not found")
        return java_dir

    @pytest.mark.slow
    def test_compile_java_project(self, java_project_dir):
        """Test that the sample Java project compiles successfully."""
        import subprocess

        # Check if Maven is available
        try:
            result = subprocess.run(["mvn", "--version"], capture_output=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Maven not available")
        except FileNotFoundError:
            pytest.skip("Maven not installed")

        # Compile the project
        result = subprocess.run(
            ["mvn", "compile", "-q"],
            cwd=java_project_dir,
            capture_output=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Compilation failed: {result.stderr.decode()}"

    @pytest.mark.slow
    def test_run_java_tests(self, java_project_dir):
        """Test that the sample Java tests run successfully."""
        import subprocess

        # Check if Maven is available
        try:
            result = subprocess.run(["mvn", "--version"], capture_output=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Maven not available")
        except FileNotFoundError:
            pytest.skip("Maven not installed")

        # Run tests
        result = subprocess.run(
            ["mvn", "test", "-q"],
            cwd=java_project_dir,
            capture_output=True,
            timeout=180,
        )

        assert result.returncode == 0, f"Tests failed: {result.stderr.decode()}"

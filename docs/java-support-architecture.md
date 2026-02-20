# Java Language Support Architecture for CodeFlash

## Executive Summary

Adding Java support to CodeFlash requires implementing the `LanguageSupport` protocol with Java-specific components for parsing, test discovery, context extraction, and test execution. The existing architecture is well-designed for multi-language support, and Java can follow the established patterns from Python and JavaScript/TypeScript.

---

## 1. Architecture Overview

### Current Language Support Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Core Optimization Pipeline                    │
│  (language-agnostic: optimizer.py, function_optimizer.py)        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   LanguageSupport     │
                    │      Protocol         │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ PythonSupport │     │JavaScriptSupport│     │   JavaSupport   │
│   (mature)    │     │  (functional)   │     │    (NEW)        │
├───────────────┤     ├─────────────────┤     ├─────────────────┤
│ - libcst      │     │ - tree-sitter   │     │ - tree-sitter   │
│ - pytest      │     │ - jest          │     │ - JUnit 5       │
│ - Jedi        │     │ - npm/yarn      │     │ - Maven/Gradle  │
└───────────────┘     └─────────────────┘     └─────────────────┘
```

### Proposed Java Module Structure

```
codeflash/languages/java/
├── __init__.py              # Module exports, register language
├── support.py               # JavaSupport class (main implementation)
├── parser.py                # Tree-sitter Java parsing utilities
├── discovery.py             # Function/method discovery
├── context_extractor.py     # Code context extraction
├── import_resolver.py       # Java import/dependency resolution
├── instrument.py            # Test instrumentation
├── test_runner.py           # JUnit test execution
├── comparator.py            # Test result comparison
├── build_tools.py           # Maven/Gradle integration
├── formatter.py             # Code formatting (google-java-format)
└── line_profiler.py         # JProfiler/async-profiler integration
```

---

## 2. Core Components

### 2.1 Language Registration

```python
# codeflash/languages/java/support.py

from codeflash.languages.base import Language, LanguageSupport
from codeflash.languages.registry import register_language

@register_language
class JavaSupport:
    @property
    def language(self) -> Language:
        return Language.JAVA  # Add to Language enum

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".java",)

    @property
    def test_framework(self) -> str:
        return "junit"

    @property
    def comment_prefix(self) -> str:
        return "//"
```

### 2.2 Language Enum Extension

```python
# codeflash/languages/base.py

class Language(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"  # NEW
```

---

## 3. Component Implementation Details

### 3.1 Parsing (tree-sitter-java)

**File: `codeflash/languages/java/parser.py`**

Tree-sitter has excellent Java support. Key node types to handle:

| Java Construct | Tree-sitter Node Type |
|----------------|----------------------|
| Class | `class_declaration` |
| Interface | `interface_declaration` |
| Method | `method_declaration` |
| Constructor | `constructor_declaration` |
| Static block | `static_initializer` |
| Lambda | `lambda_expression` |
| Anonymous class | `anonymous_class_body` |
| Annotation | `annotation` |
| Generic type | `type_parameters` |

```python
class JavaParser:
    """Tree-sitter based Java parser."""

    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(tree_sitter_java.language())

    def find_methods(self, source: str) -> list[MethodNode]:
        """Find all method declarations."""
        tree = self.parser.parse(source.encode())
        return self._walk_for_methods(tree.root_node)

    def find_classes(self, source: str) -> list[ClassNode]:
        """Find all class/interface declarations."""
        ...

    def get_method_signature(self, node: Node) -> MethodSignature:
        """Extract method signature including generics."""
        ...
```

### 3.2 Function Discovery

**File: `codeflash/languages/java/discovery.py`**

Java-specific considerations:
- Methods are always inside classes (no top-level functions)
- Need to handle: instance methods, static methods, constructors
- Interface default methods
- Annotation processing (`@Override`, `@Test`, etc.)
- Inner classes and nested methods

```python
def discover_functions(
    file_path: Path,
    criteria: FunctionFilterCriteria | None = None
) -> list[FunctionInfo]:
    """
    Discover optimizable methods in a Java file.

    Returns methods that are:
    - Public or protected (can be tested)
    - Not abstract
    - Not native
    - Not in test files
    - Not trivial (getters/setters unless specifically requested)
    """
    parser = JavaParser()
    source = file_path.read_text(encoding="utf-8")

    methods = []
    for class_node in parser.find_classes(source):
        for method in class_node.methods:
            if _should_include_method(method, criteria):
                methods.append(FunctionInfo(
                    name=method.name,
                    file_path=file_path,
                    start_line=method.start_line,
                    end_line=method.end_line,
                    parents=(ParentInfo(
                        name=class_node.name,
                        type="ClassDeclaration"
                    ),),
                    is_async=method.has_annotation("Async"),
                    is_method=True,
                    language=Language.JAVA,
                ))
    return methods
```

### 3.3 Code Context Extraction

**File: `codeflash/languages/java/context_extractor.py`**

Java context extraction must handle:
- Full class context (methods often depend on fields)
- Import statements (crucial for compilation)
- Package declarations
- Type hierarchy (extends/implements)
- Inner classes
- Static imports

```python
def extract_code_context(
    function: FunctionInfo,
    project_root: Path,
    module_root: Path | None = None
) -> CodeContext:
    """
    Extract code context for a Java method.

    Context includes:
    1. Full containing class (target method needs class context)
    2. All imports from the file
    3. Helper classes from same package
    4. Superclass/interface definitions (read-only)
    """
    source = function.file_path.read_text(encoding="utf-8")
    parser = JavaParser()

    # Extract package and imports
    package_name = parser.get_package(source)
    imports = parser.get_imports(source)

    # Get the containing class
    class_source = parser.extract_class_containing_method(
        source, function.name, function.start_line
    )

    # Find helper classes (same package, used by target class)
    helper_classes = find_helper_classes(
        function.file_path.parent,
        class_source,
        imports
    )

    return CodeContext(
        target_code=class_source,
        target_file=function.file_path,
        helper_functions=helper_classes,
        read_only_context=get_superclass_context(imports, project_root),
        imports=imports,
        language=Language.JAVA,
    )
```

### 3.4 Import/Dependency Resolution

**File: `codeflash/languages/java/import_resolver.py`**

Java import resolution is more complex:
- Explicit imports (`import com.foo.Bar;`)
- Wildcard imports (`import com.foo.*;`)
- Static imports (`import static com.foo.Bar.method;`)
- Same-package classes (implicit)
- Standard library vs external dependencies

```python
class JavaImportResolver:
    """Resolve Java imports to source files."""

    def __init__(self, project_root: Path, build_tool: BuildTool):
        self.project_root = project_root
        self.build_tool = build_tool
        self.source_roots = self._find_source_roots()
        self.classpath = build_tool.get_classpath()

    def resolve_import(self, import_stmt: str) -> ResolvedImport:
        """
        Resolve an import to its source location.

        Returns:
        - Source file path (if in project)
        - JAR location (if external dependency)
        - None (if JDK class)
        """
        ...

    def find_same_package_classes(self, package: str) -> list[Path]:
        """Find all classes in the same package."""
        ...
```

### 3.5 Test Discovery

**File: `codeflash/languages/java/support.py` (part of JavaSupport)**

Java test discovery for JUnit 5:

```python
def discover_tests(
    self,
    test_root: Path,
    source_functions: list[FunctionInfo]
) -> dict[str, list[TestInfo]]:
    """
    Discover JUnit tests that cover target methods.

    Strategy:
    1. Find test files by naming convention (*Test.java, *Tests.java)
    2. Parse test files for @Test annotated methods
    3. Analyze test code for method calls to target methods
    4. Match tests to source methods
    """
    test_files = self._find_test_files(test_root)
    test_map: dict[str, list[TestInfo]] = defaultdict(list)

    for test_file in test_files:
        parser = JavaParser()
        source = test_file.read_text()

        for test_method in parser.find_test_methods(source):
            # Find which source methods this test calls
            called_methods = parser.find_method_calls(test_method.body)

            for source_func in source_functions:
                if source_func.name in called_methods:
                    test_map[source_func.qualified_name].append(TestInfo(
                        test_name=test_method.name,
                        test_file=test_file,
                        test_class=test_method.class_name,
                    ))

    return test_map
```

### 3.6 Test Execution

**File: `codeflash/languages/java/test_runner.py`**

JUnit test execution with Maven/Gradle:

```python
class JavaTestRunner:
    """Run JUnit tests via Maven or Gradle."""

    def __init__(self, project_root: Path):
        self.build_tool = detect_build_tool(project_root)
        self.project_root = project_root

    def run_tests(
        self,
        test_classes: list[str],
        timeout: int = 60,
        capture_output: bool = True
    ) -> TestExecutionResult:
        """
        Run specified JUnit tests.

        Uses:
        - Maven: mvn test -Dtest=ClassName#methodName
        - Gradle: ./gradlew test --tests "ClassName.methodName"
        """
        if self.build_tool == BuildTool.MAVEN:
            return self._run_maven_tests(test_classes, timeout)
        else:
            return self._run_gradle_tests(test_classes, timeout)

    def _run_maven_tests(self, tests: list[str], timeout: int) -> TestExecutionResult:
        cmd = [
            "mvn", "test",
            f"-Dtest={','.join(tests)}",
            "-Dmaven.test.failure.ignore=true",
            "-DfailIfNoTests=false",
        ]
        result = subprocess.run(cmd, cwd=self.project_root, ...)
        return self._parse_surefire_reports()

    def _parse_surefire_reports(self) -> TestExecutionResult:
        """Parse target/surefire-reports/*.xml for test results."""
        ...
```

### 3.7 Code Instrumentation

**File: `codeflash/languages/java/instrument.py`**

Java instrumentation for behavior capture:

```python
class JavaInstrumenter:
    """Instrument Java code for behavior/performance capture."""

    def instrument_for_behavior(
        self,
        source: str,
        target_methods: list[str]
    ) -> str:
        """
        Add instrumentation to capture method inputs/outputs.

        Adds:
        - CodeFlash.captureInput(args) before method body
        - CodeFlash.captureOutput(result) before returns
        - Exception capture in catch blocks
        """
        parser = JavaParser()
        tree = parser.parse(source)

        # Insert capture calls using tree-sitter edit operations
        edits = []
        for method in parser.find_methods_by_name(tree, target_methods):
            edits.append(self._create_input_capture(method))
            edits.append(self._create_output_capture(method))

        return apply_edits(source, edits)

    def instrument_for_benchmarking(
        self,
        test_source: str,
        target_method: str,
        iterations: int = 1000
    ) -> str:
        """
        Add timing instrumentation to test code.

        Wraps test execution in timing loop with warmup.
        """
        ...
```

### 3.8 Build Tool Integration

**File: `codeflash/languages/java/build_tools.py`**

Maven and Gradle support:

```python
class BuildTool(Enum):
    MAVEN = "maven"
    GRADLE = "gradle"

def detect_build_tool(project_root: Path) -> BuildTool:
    """Detect whether project uses Maven or Gradle."""
    if (project_root / "pom.xml").exists():
        return BuildTool.MAVEN
    elif (project_root / "build.gradle").exists() or \
         (project_root / "build.gradle.kts").exists():
        return BuildTool.GRADLE
    raise ValueError("No Maven or Gradle build file found")

class MavenIntegration:
    """Maven build tool integration."""

    def __init__(self, project_root: Path):
        self.pom_path = project_root / "pom.xml"
        self.project_root = project_root

    def get_source_roots(self) -> list[Path]:
        """Get configured source directories."""
        # Default: src/main/java, src/test/java
        ...

    def get_classpath(self) -> list[Path]:
        """Get full classpath including dependencies."""
        result = subprocess.run(
            ["mvn", "dependency:build-classpath", "-q", "-DincludeScope=test"],
            cwd=self.project_root,
            capture_output=True
        )
        return [Path(p) for p in result.stdout.decode().split(":")]

    def compile(self, include_tests: bool = True) -> bool:
        """Compile the project."""
        cmd = ["mvn", "compile"]
        if include_tests:
            cmd.append("test-compile")
        return subprocess.run(cmd, cwd=self.project_root).returncode == 0

class GradleIntegration:
    """Gradle build tool integration."""
    # Similar implementation for Gradle
    ...
```

### 3.9 Code Replacement

**File: `codeflash/languages/java/support.py`**

```python
def replace_function(
    self,
    source: str,
    function: FunctionInfo,
    new_source: str
) -> str:
    """
    Replace a method in Java source code.

    Challenges:
    - Method might have annotations
    - Javadoc comments should be preserved/updated
    - Overloaded methods need exact signature matching
    """
    parser = JavaParser()

    # Find the exact method by line number (handles overloads)
    method_node = parser.find_method_at_line(source, function.start_line)

    # Include Javadoc if present
    start = method_node.javadoc_start or method_node.start
    end = method_node.end

    # Replace the method
    return source[:start] + new_source + source[end:]
```

### 3.10 Code Formatting

**File: `codeflash/languages/java/formatter.py`**

```python
def format_code(source: str, file_path: Path | None = None) -> str:
    """
    Format Java code using google-java-format.

    Falls back to built-in formatter if google-java-format not available.
    """
    try:
        result = subprocess.run(
            ["google-java-format", "-"],
            input=source.encode(),
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.decode()
    except FileNotFoundError:
        pass

    # Fallback: basic indentation normalization
    return normalize_indentation(source)
```

---

## 4. Test Result Comparison

### 4.1 Behavior Verification

For Java, test results comparison needs to handle:
- Object equality (`.equals()` vs reference equality)
- Collection ordering (Lists vs Sets)
- Floating point comparison with epsilon
- Exception messages and types
- Side effects (mocked interactions)

```python
# codeflash/languages/java/comparator.py

def compare_test_results(
    original_results: Path,
    candidate_results: Path,
    project_root: Path
) -> tuple[bool, list[TestDiff]]:
    """
    Compare behavior between original and optimized code.

    Uses a Java comparison utility (run via the build tool)
    that handles Java-specific equality semantics.
    """
    # Run Java-based comparison tool
    result = subprocess.run([
        "java", "-cp", get_comparison_jar(),
        "com.codeflash.Comparator",
        str(original_results),
        str(candidate_results)
    ], capture_output=True)

    diffs = json.loads(result.stdout)
    return len(diffs) == 0, [TestDiff(**d) for d in diffs]
```

---

## 5. AI Service Integration

The AI service already supports language parameter. For Java:

```python
# Called from function_optimizer.py
response = ai_service.optimize_code(
    source_code=code_context.target_code,
    dependency_code=code_context.read_only_context,
    trace_id=trace_id,
    language="java",
    language_version="17",  # or "11", "21"
    n_candidates=5,
)
```

Java-specific optimization prompts should consider:
- Stream API optimizations
- Collection choice (ArrayList vs LinkedList, HashMap vs TreeMap)
- Concurrency patterns (CompletableFuture, parallel streams)
- Memory optimization (primitive vs boxed types)
- JIT-friendly patterns

---

## 6. Configuration Detection

**File: `codeflash/languages/java/config.py`**

```python
def detect_java_version(project_root: Path) -> str:
    """Detect Java version from build configuration."""
    build_tool = detect_build_tool(project_root)

    if build_tool == BuildTool.MAVEN:
        # Check pom.xml for maven.compiler.source
        pom = ET.parse(project_root / "pom.xml")
        version = pom.find(".//maven.compiler.source")
        if version is not None:
            return version.text

    elif build_tool == BuildTool.GRADLE:
        # Check build.gradle for sourceCompatibility
        build_file = project_root / "build.gradle"
        if build_file.exists():
            content = build_file.read_text()
            match = re.search(r"sourceCompatibility\s*=\s*['\"]?(\d+)", content)
            if match:
                return match.group(1)

    # Fallback: detect from JAVA_HOME
    return detect_jdk_version()

def detect_source_roots(project_root: Path) -> list[Path]:
    """Find source code directories."""
    standard_paths = [
        project_root / "src" / "main" / "java",
        project_root / "src",
    ]
    return [p for p in standard_paths if p.exists()]

def detect_test_roots(project_root: Path) -> list[Path]:
    """Find test code directories."""
    standard_paths = [
        project_root / "src" / "test" / "java",
        project_root / "test",
    ]
    return [p for p in standard_paths if p.exists()]
```

---

## 7. Runtime Library

CodeFlash needs a Java runtime library for instrumentation:

```
codeflash-runtime-java/
├── pom.xml
├── src/main/java/com/codeflash/
│   ├── CodeFlash.java          # Main capture API
│   ├── Capture.java            # Input/output capture
│   ├── Comparator.java         # Result comparison
│   ├── Timer.java              # High-precision timing
│   └── Serializer.java         # Object serialization for comparison
```

```java
// CodeFlash.java
package com.codeflash;

public class CodeFlash {
    public static void captureInput(String methodId, Object... args) {
        // Serialize and store inputs
    }

    public static <T> T captureOutput(String methodId, T result) {
        // Serialize and store output
        return result;
    }

    public static void captureException(String methodId, Throwable e) {
        // Store exception info
    }

    public static long startTimer() {
        return System.nanoTime();
    }

    public static void recordTime(String methodId, long startTime) {
        long elapsed = System.nanoTime() - startTime;
        // Store timing
    }
}
```

---

## 8. Implementation Phases

### Phase 1: Foundation (MVP)

1. Add `Language.JAVA` to enum
2. Implement tree-sitter Java parsing
3. Basic method discovery (public methods in classes)
4. Build tool detection (Maven/Gradle)
5. Simple context extraction (single file)
6. Test discovery (JUnit 5 `@Test` methods)
7. Test execution via Maven/Gradle

### Phase 2: Full Pipeline

1. Import resolution and dependency tracking
2. Multi-file context extraction
3. Test result capture and comparison
4. Code instrumentation for behavior verification
5. Benchmarking instrumentation
6. Code formatting integr.ation

### Phase 3: Advanced Features

1. Line profiler integration (JProfiler/async-profiler)
2. Generics handling in optimization
3. Lambda and stream optimization support
4. Concurrency-aware benchmarking
5. IDE integration (Language Server)

---

## 9. Key Challenges & Considerations

### 9.1 Java-Specific Challenges

| Challenge | Solution |
|-----------|----------|
| **No top-level functions** | Always include class context |
| **Overloaded methods** | Use full signature for identification |
| **Compilation required** | Compile before running tests |
| **Build tool complexity** | Abstract via `BuildTool` interface |
| **Static typing** | Ensure type compatibility in replacements |
| **Generics** | Preserve type parameters in optimization |
| **Checked exceptions** | Maintain throws declarations |
| **Package visibility** | Handle package-private methods |

### 9.2 Performance Considerations

- **JVM Warmup**: Java needs JIT warmup before benchmarking
- **GC Noise**: Account for garbage collection in timing
- **Classloading**: First run is always slower

```python
def run_benchmark_with_warmup(
    test_method: str,
    warmup_iterations: int = 100,
    benchmark_iterations: int = 1000
) -> BenchmarkResult:
    """Run benchmark with proper JVM warmup."""
    # Warmup phase (results discarded)
    run_tests(test_method, iterations=warmup_iterations)

    # Force GC before measurement
    subprocess.run(["jcmd", str(pid), "GC.run"])

    # Actual benchmark
    return run_tests(test_method, iterations=benchmark_iterations)
```

### 9.3 Test Framework Support

| Framework | Priority | Notes |
|-----------|----------|-------|
| JUnit 5 | High | Primary target, most modern |
| JUnit 4 | Medium | Still widely used |
| TestNG | Low | Different annotation model |
| Mockito | High | Mocking support needed |
| AssertJ | Medium | Fluent assertions |

---

## 10. File Changes Summary

### New Files to Create

```
codeflash/languages/java/
├── __init__.py
├── support.py              (~800 lines)
├── parser.py               (~400 lines)
├── discovery.py            (~300 lines)
├── context_extractor.py    (~400 lines)
├── import_resolver.py      (~350 lines)
├── instrument.py           (~500 lines)
├── test_runner.py          (~400 lines)
├── comparator.py           (~200 lines)
├── build_tools.py          (~350 lines)
├── formatter.py            (~100 lines)
├── line_profiler.py        (~300 lines)
└── config.py               (~150 lines)
Total: ~4,250 lines
```

### Existing Files to Modify

| File | Changes |
|------|---------|
| `codeflash/languages/base.py` | Add `JAVA` to `Language` enum |
| `codeflash/languages/__init__.py` | Import java module |
| `codeflash/cli_cmds/init.py` | Add Java project detection |
| `codeflash/api/aiservice.py` | No changes (already supports `language` param) |
| `requirements.txt` / `pyproject.toml` | Add `tree-sitter-java` |

### External Dependencies

```toml
# pyproject.toml additions
tree-sitter-java = "^0.21.0"
```

---

## 11. Testing Strategy

### Unit Tests

```python
# tests/languages/java/test_parser.py
def test_discover_methods_in_class():
    source = '''
    public class Calculator {
        public int add(int a, int b) {
            return a + b;
        }
    }
    '''
    methods = JavaParser().find_methods(source)
    assert len(methods) == 1
    assert methods[0].name == "add"

# tests/languages/java/test_discovery.py
def test_discover_functions_filters_tests():
    # Test that test methods are excluded
    ...
```

### Integration Tests

```python
# tests/languages/java/test_integration.py
def test_full_optimization_pipeline(java_test_project):
    """End-to-end test with a real Java project."""
    support = JavaSupport()

    functions = support.discover_functions(
        java_test_project / "src/main/java/Example.java"
    )

    context = support.extract_code_context(functions[0], java_test_project)

    # Verify context is compilable
    assert compile_java(context.target_code)
```

---

## 12. LanguageSupport Protocol Reference

All methods that `JavaSupport` must implement:

### Properties

```python
@property
def language(self) -> Language: ...

@property
def file_extensions(self) -> tuple[str, ...]: ...

@property
def test_framework(self) -> str: ...

@property
def comment_prefix(self) -> str: ...
```

### Discovery Methods

```python
def discover_functions(
    self,
    file_path: Path,
    criteria: FunctionFilterCriteria | None = None
) -> list[FunctionInfo]: ...

def discover_tests(
    self,
    test_root: Path,
    source_functions: list[FunctionInfo]
) -> dict[str, list[TestInfo]]: ...
```

### Code Analysis

```python
def extract_code_context(
    self,
    function: FunctionInfo,
    project_root: Path,
    module_root: Path | None = None
) -> CodeContext: ...

def find_helper_functions(
    self,
    function: FunctionInfo,
    project_root: Path
) -> list[HelperFunction]: ...
```

### Code Transformation

```python
def replace_function(
    self,
    source: str,
    function: FunctionInfo,
    new_source: str
) -> str: ...

def format_code(
    self,
    source: str,
    file_path: Path | None = None
) -> str: ...

def normalize_code(self, source: str) -> str: ...
```

### Test Execution

```python
def run_behavioral_tests(
    self,
    test_paths: list[Path],
    test_env: dict[str, str],
    cwd: Path,
    timeout: int,
    ...
) -> tuple[Path, Any, Path | None, Path | None]: ...

def run_benchmarking_tests(
    self,
    test_paths: list[Path],
    test_env: dict[str, str],
    cwd: Path,
    timeout: int,
    ...
) -> tuple[Path, Any]: ...
```

### Instrumentation

```python
def instrument_for_behavior(
    self,
    source: str,
    functions: list[str]
) -> str: ...

def instrument_for_benchmarking(
    self,
    test_source: str,
    target_function: str
) -> str: ...

def instrument_existing_test(
    self,
    test_path: Path,
    call_positions: list[tuple[int, int]],
    ...
) -> tuple[bool, str | None]: ...
```

### Validation

```python
def validate_syntax(self, source: str) -> bool: ...
```

### Result Comparison

```python
def compare_test_results(
    self,
    original_path: Path,
    candidate_path: Path,
    project_root: Path
) -> tuple[bool, list[TestDiff]]: ...
```

---

## 13. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Java Optimization Flow                          │
└──────────────────────────────────────────────────────────────────────────┘

User runs: codeflash optimize Example.java
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Detect Build Tool            │
    │  (Maven pom.xml / Gradle)     │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Discover Methods             │
    │  (tree-sitter-java parsing)   │
    │  Filter: public, non-test     │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Extract Code Context         │
    │  - Full class with imports    │
    │  - Helper classes (same pkg)  │
    │  - Superclass definitions     │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Discover Tests               │
    │  - Find *Test.java files      │
    │  - Parse @Test annotations    │
    │  - Match to source methods    │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Run Baseline                 │
    │  - Compile (mvn/gradle)       │
    │  - Execute JUnit tests        │
    │  - Capture behavior + timing  │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  AI Optimization              │
    │  - Send to AI service         │
    │  - language="java"            │
    │  - Receive N candidates       │
    └───────────────┬───────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Candidate 1  │  ...  │  Candidate N  │
└───────┬───────┘       └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  For Each Candidate:          │
    │  1. Replace method in source  │
    │  2. Compile project           │
    │  3. Run behavior tests        │
    │  4. Compare outputs           │
    │  5. If correct: benchmark     │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Select Best Candidate        │
    │  - Correctness verified       │
    │  - Best speedup               │
    │  - Account for JVM warmup     │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Apply Optimization           │
    │  - Update source file         │
    │  - Create PR (optional)       │
    │  - Report results             │
    └───────────────────────────────┘
```

---

## 14. Conclusion

This architecture provides a comprehensive roadmap for adding Java support to CodeFlash. The modular design mirrors the existing JavaScript/TypeScript implementation pattern, making it straightforward to implement incrementally while maintaining consistency with the rest of the codebase.

Key success factors:
1. **Leverage tree-sitter** for consistent parsing approach
2. **Abstract build tools** to support both Maven and Gradle
3. **Handle JVM specifics** (warmup, GC) in benchmarking
4. **Reuse existing infrastructure** where possible (AI service, result types)
5. **Implement incrementally** following the phased approach
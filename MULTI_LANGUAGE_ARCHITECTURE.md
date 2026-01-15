# Multi-Language Architecture Proposal for Codeflash

## Executive Summary

This document proposes an architecture to extend Codeflash from Python-only to support multiple programming languages, starting with JavaScript/TypeScript. The approach uses a **hybrid abstraction strategy**: abstracting the most critical paths (discovery, test running, code replacement, context extraction) while keeping the core orchestration in Python.

---

## 1. Current Architecture Analysis

### 1.1 Core Pipeline (Language-Agnostic Concepts)
```
Discovery → Context Extraction → AI Optimization → Test Generation →
Verification → Benchmarking → Ranking → PR Creation
```

### 1.2 Python-Specific Components (Need Abstraction)

| Component | Current Implementation | Python-Specific? |
|-----------|----------------------|------------------|
| Function Discovery | LibCST + ast visitors | Yes - LibCST is Python-only |
| Code Context Extraction | Jedi for dependency resolution | Yes - Jedi is Python-only |
| Code Replacement | LibCST transformers | Yes - LibCST is Python-only |
| Test Runner | pytest subprocess | Yes - pytest is Python-only |
| Test Discovery | pytest plugin tracing | Yes |
| Tracing/Instrumentation | `sys.setprofile`, decorators | Yes - Python runtime specific |
| Code Formatting | Black, isort | Yes |
| JIT Detection | Numba, TensorFlow, JAX | Yes |

### 1.3 Language-Agnostic Components (Can Reuse)

- AI Service Client (`aiservice.py`) - just needs `language` parameter
- GitHub/PR Integration
- Ranking Algorithms (`function_ranker.py`)
- Result Type Pattern (`either.py`)
- Configuration Management
- Telemetry Infrastructure
- Core Orchestration (`optimizer.py`, `function_optimizer.py`)

---

## 2. Proposed Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Codeflash Core                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │  Optimizer  │  │ FunctionOpt  │  │  AI Service │  │ PR Creator│ │
│  └──────┬──────┘  └──────┬───────┘  └─────────────┘  └───────────┘ │
│         │                │                                          │
│         ▼                ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┤
│  │              Language Abstraction Layer                         │
│  │  ┌──────────────────────────────────────────────────────────┐  │
│  │  │  LanguageSupport Protocol                                │  │
│  │  │  - discover_functions()                                  │  │
│  │  │  - extract_code_context()                                │  │
│  │  │  - replace_function()                                    │  │
│  │  │  - run_tests()                                           │  │
│  │  │  - discover_tests()                                      │  │
│  │  │  - instrument_for_behavior()                              │  │
│  │  │  - format_code()                                         │  │
│  │  └──────────────────────────────────────────────────────────┘  │
│  └─────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ PythonSupport │    │   JSSupport   │    │   GoSupport   │
│               │    │               │    │   (future)    │
│ - LibCST      │    │ - tree-sitter │    │ - tree-sitter │
│ - Jedi        │    │ - recast      │    │ - go/ast      │
│ - pytest      │    │ - Jest/Vitest │    │ - go test     │
└───────────────┘    └───────────────┘    └───────────────┘
```

### 2.2 Core Protocol Definition

```python
# codeflash/languages/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

@dataclass
class FunctionInfo:
    """Language-agnostic function representation."""
    name: str
    qualified_name: str
    file_path: Path
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_async: bool
    is_method: bool
    class_name: str | None
    parents: list[ParentInfo]  # For nested classes/functions

@dataclass
class ParentInfo:
    """Parent scope information."""
    name: str
    type: str  # "class", "function", "module"

@dataclass
class CodeContext:
    """Code context for optimization."""
    target_code: str
    target_file: Path
    helper_functions: list[HelperFunction]
    read_only_context: str
    imports: list[str]

@dataclass
class HelperFunction:
    """Helper function dependency."""
    name: str
    qualified_name: str
    file_path: Path
    source_code: str
    start_line: int
    end_line: int

@dataclass
class TestResult:
    """Language-agnostic test result."""
    test_name: str
    test_file: Path
    passed: bool
    runtime_ns: int | None
    return_value: any
    stdout: str
    stderr: str
    error_message: str | None

@dataclass
class TestDiscoveryResult:
    """Mapping of functions to their tests."""
    function_qualified_name: str
    tests: list[TestInfo]

@dataclass
class TestInfo:
    """Test information."""
    test_name: str
    test_file: Path
    test_class: str | None


@runtime_checkable
class LanguageSupport(Protocol):
    """Protocol defining what a language implementation must provide."""

    @property
    def name(self) -> str:
        """Language identifier (e.g., 'python', 'javascript', 'typescript')."""
        ...

    @property
    def file_extensions(self) -> list[str]:
        """Supported file extensions (e.g., ['.py'], ['.js', '.ts', '.tsx'])."""
        ...

    @property
    def test_framework(self) -> str:
        """Primary test framework name (e.g., 'pytest', 'jest')."""
        ...

    # === Discovery ===

    def discover_functions(
        self,
        file_path: Path,
        filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionInfo]:
        """Find all optimizable functions in a file."""
        ...

    def discover_tests(
        self,
        test_root: Path,
        source_functions: list[FunctionInfo],
    ) -> dict[str, list[TestInfo]]:
        """Map source functions to their tests via static analysis."""
        ...

    # === Code Analysis ===

    def extract_code_context(
        self,
        function: FunctionInfo,
        project_root: Path,
        module_root: Path,
    ) -> CodeContext:
        """Extract function code and its dependencies."""
        ...

    def find_helper_functions(
        self,
        function: FunctionInfo,
        project_root: Path,
    ) -> list[HelperFunction]:
        """Find helper functions called by target function."""
        ...

    # === Code Transformation ===

    def replace_function(
        self,
        file_path: Path,
        original_function: FunctionInfo,
        new_source: str,
    ) -> str:
        """Replace function in file, return modified source."""
        ...

    def format_code(
        self,
        source: str,
        file_path: Path,
    ) -> str:
        """Format code using language-specific formatter."""
        ...

    # === Test Execution ===

    def run_tests(
        self,
        test_files: list[Path],
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> tuple[list[TestResult], Path]:
        """Run tests and return results + JUnit XML path."""
        ...

    def parse_test_results(
        self,
        junit_xml_path: Path,
        stdout: str,
    ) -> list[TestResult]:
        """Parse test results from JUnit XML and stdout."""
        ...

    # === Instrumentation ===

    def instrument_for_behavior(
        self,
        file_path: Path,
        functions: list[FunctionInfo],
    ) -> str:
        """Add tracing instrumentation to capture inputs/outputs."""
        ...

    def instrument_for_benchmarking(
        self,
        test_source: str,
        target_function: FunctionInfo,
    ) -> str:
        """Add timing instrumentation to test code."""
        ...

    # === Validation ===

    def validate_syntax(self, source: str) -> bool:
        """Check if source code is syntactically valid."""
        ...

    def normalize_code(self, source: str) -> str:
        """Normalize code for deduplication (remove comments, normalize whitespace)."""
        ...
```

---

## 3. Implementation Details

### 3.1 Tree-Sitter for Analysis (All Languages)

Use tree-sitter for consistent cross-language analysis:

```python
# codeflash/languages/treesitter_utils.py

import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Parser

LANGUAGES = {
    'python': tree_sitter_python.language(),
    'javascript': tree_sitter_javascript.language(),
    'typescript': tree_sitter_typescript.language_typescript(),
    'tsx': tree_sitter_typescript.language_tsx(),
}

class TreeSitterAnalyzer:
    """Cross-language code analysis using tree-sitter."""

    def __init__(self, language: str):
        self.parser = Parser(LANGUAGES[language])
        self.language = language

    def find_functions(self, source: str) -> list[dict]:
        """Find all function definitions in source."""
        tree = self.parser.parse(bytes(source, 'utf8'))
        # Query pattern varies by language but concept is same
        ...

    def find_imports(self, source: str) -> list[dict]:
        """Find all import statements."""
        ...

    def find_function_calls(self, source: str, within_function: str) -> list[str]:
        """Find all function calls within a function body."""
        ...

    def get_node_text(self, node, source: bytes) -> str:
        """Extract text for a tree-sitter node."""
        return source[node.start_byte:node.end_byte].decode('utf8')
```

### 3.2 Language-Specific Transformation Tools

Since tree-sitter doesn't support unparsing, use language-specific tools:

```python
# codeflash/languages/javascript/transformer.py

import subprocess
import json
from pathlib import Path

class JavaScriptTransformer:
    """JavaScript/TypeScript code transformation using jscodeshift/recast."""

    def replace_function(
        self,
        file_path: Path,
        function_name: str,
        new_source: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Replace function using jscodeshift transform."""
        # Option 1: Use jscodeshift via subprocess
        transform_script = self._generate_transform_script(
            function_name, new_source, start_line, end_line
        )
        result = subprocess.run(
            ['npx', 'jscodeshift', '-t', transform_script, str(file_path), '--dry'],
            capture_output=True, text=True
        )
        return result.stdout

        # Option 2: Text-based replacement with line numbers (simpler)
        # Since we have exact line numbers, we can do precise text replacement

    def _text_based_replace(
        self,
        source: str,
        start_line: int,
        end_line: int,
        new_source: str,
    ) -> str:
        """Simple text-based replacement using line numbers."""
        lines = source.splitlines(keepends=True)
        # Preserve indentation from original
        original_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        # Reindent new source
        new_lines = self._reindent(new_source, original_indent)
        # Replace
        return ''.join(lines[:start_line - 1] + [new_lines] + lines[end_line:])
```

### 3.3 JavaScript/TypeScript Implementation

```python
# codeflash/languages/javascript/support.py

from pathlib import Path
from codeflash.languages.base import LanguageSupport, FunctionInfo, CodeContext
from codeflash.languages.treesitter_utils import TreeSitterAnalyzer
from codeflash.languages.javascript.transformer import JavaScriptTransformer

class JavaScriptSupport(LanguageSupport):
    """JavaScript/TypeScript language support."""

    @property
    def name(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> list[str]:
        return ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']

    @property
    def test_framework(self) -> str:
        return "jest"  # or "vitest"

    def __init__(self):
        self.analyzer = TreeSitterAnalyzer('javascript')
        self.ts_analyzer = TreeSitterAnalyzer('typescript')
        self.transformer = JavaScriptTransformer()

    def discover_functions(self, file_path: Path, filter_criteria=None) -> list[FunctionInfo]:
        """Find functions using tree-sitter."""
        source = file_path.read_text()
        lang = 'typescript' if file_path.suffix in ['.ts', '.tsx'] else 'javascript'
        analyzer = self.ts_analyzer if lang == 'typescript' else self.analyzer

        functions = []
        tree = analyzer.parser.parse(bytes(source, 'utf8'))

        # Query for function declarations, arrow functions, methods
        # tree-sitter query patterns for JS/TS
        query_patterns = """
        (function_declaration name: (identifier) @name) @func
        (arrow_function) @func
        (method_definition name: (property_identifier) @name) @func
        """
        # ... process matches into FunctionInfo objects
        return functions

    def extract_code_context(
        self,
        function: FunctionInfo,
        project_root: Path,
        module_root: Path,
    ) -> CodeContext:
        """Extract context by following imports."""
        source = function.file_path.read_text()

        # 1. Find imports in the file
        imports = self._find_imports(source)

        # 2. Find function calls within target function
        calls = self._find_calls_in_function(source, function)

        # 3. Resolve which calls are local helpers
        helpers = []
        for call in calls:
            helper = self._resolve_to_local_function(call, imports, module_root)
            if helper:
                helpers.append(helper)

        # 4. Build context
        return CodeContext(
            target_code=self._extract_function_source(source, function),
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context=self._format_helpers_as_context(helpers),
            imports=imports,
        )

    def run_tests(
        self,
        test_files: list[Path],
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> tuple[list[TestResult], Path]:
        """Run Jest tests."""
        import subprocess

        junit_path = cwd / '.codeflash' / 'jest-results.xml'

        # Build Jest command
        cmd = [
            'npx', 'jest',
            '--reporters=default',
            f'--reporters=jest-junit',
            '--testPathPattern=' + '|'.join(str(f) for f in test_files),
            '--runInBand',  # Sequential for deterministic timing
            '--forceExit',
        ]

        test_env = env.copy()
        test_env['JEST_JUNIT_OUTPUT_FILE'] = str(junit_path)

        result = subprocess.run(
            cmd, cwd=cwd, env=test_env,
            capture_output=True, text=True, timeout=timeout
        )

        results = self.parse_test_results(junit_path, result.stdout)
        return results, junit_path

    def instrument_for_behavior(
        self,
        file_path: Path,
        functions: list[FunctionInfo],
    ) -> str:
        """Wrap functions with tracing HOF."""
        source = file_path.read_text()

        # Add tracing wrapper import
        tracing_import = "const { __codeflash_trace__ } = require('@codeflash/tracer');\n"

        # Wrap each function
        for func in reversed(functions):  # Reverse to preserve line numbers
            source = self._wrap_function_with_tracer(source, func)

        return tracing_import + source

    def _wrap_function_with_tracer(self, source: str, func: FunctionInfo) -> str:
        """Wrap a function with tracing instrumentation."""
        # For named functions: wrap the function
        # For arrow functions: wrap the assignment
        # This is language-specific logic
        ...
```

### 3.4 Test Discovery via Static Analysis

```python
# codeflash/languages/javascript/test_discovery.py

from pathlib import Path
from codeflash.languages.treesitter_utils import TreeSitterAnalyzer

class JestTestDiscovery:
    """Static analysis-based test discovery for Jest."""

    def __init__(self):
        self.analyzer = TreeSitterAnalyzer('javascript')

    def discover_tests(
        self,
        test_root: Path,
        source_functions: list[FunctionInfo],
    ) -> dict[str, list[TestInfo]]:
        """Map functions to tests via static analysis."""

        function_to_tests = {}

        # Find all test files
        test_files = list(test_root.rglob('*.test.js')) + \
                     list(test_root.rglob('*.test.ts')) + \
                     list(test_root.rglob('*.spec.js')) + \
                     list(test_root.rglob('*.spec.ts'))

        for test_file in test_files:
            source = test_file.read_text()

            # Find imports in test file
            imports = self._find_imports(source)

            # Find test blocks (describe, it, test)
            tests = self._find_test_blocks(source)

            # For each test, find function calls
            for test in tests:
                calls = self._find_calls_in_test(source, test)

                # Match calls to source functions
                for func in source_functions:
                    if self._function_is_called(func, calls, imports):
                        if func.qualified_name not in function_to_tests:
                            function_to_tests[func.qualified_name] = []
                        function_to_tests[func.qualified_name].append(TestInfo(
                            test_name=test.name,
                            test_file=test_file,
                            test_class=test.describe_block,
                        ))

        return function_to_tests

    def _find_imports(self, source: str) -> dict[str, str]:
        """Find import/require statements and map names to modules."""
        # Parse: import { foo } from './module'
        # Parse: const { foo } = require('./module')
        ...

    def _find_test_blocks(self, source: str) -> list[TestBlock]:
        """Find Jest test blocks (describe, it, test)."""
        # Query for: test('...', ...), it('...', ...), describe('...', ...)
        ...
```

### 3.5 Tracing Strategy for JavaScript

```javascript
// @codeflash/tracer/index.js
// This would be an npm package installed in the user's project

const fs = require('fs');
const path = require('path');

class CodeflashTracer {
    constructor(outputPath) {
        this.outputPath = outputPath;
        this.traces = [];
    }

    wrap(fn, fnName, filePath) {
        const self = this;

        // Handle async functions
        if (fn.constructor.name === 'AsyncFunction') {
            return async function(...args) {
                const start = process.hrtime.bigint();
                let result, error;
                try {
                    result = await fn.apply(this, args);
                } catch (e) {
                    error = e;
                }
                const end = process.hrtime.bigint();

                self.traces.push({
                    function: fnName,
                    file: filePath,
                    args: self.serialize(args),
                    result: error ? null : self.serialize(result),
                    error: error ? error.message : null,
                    runtime_ns: Number(end - start),
                });

                if (error) throw error;
                return result;
            };
        }

        // Handle sync functions
        return function(...args) {
            const start = process.hrtime.bigint();
            let result, error;
            try {
                result = fn.apply(this, args);
            } catch (e) {
                error = e;
            }
            const end = process.hrtime.bigint();

            self.traces.push({
                function: fnName,
                file: filePath,
                args: self.serialize(args),
                result: error ? null : self.serialize(result),
                error: error ? error.message : null,
                runtime_ns: Number(end - start),
            });

            if (error) throw error;
            return result;
        };
    }
    // saurabh's comments - Is there something more general purpose similar to python dill and pickle?
    serialize(value) {
        // Handle circular references, functions, etc.
        try {
            return JSON.stringify(value, this.replacer);
        } catch {
            return '<unserializable>';
        }
    }

    flush() {
        fs.writeFileSync(this.outputPath, JSON.stringify(this.traces, null, 2));
    }
}

module.exports = { CodeflashTracer };
```

---

## 4. File Structure

```
codeflash/
├── languages/
│   ├── __init__.py
│   ├── base.py                    # LanguageSupport protocol
│   ├── registry.py                # Language registration & detection
│   ├── treesitter_utils.py        # Shared tree-sitter utilities
│   │
│   ├── python/
│   │   ├── __init__.py
│   │   ├── support.py             # PythonSupport implementation
│   │   ├── discovery.py           # Function discovery (LibCST)
│   │   ├── context.py             # Context extraction (Jedi)
│   │   ├── transformer.py         # Code replacement (LibCST)
│   │   ├── test_runner.py         # pytest execution
│   │   └── tracer.py              # Python tracing
│   │
│   ├── javascript/
│   │   ├── __init__.py
│   │   ├── support.py             # JavaScriptSupport implementation
│   │   ├── discovery.py           # Function discovery (tree-sitter)
│   │   ├── context.py             # Context extraction (tree-sitter + imports)
│   │   ├── transformer.py         # Code replacement (recast/text-based)
│   │   ├── test_runner.py         # Jest execution
│   │   └── tracer.py              # JS tracing instrumentation
│   │
│   └── typescript/                # Extends JavaScript with TS specifics
│       ├── __init__.py
│       └── support.py
│
├── models/
│   ├── models.py                  # Existing models (updated for multi-lang)
│   └── language_models.py         # New language-agnostic models
│
└── ... (existing structure)
```

---

## 5. Key Changes to Existing Code

### 5.1 Language Detection & Registry

```python
# codeflash/languages/registry.py

from pathlib import Path
from typing import Type
from codeflash.languages.base import LanguageSupport

_LANGUAGE_REGISTRY: dict[str, Type[LanguageSupport]] = {}

def register_language(cls: Type[LanguageSupport]) -> Type[LanguageSupport]:
    """Decorator to register a language implementation."""
    instance = cls()
    for ext in instance.file_extensions:
        _LANGUAGE_REGISTRY[ext] = cls
    return cls

def get_language_for_file(file_path: Path) -> LanguageSupport:
    """Get language support for a file based on extension."""
    ext = file_path.suffix.lower()
    if ext not in _LANGUAGE_REGISTRY:
        raise ValueError(f"Unsupported file extension: {ext}")
    return _LANGUAGE_REGISTRY[ext]()

def detect_project_language(project_root: Path, module_root: Path) -> str:
    """Detect primary language of project."""
    # Count files by extension
    extension_counts = {}
    for file in module_root.rglob('*'):
        if file.is_file():
            ext = file.suffix.lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

    # Return most common supported language
    for ext in sorted(extension_counts, key=extension_counts.get, reverse=True):
        if ext in _LANGUAGE_REGISTRY:
            return _LANGUAGE_REGISTRY[ext]().name

    raise ValueError("No supported language detected in project")
```

### 5.2 Update FunctionToOptimize

```python
# codeflash/discovery/functions_to_optimize.py

@dataclass(frozen=True)
class FunctionToOptimize:
    """Language-agnostic function representation."""
    function_name: str
    file_path: Path
    parents: list[FunctionParent]
    starting_line: int | None = None
    ending_line: int | None = None
    starting_col: int | None = None  # NEW: for precise location
    ending_col: int | None = None    # NEW: for precise location
    is_async: bool = False
    language: str = "python"         # NEW: language identifier

    @property
    def qualified_name(self) -> str:
        if not self.parents:
            return self.function_name
        parent_path = ".".join(parent.name for parent in self.parents)
        return f"{parent_path}.{self.function_name}"
```

### 5.3 Update CodeStringsMarkdown

```python
# codeflash/models/models.py

class CodeStringsMarkdown(BaseModel):
    code_strings: list[CodeString] = []
    language: str = "python"  # NEW: language for markdown formatting

    @property
    def markdown(self) -> str:
        """Returns Markdown-formatted code blocks with correct language tag."""
        lang_tag = self.language  # 'python', 'javascript', 'typescript', etc.
        return "\n".join([
            f"```{lang_tag}{':' + cs.file_path.as_posix() if cs.file_path else ''}\n{cs.code.strip()}\n```"
            for cs in self.code_strings
        ])
```

### 5.4 Update Optimizer to Use Language Support

```python
# codeflash/optimization/optimizer.py

from codeflash.languages.registry import get_language_for_file, detect_project_language

class Optimizer:
    def __init__(self, args, ...):
        self.args = args
        # Detect or use specified language
        self.language = detect_project_language(
            args.project_root,
            args.module_root
        )
        self.lang_support = get_language_for_file(
            Path(args.module_root) / f"dummy.{self._get_primary_extension()}"
        )

    def get_optimizable_functions(self) -> dict[Path, list[FunctionToOptimize]]:
        """Use language-specific discovery."""
        functions = {}
        for file_path in self._get_source_files():
            lang = get_language_for_file(file_path)
            discovered = lang.discover_functions(file_path)
            functions[file_path] = [
                FunctionToOptimize(
                    function_name=f.name,
                    file_path=f.file_path,
                    parents=f.parents,
                    starting_line=f.start_line,
                    ending_line=f.end_line,
                    is_async=f.is_async,
                    language=lang.name,
                )
                for f in discovered
            ]
        return functions
```

### 5.5 Update AI Service Request

```python
# codeflash/api/aiservice.py

def optimize_code(
    self,
    source_code: str,
    dependency_code: str,
    trace_id: str,
    is_async: bool,
    n_candidates: int,
    language: str = "python",  # NEW: language parameter
    ...
) -> Result[list[OptimizedCandidate], str]:
    """Request optimization from AI service."""
    payload = {
        "source_code": source_code,
        "dependency_code": dependency_code,
        "trace_id": trace_id,
        "is_async": is_async,
        "n_candidates": n_candidates,
        "language": language,  # Backend handles language-specific prompts
        ...
    }
    # ... rest of implementation
```

---

## 6. Configuration Updates

### 6.1 pyproject.toml Schema

```toml
[tool.codeflash]
# Existing fields
module-root = "src"
tests-root = "tests"

# New optional field (auto-detected if not specified)
language = "javascript"  # or "python", "typescript", etc.

# Language-specific settings
[tool.codeflash.javascript]
test-framework = "jest"  # or "vitest", "mocha"
test-pattern = "**/*.test.{js,ts}"
formatter = "prettier"

[tool.codeflash.python]
test-framework = "pytest"
formatter-cmds = ["black", "isort"]
```

---

## 7. Implementation Phases

### Phase 1: Core Abstraction (Week 1-2)
1. Create `LanguageSupport` protocol in `codeflash/languages/base.py`
2. Create language registry and detection
3. Refactor `FunctionToOptimize` to be language-agnostic
4. Update `CodeStringsMarkdown` to support language tags
5. Create `PythonSupport` by wrapping existing code

### Phase 2: Tree-Sitter Integration (Week 2-3)
1. Add tree-sitter dependencies
2. Create `TreeSitterAnalyzer` utility class
3. Implement tree-sitter based function discovery
4. Implement tree-sitter based import analysis

### Phase 3: JavaScript Support (Week 3-5)
1. Create `JavaScriptSupport` class
2. Implement function discovery for JS/TS
3. Implement code context extraction via import following
4. Implement text-based code replacement
5. Implement Jest test runner integration
6. Implement static test discovery

### Phase 4: Tracing & Instrumentation (Week 5-6)
1. Create `@codeflash/tracer` npm package
2. Implement JS function wrapping for tracing
3. Implement replay test generation for JS
4. Test end-to-end tracing workflow

### Phase 5: Integration & Testing (Week 6-7)
1. Update CLI to handle language parameter
2. Update configuration parsing
3. Create integration tests
4. Documentation updates

---

## 8. Design Decisions (Finalized)

### 8.1 Code Replacement Strategy
**Status: DECIDED** - See Section 11 for experiment results.

**Decision: Hybrid Approach (C)** - Tree-sitter for analysis + text-based replacement

**Tested Approaches**:
- (A) jscodeshift/recast - Requires Node.js, adds complexity
- (B) Text-based - Simple, 100% pass rate on 19 test cases
- (C) Hybrid - Tree-sitter analysis + text replacement, 100% pass rate

**Why Hybrid**:
- Tree-sitter provides accurate function boundaries for all JS/TS constructs
- Text-based replacement is simple, fast, and handles all edge cases
- No Node.js dependency required
- Syntax validation possible via tree-sitter after replacement

### 8.2 Return Value Capture
**Decision: Option B** - Instrument test code to capture return values.

**Implementation**:
- Inject code at the start/end of each test to capture return values
- For return values, prefer sqlite db to store the results. This is similar to the current implementation.
- Parse both JUnit XML (pass/fail, timing) and sqlite for full verification

### 8.3 TypeScript Handling
**Decision: Option A** - Separate language implementation that extends JavaScript.

**Implementation**:
```python
class TypeScriptSupport(JavaScriptSupport):
    """TypeScript extends JavaScript with type-aware differences."""

    @property
    def name(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> list[str]:
        return ['.ts', '.tsx']

    # Override methods where TypeScript differs from JavaScript
    def _get_parser(self):
        return TreeSitterAnalyzer('typescript')
```

### 8.4 Monorepo Support
**Decision**: Single language per module configuration.

**Implementation**:
- Each `[tool.codeflash]` section in `pyproject.toml` configures one module
- Language is detected from `module-root` or explicitly specified
- For multi-language monorepos, users run codeflash separately per module

---

## 9. Dependencies

### Python Dependencies (pyproject.toml)
```toml
[project.dependencies]
tree-sitter = ">=0.21.0"
tree-sitter-python = ">=0.21.0"
tree-sitter-javascript = ">=0.21.0"
tree-sitter-typescript = ">=0.21.0"
```

### Node.js Dependencies (for JS/TS projects)
```json
{
  "devDependencies": {
    "@codeflash/tracer": "^1.0.0",
    "jest-junit": "^16.0.0"
  }
}
```

---

## 10. Success Criteria

1. **Functional**: Can optimize a JavaScript function end-to-end
2. **Correct**: All existing Python tests pass
3. **Extensible**: Adding a new language requires only implementing `LanguageSupport`
4. **Maintainable**: Core orchestration code has no language-specific logic
5. **Performant**: No significant regression in Python optimization speed

---

## 11. Code Replacement Experiment Results

**Experiment Date**: 2026-01-14

### 11.1 Approaches Tested

| Approach | Description | Dependencies |
|----------|-------------|--------------|
| **A: jscodeshift** | AST-based via Node.js subprocess | Node.js, npm |
| **B: Text-Based** | Pure Python line manipulation | None |
| **C: Hybrid** | Tree-sitter analysis + text replacement | tree-sitter |

### 11.2 Test Cases

19 test cases covering:
- Basic function declarations
- Arrow functions (const, one-liner)
- Class methods and static methods
- Async functions
- TypeScript typed functions and generics
- Functions with JSDoc and inline comments
- Nested functions
- Export patterns (named, default)
- Decorated methods
- Edge cases (first/last/only function in file)
- Deep indentation scenarios

### 11.3 Results

| Approach | Passed | Failed | Pass Rate | Total Time |
|----------|--------|--------|-----------|------------|
| **B: Text-Based** | 19 | 0 | **100%** | 0.04ms |
| **C: Hybrid** | 19 | 0 | **100%** | 0.08ms |
| A: jscodeshift | - | - | - | (requires npm setup) |

### 11.4 Decision

**Selected Approach: Hybrid (C) with Text-Based Replacement**

**Rationale**:
1. **Tree-sitter for analysis**: Use tree-sitter to find function boundaries, understand code structure, and validate syntax
2. **Text-based for replacement**: Use simple line-based text manipulation for the actual code replacement
3. **No Node.js dependency**: Entire codeflash CLI stays in Python, no subprocess overhead

**Implementation Strategy**:
```python
class JavaScriptSupport:
    def replace_function(self, file_path, function: FunctionInfo, new_source: str) -> str:
        source = file_path.read_text()

        # Tree-sitter provides precise line numbers from discovery phase
        # FunctionInfo already has start_line, end_line from tree-sitter analysis

        # Text-based replacement using line numbers
        lines = source.splitlines(keepends=True)
        before = lines[:function.start_line - 1]
        after = lines[function.end_line:]

        # Handle indentation adjustment
        new_lines = self._adjust_indentation(new_source, function.start_line, lines)

        return ''.join(before + new_lines + after)
```

### 11.5 Key Findings

1. **Text-based replacement is sufficient**: With accurate line numbers from tree-sitter, simple text manipulation handles all edge cases correctly.

2. **Tree-sitter adds value for analysis, not transformation**: Tree-sitter is valuable for:
   - Finding function boundaries accurately
   - Understanding code structure (nested functions, classes)
   - Syntax validation of results
   - But NOT needed for the replacement itself

3. **No external dependencies needed**: jscodeshift would require Node.js subprocess calls, adding complexity and latency. The text-based approach works entirely in Python.

4. **Indentation handling is critical**: The key to correct replacement is:
   - Detecting original function's indentation
   - Adjusting new function's indentation to match
   - Preserving surrounding whitespace

### 11.6 Experiment Files

Experiments are located in: `experiments/code_replacement/`
- `test_cases.py` - 19 test cases covering various scenarios
- `approach_b_text_based.py` - Pure Python text-based implementation
- `approach_c_hybrid.py` - Tree-sitter + text-based implementation
- `run_experiments.py` - Test runner and report generator
- `EXPERIMENT_RESULTS.md` - Detailed results
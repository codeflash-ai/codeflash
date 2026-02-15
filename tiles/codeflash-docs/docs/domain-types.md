# Domain Types

Core data types used throughout the codeflash optimization pipeline.

## Function Representation

### `FunctionToOptimize` (`models/function_types.py`)

The canonical dataclass representing a function candidate for optimization. Works across Python, JavaScript, and TypeScript.

Key fields:
- `function_name: str` — The function name
- `file_path: Path` — Absolute file path where the function is located
- `parents: list[FunctionParent]` — Parent scopes (classes/functions), each with `name` and `type`
- `starting_line / ending_line: Optional[int]` — Line range (1-indexed)
- `is_async: bool` — Whether the function is async
- `is_method: bool` — Whether it belongs to a class
- `language: str` — Programming language (default: `"python"`)

Key properties:
- `qualified_name` — Full dotted name including parent classes (e.g., `MyClass.my_method`)
- `top_level_parent_name` — Name of outermost parent, or function name if no parents
- `class_name` — Immediate parent class name, or `None`

### `FunctionParent` (`models/function_types.py`)

Represents a parent scope: `name: str` (e.g., `"MyClass"`) and `type: str` (e.g., `"ClassDef"`).

### `FunctionSource` (`models/models.py`)

Represents a resolved function with source code. Used for helper functions in context extraction.

Fields: `file_path`, `qualified_name`, `fully_qualified_name`, `only_function_name`, `source_code`, `jedi_definition`.

## Code Representation

### `CodeString` (`models/models.py`)

A single code block with validated syntax:
- `code: str` — The source code
- `file_path: Optional[Path]` — Origin file path
- `language: str` — Language for validation (default: `"python"`)

Validates syntax on construction via `model_validator`.

### `CodeStringsMarkdown` (`models/models.py`)

A collection of `CodeString` blocks — the primary format for passing code through the pipeline.

Key properties:
- `.flat` — Combined source code with file-path comment prefixes (e.g., `# file: path/to/file.py`)
- `.markdown` — Markdown-formatted with fenced code blocks: `` ```python:filepath\ncode\n``` ``
- `.file_to_path()` — Dict mapping file path strings to code

Static method:
- `parse_markdown_code(markdown_code, expected_language)` — Parses markdown code blocks back into `CodeStringsMarkdown`

## Optimization Context

### `CodeOptimizationContext` (`models/models.py`)

Holds all code context needed for optimization:
- `read_writable_code: CodeStringsMarkdown` — Code the LLM can modify
- `read_only_context_code: str` — Reference-only dependency code
- `testgen_context: CodeStringsMarkdown` — Context for test generation
- `hashing_code_context: str` / `hashing_code_context_hash: str` — For deduplication
- `helper_functions: list[FunctionSource]` — Helper functions in the writable code
- `preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]]` — Objects that already exist in the code

### `CodeContextType` enum (`models/models.py`)

Defines context categories: `READ_WRITABLE`, `READ_ONLY`, `TESTGEN`, `HASHING`.

## Candidates

### `OptimizedCandidate` (`models/models.py`)

A generated code variant:
- `source_code: CodeStringsMarkdown` — The optimized code
- `explanation: str` — LLM explanation of the optimization
- `optimization_id: str` — Unique identifier
- `source: OptimizedCandidateSource` — How it was generated
- `parent_id: str | None` — ID of parent candidate (for refinements/repairs)
- `model: str | None` — Which LLM model generated it

### `OptimizedCandidateSource` enum (`models/models.py`)

How a candidate was generated: `OPTIMIZE`, `OPTIMIZE_LP` (line profiler), `REFINE`, `REPAIR`, `ADAPTIVE`, `JIT_REWRITE`.

### `CandidateEvaluationContext` (`models/models.py`)

Tracks state during candidate evaluation:
- `speedup_ratios` / `optimized_runtimes` / `is_correct` — Per-candidate results
- `ast_code_to_id` — Deduplication map (normalized AST → first seen candidate)
- `valid_optimizations` — Candidates that passed all checks

Key methods: `record_failed_candidate()`, `record_successful_candidate()`, `handle_duplicate_candidate()`, `register_new_candidate()`.

## Baseline & Results

### `OriginalCodeBaseline` (`models/models.py`)

Baseline measurements for the original code:
- `behavior_test_results: TestResults` / `benchmarking_test_results: TestResults`
- `line_profile_results: dict`
- `runtime: int` — Total runtime in nanoseconds
- `coverage_results: Optional[CoverageData]`

### `BestOptimization` (`models/models.py`)

The winning candidate after evaluation:
- `candidate: OptimizedCandidate`
- `helper_functions: list[FunctionSource]`
- `code_context: CodeOptimizationContext`
- `runtime: int`
- `winning_behavior_test_results` / `winning_benchmarking_test_results: TestResults`

## Test Types

### `TestType` enum (`models/test_type.py`)

- `EXISTING_UNIT_TEST` (1) — Pre-existing tests from the codebase
- `INSPIRED_REGRESSION` (2) — Tests inspired by existing tests
- `GENERATED_REGRESSION` (3) — AI-generated regression tests
- `REPLAY_TEST` (4) — Tests from recorded benchmark data
- `CONCOLIC_COVERAGE_TEST` (5) — Coverage-guided tests
- `INIT_STATE_TEST` (6) — Class init state verification

### `TestFile` / `TestFiles` (`models/models.py`)

`TestFile` represents a single test file with `instrumented_behavior_file_path`, optional `benchmarking_file_path`, `original_file_path`, `test_type`, and `tests_in_file`.

`TestFiles` is a collection with lookup methods: `get_by_type()`, `get_by_original_file_path()`, `get_test_type_by_instrumented_file_path()`.

### `TestResults` (`models/models.py`)

Collection of `FunctionTestInvocation` results with indexed lookup. Key methods:
- `add(invocation)` — Deduplicated insert
- `total_passed_runtime()` — Sum of minimum runtimes per test case (nanoseconds)
- `number_of_loops()` — Max loop index across all results
- `usable_runtime_data_by_test_case()` — Dict of invocation ID → list of runtimes

## Result Type

### `Result[L, R]` / `Success` / `Failure` (`either.py`)

Functional error handling type:
- `Success(value)` — Wraps a successful result
- `Failure(error)` — Wraps an error
- `result.is_successful()` / `result.is_failure()` — Check type
- `result.unwrap()` — Get success value (raises if Failure)
- `result.failure()` — Get failure value (raises if Success)
- `is_successful(result)` — Module-level helper function

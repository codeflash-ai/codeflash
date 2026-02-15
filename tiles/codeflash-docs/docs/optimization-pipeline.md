# Optimization Pipeline

Step-by-step data flow from function discovery to PR creation.

## 1. Entry Point: `Optimizer.run()` (`optimization/optimizer.py`)

The `Optimizer` class is initialized with CLI args and creates:
- `TestConfig` with test roots, project root, pytest command
- `AiServiceClient` for AI service communication
- Optional `LocalAiServiceClient` for experiments

`run()` orchestrates the full pipeline: discovers functions, optionally ranks them, then optimizes each in turn.

## 2. Function Discovery (`discovery/functions_to_optimize.py`)

`FunctionVisitor` traverses source files to find optimizable functions, producing `FunctionToOptimize` instances. Filters include:
- Skipping functions that are too small or trivial
- Skipping previously optimized functions (via `was_function_previously_optimized()`)
- Applying user-configured include/exclude patterns

## 3. Function Ranking (`benchmarking/function_ranker.py`)

When trace data is available, `FunctionRanker` ranks functions by **addressable time** — the time a function spends that could be optimized (own time + callee time / call count). Functions below `DEFAULT_IMPORTANCE_THRESHOLD=0.001` are skipped.

## 4. Per-Function Optimization: `FunctionOptimizer` (`optimization/function_optimizer.py`)

For each function, `FunctionOptimizer.optimize_function()` runs the full optimization loop:

### 4a. Context Extraction (`context/code_context_extractor.py`)

Extracts `CodeOptimizationContext` containing:
- `read_writable_code` — Code the LLM can modify (the function + helpers)
- `read_only_context_code` — Dependency code for reference only
- `testgen_context` — Context for test generation (may include imported class definitions)

Token limits are enforced: `OPTIMIZATION_CONTEXT_TOKEN_LIMIT=16000` and `TESTGEN_CONTEXT_TOKEN_LIMIT=16000`. Functions exceeding these are rejected.

### 4b. Concurrent Test Generation + LLM Optimization

These run in parallel using `concurrent.futures`:
- **Test generation**: Generates regression tests from the function context
- **LLM optimization**: Sends `read_writable_code.markdown` + `read_only_context_code` to the AI service

The number of candidates depends on effort level (see Configuration docs).

### 4c. Candidate Evaluation

For each `OptimizedCandidate`:

1. **Deduplication**: Normalize code AST and check against `CandidateEvaluationContext.ast_code_to_id`. If duplicate, copy results from previous evaluation.

2. **Code replacement**: Replace the original function with the candidate using `replace_function_definitions_in_module()`.

3. **Behavioral testing**: Run instrumented tests in subprocess. The custom pytest plugin applies deterministic patches. Compare return values, stdout, and pass/fail status against the original baseline.

4. **Benchmarking**: If behavior matches, run performance tests with looping (`TOTAL_LOOPING_TIME=10s`). Calculate speedup ratio.

5. **Validation**: Candidate must beat `MIN_IMPROVEMENT_THRESHOLD=0.05` (5% speedup) and pass stability checks.

### 4d. Refinement & Repair

- **Repair**: If fewer than `MIN_CORRECT_CANDIDATES=2` pass, failed candidates can be repaired via `AIServiceCodeRepairRequest` (sends test diffs to LLM).
- **Refinement**: Top valid candidates are refined via `AIServiceRefinerRequest` (sends runtime data, line profiler results).
- **Adaptive**: At HIGH effort, additional adaptive optimization rounds via `AIServiceAdaptiveOptimizeRequest`.

### 4e. Best Candidate Selection

The winning candidate is selected by:
1. Highest speedup ratio
2. For tied speedups, shortest diff length from original
3. Refinement candidates use weighted ranking: `(2 * runtime_rank + 1 * diff_rank)`

Result is a `BestOptimization` with the candidate, context, test results, and runtime.

## 5. PR Creation (`github/`)

If a winning candidate is found, a PR is created with:
- The optimized code diff
- Performance benchmark details
- Explanation from the LLM

## Worktree Mode

When `--worktree` is enabled, optimization runs in an isolated git worktree (`code_utils/git_worktree_utils.py`). This allows parallel optimization without affecting the working tree. Changes are captured as patch files.
